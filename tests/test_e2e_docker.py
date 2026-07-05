"""End-to-end tests against a live local Docker Compose deployment.

These are pure black-box HTTP tests (httpx + pytest only, no imports from
lab_ops_accelerator) so they can run from any Python environment that can reach the
stack's published ports -- the host machine, or a throwaway container using
`host.docker.internal` when the host has no local Python.

Bring the stack up first:

    docker compose up --build -d

Then, from a Python environment with `pytest` and `httpx` installed:

    E2E=1 pytest tests/test_e2e_docker.py -v

Override endpoints with E2E_BASE_URL / E2E_PROMETHEUS_URL / E2E_GRAFANA_URL if not
running against plain localhost (e.g. from a helper container, point these at
http://host.docker.internal:<port> instead).
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
PROMETHEUS_URL = os.environ.get("E2E_PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL = os.environ.get("E2E_GRAFANA_URL", "http://localhost:3000")

pytestmark = pytest.mark.skipif(
    os.environ.get("E2E") != "1",
    reason="set E2E=1 to run against a live `docker compose up` stack",
)


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
        yield c


def _specimen_event(**overrides) -> dict:
    event = {
        "specimen_id": f"E2E-{uuid.uuid4().hex[:10]}",
        "patient_id": "PAT-E2E",
        "order_id": f"ORD-{uuid.uuid4().hex[:8]}",
        "test_code": "NIPT-PANORAMA",
        "collection_timestamp": "2024-03-15T08:00:00Z",
        "received_timestamp": "2024-03-15T09:00:00Z",
        "tube_type": "EDTA",
        "volume_ml": 0.3,
        "temperature_c": 4.0,
        "exception_flags": ["insufficient_volume", "below_minimum_threshold"],
        "raw_lims_payload": {"qc_status": "FAILED", "qc_flags": ["INSUFF_VOL"]},
    }
    event.update(overrides)
    return {"specimen_event": event}


class TestHealthAndReadiness:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ready_reports_real_dependency_state(self, client):
        resp = client.get("/v1/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["database"] == "ok"
        assert body["knowledge_base_seeded"] is True
        assert body["protocol_count"] >= 1
        assert body["llm_provider_configured"] is True

    def test_metrics_exposes_expected_series(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        for series in [
            "exceptions_processed_total",
            "hitl_rate",
            "supervisor_overrides_total",
            "exception_resolution_seconds",
        ]:
            assert series in body


class TestRequestValidation:
    def test_process_missing_required_fields_returns_422(self, client):
        resp = client.post("/v1/process", json={"specimen_event": {"specimen_id": "X"}})
        assert resp.status_code == 422

    def test_resume_unknown_thread_returns_404(self, client):
        resp = client.post("/v1/resume", json={
            "thread_id": f"spec-{uuid.uuid4().hex[:12]}",
            "decision": "reject",
            "rationale": "n/a",
            "reviewer_id": "e2e",
        })
        assert resp.status_code == 404

    def test_resume_invalid_decision_returns_422(self, client):
        resp = client.post("/v1/resume", json={
            "thread_id": f"spec-{uuid.uuid4().hex[:12]}",
            "decision": "not_a_real_disposition",
            "rationale": "n/a",
            "reviewer_id": "e2e",
        })
        assert resp.status_code == 422


class TestSpecimenExceptionWorkflow:
    def test_process_returns_well_formed_response(self, client):
        resp = client.post("/v1/process", json=_specimen_event())
        assert resp.status_code == 200
        body = resp.json()
        assert body["thread_id"].startswith("spec-")
        assert body["status"] in {"resolved", "pending_review"}
        if body["status"] == "resolved":
            assert body["disposition"] is not None
            assert body["notification_sent"] is True
            assert 0.0 <= body["confidence"] <= 1.0
        else:
            assert body["agent_recommendation"] is not None
            assert 0.0 <= body["confidence"] <= 1.0
            assert body["review_url"] == f"/v1/review/{body['thread_id']}"

    def test_full_hitl_round_trip(self, client):
        """Drive a specimen event to pending_review, then resolve it as a supervisor.

        Confidence is a real model judgment call, not something the test controls
        directly, so this retries a few distinct low-confidence-prone payloads
        (varying exception flags) until one lands in pending_review.
        """
        candidate_flags = [
            ["insufficient_volume", "below_minimum_threshold"],
            ["hemolysis", "ambiguous_grade"],
            ["temperature_excursion"],
            ["contamination", "iv_fluid_suspected"],
            ["labeling_error"],
        ]
        thread_id = None
        for flags in candidate_flags:
            resp = client.post("/v1/process", json=_specimen_event(exception_flags=flags))
            assert resp.status_code == 200
            body = resp.json()
            if body["status"] == "pending_review":
                thread_id = body["thread_id"]
                break

        if thread_id is None:
            pytest.skip("model auto-resolved every attempt with high confidence; no HITL case to resume")

        resume_resp = client.post("/v1/resume", json={
            "thread_id": thread_id,
            "decision": "reject",
            "rationale": "Supervisor confirms rejection is appropriate for this exception.",
            "reviewer_id": "e2e-supervisor",
        })
        assert resume_resp.status_code == 200
        resume_body = resume_resp.json()
        assert resume_body["thread_id"] == thread_id
        assert resume_body["status"] == "resolved"
        assert resume_body["final_disposition"] == "reject"
        assert resume_body["notification_sent"] is True

        # A resolved thread cannot be resumed a second time.
        second_resume = client.post("/v1/resume", json={
            "thread_id": thread_id,
            "decision": "reject",
            "rationale": "duplicate resume attempt",
            "reviewer_id": "e2e-supervisor",
        })
        assert second_resume.status_code == 404

    def test_supervisor_override_is_tracked_in_metrics(self, client):
        """Force an override (resume with a decision different from the recommendation)
        and confirm supervisor_overrides_total increases."""
        resp = client.post("/v1/process", json=_specimen_event(
            exception_flags=["clotted", "visible_clot"],
            volume_ml=3.0,
            raw_lims_payload={"qc_status": "FAILED", "qc_flags": ["CLOTTED"]},
        ))
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] != "pending_review":
            pytest.skip("model auto-resolved with high confidence; no supervisor decision to override")

        before = float(_metric_value(client, "supervisor_overrides_total"))

        # Deliberately pick a decision unlikely to match the agent's own recommendation.
        override_decision = "accept_with_notation" if body["agent_recommendation"] != "accept_with_notation" else "escalate"
        resume_resp = client.post("/v1/resume", json={
            "thread_id": body["thread_id"],
            "decision": override_decision,
            "rationale": "Supervisor override for e2e coverage.",
            "reviewer_id": "e2e-supervisor",
        })
        assert resume_resp.status_code == 200

        after = float(_metric_value(client, "supervisor_overrides_total"))
        assert after == before + 1


def _metric_value(client: httpx.Client, metric_name: str) -> str:
    body = client.get("/metrics").text
    for line in body.splitlines():
        if line.startswith(metric_name + " "):
            return line.split()[-1]
    raise AssertionError(f"metric {metric_name} not found in /metrics output")


def _resolve(client: httpx.Client, body: dict) -> dict:
    """If /v1/process returned pending_review, resolve it via /v1/resume so every
    category-coverage case reaches a final disposition, not just an open thread."""
    if body["status"] == "resolved":
        return body
    resume_resp = client.post("/v1/resume", json={
        "thread_id": body["thread_id"],
        "decision": body["agent_recommendation"],
        "rationale": "Supervisor concurs with the agent's recommendation.",
        "reviewer_id": "e2e-supervisor",
    })
    assert resume_resp.status_code == 200
    resolved = resume_resp.json()
    assert resolved["status"] == "resolved"
    assert resolved["notification_sent"] is True
    return resolved


# One representative specimen event per exception category the intake classifier
# and protocol knowledge base support -- see samples/protocols/*.json and
# graph/state.py::ExceptionType. Exercises every category through the real,
# deployed /v1/process endpoint rather than just a couple of hand-picked flags.
EXCEPTION_CATEGORY_CASES = [
    pytest.param(
        {
            "test_code": "NIPT-PANORAMA",
            "tube_type": "EDTA",
            "volume_ml": 0.3,
            "temperature_c": 4.0,
            "exception_flags": ["insufficient_volume"],
            "raw_lims_payload": {"qc_flags": ["INSUFF_VOL"], "received_volume_ul": 300, "minimum_required_ul": 500},
        },
        id="insufficient_volume",
    ),
    pytest.param(
        {
            "test_code": "HORIZON-CARRIER",
            "tube_type": "SST",
            "volume_ml": 5.0,
            "temperature_c": 4.0,
            "exception_flags": ["wrong_tube_type"],
            "raw_lims_payload": {"required_tube": "EDTA", "received_tube": "SST"},
        },
        id="wrong_tube",
    ),
    pytest.param(
        {
            "test_code": "SIGNATERA-MRD",
            "tube_type": "EDTA",
            "volume_ml": 8.5,
            "temperature_c": 4.0,
            "exception_flags": ["severe_hemolysis", "grade_3_plus"],
            "raw_lims_payload": {"hemolysis_index": 4, "visual_inspection": "dark_red"},
        },
        id="hemolysis",
    ),
    pytest.param(
        {
            "test_code": "BASIC-METABOLIC-PANEL",
            "tube_type": "SST",
            "volume_ml": 5.0,
            "temperature_c": 4.0,
            "exception_flags": ["lipemia", "turbid_serum"],
            "raw_lims_payload": {"visual_inspection": "markedly_turbid"},
        },
        id="lipemia",
    ),
    pytest.param(
        {
            "test_code": "NIPT-PANORAMA",
            "tube_type": "EDTA",
            "volume_ml": 10.0,
            "temperature_c": 28.0,
            "exception_flags": ["temperature_excursion", "transit_delay"],
            "raw_lims_payload": {"max_temp_logged": 28.2, "excursion_duration_hours": 6},
        },
        id="temperature_excursion",
    ),
    pytest.param(
        {
            "test_code": "HORIZON-CARRIER",
            "tube_type": "EDTA",
            "volume_ml": 6.0,
            "temperature_c": 4.0,
            "exception_flags": ["clotted_specimen"],
            "raw_lims_payload": {"visual_inspection": "clot_visible", "inversion_count": 0},
        },
        id="clotted",
    ),
    pytest.param(
        {
            "test_code": "BLOOD-CULTURE",
            "tube_type": "SST",
            "volume_ml": 4.0,
            "temperature_c": 4.0,
            "exception_flags": ["contamination_suspected", "iv_fluid_suspected"],
            "raw_lims_payload": {"draw_site": "active_iv_line", "visual_inspection": "diluted"},
        },
        id="contamination",
    ),
    pytest.param(
        {
            "test_code": "NIPT-PANORAMA",
            "tube_type": "EDTA",
            "volume_ml": 5.0,
            "temperature_c": 4.0,
            "exception_flags": ["labeling_mismatch", "missing_second_identifier"],
            "raw_lims_payload": {"label_status": "identifier_mismatch"},
        },
        id="labeling_error",
    ),
]


class TestAllExceptionCategories:
    """Drives every supported exception category through the real, deployed
    /v1/process endpoint (and /v1/resume when the model requests HITL), confirming
    the full pipeline -- LLM classification, RAG protocol retrieval, routing,
    Postgres-checkpointed HITL, and MCP notification dispatch -- completes for
    each category without error and reaches a final disposition."""

    @pytest.mark.parametrize("overrides", EXCEPTION_CATEGORY_CASES)
    def test_category_resolves_end_to_end(self, client, overrides):
        resp = client.post("/v1/process", json=_specimen_event(**overrides))
        assert resp.status_code == 200
        body = resp.json()
        assert body["thread_id"].startswith("spec-")
        assert body["status"] in {"resolved", "pending_review"}

        resolved = _resolve(client, body)
        assert resolved["notification_sent"] is True
        final_disposition = resolved.get("disposition") or resolved.get("final_disposition")
        assert final_disposition in {
            "retest_required",
            "reject",
            "accept_with_notation",
            "escalate",
        }


class TestObservabilityStack:
    def test_prometheus_is_reachable(self):
        resp = httpx.get(f"{PROMETHEUS_URL}/-/healthy", timeout=10.0)
        assert resp.status_code == 200

    def test_grafana_is_reachable(self):
        resp = httpx.get(f"{GRAFANA_URL}/api/health", timeout=10.0)
        assert resp.status_code == 200

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lab_ops_guardian.graph.state import (
    Disposition,
    ExceptionType,
    SpecimenEvent,
    WorkflowState,
)
from lab_ops_guardian.nodes.exception_router import route_exception
from lab_ops_guardian.nodes.intake_classifier import classify_intake


def _make_state(exception_flags=None, volume_ml=5.0, temp_c=4.0, test_code="NIPT-PANORAMA"):
    return WorkflowState(
        specimen_event=SpecimenEvent(
            specimen_id="TEST-001",
            patient_id="PAT-TEST",
            order_id="ORD-TEST",
            test_code=test_code,
            collection_timestamp="2024-01-15T08:00:00Z",
            received_timestamp="2024-01-15T10:00:00Z",
            tube_type="EDTA",
            volume_ml=volume_ml,
            temperature_c=temp_c,
            exception_flags=exception_flags or [],
        )
    )


def _mock_bedrock_classify(exception_type: str):
    mock = MagicMock()
    mock.invoke_model.return_value = {
        "body": MagicMock(
            read=lambda: json.dumps({
                "content": [{"text": json.dumps({"exception_type": exception_type, "reasoning": "mock"})}],
                "usage": {"input_tokens": 100, "output_tokens": 30},
            }).encode()
        )
    }
    return mock


def _mock_bedrock_route(disposition: str, confidence: float):
    mock = MagicMock()
    mock.invoke_model.return_value = {
        "body": MagicMock(
            read=lambda: json.dumps({
                "content": [{"text": json.dumps({"disposition": disposition, "confidence": confidence, "reasoning": "mock"})}],
                "usage": {"input_tokens": 150, "output_tokens": 40},
            }).encode()
        )
    }
    return mock


class TestIntakeClassifier:
    def test_classifies_insufficient_volume(self):
        state = _make_state(exception_flags=["insufficient_volume"], volume_ml=0.3)
        with patch("boto3.client", return_value=_mock_bedrock_classify("insufficient_volume")):
            result = classify_intake(state)
        assert result.exception_type == ExceptionType.INSUFFICIENT_VOLUME
        assert result.prompt_tokens > 0

    def test_classifies_wrong_tube(self):
        state = _make_state(exception_flags=["wrong_tube_type"])
        with patch("boto3.client", return_value=_mock_bedrock_classify("wrong_tube")):
            result = classify_intake(state)
        assert result.exception_type == ExceptionType.WRONG_TUBE

    def test_falls_back_to_unknown_on_bad_json(self):
        mock = MagicMock()
        mock.invoke_model.return_value = {
            "body": MagicMock(
                read=lambda: json.dumps({
                    "content": [{"text": "not valid json at all"}],
                    "usage": {"input_tokens": 50, "output_tokens": 10},
                }).encode()
            )
        }
        state = _make_state()
        with patch("boto3.client", return_value=mock):
            result = classify_intake(state)
        assert result.exception_type == ExceptionType.UNKNOWN


class TestExceptionRouter:
    def test_auto_routes_high_confidence(self):
        state = _make_state()
        state = state.model_copy(update={
            "exception_type": ExceptionType.INSUFFICIENT_VOLUME,
            "protocol_id": "SOP-LAB-047",
            "protocol_text": "Insufficient volume: request recollection.",
            "qc_flags": [],
        })
        with patch("boto3.client", return_value=_mock_bedrock_route("retest_required", 0.95)):
            result = route_exception(state)
        assert result.recommended_disposition == Disposition.RETEST_REQUIRED
        assert result.requires_human_review is False
        assert result.confidence == 0.95

    def test_triggers_hitl_below_threshold(self):
        state = _make_state()
        state = state.model_copy(update={
            "exception_type": ExceptionType.HEMOLYSIS,
            "protocol_id": "SOP-LAB-012",
            "protocol_text": "Hemolysis ambiguous grade — see protocol.",
            "qc_flags": [],
        })
        with patch("boto3.client", return_value=_mock_bedrock_route("escalate", 0.55)):
            with patch("lab_ops_guardian.nodes.exception_router.interrupt"):
                result = route_exception(state)
        assert result.requires_human_review is True
        assert result.confidence == 0.55

    def test_confidence_at_exact_threshold_does_not_trigger_hitl(self):
        state = _make_state()
        state = state.model_copy(update={
            "exception_type": ExceptionType.CLOTTED,
            "protocol_id": "SOP-LAB-031",
            "protocol_text": "Clotted specimen: retest.",
            "qc_flags": [],
        })
        with patch("boto3.client", return_value=_mock_bedrock_route("retest_required", 0.80)):
            result = route_exception(state)
        assert result.requires_human_review is False

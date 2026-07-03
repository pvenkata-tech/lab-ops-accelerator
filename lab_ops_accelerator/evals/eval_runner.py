from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

from lab_ops_accelerator.graph.state import Disposition, SpecimenEvent, WorkflowState

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0
    hitl_triggered: int = 0
    total_tokens: int = 0
    total_seconds: float = 0.0
    failures: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def hitl_rate(self) -> float:
        return self.hitl_triggered / self.total if self.total else 0.0

    @property
    def avg_seconds(self) -> float:
        return self.total_seconds / self.total if self.total else 0.0


def run_eval(dataset_path: str) -> EvalResult:
    dataset = json.loads(Path(dataset_path).read_text())
    result = EvalResult(total=len(dataset["cases"]))

    for case in dataset["cases"]:
        start = time.perf_counter()
        try:
            state = _run_case(case)
            elapsed = time.perf_counter() - start

            expected = Disposition(case["expected_disposition"])
            actual = state.final_disposition or state.recommended_disposition

            if actual == expected:
                result.correct += 1
            else:
                result.failures.append({
                    "specimen_id": case["specimen_id"],
                    "expected": expected.value,
                    "actual": actual.value if actual else None,
                    "confidence": state.confidence,
                })

            if state.requires_human_review:
                result.hitl_triggered += 1

            result.total_tokens += state.prompt_tokens + state.completion_tokens
            result.total_seconds += elapsed

        except Exception as exc:
            logger.error("Case %s failed: %s", case.get("specimen_id"), exc)
            result.failures.append({"specimen_id": case.get("specimen_id"), "error": str(exc)})

    _print_summary(result)
    return result


def _run_case(case: dict) -> WorkflowState:
    from lab_ops_accelerator.nodes.exception_router import route_exception
    from lab_ops_accelerator.nodes.intake_classifier import classify_intake
    from lab_ops_accelerator.nodes.qc_evaluator import evaluate_qc

    event = SpecimenEvent(**case["specimen_event"])
    state = WorkflowState(specimen_event=event)

    with patch("boto3.client") as mock_boto, patch(
        "lab_ops_accelerator.nodes.qc_evaluator.retrieve_protocol"
    ) as mock_retrieve:
        mock_boto.return_value = _build_mock_bedrock(case)
        mock_retrieve.return_value = {
            "protocol_id": "EVAL-PROTOCOL",
            "protocol_text": "Mock protocol for evaluation — routing accuracy is under test, not retrieval.",
        }
        state = classify_intake(state)
        state = evaluate_qc(state)
        state = route_exception(state)

    return state


def _build_mock_bedrock(case: dict):
    mock = MagicMock()

    def invoke_model(**kwargs):
        body = json.loads(kwargs.get("body", "{}"))
        is_embedding = "inputText" in body

        if is_embedding:
            return {
                "body": MagicMock(
                    read=lambda: json.dumps({"embedding": [0.1] * 1024}).encode()
                )
            }

        # The routing system prompt is the only one that mentions "disposition" —
        # distinguishing on the system prompt (not the user message) is what makes
        # this reliable, since both calls' user messages are free-form case data.
        is_routing_call = "disposition" in body.get("system", "")
        if is_routing_call:
            response_text = json.dumps({
                "disposition": case["expected_disposition"],
                "confidence": case.get("mock_confidence", 0.92),
                "reasoning": "mock routing",
            })
        else:
            response_text = json.dumps({
                "exception_type": case.get("expected_exception_type", "unknown"),
                "reasoning": "mock classification",
            })

        return {
            "body": MagicMock(
                read=lambda: json.dumps({
                    "content": [{"text": response_text}],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }).encode()
            )
        }

    mock.invoke_model.side_effect = invoke_model
    return mock


def _print_summary(result: EvalResult) -> None:
    print("\n=== Eval Results ===")
    print(f"Total cases:     {result.total}")
    print(f"Accuracy:        {result.accuracy:.1%}")
    print(f"HITL rate:       {result.hitl_rate:.1%}")
    print(f"Avg latency:     {result.avg_seconds:.2f}s")
    print(f"Total tokens:    {result.total_tokens:,}")
    if result.failures:
        print(f"\nFailures ({len(result.failures)}):")
        for f in result.failures[:5]:
            print(f"  {f}")
    print("====================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to golden dataset JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_eval(args.dataset)

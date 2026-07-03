from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lab_ops_accelerator.evals.eval_runner import run_eval


GOLDEN_DATASET = Path(__file__).parent.parent / "samples" / "golden_dataset.json"
ACCURACY_FLOOR = 0.80


@pytest.mark.skipif(
    not GOLDEN_DATASET.exists(),
    reason="golden_dataset.json not found",
)
def test_eval_accuracy_above_floor():
    result = run_eval(str(GOLDEN_DATASET))
    assert result.accuracy >= ACCURACY_FLOOR, (
        f"Eval accuracy {result.accuracy:.1%} is below floor {ACCURACY_FLOOR:.1%}. "
        f"Failures: {result.failures}"
    )


@pytest.mark.skipif(
    not GOLDEN_DATASET.exists(),
    reason="golden_dataset.json not found",
)
def test_eval_hitl_rate_reasonable():
    result = run_eval(str(GOLDEN_DATASET))
    assert result.hitl_rate <= 0.40, (
        f"HITL rate {result.hitl_rate:.1%} is unexpectedly high — check confidence threshold or prompt."
    )

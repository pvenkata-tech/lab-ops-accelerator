from __future__ import annotations

import json
import logging

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.graph.state import Disposition, WorkflowState
from lab_ops_accelerator.llm import get_llm_client
from lab_ops_accelerator.llm.parsing import strip_code_fence

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a lab operations routing agent. Given a specimen exception and the retrieved \
handling protocol, determine the correct disposition and your confidence (0.0–1.0).

Dispositions:
- retest_required: collect a new specimen
- reject: specimen cannot be processed; no retest needed
- accept_with_notation: process with a quality note in the report
- escalate: requires pathologist or medical director review

Respond with valid JSON only:
{
  "disposition": "<disposition>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}
"""


def route_exception(state: WorkflowState) -> WorkflowState:
    settings = get_settings()

    user_message = (
        f"Exception type: {state.exception_type.value if state.exception_type else 'unknown'}\n"
        f"QC flags: {', '.join(state.qc_flags) or 'none'}\n"
        f"Protocol ID: {state.protocol_id}\n"
        f"Protocol text:\n{state.protocol_text}\n"
        f"Classification reasoning: {state.classification_reasoning}"
    )

    client = get_llm_client(settings)
    # 1024, not 256: models with extended/adaptive thinking spend part of the
    # token budget on hidden reasoning before the visible JSON answer, and a
    # too-small budget truncates the JSON mid-object.
    response = client.invoke(_SYSTEM_PROMPT, user_message, max_tokens=1024)

    try:
        parsed = json.loads(strip_code_fence(response.text))
        disposition = Disposition(parsed["disposition"])
        confidence = float(parsed["confidence"])
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse routing response: %s — raw: %s", exc, response.text)
        disposition = Disposition.ESCALATE
        confidence = 0.0
        reasoning = response.text

    requires_hitl = confidence < settings.hitl_confidence_threshold

    return state.model_copy(update={
        "recommended_disposition": disposition,
        "confidence": confidence,
        "routing_reasoning": reasoning,
        "requires_human_review": requires_hitl,
        "prompt_tokens": state.prompt_tokens + response.input_tokens,
        "completion_tokens": state.completion_tokens + response.output_tokens,
    })

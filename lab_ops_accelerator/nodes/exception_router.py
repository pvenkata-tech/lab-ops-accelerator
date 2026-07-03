from __future__ import annotations

import json
import logging

import boto3
from langgraph.types import interrupt

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.graph.state import Disposition, WorkflowState
from lab_ops_accelerator.observability.metrics import HITL_RATE

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

    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    response = client.invoke_model(
        modelId=settings.bedrock_claude_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }),
    )

    body = json.loads(response["body"].read())
    text = body["content"][0]["text"].strip()
    usage = body.get("usage", {})

    try:
        parsed = json.loads(text)
        disposition = Disposition(parsed["disposition"])
        confidence = float(parsed["confidence"])
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse routing response: %s — raw: %s", exc, text)
        disposition = Disposition.ESCALATE
        confidence = 0.0
        reasoning = text

    requires_hitl = confidence < settings.hitl_confidence_threshold

    if requires_hitl:
        HITL_RATE.set(1)
        interrupt({
            "reason": "low_confidence",
            "agent_recommendation": disposition.value,
            "confidence": confidence,
            "protocol_id": state.protocol_id,
            "reasoning": reasoning,
        })

    return state.model_copy(update={
        "recommended_disposition": disposition,
        "confidence": confidence,
        "routing_reasoning": reasoning,
        "requires_human_review": requires_hitl,
        "prompt_tokens": state.prompt_tokens + usage.get("input_tokens", 0),
        "completion_tokens": state.completion_tokens + usage.get("output_tokens", 0),
    })

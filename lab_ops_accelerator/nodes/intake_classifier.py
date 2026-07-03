from __future__ import annotations

import json
import logging

import boto3

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.graph.state import ExceptionType, WorkflowState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a lab operations AI assistant. Your job is to classify a specimen exception \
from a LIMS event payload into one of the following exception types:

- insufficient_volume
- wrong_tube
- hemolysis
- lipemia
- temperature_excursion
- clotted
- contamination
- labeling_error
- unknown

Respond with valid JSON only:
{
  "exception_type": "<type>",
  "reasoning": "<one sentence>"
}

If the flags clearly indicate multiple issues, pick the highest-severity one.
"""


def classify_intake(state: WorkflowState) -> WorkflowState:
    settings = get_settings()
    event = state.specimen_event

    user_message = (
        f"Specimen ID: {event.specimen_id}\n"
        f"Test code: {event.test_code}\n"
        f"Tube type: {event.tube_type}\n"
        f"Volume (mL): {event.volume_ml}\n"
        f"Temperature (°C): {event.temperature_c}\n"
        f"Exception flags: {', '.join(event.exception_flags) or 'none'}\n"
        f"Raw LIMS payload: {json.dumps(event.raw_lims_payload, indent=2)}"
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
        exception_type = ExceptionType(parsed["exception_type"])
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse intake classification: %s — raw: %s", exc, text)
        exception_type = ExceptionType.UNKNOWN
        reasoning = text

    return state.model_copy(update={
        "exception_type": exception_type,
        "classification_reasoning": reasoning,
        "model_id": settings.bedrock_claude_model_id,
        "prompt_tokens": state.prompt_tokens + usage.get("input_tokens", 0),
        "completion_tokens": state.completion_tokens + usage.get("output_tokens", 0),
    })

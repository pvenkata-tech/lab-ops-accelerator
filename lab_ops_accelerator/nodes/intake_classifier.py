from __future__ import annotations

import json
import logging

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.graph.state import ExceptionType, WorkflowState
from lab_ops_accelerator.llm import get_llm_client

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

    client = get_llm_client(settings)
    response = client.invoke(_SYSTEM_PROMPT, user_message, max_tokens=256)

    try:
        parsed = json.loads(response.text)
        exception_type = ExceptionType(parsed["exception_type"])
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse intake classification: %s — raw: %s", exc, response.text)
        exception_type = ExceptionType.UNKNOWN
        reasoning = response.text

    return state.model_copy(update={
        "exception_type": exception_type,
        "classification_reasoning": reasoning,
        "model_id": response.model_id,
        "prompt_tokens": state.prompt_tokens + response.input_tokens,
        "completion_tokens": state.completion_tokens + response.output_tokens,
    })

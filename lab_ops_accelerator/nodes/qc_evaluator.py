from __future__ import annotations

import logging

from lab_ops_accelerator.graph.state import WorkflowState
from lab_ops_accelerator.rag.retriever import retrieve_protocol

logger = logging.getLogger(__name__)


def evaluate_qc(state: WorkflowState) -> WorkflowState:
    """Retrieve the applicable handling protocol for the classified exception."""
    if state.exception_type is None:
        return state.model_copy(update={"error": "Cannot evaluate QC without exception_type"})

    query = (
        f"specimen exception type: {state.exception_type.value} "
        f"tube: {state.specimen_event.tube_type} "
        f"test: {state.specimen_event.test_code}"
    )

    result = retrieve_protocol(query)

    flags: list[str] = []
    if state.specimen_event.volume_ml is not None and state.specimen_event.volume_ml < 0.5:
        flags.append("critically_low_volume")
    if (
        state.specimen_event.temperature_c is not None
        and not (2.0 <= state.specimen_event.temperature_c <= 8.0)
    ):
        flags.append("temperature_out_of_range")

    return state.model_copy(update={
        "protocol_id": result["protocol_id"],
        "protocol_text": result["protocol_text"],
        "qc_flags": flags,
    })

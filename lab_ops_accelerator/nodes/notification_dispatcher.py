from __future__ import annotations

import logging

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.graph.state import Disposition, SupervisorDecision, WorkflowState
from lab_ops_accelerator.observability.metrics import EXCEPTIONS_PROCESSED, SUPERVISOR_OVERRIDES
from lab_ops_accelerator.tools.mcp_client import call_tool

logger = logging.getLogger(__name__)


async def dispatch_notification(state: WorkflowState) -> WorkflowState:
    """Send notifications and update LIMS after disposition is determined.

    Both upstream systems are reached through their MCP servers, not bespoke SDKs — the
    orchestrator only knows tool names and arguments, so a new upstream source is a new
    MCP server, not a change here.
    """
    settings = get_settings()

    final_disposition = _resolve_final_disposition(state)
    override_occurred = _check_override(state, final_disposition)

    await _update_lims(settings, state, final_disposition)
    await _notify_ehr(settings, state, final_disposition)

    if override_occurred:
        SUPERVISOR_OVERRIDES.inc()

    EXCEPTIONS_PROCESSED.labels(
        exception_type=state.exception_type.value if state.exception_type else "unknown",
        disposition=final_disposition.value,
    ).inc()

    return state.model_copy(update={
        "final_disposition": final_disposition,
        "notification_sent": True,
        "lims_updated": True,
    })


def _resolve_final_disposition(state: WorkflowState) -> Disposition:
    if state.supervisor_decision is not None:
        return state.supervisor_decision.decision
    if state.recommended_disposition is not None:
        return state.recommended_disposition
    return Disposition.ESCALATE


def _check_override(state: WorkflowState, final: Disposition) -> bool:
    if state.supervisor_decision and state.recommended_disposition:
        return final != state.recommended_disposition
    return False


async def _update_lims(settings, state: WorkflowState, disposition: Disposition) -> None:
    try:
        await call_tool(
            settings.lims_mcp_server_url,
            "update_specimen_disposition",
            {
                "specimen_id": state.specimen_event.specimen_id,
                "order_id": state.specimen_event.order_id,
                "disposition": disposition.value,
                "protocol_applied": state.protocol_id,
                "confidence": state.confidence,
                "requires_retest": disposition == Disposition.RETEST_REQUIRED,
            },
        )
    except Exception as exc:
        logger.error("LIMS MCP update failed for %s: %s", state.specimen_event.specimen_id, exc)
        raise


async def _notify_ehr(settings, state: WorkflowState, disposition: Disposition) -> None:
    try:
        await call_tool(
            settings.ehr_mcp_server_url,
            "notify_physician",
            {
                "order_id": state.specimen_event.order_id,
                "patient_id": state.specimen_event.patient_id,
                "disposition": disposition.value,
                "message": _build_physician_message(state, disposition),
            },
        )
    except Exception as exc:
        logger.error("EHR MCP notification failed for order %s: %s", state.specimen_event.order_id, exc)
        raise


def _build_physician_message(state: WorkflowState, disposition: Disposition) -> str:
    base = (
        f"Specimen {state.specimen_event.specimen_id} for order "
        f"{state.specimen_event.order_id} ({state.specimen_event.test_code}) "
        f"could not be processed as received."
    )
    if disposition == Disposition.RETEST_REQUIRED:
        return base + " A new specimen collection is required. Please arrange recollection."
    if disposition == Disposition.REJECT:
        return base + " The specimen has been rejected. No retest is indicated per protocol."
    if disposition == Disposition.ACCEPT_WITH_NOTATION:
        return (
            base
            + " The specimen has been accepted with a quality notation. "
            "Results will include a QC comment."
        )
    return base + " This case has been escalated for medical director review."

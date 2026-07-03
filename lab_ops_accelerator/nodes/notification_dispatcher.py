from __future__ import annotations

import logging

import httpx

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.graph.state import Disposition, SupervisorDecision, WorkflowState
from lab_ops_accelerator.observability.metrics import EXCEPTIONS_PROCESSED, SUPERVISOR_OVERRIDES

logger = logging.getLogger(__name__)


def dispatch_notification(state: WorkflowState) -> WorkflowState:
    """Send notifications and update LIMS after disposition is determined."""
    settings = get_settings()

    final_disposition = _resolve_final_disposition(state)
    override_occurred = _check_override(state, final_disposition)

    _update_lims(settings, state, final_disposition)
    _notify_ehr(settings, state, final_disposition)

    if override_occurred:
        SUPERVISOR_OVERRIDES.inc()

    EXCEPTIONS_PROCESSED.inc()

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


def _update_lims(settings, state: WorkflowState, disposition: Disposition) -> None:
    payload = {
        "specimen_id": state.specimen_event.specimen_id,
        "order_id": state.specimen_event.order_id,
        "disposition": disposition.value,
        "protocol_applied": state.protocol_id,
        "confidence": state.confidence,
        "requires_retest": disposition == Disposition.RETEST_REQUIRED,
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                f"{settings.lims_api_base_url}/specimens/{state.specimen_event.specimen_id}/disposition",
                json=payload,
                headers={"X-API-Key": settings.lims_api_key},
            )
    except httpx.HTTPError as exc:
        logger.error("LIMS update failed for %s: %s", state.specimen_event.specimen_id, exc)
        raise


def _notify_ehr(settings, state: WorkflowState, disposition: Disposition) -> None:
    message = _build_physician_message(state, disposition)
    payload = {
        "order_id": state.specimen_event.order_id,
        "patient_id": state.specimen_event.patient_id,
        "notification_type": "specimen_exception",
        "disposition": disposition.value,
        "message": message,
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                settings.ehr_webhook_url,
                json=payload,
                headers={"X-API-Key": settings.ehr_api_key},
            )
    except httpx.HTTPError as exc:
        logger.error("EHR notification failed for order %s: %s", state.specimen_event.order_id, exc)
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

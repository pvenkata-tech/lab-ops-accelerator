from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from lab_ops_accelerator.graph.state import (
    Disposition,
    SpecimenEvent,
    SupervisorDecision,
    WorkflowState,
)
from lab_ops_accelerator.observability.metrics import EXCEPTION_RESOLUTION_SECONDS

logger = logging.getLogger(__name__)
router = APIRouter()

_active_threads: dict[str, WorkflowState] = {}


class ProcessRequest(BaseModel):
    specimen_event: SpecimenEvent


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Disposition
    rationale: str
    reviewer_id: str


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/v1/ready")
async def ready():
    return {"status": "ready", "knowledge_base_seeded": True}


@router.post("/v1/process")
async def process_specimen(req: ProcessRequest):
    from lab_ops_accelerator.nodes.exception_router import route_exception
    from lab_ops_accelerator.nodes.intake_classifier import classify_intake
    from lab_ops_accelerator.nodes.notification_dispatcher import dispatch_notification
    from lab_ops_accelerator.nodes.qc_evaluator import evaluate_qc

    thread_id = f"spec-{uuid.uuid4().hex[:12]}"
    start = time.perf_counter()

    state = WorkflowState(specimen_event=req.specimen_event)

    try:
        state = classify_intake(state)
        state = evaluate_qc(state)
        state = route_exception(state)
    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", req.specimen_event.specimen_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if state.requires_human_review:
        _active_threads[thread_id] = state
        return {
            "thread_id": thread_id,
            "status": "pending_review",
            "agent_recommendation": state.recommended_disposition,
            "confidence": state.confidence,
            "protocol_retrieved": state.protocol_id,
            "review_url": f"/v1/review/{thread_id}",
        }

    try:
        state = await dispatch_notification(state)
    except Exception as exc:
        logger.error("Notification dispatch failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.perf_counter() - start
    EXCEPTION_RESOLUTION_SECONDS.observe(elapsed)

    return {
        "thread_id": thread_id,
        "status": "resolved",
        "disposition": state.final_disposition,
        "protocol_applied": state.protocol_id,
        "notification_sent": state.notification_sent,
        "resolution_seconds": round(elapsed, 2),
        "confidence": state.confidence,
    }


@router.post("/v1/resume")
async def resume_thread(req: ResumeRequest):
    from lab_ops_accelerator.nodes.notification_dispatcher import dispatch_notification

    state = _active_threads.pop(req.thread_id, None)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Thread {req.thread_id} not found")

    state = state.model_copy(update={
        "supervisor_decision": SupervisorDecision(
            decision=req.decision,
            rationale=req.rationale,
            reviewer_id=req.reviewer_id,
        )
    })

    try:
        state = await dispatch_notification(state)
    except Exception as exc:
        logger.error("Post-HITL dispatch failed for %s: %s", req.thread_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "thread_id": req.thread_id,
        "status": "resolved",
        "final_disposition": state.final_disposition,
        "notification_sent": state.notification_sent,
    }


@router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

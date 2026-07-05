from __future__ import annotations

import asyncio
import logging
import time
import uuid

import psycopg
from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.graph.state import Disposition, SpecimenEvent, WorkflowState
from lab_ops_accelerator.llm import get_llm_client
from lab_ops_accelerator.observability.metrics import EXCEPTION_RESOLUTION_SECONDS

logger = logging.getLogger(__name__)
router = APIRouter()


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


def _check_ready(settings) -> dict:
    checks: dict = {}

    try:
        with psycopg.connect(settings.checkpoint_database_url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"

    try:
        with psycopg.connect(settings.checkpoint_database_url, connect_timeout=3) as conn:
            row = conn.execute("SELECT count(*) FROM protocols").fetchone()
        count = row[0] if row else 0
        checks["knowledge_base_seeded"] = count > 0
        checks["protocol_count"] = count
    except Exception as exc:
        checks["knowledge_base_seeded"] = False
        checks["knowledge_base_error"] = str(exc)

    try:
        get_llm_client(settings)
        checks["llm_provider_configured"] = True
    except Exception as exc:
        checks["llm_provider_configured"] = False
        checks["llm_error"] = str(exc)

    return checks


@router.get("/v1/ready")
async def ready():
    settings = get_settings()
    checks = await asyncio.to_thread(_check_ready, settings)

    healthy = checks.get("database") == "ok" and checks.get("llm_provider_configured", False)
    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if healthy else "not_ready", **checks},
    )


@router.post("/v1/process")
async def process_specimen(req: ProcessRequest, request: Request):
    graph = request.app.state.graph
    thread_id = f"spec-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    start = time.perf_counter()

    initial_state = WorkflowState(specimen_event=req.specimen_event)

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", req.specimen_event.specimen_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        return {
            "thread_id": thread_id,
            "status": "pending_review",
            "agent_recommendation": payload["agent_recommendation"],
            "confidence": payload["confidence"],
            "protocol_retrieved": payload["protocol_id"],
            "review_url": f"/v1/review/{thread_id}",
        }

    elapsed = time.perf_counter() - start
    EXCEPTION_RESOLUTION_SECONDS.observe(elapsed)

    return {
        "thread_id": thread_id,
        "status": "resolved",
        "disposition": result["final_disposition"],
        "protocol_applied": result["protocol_id"],
        "notification_sent": result["notification_sent"],
        "resolution_seconds": round(elapsed, 2),
        "confidence": result["confidence"],
    }


@router.post("/v1/resume")
async def resume_thread(req: ResumeRequest, request: Request):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": req.thread_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=404, detail=f"Thread {req.thread_id} not found or not pending review")

    resume_value = {
        "decision": req.decision.value,
        "rationale": req.rationale,
        "reviewer_id": req.reviewer_id,
    }

    try:
        result = await graph.ainvoke(Command(resume=resume_value), config=config)
    except Exception as exc:
        logger.error("Post-HITL dispatch failed for %s: %s", req.thread_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "thread_id": req.thread_id,
        "status": "resolved",
        "final_disposition": result["final_disposition"],
        "notification_sent": result["notification_sent"],
    }


@router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

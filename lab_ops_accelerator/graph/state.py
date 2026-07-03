from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ExceptionType(str, Enum):
    INSUFFICIENT_VOLUME = "insufficient_volume"
    WRONG_TUBE = "wrong_tube"
    HEMOLYSIS = "hemolysis"
    LIPEMIA = "lipemia"
    TEMPERATURE_EXCURSION = "temperature_excursion"
    CLOTTED = "clotted"
    CONTAMINATION = "contamination"
    LABELING_ERROR = "labeling_error"
    UNKNOWN = "unknown"


class Disposition(str, Enum):
    RETEST_REQUIRED = "retest_required"
    REJECT = "reject"
    ACCEPT_WITH_NOTATION = "accept_with_notation"
    ESCALATE = "escalate"
    PENDING_REVIEW = "pending_review"


class SpecimenEvent(BaseModel):
    specimen_id: str
    patient_id: str
    order_id: str
    test_code: str
    collection_timestamp: str
    received_timestamp: str
    tube_type: str
    volume_ml: Optional[float] = None
    temperature_c: Optional[float] = None
    exception_flags: list[str] = Field(default_factory=list)
    raw_lims_payload: dict = Field(default_factory=dict)


class SupervisorDecision(BaseModel):
    decision: Disposition
    rationale: str
    reviewer_id: str


class WorkflowState(BaseModel):
    """Mutable state threaded through the LangGraph workflow."""

    # Input
    specimen_event: SpecimenEvent

    # Populated by intake classifier
    exception_type: Optional[ExceptionType] = None
    classification_reasoning: Optional[str] = None

    # Populated by QC evaluator
    protocol_id: Optional[str] = None
    protocol_text: Optional[str] = None
    qc_flags: list[str] = Field(default_factory=list)

    # Populated by exception router
    recommended_disposition: Optional[Disposition] = None
    confidence: Optional[float] = None
    routing_reasoning: Optional[str] = None

    # HITL
    requires_human_review: bool = False
    supervisor_decision: Optional[SupervisorDecision] = None

    # Final output
    final_disposition: Optional[Disposition] = None
    notification_sent: bool = False
    lims_updated: bool = False
    error: Optional[str] = None

    # Audit
    model_id: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    resolution_seconds: Optional[float] = None

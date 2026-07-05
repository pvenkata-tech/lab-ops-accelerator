from __future__ import annotations

import logging

from langgraph.types import interrupt

from lab_ops_accelerator.graph.state import SupervisorDecision, WorkflowState
from lab_ops_accelerator.observability.metrics import HITL_RATE

logger = logging.getLogger(__name__)


def human_review(state: WorkflowState) -> WorkflowState:
    """Pause the graph for supervisor input.

    This node does no expensive work before calling interrupt() — on resume, LangGraph
    re-runs the node function from the top, so anything placed before the interrupt
    call (like an LLM invocation) would otherwise be repeated. The LLM call for the
    routing recommendation lives in exception_router, upstream of this node.
    """
    HITL_RATE.set(1)

    decision = interrupt({
        "reason": "low_confidence",
        "agent_recommendation": state.recommended_disposition.value if state.recommended_disposition else None,
        "confidence": state.confidence,
        "protocol_id": state.protocol_id,
        "reasoning": state.routing_reasoning,
    })

    return state.model_copy(update={
        "supervisor_decision": SupervisorDecision(**decision),
    })

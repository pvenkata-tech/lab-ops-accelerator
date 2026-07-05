from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from lab_ops_accelerator.graph.state import WorkflowState
from lab_ops_accelerator.nodes.exception_router import route_exception
from lab_ops_accelerator.nodes.human_review import human_review
from lab_ops_accelerator.nodes.intake_classifier import classify_intake
from lab_ops_accelerator.nodes.notification_dispatcher import dispatch_notification
from lab_ops_accelerator.nodes.qc_evaluator import evaluate_qc

logger = logging.getLogger(__name__)


def _should_go_to_hitl(state: WorkflowState) -> str:
    if state.requires_human_review:
        return "hitl"
    return "dispatch"


def build_graph(checkpointer: Any) -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("classify", classify_intake)
    graph.add_node("evaluate_qc", evaluate_qc)
    graph.add_node("route", route_exception)
    graph.add_node("human_review", human_review)
    graph.add_node("dispatch", dispatch_notification)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "evaluate_qc")
    graph.add_edge("evaluate_qc", "route")
    graph.add_conditional_edges(
        "route",
        _should_go_to_hitl,
        {"hitl": "human_review", "dispatch": "dispatch"},
    )
    graph.add_edge("human_review", "dispatch")
    graph.add_edge("dispatch", END)

    return graph.compile(checkpointer=checkpointer)

"""RCA LangGraph: classify input (plan JSON vs log text), then branch-specific agents."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agents.classify import classify_input_agent
from agents.extracted_signals_summary_agent import extracted_signals_summary_agent
from agents.log_analyser_agent import log_analyser_agent
from agents.log_pipeline_agent import log_signal_pipeline_agent
from agents.plan_analysis_agent import plan_analysis_agent
from agents.plan_narrative_agent import plan_narrative_agent
from agents.plan_preprocess_agent import plan_preprocess_agent
from agents.recommendation_agent import recommendation_agent
from agents.root_cause_agent import root_cause_agent
from graphs.node_logging import with_step_logging
from graphs.state import GraphState


def get_initial_state(raw_log: str = "", *, stop_after_preprocess: bool = False) -> dict[str, Any]:
    """Full initial state compatible with :class:`GraphState` (all keys present)."""
    return {
        "stop_after_preprocess": stop_after_preprocess,
        "raw_log": raw_log,
        "preprocess_error_hits": [],
        "preprocess_warning_hits": [],
        "preprocess_summary": "",
        "preprocess_llm_digest": "",
        "input_kind": "log",
        "plan_preprocess_errors": [],
        "plan_preprocess_warnings": [],
        "plan_extracted_warnings": [],
        "plan_extracted_errors": [],
        "plan_extracted_messages": [],
        "plan_skeleton_outline": "",
        "plan_skeleton_summary": "",
        "plan_messages_list": [],
        "plan_json": "",
        "parsed_log": {},
        "detected_patterns": [],
        "root_cause": "",
        "confidence": 0.0,
        "severity": "",
        "recommendation": "",
        "recommendation_summary": "",
        "findings_summary": "",
        "history": [],
        "plan_feasible": None,
        "plan_warnings_list": [],
        "plan_errors_list": [],
        "plan_deep_analysis": "",
        "log_errors_analysis": "",
        "log_warnings_analysis": "",
        "log_analysis_insight": "",
        "log_analysis_plan_failure": "",
        "log_analysis_recommendation": "",
        "memory_context": "",
        "memory_relevant_incidents": [],
    }


def _route_after_classify(state: GraphState) -> Literal["plan_preprocess", "log_signal_pipeline"]:
    return "plan_preprocess" if state.get("input_kind") == "plan" else "log_signal_pipeline"


def _route_after_plan_preprocess(state: GraphState) -> Literal["plan_analysis", "__end__"]:
    if state.get("stop_after_preprocess"):
        return END
    return "plan_analysis"


def _route_after_log_signal(state: GraphState) -> Literal["log_analyser", "__end__"]:
    if state.get("stop_after_preprocess"):
        return END
    return "log_analyser"


def build_graph() -> StateGraph:
    """Build ``StateGraph(GraphState)`` with distinct agent nodes per branch.

    **Plan (JSON):** ``classify_input`` → ``plan_preprocess`` → ``plan_analysis`` (LLM) → ``plan_narrative``
    (LLM: findings, root cause, ``recommendation_summary`` + ``recommendation`` grounded in merged errors/warnings/messages and feasibility) → END.
    Preprocess-only: after ``plan_preprocess`` → END when ``stop_after_preprocess`` is true.

    **Log (text):** ``log_signal_pipeline`` → ``log_analyser`` (Claude Sonnet on extractions) → ``log_findings_summary``
    → ``root_cause`` → ``recommendation`` → END. Preprocess-only ends after ``log_signal_pipeline``.

    Set ``RCA_GRAPH_QUIET=1`` to disable per-step print logging.
    """
    g = StateGraph(GraphState)
    g.add_node("classify_input", with_step_logging("classify_input", classify_input_agent))
    g.add_node("plan_preprocess", with_step_logging("plan_preprocess", plan_preprocess_agent))
    g.add_node("plan_analysis", with_step_logging("plan_analysis", plan_analysis_agent))
    g.add_node("plan_narrative", with_step_logging("plan_narrative", plan_narrative_agent))
    g.add_node("log_signal_pipeline", with_step_logging("log_signal_pipeline", log_signal_pipeline_agent))
    g.add_node("log_analyser", with_step_logging("log_analyser", log_analyser_agent))
    g.add_node(
        "log_findings_summary",
        with_step_logging("log_findings_summary", extracted_signals_summary_agent),
    )
    g.add_node("root_cause", with_step_logging("root_cause", root_cause_agent))
    g.add_node("recommendation", with_step_logging("recommendation", recommendation_agent))

    g.add_edge(START, "classify_input")
    g.add_conditional_edges(
        "classify_input",
        _route_after_classify,
        {"plan_preprocess": "plan_preprocess", "log_signal_pipeline": "log_signal_pipeline"},
    )
    g.add_conditional_edges(
        "plan_preprocess",
        _route_after_plan_preprocess,
        {"plan_analysis": "plan_analysis", END: END},
    )
    g.add_edge("plan_analysis", "plan_narrative")
    g.add_edge("plan_narrative", END)
    g.add_conditional_edges(
        "log_signal_pipeline",
        _route_after_log_signal,
        {"log_analyser": "log_analyser", END: END},
    )
    g.add_edge("log_analyser", "log_findings_summary")
    g.add_edge("log_findings_summary", "root_cause")
    g.add_edge("root_cause", "recommendation")
    g.add_edge("recommendation", END)
    return g


def get_runnable_graph(*, with_checkpointer: bool = True):
    """Return a compiled graph ready for ``invoke`` / ``ainvoke`` / ``stream``.

    When ``with_checkpointer`` is True, uses :func:`memory.checkpoint.get_checkpointer`
    so ``thread_id`` works in ``config["configurable"]``.
    """
    builder = build_graph()
    checkpointer = None
    if with_checkpointer:
        from memory.checkpoint import get_checkpointer

        checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)

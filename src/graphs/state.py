"""Shared graph state schema for RCA (root cause analysis) workflows."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """State passed between LangGraph nodes for log analysis and RCA."""

    stop_after_preprocess: bool
    raw_log: str
    preprocess_error_hits: list[str]
    preprocess_warning_hits: list[str]
    preprocess_summary: str
    preprocess_llm_digest: str
    input_kind: str
    plan_preprocess_errors: list[str]
    plan_preprocess_warnings: list[str]
    plan_extracted_warnings: list[str]
    plan_extracted_errors: list[str]
    plan_extracted_messages: list[str]
    plan_skeleton_outline: str
    plan_skeleton_summary: str
    plan_messages_list: list[str]
    plan_json: str
    parsed_log: dict[str, Any]
    detected_patterns: Annotated[list[Any], add]
    root_cause: str
    confidence: float
    severity: str
    recommendation: str
    recommendation_summary: str
    findings_summary: str
    history: Annotated[list[str], add]
    plan_feasible: bool | None
    plan_warnings_list: list[str]
    plan_errors_list: list[str]
    plan_deep_analysis: str
    log_errors_analysis: str
    log_warnings_analysis: str
    # Log analyser (Claude Sonnet) — after pipeline, before findings summary
    log_analysis_insight: str
    log_analysis_plan_failure: str
    log_analysis_recommendation: str
    # Persistent store (``memory/store.json``): loaded before pattern detection on log path
    memory_context: str
    memory_relevant_incidents: list[dict[str, Any]]

"""FastAPI surface for the RCA LangGraph workflow."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from graphs.diagram import (
    mermaid_execution_path,
    mermaid_reference_topology,
    ordered_nodes_from_step_models,
)
from graphs.trace import ainvoke_with_trace, legend_for_input_kind
from graphs.workflow import get_initial_state, get_runnable_graph
from utils.config import get_config
from utils.health import check_ollama_connection

logger = logging.getLogger("rca.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Verify Ollama (local) or Anthropic API key (Claude) before serving when strict."""
    logging.getLogger("rca.tools").setLevel(logging.INFO)
    cfg = get_config()
    provider = str(cfg.get("llm_provider") or "ollama")

    if provider == "anthropic":
        key = str(cfg.get("anthropic_api_key") or "").strip()
        if key:
            logger.info(
                "Startup: using Anthropic Claude API (model=%s).",
                cfg.get("model"),
            )
        else:
            logger.error(
                "Startup: anthropic selected but ANTHROPIC_API_KEY is empty. "
                "Add it to rca-agent/.env (copy .env.example), or export ANTHROPIC_API_KEY. "
                "Ensure python-dotenv is installed (pip install python-dotenv). "
                "POST/PUT /analyze will fail until the key is set."
            )
            strict = os.environ.get("RCA_STRICT_ANTHROPIC", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if strict:
                logger.critical(
                    "RCA_STRICT_ANTHROPIC is enabled — aborting startup (set key or use ollama)."
                )
                sys.exit(1)
    else:
        base_url = cfg["base_url"]
        ok = check_ollama_connection(silent=True)
        if ok:
            logger.info("Startup: Ollama is reachable at %s (GET /api/tags OK).", base_url)
        else:
            logger.error(
                "Startup: cannot reach Ollama at %s — GET /api/tags failed. "
                "Check the server and OLLAMA_BASE_URL.",
                base_url,
            )
            strict = os.environ.get("RCA_STRICT_OLLAMA", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if strict:
                logger.critical(
                    "RCA_STRICT_OLLAMA is enabled — aborting startup (set to 0 to only warn)."
                )
                sys.exit(1)
            logger.warning(
                "Startup: continuing without Ollama; POST/PUT /analyze may fail until it is available."
            )
    yield


class AnalyzeRequest(BaseModel):
    log: str = Field(..., description="Raw log text to analyze")


class GraphTraceStep(BaseModel):
    step: int
    node: str
    keys: list[str] = Field(default_factory=list)
    summary: str = ""


class AnalyzeResponse(BaseModel):
    input_kind: str = Field(default="log", description="plan (JSON) or log (plain text)")
    root_cause: str
    confidence: float
    severity: str
    recommendation: str
    recommendation_summary: str = Field(
        default="",
        description="Short distilled signal the recommendation acts on (log path, recommendation step)",
    )
    findings_summary: str = Field(
        default="",
        description="Summary of findings (plan or log); log path ties to timestamps/PIDs when present",
    )
    plan_feasible: bool | None = Field(
        default=None,
        description="True if JSON plan looks feasible, False if infeasible; null for log input",
    )
    plan_warnings: list[str] = Field(default_factory=list)
    plan_errors: list[str] = Field(default_factory=list)
    plan_messages: list[str] = Field(
        default_factory=list,
        description="Notes/messages from issue-like keys + LLM (plan path)",
    )
    plan_analysis: str = Field(
        default="",
        description="Feasibility (digest assessment) line and compact JSON outline; five-sentence skeleton is plan_skeleton_summary (plan path)",
    )
    plan_skeleton_summary: str = Field(
        default="",
        description="Five-sentence prose summary of plan JSON shape (plan path, from preprocess)",
    )
    log_errors_analysis: str = Field(
        default="",
        description="Summary of error-class lines (log input only)",
    )
    log_warnings_analysis: str = Field(
        default="",
        description="Summary of warning-class lines (log input only)",
    )
    log_analysis_insight: str = Field(
        default="",
        description="Claude Sonnet interpretation of extracted signals (log path, after pipeline)",
    )
    log_analysis_plan_failure: str = Field(
        default="",
        description="How plan generation / solver failed per log analyser (log path)",
    )
    log_analysis_recommendation: str = Field(
        default="",
        description="Recommendation from log analyser step before findings summary (log path)",
    )
    preprocess_summary: str = Field(
        default="",
        description="Heuristic ERROR/WARNING line scan before LLM (log path only)",
    )
    preprocess_llm_digest: str = Field(
        default="",
        description="Compact digest sent to early LLM steps (log path only); subset of pre-scan for token savings",
    )
    preprocess_error_hits: list[str] = Field(
        default_factory=list,
        description="Sample lines matching error/failure patterns (log path)",
    )
    preprocess_warning_hits: list[str] = Field(
        default_factory=list,
        description="Sample lines matching warning patterns (log path)",
    )
    preprocess_only: bool = Field(
        default=False,
        description="True when the run stopped after preprocess (log or plan); no LLM nodes ran",
    )
    plan_preprocess_errors: list[str] = Field(
        default_factory=list,
        description="Structural JSON checks before plan LLM (plan path); set when preprocess_only or before merge",
    )
    plan_preprocess_warnings: list[str] = Field(
        default_factory=list,
        description="Structural JSON warnings before plan LLM (plan path)",
    )
    trace: list[GraphTraceStep] | None = Field(
        default=None,
        description="Per-node execution trace (only when include_trace=true)",
    )
    trace_legend: str | None = Field(
        default=None,
        description="High-level graph path for this input kind (only with trace)",
    )
    trace_mermaid_reference: str | None = Field(
        default=None,
        description="Mermaid source: reference topology for plan or log branch (only with trace)",
    )
    trace_mermaid_execution: str | None = Field(
        default=None,
        description="Mermaid source: linear path for this run (only with trace)",
    )


_compiled = None


def _str_list(val: object) -> list[str]:
    if not isinstance(val, list):
        return []
    return [str(x) for x in val]


def _graph():
    global _compiled
    if _compiled is None:
        _compiled = get_runnable_graph(with_checkpointer=False)
    return _compiled


def _result_to_response(result: dict, *, trace: list[GraphTraceStep] | None) -> AnalyzeResponse:
    pw = result.get("plan_warnings_list")
    pe = result.get("plan_errors_list")
    pm = result.get("plan_messages_list")
    if not isinstance(pw, list):
        pw = []
    if not isinstance(pe, list):
        pe = []
    if not isinstance(pm, list):
        pm = []
    input_kind = str(result.get("input_kind", "log"))
    trace_legend: str | None = None
    trace_mermaid_reference: str | None = None
    trace_mermaid_execution: str | None = None
    if trace is not None:
        trace_legend = legend_for_input_kind(input_kind)
        trace_mermaid_reference = mermaid_reference_topology(input_kind=input_kind)
        trace_mermaid_execution = mermaid_execution_path(ordered_nodes_from_step_models(trace))
    return AnalyzeResponse(
        input_kind=input_kind,
        root_cause=str(result.get("root_cause", "")),
        confidence=float(result.get("confidence", 0.0)),
        severity=str(result.get("severity", "")),
        recommendation=str(result.get("recommendation", "")),
        recommendation_summary=str(result.get("recommendation_summary", "")),
        findings_summary=str(result.get("findings_summary", "")),
        plan_feasible=result.get("plan_feasible"),
        plan_warnings=[str(x) for x in pw],
        plan_errors=[str(x) for x in pe],
        plan_messages=[str(x) for x in pm],
        plan_analysis=str(result.get("plan_deep_analysis", "")),
        plan_skeleton_summary=str(result.get("plan_skeleton_summary", "")),
        log_errors_analysis=str(result.get("log_errors_analysis", "")),
        log_warnings_analysis=str(result.get("log_warnings_analysis", "")),
        log_analysis_insight=str(result.get("log_analysis_insight", "")),
        log_analysis_plan_failure=str(result.get("log_analysis_plan_failure", "")),
        log_analysis_recommendation=str(result.get("log_analysis_recommendation", "")),
        preprocess_summary=str(result.get("preprocess_summary", "")),
        preprocess_llm_digest=str(result.get("preprocess_llm_digest", "")),
        preprocess_error_hits=_str_list(result.get("preprocess_error_hits")),
        preprocess_warning_hits=_str_list(result.get("preprocess_warning_hits")),
        preprocess_only=bool(result.get("stop_after_preprocess")),
        plan_preprocess_errors=_str_list(result.get("plan_preprocess_errors")),
        plan_preprocess_warnings=_str_list(result.get("plan_preprocess_warnings")),
        trace=trace,
        trace_legend=trace_legend,
        trace_mermaid_reference=trace_mermaid_reference,
        trace_mermaid_execution=trace_mermaid_execution,
    )


app = FastAPI(title="RCA Agent", lifespan=lifespan)


@app.api_route("/analyze", methods=["POST", "PUT"], response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    preprocess_only: bool = Query(
        False,
        description="If true, run only classify + preprocess (log or plan) and return; no LLM steps.",
    ),
    include_trace: bool = Query(
        False,
        description="If true, include per-node graph trace (same analysis run, slower payload).",
    ),
) -> AnalyzeResponse:
    initial = get_initial_state(body.log, stop_after_preprocess=preprocess_only)
    g = _graph()
    if include_trace:
        result, raw_trace = await ainvoke_with_trace(g, initial)
        trace = [
            GraphTraceStep(
                step=t["step"],
                node=t["node"],
                keys=t["keys"],
                summary=t["summary"],
            )
            for t in raw_trace
        ]
        return _result_to_response(result, trace=trace)
    result = await g.ainvoke(initial)
    return _result_to_response(result, trace=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apis.app:app", host="0.0.0.0", port=8000)

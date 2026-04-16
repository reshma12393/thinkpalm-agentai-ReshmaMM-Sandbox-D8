"""Dedicated log analyser: Claude Sonnet interprets extracted errors/warnings/exceptions for plan-failure insight."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from graphs.state import GraphState
from tools.log_analyser_payload_tool import build_log_analyser_payload
from utils.agent_debug import log_input_state, log_llm_response, log_parsed_output
from utils.llm import get_log_analyser_llm, invoke_llm_chain
from utils.parser import extract_llm_json_object, parse_llm_json
from utils.tool_logging import log_tool_run


class _LogAnalyserOut(BaseModel):
    insight: str = ""
    plan_generation_failure: str = ""
    recommendation: str = ""


_LOG_ANALYSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert log analyst. You receive **only** structured extractions from a pipeline "
            "(error/warning narratives, preprocess samples, parsed fields, patterns, optional raw excerpt, exception hints).\n\n"
            "Your job:\n"
            "1. **insight**: Clear explanation of what the log evidence shows—components involved, failure mode, and how "
            "error vs warning material relates.\n"
            "2. **plan_generation_failure**: If the text relates to **plan generation**, optimization, solver output, "
            "feasibility, tolerances, constraints, cargo/load, or JSON plan objects, explain **how that process failed or "
            "degraded** (or what warning implies for the plan). If there is no indication of planning/plan-generation, "
            'answer exactly: `Not indicated in the supplied excerpts.`\n'
            "3. **recommendation**: Specific next actions grounded in the excerpts (what to change, verify, or re-run)—"
            "not generic advice like only \"check logs\".\n\n"
            "Output **strict JSON only** with exactly these keys:\n"
            '{"insight":"...","plan_generation_failure":"...","recommendation":"..."}\n'
            "No markdown fences, no text outside JSON.",
        ),
        ("human", "{payload}"),
    ]
)


def _heuristic_log_analysis(state: dict[str, Any]) -> dict[str, str]:
    le = (state.get("log_errors_analysis") or "").strip()
    lw = (state.get("log_warnings_analysis") or "").strip()
    insight = (
        "Heuristic summary (LLM unavailable or invalid JSON): "
        + (le[:1200] if le else "")
        + (" " + lw[:800] if lw else "")
    ).strip() or "No error or warning narrative was extracted."
    pg = (
        "Not indicated in the supplied excerpts."
        if "plan" not in (le + lw).lower() and "feasible" not in (le + lw).lower()
        else "Review extracted error/warning text for plan or feasibility cues; full analysis requires the model."
    )
    rec = (
        "Use the extracted error and warning lines above to target the failing subsystem; reproduce with the same inputs "
        "after addressing the first concrete exception or limit mentioned."
    )
    return {"insight": insight, "plan_generation_failure": pg, "recommendation": rec}


def log_analyser_agent(state: GraphState) -> dict[str, Any]:
    payload_dict = build_log_analyser_payload(dict(state))
    log_tool_run("log_analyser", "build_log_analyser_payload", payload_dict)
    payload = json.dumps(payload_dict, ensure_ascii=False)
    log_input_state(
        "log_analyser",
        {
            "payload_chars": len(payload),
            "model": "RCA_LOG_ANALYSER_MODEL / Sonnet when anthropic",
        },
    )

    chain = _LOG_ANALYSER_PROMPT | get_log_analyser_llm()
    content = invoke_llm_chain(chain, {"payload": payload[:300_000]})
    log_llm_response("log_analyser", content)

    used_fallback = extract_llm_json_object(content) is None
    data = parse_llm_json(content, default={})
    try:
        validated = _LogAnalyserOut.model_validate(
            {
                "insight": str(data.get("insight", "") or ""),
                "plan_generation_failure": str(data.get("plan_generation_failure", "") or ""),
                "recommendation": str(data.get("recommendation", "") or ""),
            }
        )
    except ValidationError:
        used_fallback = True
        validated = _LogAnalyserOut.model_validate({})

    ins = validated.insight.strip()
    pgf = validated.plan_generation_failure.strip()
    rec = validated.recommendation.strip()
    if not ins and not pgf and not rec:
        used_fallback = True
        h = _heuristic_log_analysis(dict(state))
        ins, pgf, rec = h["insight"], h["plan_generation_failure"], h["recommendation"]

    hist = ["log_analyser: Claude Sonnet (RCA_LOG_ANALYSER_MODEL when anthropic)"]
    if used_fallback:
        hist.append("log_analyser: json_or_empty_heuristic")

    result = {
        "log_analysis_insight": ins,
        "log_analysis_plan_failure": pgf,
        "log_analysis_recommendation": rec,
        "history": hist,
    }
    log_parsed_output("log_analyser", {**result, "json_fallback": used_fallback})
    return result

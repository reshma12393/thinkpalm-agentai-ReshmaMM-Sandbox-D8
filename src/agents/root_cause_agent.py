"""Agent: infer root cause and confidence from parsed log + patterns (LLM)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from graphs.state import GraphState
from tools.rca_signal_summary_tool import rca_signal_summary_tool
from utils.agent_debug import log_input_state, log_llm_response, log_parsed_output
from utils.llm import get_llm, invoke_llm_chain
from utils.log_signals_fallback import heuristic_root_cause
from utils.parser import extract_llm_json_object, parse_llm_json
from utils.rca_text_format import normalize_to_n_lines
from utils.tool_logging import log_tool_run


class _RootCauseSchema(BaseModel):
    root_cause: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


_ROOT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Infer the most likely root cause **as a description of what failed**, grounded in **error types / warning themes** "
            "you identify from the payload (parsed_log, tool_signal_summary, error_log_analysis, warning_log_analysis, patterns, "
            "findings_summary, log_analysis_*). Name those themes in your own words from the evidence—do not assume labels not "
            "supported by the text.\n"
            "Treat findings_summary (often three lines) as the synthesized view; align with error_log_analysis and warning_log_analysis.\n"
            "When log_analysis_insight or log_analysis_plan_failure are present, align with them unless raw text clearly contradicts.\n\n"
            "Output **root_cause** as **exactly three lines**, separated by newline characters (\\n). "
            "Each line is plain prose: describe the failure mechanism, contributing factors, and confidence-facing summary—still "
            "only from this payload.\n\n"
            "Output only a single JSON object with exactly these keys:\n"
            '- "root_cause": string (three lines with newline characters inside the JSON string)\n'
            '- "confidence": number between 0 and 1\n\n'
            "Rules:\n"
            "- confidence must be in [0, 1].\n"
            "- Higher confidence when detected_patterns strongly support the diagnosis (multiple clear matches, alignment with error_type/module/keywords).\n"
            "- Lower confidence when patterns are empty, vague, or contradictory.\n"
            "No markdown, no code fences, no text outside the JSON object.",
        ),
        ("human", "{payload}"),
    ]
)


def _coerce_root_cause(d: dict[str, Any]) -> dict[str, Any]:
    rc = str(d.get("root_cause", "") or "").strip()
    try:
        c = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        c = 0.0
    c = max(0.0, min(1.0, c))
    return {"root_cause": rc, "confidence": c}


def _fallback_root_empty() -> dict[str, Any]:
    return {"root_cause": "", "confidence": 0.0}


def root_cause_agent(state: GraphState) -> dict[str, Any]:
    parsed: dict[str, Any] = state.get("parsed_log") or {}
    if not isinstance(parsed, dict):
        parsed = {}
    patterns: list[Any] = list(state.get("detected_patterns") or [])

    payload_obj: dict[str, Any] = {"parsed_log": parsed, "detected_patterns": patterns}
    le = (state.get("log_errors_analysis") or "").strip()
    lw = (state.get("log_warnings_analysis") or "").strip()
    # Always include keys so the model sees empty vs missing; cap size for very large logs.
    _max_ew = 120_000
    payload_obj["error_log_analysis"] = le[:_max_ew] + ("…" if len(le) > _max_ew else "")
    payload_obj["warning_log_analysis"] = lw[:_max_ew] + ("…" if len(lw) > _max_ew else "")
    pre = (state.get("preprocess_llm_digest") or state.get("preprocess_summary") or "").strip()
    if pre:
        payload_obj["preprocess_scan"] = pre
    mem = (state.get("memory_context") or "").strip()
    if mem:
        payload_obj["similar_past_incidents"] = mem[:8000]

    fs = (state.get("findings_summary") or "").strip()
    if fs:
        payload_obj["findings_summary"] = fs[:12000]

    lai = (state.get("log_analysis_insight") or "").strip()
    lap = (state.get("log_analysis_plan_failure") or "").strip()
    lar = (state.get("log_analysis_recommendation") or "").strip()
    if lai:
        payload_obj["log_analysis_insight"] = lai[:8000]
    if lap:
        payload_obj["log_analysis_plan_failure"] = lap[:8000]
    if lar:
        payload_obj["log_analysis_recommendation"] = lar[:8000]

    tool_summary = rca_signal_summary_tool(
        parsed,
        patterns,
        log_errors_analysis=le,
        log_warnings_analysis=lw,
        memory_present=bool(mem),
    )
    log_tool_run("root_cause", "rca_signal_summary_tool", tool_summary)
    payload_obj["tool_signal_summary"] = tool_summary

    payload = json.dumps(payload_obj, ensure_ascii=False)
    log_input_state(
        "root_cause",
        {
            "parsed_log": parsed,
            "detected_patterns": patterns,
            "has_error_log_analysis": bool(le),
            "has_warning_log_analysis": bool(lw),
            "has_memory_context": bool(mem),
            "has_findings_summary": bool(fs),
        },
    )

    chain = _ROOT_PROMPT | get_llm()
    content = invoke_llm_chain(chain, {"payload": payload})
    log_llm_response("root_cause", content)

    used_fallback = extract_llm_json_object(content) is None
    data = parse_llm_json(content, default=_fallback_root_empty())
    data = _coerce_root_cause(data)

    try:
        validated = _RootCauseSchema.model_validate(data)
    except ValidationError:
        used_fallback = True
        validated = _RootCauseSchema.model_validate(_fallback_root_empty())

    rc_text = str(validated.root_cause or "").strip()
    conf_val = float(validated.confidence)
    if not rc_text:
        rc_text, conf_val = heuristic_root_cause(dict(state))
    rc_text = normalize_to_n_lines(rc_text, 3)

    hist = ["root_cause: inferred (3-line, type-grounded)"]
    if used_fallback:
        hist.append("root_cause: json_fallback")
    if not str(validated.root_cause or "").strip():
        hist.append("root_cause: filled_from_preprocess_or_excerpt")

    result = {
        "root_cause": rc_text,
        "confidence": conf_val,
        "history": hist,
    }
    log_parsed_output(
        "root_cause",
        {
            "root_cause": result["root_cause"],
            "confidence": result["confidence"],
            "history": result["history"],
            "json_fallback": used_fallback,
        },
    )
    return result

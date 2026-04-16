"""LLM: summarize first-level extracted signals (preprocess + error/warning narratives) into findings_summary."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from graphs.state import GraphState
from tools.extracted_signals_payload_tool import build_extracted_signals_payload
from utils.agent_debug import log_input_state, log_llm_response, log_parsed_output
from utils.llm import get_llm, invoke_llm_chain
from utils.log_signals_fallback import heuristic_extracted_signals_summary
from utils.parser import extract_llm_json_object, parse_llm_json
from utils.rca_text_format import normalize_to_n_lines
from utils.tool_logging import log_tool_run


class _SummaryOut(BaseModel):
    findings_summary: str = ""


_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write a clear findings summary from the JSON payload only. The payload contains:\n"
            "- **log_errors_analysis** and **log_warnings_analysis** (first in the JSON): full severity narratives from the "
            "prior log-error/warning step—these are the primary source for what went wrong vs what was warned; "
            "you MUST read and reflect their substantive content (facts, messages, components), not only the preprocess lists.\n"
            "- **log_analysis_insight**, **log_analysis_plan_failure**, **log_analysis_recommendation**: output of the prior "
            "**log_analyser** step (Claude Sonnet on the same extractions)—integrate this interpretation with the raw narratives; "
            "if it explains plan-generation or solver failure, reflect that in your findings_summary without duplicating it verbatim.\n"
            "- preprocess_error_hits / preprocess_warning_hits: heuristic line samples (supporting)\n"
            "- preprocess_summary and preprocess_llm_digest: scan overview\n"
            "- parsed_log and detected_patterns: structured extraction\n"
            "- memory_context_preview: optional similar past incidents (weak context)\n\n"
            "First **infer the kinds of failures and warnings** present (from parsed_log.error_type, keywords, narratives, "
            "patterns, and log_analyser fields)—describe them in your own words based only on this payload; do not rely on a fixed "
            "taxonomy supplied outside the JSON.\n\n"
            "Output **findings_summary** as **exactly three lines**, separated by newline characters (\\n). "
            "Line 1: primary error/failure characterization. Line 2: notable warnings or secondary issues. "
            "Line 3: one-line overall assessment. Each line must be plain prose (no bullet prefixes).\n\n"
            "Do not give remediation steps or recommendations here—only describe findings.\n\n"
            "If warning-class text appears together with success or feasibility indicators, state both sides briefly across the three lines.\n\n"
            "Output a single JSON object {\"findings_summary\": \"...\"}. The string value must be **three physical lines** "
            "(embed newline characters between lines inside the JSON string).\n"
            "No markdown fences, no text outside JSON.",
        ),
        ("human", "{payload}"),
    ]
)


def extracted_signals_summary_agent(state: GraphState) -> dict[str, Any]:
    payload_dict = build_extracted_signals_payload(dict(state))
    log_tool_run("log_findings_summary", "build_extracted_signals_payload", payload_dict)
    payload = json.dumps(payload_dict, ensure_ascii=False)
    log_input_state(
        "log_findings_summary",
        {"payload_chars": len(payload), "error_hits": len(payload_dict.get("preprocess_error_hits") or [])},
    )

    chain = _SUMMARY_PROMPT | get_llm()
    # Budget: preprocess lists are capped in build_extracted_signals_payload; keep analyses + rest for the model.
    content = invoke_llm_chain(chain, {"payload": payload[:280_000]})
    log_llm_response("log_findings_summary", content)

    used_fallback = extract_llm_json_object(content) is None
    data = parse_llm_json(content, default={"findings_summary": ""})
    fs = str(data.get("findings_summary", "") or "").strip()

    try:
        validated = _SummaryOut.model_validate({"findings_summary": fs})
    except ValidationError:
        used_fallback = True
        validated = _SummaryOut.model_validate({"findings_summary": ""})

    out_text = validated.findings_summary.strip()
    if not out_text:
        used_fallback = True
        out_text = heuristic_extracted_signals_summary(dict(state))
    out_text = normalize_to_n_lines(out_text, 3)

    hist = ["log_findings_summary: LLM summary of extracted signals (3-line findings)"]
    if used_fallback:
        hist.append("log_findings_summary: json_or_empty_fallback")

    result = {
        "findings_summary": out_text,
        "history": hist,
    }
    log_parsed_output("log_findings_summary", {**result, "json_fallback": used_fallback})
    return result

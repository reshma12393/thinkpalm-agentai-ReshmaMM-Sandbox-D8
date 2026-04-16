"""Agent: log signal pipeline (preprocess → parse → ERROR/WARN → pattern detection)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from graphs.state import GraphState
from memory.relevance import pattern_hints_from_incidents
from tools.heuristic_parse_hints_tool import heuristic_parse_hints_tool
from tools.memory_retrieval_tool import load_relevant_incidents_tool
from tools.pattern_lookup import lookup_failure_patterns_tool
from tools.preprocess_scan_tool import preprocess_scan_tool
from tools.severity_line_buckets_tool import severity_line_buckets_tool
from utils.agent_debug import log_input_state, log_llm_response, log_parsed_output
from utils.llm import get_llm, invoke_llm_chain
from utils.log_preprocess import excerpt_log_for_llm
from utils.tool_logging import log_tool_run
from utils.log_signals_fallback import preprocess_error_warning_blocks
from utils.parser import extract_llm_json_object, parse_llm_json


def _run_preprocess_log(state: GraphState) -> dict[str, Any]:
    raw = state.get("raw_log") or ""
    scan_out = preprocess_scan_tool(raw)
    log_tool_run("log_preprocess", "preprocess_scan_tool", scan_out)
    hist = [
        "preprocess_log: "
        f"error_like_total={scan_out.get('error_line_total', 0)} "
        f"stored={len(scan_out.get('preprocess_error_hits') or [])}, "
        f"warning_like_total={scan_out.get('warning_line_total', 0)} "
        f"stored={len(scan_out.get('preprocess_warning_hits') or [])}; "
        f"llm_digest_chars={len(scan_out.get('preprocess_llm_digest') or '')}"
    ]
    if not raw.strip():
        hist = ["preprocess_log: empty raw_log"]
    return {
        "preprocess_error_hits": scan_out["preprocess_error_hits"],
        "preprocess_warning_hits": scan_out["preprocess_warning_hits"],
        "preprocess_summary": scan_out["preprocess_summary"],
        "preprocess_llm_digest": scan_out["preprocess_llm_digest"],
        "history": hist,
    }


class _ParsedLogSchema(BaseModel):
    error_type: str = ""
    module: str = ""
    keywords: list[str] = Field(default_factory=list)


_LOG_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Refine structured fields from the log. A deterministic **tool_output** block (heuristic hints) "
            "is provided first—use it when consistent with log evidence; correct it when the log contradicts it.\n\n"
            "Output only a single JSON object with exactly these keys:\n"
            '- "error_type": string\n'
            '- "module": string\n'
            '- "keywords": array of strings\n\n'
            "Rules: no markdown, no code fences, no commentary—JSON only.",
        ),
        ("human", "{log}"),
    ]
)


def _coerce_parsed(d: dict[str, Any]) -> dict[str, Any]:
    kw = d.get("keywords", [])
    if not isinstance(kw, list):
        kw = []
    kw = [str(x).strip() for x in kw if str(x).strip()]
    return {
        "error_type": str(d.get("error_type", "") or "").strip(),
        "module": str(d.get("module", "") or "").strip(),
        "keywords": kw[:24],
    }


def _fallback_parsed_empty() -> dict[str, Any]:
    return {"error_type": "", "module": "", "keywords": []}


def _run_log_parser(state: GraphState) -> dict[str, Any]:
    raw = (state.get("raw_log") or "").strip()
    if not raw:
        h = heuristic_parse_hints_tool("")
        log_tool_run("log_parser", "heuristic_parse_hints_tool", h)
        log_input_state("log_parser", {"raw_log": "", "note": "empty input; skipping LLM"})
        log_llm_response("log_parser", "[skipped — empty raw_log]")
        out = {
            "parsed_log": {"error_type": "", "module": "", "keywords": []},
            "history": ["log_parser: empty raw_log"],
        }
        log_parsed_output("log_parser", out)
        return out

    digest = (state.get("preprocess_llm_digest") or "").strip()
    legacy = (state.get("preprocess_summary") or "").strip()
    body = excerpt_log_for_llm(raw, max_total=24_000)
    if digest:
        log_text = (
            "[Initial pattern processing — compact digest + raw excerpt; use digest for severity hints]\n\n"
            f"{digest}\n\n---\n\n[Raw log excerpt (head+tail if long)]\n\n{body}"
        )
    elif legacy:
        log_text = (
            "[Pre-scan — heuristic ERROR/WARNING line patterns]\n"
            f"{legacy}\n\n---\n\n{body}"
        )
    else:
        log_text = body[:50_000]

    tool_hints = heuristic_parse_hints_tool(raw)
    log_tool_run("log_parser", "heuristic_parse_hints_tool", tool_hints)

    log_text = (
        "## tool_output (deterministic)\n"
        + json.dumps(tool_hints, ensure_ascii=False, indent=2)
        + "\n\n## log_material\n"
        + log_text[:52_000]
    )

    log_input_state(
        "log_parser",
        {
            "raw_log_preview": raw[:2000] + ("..." if len(raw) > 2000 else ""),
            "raw_log_chars": len(raw),
            "preprocess_digest_chars": len(digest),
            "preprocess_in_prompt": bool(digest or legacy),
            "tool": "heuristic_parse_hints_tool",
        },
    )

    chain = _LOG_PARSE_PROMPT | get_llm()
    content = invoke_llm_chain(chain, {"log": log_text[:52_000]})
    log_llm_response("log_parser", content)

    used_fallback = extract_llm_json_object(content) is None
    data = parse_llm_json(content, default=_fallback_parsed_empty())
    data = _coerce_parsed(data)

    try:
        validated = _ParsedLogSchema.model_validate(data)
    except ValidationError:
        used_fallback = True
        validated = _ParsedLogSchema.model_validate(_fallback_parsed_empty())

    hist = ["log_parser: parsed"]
    if used_fallback:
        hist.append("log_parser: json_fallback")

    result = {
        "parsed_log": {
            "error_type": validated.error_type,
            "module": validated.module,
            "keywords": list(validated.keywords),
        },
        "history": hist,
    }
    log_parsed_output(
        "log_parser",
        {
            "parsed_log": result["parsed_log"],
            "history": result["history"],
            "json_fallback": used_fallback,
        },
    )
    return result


class _LogEWSchema(BaseModel):
    error_log_analysis: str = ""
    warning_log_analysis: str = ""


_LOG_EW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyze log text and write **short prose summaries**—not raw log dumps, not bullet lists of every line, "
            "and do not paste large JSON or tool_output verbatim.\n\n"
            "A deterministic **tool_output** buckets error-like vs warning-like lines—use it only to infer **themes** "
            "and severity; synthesize what happened in plain language.\n\n"
            "- **error_log_analysis**: 2–10 sentences summarizing failures, exceptions, fatal/ERROR lines, timeouts, "
            "exits—**what went wrong and in what order**, using your own words. Quote at most one **short** fragment "
            "(≤120 chars) only if essential.\n"
            "- **warning_log_analysis**: 2–10 sentences summarizing warnings, deprecations, retries, pressure—same style.\n\n"
            "If a category has no relevant evidence, say so in one sentence.\n\n"
            "Output only one JSON object with exactly these keys:\n"
            '- "error_log_analysis": string\n'
            '- "warning_log_analysis": string\n\n'
            "No markdown, no code fences, JSON only.",
        ),
        ("human", "{log_text}"),
    ]
)


def _fallback_ew() -> dict[str, Any]:
    return {"error_log_analysis": "", "warning_log_analysis": ""}


def _run_log_error_warning(state: GraphState) -> dict[str, Any]:
    raw = (state.get("raw_log") or "").strip()
    if not raw:
        b = severity_line_buckets_tool("")
        log_tool_run("log_error_warning", "severity_line_buckets_tool", b)
        log_input_state("log_error_warning", {"raw_log": "", "note": "empty"})
        log_llm_response("log_error_warning", "[skipped — empty raw_log]")
        out = {
            "log_errors_analysis": "",
            "log_warnings_analysis": "",
            "history": ["log_error_warning: skipped empty raw_log"],
        }
        log_parsed_output("log_error_warning", out)
        return out

    digest = (state.get("preprocess_llm_digest") or "").strip()
    legacy = (state.get("preprocess_summary") or "").strip()
    body = excerpt_log_for_llm(raw, max_total=24_000)
    if digest:
        log_text = (
            "[Initial pattern processing — compact digest + raw excerpt]\n\n"
            f"{digest}\n\n---\n\n[Raw log excerpt]\n\n{body}"
        )
    elif legacy:
        log_text = (
            "[Pre-scan — heuristic ERROR/WARNING patterns]\n"
            f"{legacy}\n\n---\n\n{body}"
        )
    else:
        log_text = body[:50_000]

    buckets = severity_line_buckets_tool(raw)
    log_tool_run("log_error_warning", "severity_line_buckets_tool", buckets)

    log_text = (
        "## tool_output (deterministic)\n"
        + json.dumps(buckets, ensure_ascii=False, indent=2)
        + "\n\n## full_log_for_narrative\n"
        + log_text[:52_000]
    )

    log_input_state(
        "log_error_warning",
        {
            "raw_log_preview": raw[:2000] + ("..." if len(raw) > 2000 else ""),
            "preprocess_digest_chars": len(digest),
            "preprocess_in_prompt": bool(digest or legacy),
            "tool": "severity_line_buckets_tool",
        },
    )
    chain = _LOG_EW_PROMPT | get_llm()
    content = invoke_llm_chain(chain, {"log_text": log_text[:52_000]})
    log_llm_response("log_error_warning", content)

    used_fallback = extract_llm_json_object(content) is None
    data = parse_llm_json(content, default=_fallback_ew())
    err = str(data.get("error_log_analysis", "") or "").strip()
    warn = str(data.get("warning_log_analysis", "") or "").strip()

    try:
        validated = _LogEWSchema.model_validate(
            {"error_log_analysis": err, "warning_log_analysis": warn}
        )
    except ValidationError:
        used_fallback = True
        validated = _LogEWSchema.model_validate(_fallback_ew())

    err_out = str(validated.error_log_analysis or "").strip()
    warn_out = str(validated.warning_log_analysis or "").strip()
    e_heur, w_heur = preprocess_error_warning_blocks(dict(state))
    if not err_out and e_heur:
        err_out = e_heur
    if not warn_out and w_heur:
        warn_out = w_heur

    hist = ["log_error_warning: error vs warning split"]
    if used_fallback:
        hist.append("log_error_warning: json_fallback")

    result = {
        "log_errors_analysis": err_out,
        "log_warnings_analysis": warn_out,
        "history": hist,
    }
    log_parsed_output("log_error_warning", result)
    return result


def _run_load_relevant_memory(state: GraphState) -> dict[str, Any]:
    """Load ranked past incidents from disk for pattern detection and downstream RCA."""
    snap = dict(state)
    mem_out = load_relevant_incidents_tool(snap, limit=5)
    log_tool_run("log_memory", "load_relevant_incidents_tool", mem_out)
    incidents = mem_out.get("memory_relevant_incidents") or []
    if incidents:
        hist = [f"memory: {len(incidents)} past incident(s) selected for pattern + root-cause context"]
    else:
        hist = ["memory: no past incidents (empty store or no overlap)"]
    return {
        "memory_relevant_incidents": incidents,
        "memory_context": str(mem_out.get("memory_context") or ""),
        "history": hist,
    }


_PATTERN_LLM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Suggest failure-pattern labels. The JSON **payload** includes:\n"
            "- **parsed_log** — structured fields;\n"
            "- **tool_dictionary_patterns** — deterministic labels from a Python dictionary (may be empty);\n"
            "- optionally **similar_past_incidents** and **prior_pattern_labels_from_memory**.\n"
            "Reuse or align with dictionary / memory labels when evidence fits; otherwise propose concise new labels.\n\n"
            "Output only: {\"patterns\": [\"...\", ...]} — empty array if insufficient evidence. No markdown or extra text.",
        ),
        ("human", "{payload}"),
    ]
)


def _tokens_from_parsed_log(parsed: dict[str, Any]) -> list[str]:
    keywords = list(parsed.get("keywords") or [])
    for key in ("error_type", "module"):
        v = parsed.get(key)
        if v is not None and str(v).strip():
            keywords.append(str(v))
    return keywords


def _coerce_patterns(data: dict[str, Any]) -> list[str]:
    raw = data.get("patterns", [])
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for p in raw:
        s = str(p).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:12]


def _llm_infer_patterns(
    parsed: dict[str, Any],
    *,
    tool_dictionary_patterns: list[str],
    memory_context: str = "",
    memory_hints: list[str] | None = None,
) -> tuple[list[str], bool]:
    chain = _PATTERN_LLM_PROMPT | get_llm(temperature=0)
    payload_obj: dict[str, Any] = {
        "parsed_log": parsed,
        "tool_dictionary_patterns": list(tool_dictionary_patterns),
    }
    if memory_context.strip():
        payload_obj["similar_past_incidents"] = memory_context.strip()[:6000]
    mh = memory_hints or []
    if mh:
        payload_obj["prior_pattern_labels_from_memory"] = mh[:16]
    payload = json.dumps(payload_obj, ensure_ascii=False)
    content = invoke_llm_chain(chain, {"payload": payload})
    log_llm_response("pattern_detection", content)

    data = parse_llm_json(content, default={"patterns": []})
    json_failed = extract_llm_json_object(content) is None
    return _coerce_patterns(data), json_failed


def _merge_patterns(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for label in group:
            s = str(label).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= 20:
                return out
    return out


def _run_pattern_detection(state: GraphState) -> dict[str, Any]:
    parsed = state.get("parsed_log") or {}
    if not isinstance(parsed, dict):
        parsed = {}

    incidents = state.get("memory_relevant_incidents") or []
    if not isinstance(incidents, list):
        incidents = []
    mem_ctx = (state.get("memory_context") or "").strip()
    memory_hints = pattern_hints_from_incidents([x for x in incidents if isinstance(x, dict)])

    tokens = _tokens_from_parsed_log(parsed)
    log_input_state(
        "pattern_detection",
        {
            "parsed_log": parsed,
            "tokens": tokens,
            "memory_incident_count": len(incidents),
            "memory_hint_count": len(memory_hints),
        },
    )

    tool_patterns = lookup_failure_patterns_tool(tokens)
    log_tool_run("pattern_detection", "lookup_failure_patterns_tool", tool_patterns)

    llm_patterns: list[str] = []
    llm_json_failed = False
    if not tool_patterns:
        llm_patterns, llm_json_failed = _llm_infer_patterns(
            parsed,
            tool_dictionary_patterns=tool_patterns,
            memory_context=mem_ctx,
            memory_hints=memory_hints,
        )
    else:
        log_llm_response(
            "pattern_detection",
            "[skipped — dictionary rules produced a match; LLM not invoked]",
        )

    detected = _merge_patterns(tool_patterns, llm_patterns, memory_hints)

    parts = [f"rules={len(tool_patterns)}", f"llm={len(llm_patterns)}", f"merged={len(detected)}"]
    if llm_json_failed:
        parts.append("json_fallback")
    hist = [f"pattern_detection: {' '.join(parts)}"]
    result = {
        "detected_patterns": detected,
        "history": hist,
    }
    log_parsed_output(
        "pattern_detection",
        {
            "tool_patterns": tool_patterns,
            "llm_patterns": llm_patterns,
            "memory_pattern_hints": memory_hints,
            "detected_patterns": detected,
            "history": result["history"],
        },
    )
    return result


def _merge_log_patch(acc: dict[str, Any], patch: dict[str, Any]) -> None:
    for k, v in patch.items():
        if k == "history":
            cur = list(acc.get("history") or [])
            acc["history"] = cur + (v if isinstance(v, list) else [v])
        elif k == "detected_patterns":
            cur = list(acc.get("detected_patterns") or [])
            acc["detected_patterns"] = cur + (v if isinstance(v, list) else [v])
        else:
            acc[k] = v


def log_signal_pipeline_agent(state: GraphState) -> dict[str, Any]:
    if state.get("stop_after_preprocess"):
        return _run_preprocess_log(state)

    steps = (
        _run_preprocess_log,
        _run_log_parser,
        _run_log_error_warning,
        _run_load_relevant_memory,
        _run_pattern_detection,
    )
    acc: dict[str, Any] = {}
    working: dict[str, Any] = dict(state)

    for fn in steps:
        patch = fn(working)
        _merge_log_patch(acc, patch)
        working.update(patch)

    acc.setdefault("history", []).append(
        "log_signal_pipeline: preprocess → log_parser → log_error_warning → memory → pattern_detection"
    )
    return acc

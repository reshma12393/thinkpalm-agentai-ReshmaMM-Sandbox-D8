"""Agent: strict LLM recommendation grounded in parsed_log, patterns, error/warning summaries, root_cause."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from graphs.state import GraphState
from memory.store import save_incident
from tools.log_context_hints_tool import extract_log_context_anchors_tool
from tools.rca_signal_summary_tool import rca_signal_summary_tool
from tools.severity import severity_baseline_tool
from utils.agent_debug import log_input_state, log_llm_response, log_parsed_output
from utils.llm import get_recommendation_llm, invoke_llm_chain
from utils.log_signals_fallback import _lines_from_hits, heuristic_findings_summary
from utils.parser import extract_llm_json_object, parse_llm_json
from utils.rca_text_format import normalize_to_n_lines
from utils.tool_logging import log_tool_run


class _RecommendationSchema(BaseModel):
    recommendation: str = ""
    summary: str = ""
    severity: Literal["High", "Medium", "Low"] = Field(default="Medium")


_REC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert system debugging assistant.\n\n"
            "**Findings** (below) is the authoritative synthesis of what was already observed—your `summary` and "
            "`recommendation` MUST read as a direct, proper reply to that narrative: address the same issues, severity, "
            "and entities the findings describe. Do not contradict Findings unless you explain why a narrower technical fix still applies.\n\n"
            "You MUST also ground specifics in: Error Type, Module, Keywords, Detected Patterns, error/warning excerpts, "
            "Root Cause, and (when present) the **Log analyser** fields (insight / plan-generation failure / its recommendation).\n\n"
            "You MUST NOT give generic or filler advice. Forbidden examples (do not use, even paraphrased): "
            '"check the logs", "review configuration", "retry the process", "verify settings", '
            '"monitor the system", "look into it", "ensure things are correct", "contact support" '
            "without naming a concrete artifact from the Given section.\n\n"
            "You MUST:\n"
            "- In `summary`: **exactly three lines** (plain prose, separated by newline characters). Restate the issue using "
            "**the same error/warning themes** as Findings and Root Cause (error type, keywords, patterns)—no generic filler.\n"
            "- In `recommendation`: **exactly three lines** of concrete next steps that **follow from** that summary and the "
            "named error/warning classes; each line should be actionable and reference at least one anchor from Error Type, "
            "Module, Keywords, or Detected Patterns where possible.\n\n"
            "Output STRICT JSON ONLY — exactly one object. `summary` and `recommendation` string values must each be "
            "**three physical lines** with newlines embedded in the JSON strings. Severity as shown.\n"
            '{"recommendation":"...","summary":"...","severity":"High"|"Medium"|"Low"}\n',
        ),
        (
            "human",
            "Given:\n"
            "- Findings (authoritative synthesis to reply to): {findings_summary}\n"
            "- Error Type: {error_type}\n"
            "- Module: {module}\n"
            "- Keywords: {keywords}\n"
            "- Detected Patterns: {detected_patterns}\n"
            "- Errors (narrative summary excerpt): {errors_summary}\n"
            "- Warnings (narrative summary excerpt): {warnings_summary}\n"
            "- Root Cause: {root_cause}\n"
            "- Log analyser insight (Claude Sonnet on extractions): {log_analysis_insight}\n"
            "- Plan generation failure (from log analyser): {log_analysis_plan_failure}\n"
            "- Log analyser recommendation (prior step): {log_analysis_recommendation}\n\n"
            "Produce `summary` and `recommendation` as a proper response to **Findings**, using the structured fields for precision; "
            "merge log analyser insight where it adds specificity.",
        ),
    ]
)


def _format_keywords(parsed: dict[str, Any]) -> str:
    kw = parsed.get("keywords")
    if isinstance(kw, list):
        parts = [str(x).strip() for x in kw if str(x).strip()]
        return ", ".join(parts) if parts else "(none)"
    s = str(kw or "").strip()
    return s if s else "(none)"


def _signal_excerpt(text: str, *, max_chars: int = 2800) -> str:
    t = (text or "").strip()
    if not t:
        return "(none present)"
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _normalize_severity(raw: Any, tool_baseline: str) -> Literal["High", "Medium", "Low"]:
    s = str(raw or "").strip().lower()
    if s == "high":
        return "High"
    if s == "medium":
        return "Medium"
    if s == "low":
        return "Low"
    if tool_baseline == "High":
        return "High"
    if tool_baseline == "Medium":
        return "Medium"
    return "Medium"


def _coerce_rec(data: dict[str, Any], tool_baseline: str) -> dict[str, Any]:
    rec = str(data.get("recommendation", "") or "").strip()
    summ = str(data.get("summary", "") or "").strip()
    sev = _normalize_severity(data.get("severity"), tool_baseline)
    return {"recommendation": rec, "summary": summ, "severity": sev}


def _fallback_rec(tool_baseline: str) -> dict[str, Any]:
    sev = _normalize_severity(None, tool_baseline)
    return {"recommendation": "", "summary": "", "severity": sev}


def _trim_pattern_strings(patterns: list[Any], *, max_items: int = 48, max_each: int = 400) -> list[str]:
    out: list[str] = []
    for x in patterns:
        s = str(x).strip()
        if not s:
            continue
        if len(s) > max_each:
            s = s[: max_each - 1] + "…"
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _snippet(text: str, max_chars: int) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t:
        return ""
    return t[:max_chars] + ("…" if len(t) > max_chars else "")


def _heuristic_grounded_stub(
    state: dict[str, Any],
    parsed: dict[str, Any],
    pattern_strings: list[str],
    root_cause: str,
    findings: str,
) -> tuple[str, str]:
    """Fallback: three-line summary + recommendation grounded in extracts when the model output is empty."""
    le = (state.get("log_errors_analysis") or "").strip()
    lw = (state.get("log_warnings_analysis") or "").strip()
    pe = _lines_from_hits("preprocess_error_hits", state, cap=6)
    pw = _lines_from_hits("preprocess_warning_hits", state, cap=6)
    pat_join = "; ".join(pattern_strings[:8]) if pattern_strings else ""

    et = str(parsed.get("error_type") or "").strip()
    mod = str(parsed.get("module") or "").strip()
    kw = _format_keywords(parsed)
    fd = (findings or "").strip()
    rc = (root_cause or "").strip()

    has_extracts = bool(le or lw or pe or pw or pat_join)

    if et and mod:
        s1 = f"Classifier: error_type «{et}», module «{mod}», keywords «{kw}»."
    elif has_extracts:
        bits: list[str] = []
        if pat_join:
            bits.append("patterns: " + _snippet(pat_join, 220))
        if pe:
            bits.append("pre-scan error samples: " + " | ".join(_snippet(x, 100) for x in pe[:2]))
        if pw:
            bits.append("pre-scan warning samples: " + " | ".join(_snippet(x, 100) for x in pw[:2]))
        s1 = "Extracted signals (parser tags incomplete). " + (" ".join(bits) if bits else _snippet(le or lw, 280))
    else:
        s1 = f"Classifier: error_type «{et or '—'}», module «{mod or '—'}», keywords «{kw}»."

    if le:
        s2 = "Error-class narrative (extracted): " + _snippet(le, 420)
    elif pe:
        s2 = "Pre-scan error/failure-like lines: " + " | ".join(_snippet(x, 140) for x in pe[:4])
    else:
        s2 = "Error-class narrative: (none in state). " + (_snippet(fd, 200) if fd else "Add ERROR/exception lines or re-run pipeline.")

    if lw:
        s3 = "Warning-class narrative (extracted): " + _snippet(lw, 420)
    elif pat_join:
        s3 = "Detected patterns (for triage): " + _snippet(pat_join, 420)
    elif pw:
        s3 = "Pre-scan warning-like lines: " + " | ".join(_snippet(x, 140) for x in pw[:4])
    elif rc:
        s3 = "Diagnosis tie-in: " + _snippet(rc, 420)
    elif fd:
        s3 = "Findings tie-in: " + _snippet(fd, 420)
    else:
        s3 = "No warning narrative or patterns in state; inspect raw log for context."

    summary = normalize_to_n_lines("\n".join([s1, s2, s3]), 3)

    if le:
        r1 = "Prioritize the failure described in the error-class extract; fix the component or dependency it names, then retest."
    elif pe:
        r1 = "Investigate the pre-scan error-class sample lines above (timestamps/stack); correct the first failing subsystem they imply."
    else:
        r1 = "Capture ERROR/exception blocks with full stack traces, map them to owning services or modules, then apply a targeted fix."

    if lw or pw:
        r2 = "Resolve or justify the warning/pre-scan lines so noise does not mask the primary failure on the next run."
    elif pat_join:
        r2 = "Validate configuration and code paths that match the detected patterns above before redeploying."
    else:
        r2 = "Tighten logging around the failing operation so the next run yields a clearer error signature."

    r3 = "Re-run the same workload and confirm the failure signature clears."
    rec = normalize_to_n_lines("\n".join([r1, r2, r3]), 3)
    return summary, rec


def recommendation_agent(state: GraphState) -> dict[str, Any]:
    parsed_raw = state.get("parsed_log")
    parsed: dict[str, Any] = parsed_raw if isinstance(parsed_raw, dict) else {}
    error_type = str(parsed.get("error_type") or "")
    module_s = str(parsed.get("module") or "") or "(none)"
    tool_baseline = severity_baseline_tool(error_type)
    log_tool_run("recommendation", "severity_baseline_tool", tool_baseline)

    raw_log = str(state.get("raw_log") or "")
    anchors = extract_log_context_anchors_tool(raw_log)
    log_tool_run("recommendation", "extract_log_context_anchors_tool", anchors)

    root_cause = (state.get("root_cause") or "").strip()
    findings_prior = (state.get("findings_summary") or "").strip()
    if not findings_prior:
        findings_prior = heuristic_findings_summary(dict(state), log_anchors=anchors)

    patterns: list[Any] = list(state.get("detected_patterns") or [])
    pattern_strings = _trim_pattern_strings(patterns)
    le = (state.get("log_errors_analysis") or "").strip()
    lw = (state.get("log_warnings_analysis") or "").strip()
    mem = (state.get("memory_context") or "").strip()

    tool_signal_summary = rca_signal_summary_tool(
        parsed,
        patterns,
        log_errors_analysis=le,
        log_warnings_analysis=lw,
        memory_present=bool(mem),
    )
    log_tool_run("recommendation", "rca_signal_summary_tool", tool_signal_summary)

    keywords_s = _format_keywords(parsed)
    patterns_s = ", ".join(pattern_strings) if pattern_strings else "(none)"
    errors_summary = _signal_excerpt(le)
    warnings_summary = _signal_excerpt(lw)
    findings_for_prompt = findings_prior[:10_000] + ("…" if len(findings_prior) > 10_000 else "")
    lai = (state.get("log_analysis_insight") or "").strip()
    lap = (state.get("log_analysis_plan_failure") or "").strip()
    lar = (state.get("log_analysis_recommendation") or "").strip()
    lai_p = (lai[:6000] + ("…" if len(lai) > 6000 else "")) or "(none)"
    lap_p = (lap[:6000] + ("…" if len(lap) > 6000 else "")) or "(none)"
    lar_p = (lar[:6000] + ("…" if len(lar) > 6000 else "")) or "(none)"

    log_input_state(
        "recommendation",
        {
            "error_type": error_type,
            "module": module_s,
            "keywords": keywords_s[:500],
            "detected_patterns_n": len(pattern_strings),
            "findings_summary_chars": len(findings_prior),
            "root_cause": root_cause[:200],
            "tool_severity_baseline": tool_baseline,
        },
    )

    chain = _REC_PROMPT | get_recommendation_llm()
    content = invoke_llm_chain(
        chain,
        {
            "findings_summary": findings_for_prompt or "(none — derive from fields below)",
            "error_type": error_type or "(none)",
            "module": module_s,
            "keywords": keywords_s,
            "detected_patterns": patterns_s,
            "errors_summary": errors_summary,
            "warnings_summary": warnings_summary,
            "root_cause": root_cause or "(none)",
            "log_analysis_insight": lai_p,
            "log_analysis_plan_failure": lap_p,
            "log_analysis_recommendation": lar_p,
        },
    )
    log_llm_response("recommendation", content)

    used_fallback = extract_llm_json_object(content) is None
    data = parse_llm_json(content, default=_fallback_rec(tool_baseline))
    data = _coerce_rec(data, tool_baseline)

    try:
        validated = _RecommendationSchema.model_validate(data)
    except ValidationError:
        used_fallback = True
        validated = _RecommendationSchema.model_validate(_fallback_rec(tool_baseline))

    rec_text = str(validated.recommendation or "").strip()
    summary_text = str(validated.summary or "").strip()

    if not rec_text or not summary_text:
        hs, hr = _heuristic_grounded_stub(dict(state), parsed, pattern_strings, root_cause, findings_prior)
        if not summary_text:
            summary_text = hs
        if not rec_text or len(rec_text) < 40:
            rec_text = hr
    summary_text = normalize_to_n_lines(summary_text, 3)
    rec_text = normalize_to_n_lines(rec_text, 3)

    hist = [
        f"recommendation: severity_tool={tool_baseline}",
        "recommendation: model=RCA_RECOMMENDATION_MODEL / claude-opus-4-6 (anthropic)",
        "recommendation: strict_prompt(findings_summary+parsed_log+patterns+error/warn excerpts+root_cause)",
    ]
    if used_fallback:
        hist.append("recommendation: json_fallback")
    if not str(validated.recommendation or "").strip():
        hist.append("recommendation: filled_from_heuristic_stub")

    fs_final = (state.get("findings_summary") or "").strip() or findings_prior

    if not state.get("stop_after_preprocess"):
        save_incident(
            {
                "raw_log": raw_log,
                "root_cause": str(state.get("root_cause", "")),
                "recommendation": rec_text,
                "detected_patterns": list(state.get("detected_patterns") or []),
                "parsed_log": parsed,
            }
        )
        hist.append("memory: incident appended to persistent store")

    result = {
        "findings_summary": fs_final,
        "recommendation": rec_text,
        "recommendation_summary": summary_text,
        "severity": validated.severity,
        "history": hist,
    }
    log_parsed_output(
        "recommendation",
        {
            "findings_summary": result["findings_summary"][:500] + ("..." if len(result["findings_summary"]) > 500 else ""),
            "recommendation": result["recommendation"],
            "recommendation_summary": result["recommendation_summary"][:400],
            "severity": result["severity"],
            "history": result["history"],
            "json_fallback": used_fallback,
        },
    )
    return result

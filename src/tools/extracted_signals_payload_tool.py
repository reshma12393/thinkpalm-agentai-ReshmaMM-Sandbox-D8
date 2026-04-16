"""Tool: bundle first-level log extractions for the summary LLM (no LLM)."""

from __future__ import annotations

from typing import Any

# Keep preprocess lists bounded so severity narratives (LLM analyses) are never squeezed out of the prompt.
_MAX_PREPROCESS_LINES = 96
_MAX_PREPROCESS_LINE_CHARS = 800
_MAX_ANALYSIS_CHARS = 100_000
_MAX_LOG_ANALYSIS_CHARS = 12_000


def _trim_lines(raw: Any, *, max_lines: int, max_each: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if not s:
            continue
        if len(s) > max_each:
            s = s[: max_each - 1] + "…"
        out.append(s)
        if len(out) >= max_lines:
            break
    return out


def _cap_analysis_text(s: str, *, max_chars: int) -> str:
    t = (s or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def build_extracted_signals_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Structured payload: preprocess hits, scans, error/warning narratives, parse, patterns, optional memory.

    **log_errors_analysis** and **log_warnings_analysis** are placed first and capped last—they are the
    authoritative severity narratives from ``log_error_warning`` and must stay visible to the summary LLM.
    """
    le = _cap_analysis_text(str(state.get("log_errors_analysis") or ""), max_chars=_MAX_ANALYSIS_CHARS)
    lw = _cap_analysis_text(str(state.get("log_warnings_analysis") or ""), max_chars=_MAX_ANALYSIS_CHARS)
    lai = _cap_analysis_text(str(state.get("log_analysis_insight") or ""), max_chars=_MAX_LOG_ANALYSIS_CHARS)
    lap = _cap_analysis_text(str(state.get("log_analysis_plan_failure") or ""), max_chars=_MAX_LOG_ANALYSIS_CHARS)
    lar = _cap_analysis_text(str(state.get("log_analysis_recommendation") or ""), max_chars=_MAX_LOG_ANALYSIS_CHARS)
    return {
        "log_errors_analysis": le,
        "log_warnings_analysis": lw,
        "log_analysis_insight": lai,
        "log_analysis_plan_failure": lap,
        "log_analysis_recommendation": lar,
        "preprocess_error_hits": _trim_lines(
            state.get("preprocess_error_hits"),
            max_lines=_MAX_PREPROCESS_LINES,
            max_each=_MAX_PREPROCESS_LINE_CHARS,
        ),
        "preprocess_warning_hits": _trim_lines(
            state.get("preprocess_warning_hits"),
            max_lines=_MAX_PREPROCESS_LINES,
            max_each=_MAX_PREPROCESS_LINE_CHARS,
        ),
        "preprocess_summary": str(state.get("preprocess_summary") or ""),
        "preprocess_llm_digest": str(state.get("preprocess_llm_digest") or ""),
        "parsed_log": state.get("parsed_log") if isinstance(state.get("parsed_log"), dict) else {},
        "detected_patterns": list(state.get("detected_patterns") or []),
        "memory_context_preview": str(state.get("memory_context") or "")[:4000],
    }

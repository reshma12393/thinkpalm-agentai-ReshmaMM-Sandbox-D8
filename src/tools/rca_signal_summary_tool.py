"""Tool: compact non-LLM summary of signals for root-cause reasoning."""

from __future__ import annotations

from typing import Any


def rca_signal_summary_tool(
    parsed_log: dict[str, Any],
    detected_patterns: list[Any],
    *,
    log_errors_analysis: str = "",
    log_warnings_analysis: str = "",
    memory_present: bool = False,
) -> dict[str, Any]:
    """Aggregate structured fields into a small dict for the root-cause LLM (tool output only)."""
    et = str(parsed_log.get("error_type") or "").strip()
    mod = str(parsed_log.get("module") or "").strip()
    kw = parsed_log.get("keywords") if isinstance(parsed_log.get("keywords"), list) else []
    kw_s = [str(x).strip() for x in kw if str(x).strip()][:16]
    pat = [str(x).strip() for x in (detected_patterns or []) if str(x).strip()][:16]
    return {
        "parsed_error_type": et,
        "parsed_module": mod,
        "parsed_keywords": kw_s,
        "detected_patterns": pat,
        "error_analysis_chars": len(log_errors_analysis or ""),
        "warning_analysis_chars": len(log_warnings_analysis or ""),
        "similar_incidents_available": bool(memory_present),
    }

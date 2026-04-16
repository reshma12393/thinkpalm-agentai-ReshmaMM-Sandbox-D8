"""Tool: bucket error-like vs warning-like lines via same heuristics as preprocess — no LLM."""

from __future__ import annotations

from typing import Any

from utils.log_preprocess import scan_log_text


def severity_line_buckets_tool(raw_log: str) -> dict[str, Any]:
    """Return counts and capped samples for error vs warning buckets."""
    raw = (raw_log or "").strip()
    if not raw:
        return {
            "error_line_total": 0,
            "warning_line_total": 0,
            "error_samples": [],
            "warning_samples": [],
        }

    scan = scan_log_text(raw)
    return {
        "error_line_total": scan.error_line_total,
        "warning_line_total": scan.warning_line_total,
        "error_samples": [str(x)[:400] for x in scan.error_hits[:12]],
        "warning_samples": [str(x)[:400] for x in scan.warning_hits[:12]],
    }

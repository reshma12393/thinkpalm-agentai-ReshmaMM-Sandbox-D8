"""Tool: heuristic log scan (regex / line classification) — no LLM."""

from __future__ import annotations

from typing import Any

from utils.log_preprocess import build_preprocess_llm_digest, scan_log_text


def preprocess_scan_tool(raw_log: str) -> dict[str, Any]:
    """Run preprocess scan over raw log text.

    Returns structured fields consumed by downstream agents and prompts.
    """
    raw = raw_log or ""
    if not (raw or "").strip():
        return {
            "preprocess_error_hits": [],
            "preprocess_warning_hits": [],
            "preprocess_summary": "Pre-scan: empty input.",
            "preprocess_llm_digest": "",
            "error_line_total": 0,
            "warning_line_total": 0,
        }

    scan = scan_log_text(raw)
    digest = build_preprocess_llm_digest(scan)
    return {
        "preprocess_error_hits": scan.error_hits,
        "preprocess_warning_hits": scan.warning_hits,
        "preprocess_summary": scan.summary,
        "preprocess_llm_digest": digest,
        "error_line_total": scan.error_line_total,
        "warning_line_total": scan.warning_line_total,
    }

"""Severity classification from error_type string."""

from __future__ import annotations


def calculate_severity(error_type: str) -> str:
    """Map error_type text to High or Medium (case-insensitive substring match)."""
    t = (error_type or "").lower()
    if "failure" in t or "critical" in t:
        return "High"
    return "Medium"


def severity_baseline_tool(error_type: str) -> str:
    """Tool entrypoint for recommendation step (wraps :func:`calculate_severity`)."""
    return calculate_severity(error_type)

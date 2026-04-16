"""Keyword-to-failure-pattern lookup (dictionary-backed)."""

from __future__ import annotations

from .pattern_triggers import LOG_SUBSTRING_TRIGGERS

# Backwards-compatible name: substring match on normalized keyword text -> label
_PATTERN_TRIGGERS: dict[str, str] = LOG_SUBSTRING_TRIGGERS


def lookup_pattern(keywords: list) -> list:
    """Map keyword strings to known failure patterns; order preserved, deduplicated."""
    if not keywords:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        s = str(kw).lower().strip()
        if not s:
            continue
        for needle, label in _PATTERN_TRIGGERS.items():
            if needle in s and label not in seen:
                seen.add(label)
                out.append(label)
    return out


def lookup_failure_patterns_tool(keywords: list[str]) -> list[str]:
    """Tool entrypoint: deterministic pattern labels from keywords (wraps :func:`lookup_pattern`)."""
    print(f"Calling lookup_pattern with keywords: {keywords!r}", flush=True)
    return lookup_pattern(keywords)

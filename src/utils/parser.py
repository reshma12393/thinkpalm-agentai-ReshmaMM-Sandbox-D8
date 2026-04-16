"""Safely extract a JSON object from LLM text."""

from __future__ import annotations

import copy
import json
import re
from typing import Any


def _json_candidates(text: str) -> list[str]:
    """Ordered substrings to try with :func:`json.loads`."""
    t = (text or "").strip()
    out: list[str] = []
    if not t:
        return out

    out.append(t)

    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.DOTALL)
    if m:
        out.append(m.group(1).strip())

    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        out.append(t[start : end + 1])

    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def extract_llm_json_object(response: str) -> dict[str, Any] | None:
    """Return the first JSON object parsed from ``response``, or ``None`` if none."""
    for cand in _json_candidates(response):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_llm_json(response: str, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract the first JSON object from LLM output.

    Tries the full string, fenced `` ```json `` `` blocks, and a ``{``…``}`` slice.

    On failure, returns a deep copy of ``default`` if given, otherwise ``{}``.
    Malformed JSON is skipped until a candidate parses.
    """
    got = extract_llm_json_object(response)
    if got is not None:
        return got
    if default is not None:
        return copy.deepcopy(default)
    return {}

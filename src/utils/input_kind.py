"""Detect whether raw input is JSON (plan) or plain text (log)."""

from __future__ import annotations

import json
from typing import Any, Literal


def classify_input(raw: str) -> tuple[Literal["plan", "log"], Any | None]:
    """Return ``(\"plan\", parsed)`` for JSON object/array, else ``(\"log\", None)``.

    Root-level JSON strings/numbers/bools are treated as **log** text so that
    pasted prose is not misclassified as a plan.
    """
    s = (raw or "").strip()
    if not s:
        return "log", None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return "log", None
    if isinstance(obj, (dict, list)):
        return "plan", obj
    return "log", None

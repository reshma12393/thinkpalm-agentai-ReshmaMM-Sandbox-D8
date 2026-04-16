"""Classify input: JSON plan vs plain-text log."""

from __future__ import annotations

import json
from typing import Any

from graphs.state import GraphState
from utils.input_kind import classify_input


def classify_input_agent(state: GraphState) -> dict[str, Any]:
    raw = state.get("raw_log") or ""
    kind, payload = classify_input(raw)
    hist = [f"classify_input: input_kind={kind}"]
    out: dict[str, Any] = {"input_kind": kind, "history": hist}
    if kind == "plan" and payload is not None:
        out["plan_json"] = json.dumps(payload, ensure_ascii=False)[:100_000]
    else:
        out["plan_json"] = ""
    return out

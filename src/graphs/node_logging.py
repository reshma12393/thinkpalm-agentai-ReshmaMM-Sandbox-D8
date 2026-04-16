"""Wrap LangGraph node callables with stdout logging for debugging."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from graphs.state import GraphState


def _quiet() -> bool:
    return os.getenv("RCA_GRAPH_QUIET", "").lower() in ("1", "true", "yes")


def _trunc(text: str, max_len: int = 160) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _format_state_summary(state: dict[str, Any]) -> str:
    """Compact view of shared state relevant for RCA debugging."""
    chunks: list[str] = []
    if "raw_log" in state and state["raw_log"]:
        chunks.append(f"raw_log={_trunc(str(state['raw_log']), 120)!r}")
    pl = state.get("parsed_log")
    if isinstance(pl, dict) and pl:
        chunks.append("parsed_log=" + _trunc(json.dumps(pl, ensure_ascii=False), 200))
    dp = state.get("detected_patterns")
    if isinstance(dp, list) and dp:
        chunks.append(f"detected_patterns={dp!r}")
    if state.get("root_cause"):
        chunks.append(f"root_cause={_trunc(str(state['root_cause']), 120)!r}")
    if state.get("confidence") is not None:
        chunks.append(f"confidence={state['confidence']!r}")
    if state.get("severity"):
        chunks.append(f"severity={state['severity']!r}")
    if state.get("recommendation"):
        chunks.append(f"recommendation={_trunc(str(state['recommendation']), 120)!r}")
    if state.get("input_kind"):
        chunks.append(f"input_kind={state['input_kind']!r}")
    if state.get("preprocess_summary"):
        chunks.append(f"preprocess={_trunc(str(state['preprocess_summary']), 80)!r}")
    if state.get("plan_feasible") is not None:
        chunks.append(f"plan_feasible={state['plan_feasible']!r}")
    if state.get("log_errors_analysis"):
        chunks.append(f"log_errors_analysis={_trunc(str(state['log_errors_analysis']), 100)!r}")
    if state.get("log_warnings_analysis"):
        chunks.append(f"log_warnings_analysis={_trunc(str(state['log_warnings_analysis']), 100)!r}")
    hist = state.get("history") or []
    if hist:
        chunks.append(f"history_tail={hist[-2:]!r}")
    return " | ".join(chunks) if chunks else "(empty)"


def _format_update(update: dict[str, Any]) -> str:
    """Summarize node return dict (partial state update)."""
    if not update:
        return "{}"
    parts: list[str] = []
    for k, v in update.items():
        if k == "history":
            parts.append(f"history+={_trunc(repr(v), 240)}")
            continue
        if isinstance(v, dict):
            parts.append(f"{k}={_trunc(json.dumps(v, ensure_ascii=False), 180)}")
        elif isinstance(v, list):
            parts.append(f"{k}={_trunc(repr(v), 200)}")
        else:
            parts.append(f"{k}={v!r}")
    return " | ".join(parts)


def with_step_logging(
    node_id: str,
    agent_fn: Callable[[GraphState], dict[str, Any]],
) -> Callable[[GraphState], dict[str, Any]]:
    """Delegate to ``agent_fn`` (uses ``utils.llm.get_llm()``); log inputs/outputs."""

    def _node(state: GraphState) -> dict[str, Any]:
        if not _quiet():
            print(f"\n[rca-graph] >>> enter node={node_id!r}")
            print(f"[rca-graph]     state_in: {_format_state_summary(state)}")
        out = agent_fn(state)
        if not _quiet():
            print(f"[rca-graph]     update_out: {_format_update(out)}")
            print(f"[rca-graph] <<< exit  node={node_id!r}")
        return out

    return _node

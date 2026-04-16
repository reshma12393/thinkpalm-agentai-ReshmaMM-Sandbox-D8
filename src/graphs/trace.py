"""Collect LangGraph node-by-node trace via streaming (single run)."""

from __future__ import annotations

from typing import Any

from graphs.node_logging import _format_update

GRAPH_LEGEND_PLAN = (
    "Plan branch: **classify_input** → **plan_preprocess** → **plan_analysis** (LLM) → **plan_narrative** "
    "(natural-language findings_summary, root_cause, recommendation). "
    "Preprocess-only: **plan_preprocess** → END."
)
GRAPH_LEGEND_LOG = (
    "Log branch: **classify_input** → **log_signal_pipeline** → **log_analyser** (Claude Sonnet on errors/warnings/exceptions) → "
    "**log_findings_summary** → **root_cause** → **recommendation** → END. "
    "Preprocess-only: pipeline → END."
)


def legend_for_input_kind(input_kind: str) -> str:
    return GRAPH_LEGEND_PLAN if input_kind == "plan" else GRAPH_LEGEND_LOG


def _unpack_astream_chunk(item: Any) -> tuple[str, Any] | None:
    """Normalize LangGraph ``astream`` chunks to ``(mode, payload)``.

    Handles:
    - ``(mode, payload)`` — usual shape when ``stream_mode`` is a list of modes
    - ``(namespace, mode, payload)`` — when subgraph streaming or newer runners emit a namespace prefix
    - ``{"type": "updates"|"values", "data": ...}`` — LangGraph v2 stream records
    """
    if isinstance(item, dict):
        t = item.get("type")
        data = item.get("data")
        if t == "values" and isinstance(data, dict):
            return "values", data
        if t == "updates" and isinstance(data, dict):
            return "updates", data
        return None
    if not isinstance(item, tuple) or len(item) < 2:
        return None
    if len(item) == 2:
        mode, payload = item[0], item[1]
        return str(mode), payload
    if len(item) == 3:
        _ns, mode, payload = item
        return str(mode), payload
    return None


async def ainvoke_with_trace(
    compiled: Any,
    initial_state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the graph once; return final state and ordered trace steps.

    Each trace step: ``step``, ``node``, ``keys``, ``summary``.
    """
    trace_steps: list[dict[str, Any]] = []
    step_idx = 0
    final_state: dict[str, Any] | None = None

    async for raw in compiled.astream(
        initial_state,
        stream_mode=["updates", "values"],
    ):
        unpacked = _unpack_astream_chunk(raw)
        if unpacked is None:
            continue
        mode, payload = unpacked
        if mode == "values" and isinstance(payload, dict):
            final_state = payload
            continue
        if mode != "updates" or not isinstance(payload, dict):
            continue
        for node_name, update in payload.items():
            if str(node_name).startswith("__"):
                continue
            keys: list[str] = []
            summary = ""
            if isinstance(update, dict):
                keys = list(update.keys())
                summary = _format_update(update)
            else:
                summary = repr(update)[:500]
            trace_steps.append(
                {
                    "step": step_idx,
                    "node": str(node_name),
                    "keys": keys,
                    "summary": summary,
                }
            )
            step_idx += 1

    if final_state is None:
        raise RuntimeError("Graph produced no final state (values stream empty).")
    return final_state, trace_steps

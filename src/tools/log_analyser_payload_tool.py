"""Bundle extracted log signals for the dedicated log analyser LLM (Claude Sonnet)."""

from __future__ import annotations

from typing import Any

from tools.extracted_signals_payload_tool import build_extracted_signals_payload

_MAX_RAW_EXCERPT = 24_000


def _exception_and_failure_hints(parsed: dict[str, Any]) -> str:
    """Pull exception / failure-like strings from structured parse."""
    if not isinstance(parsed, dict):
        return ""
    chunks: list[str] = []
    for key in (
        "stack_trace",
        "exception",
        "error_message",
        "message",
        "detail",
        "failure_reason",
        "stderr",
    ):
        v = parsed.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            cap = 4000 if key == "stack_trace" else 2000
            if len(s) > cap:
                s = s[: cap - 1] + "…"
            chunks.append(f"{key}: {s}")
    return "\n".join(chunks) if chunks else ""


def build_log_analyser_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Same extractions as findings summary, plus raw excerpt and explicit exception hints."""
    base = build_extracted_signals_payload(state)
    raw = str(state.get("raw_log") or "")
    excerpt = raw[:_MAX_RAW_EXCERPT] + ("…" if len(raw) > _MAX_RAW_EXCERPT else "")

    pl = base.get("parsed_log")
    parsed = pl if isinstance(pl, dict) else {}
    hints = _exception_and_failure_hints(parsed)

    out: dict[str, Any] = dict(base)
    out["exception_and_parse_hints"] = hints
    out["raw_log_excerpt"] = excerpt
    out["task_focus"] = (
        "Interpret errors, warnings, exceptions, and patterns in the context of plan generation / optimization / "
        "solver runs when the text suggests it (feasibility, tolerance, constraints, cargo, JSON plan objects)."
    )
    return out

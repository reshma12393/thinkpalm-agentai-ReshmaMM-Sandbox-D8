"""Backward-compatible deterministic plan → RCA mapping (same as :func:`utils.plan_rca_fallback.deterministic_plan_rca`)."""

from __future__ import annotations

from typing import Any

from graphs.state import GraphState
from utils.plan_rca_fallback import deterministic_plan_rca


def plan_finalize_agent(state: GraphState) -> dict[str, Any]:
    """Map plan fields to RCA output without LLM (used for tests or direct calls)."""
    out = deterministic_plan_rca(dict(state))
    out["history"] = ["plan_finalize: deterministic RCA mapping (compat)"]
    return out

"""Structured INFO logging for deterministic tool calls (separate from LLM)."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("rca.tools")

_MAX = 12000


def _fmt(out: Any) -> str:
    if isinstance(out, (dict, list)):
        s = json.dumps(out, ensure_ascii=False, default=str, indent=2)
    else:
        s = str(out)
    if len(s) > _MAX:
        return s[:_MAX] + "\n... [truncated]"
    return s


def log_tool_run(agent: str, tool_name: str, output: Any) -> None:
    """Log which tool ran and what it returned (always at INFO for observability)."""
    logger.info("tool_call agent=%s tool=%s output=%s", agent, tool_name, _fmt(output))

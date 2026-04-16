"""Optional per-agent debug prints (LLM / JSON troubleshooting).

Uses ``get_config()['debug']`` (set via ``DEBUG`` or ``RCA_DEBUG`` in :mod:`utils.config` only).
"""

from __future__ import annotations

import json
from typing import Any

from utils.config import get_config

_MAX_CHARS = 8000


def agent_debug_enabled() -> bool:
    return bool(get_config()["debug"])


def _fmt(obj: Any) -> str:
    if isinstance(obj, (dict, list)):
        s = json.dumps(obj, ensure_ascii=False, default=str, indent=2)
    else:
        s = str(obj)
    if len(s) > _MAX_CHARS:
        return s[:_MAX_CHARS] + "\n... [truncated]"
    return s


def log_input_state(agent: str, state_slice: dict[str, Any]) -> None:
    """Print relevant input fields from graph state."""
    if not agent_debug_enabled():
        return
    print(f"\n[RCA_DEBUG] agent={agent} phase=input_state")
    print(_fmt(state_slice))


def log_llm_response(agent: str, text: str) -> None:
    """Print raw model text (before JSON parse)."""
    if not agent_debug_enabled():
        return
    print(f"\n[RCA_DEBUG] agent={agent} phase=llm_response")
    print(_fmt(text))


def log_parsed_output(agent: str, data: Any) -> None:
    """Print coerced / validated structure returned to the graph."""
    if not agent_debug_enabled():
        return
    print(f"\n[RCA_DEBUG] agent={agent} phase=parsed_output")
    print(_fmt(data))

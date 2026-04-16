"""Tool: load ranked past incidents for log RCA (disk-backed)."""

from __future__ import annotations

from typing import Any

from memory.relevance import format_memory_context, select_relevant_incidents


def load_relevant_incidents_tool(state_snapshot: dict[str, Any], *, limit: int = 5) -> dict[str, Any]:
    """Return selected incidents and formatted context string for downstream LLM payloads."""
    incidents = select_relevant_incidents(state_snapshot, limit=limit, min_score=0.03)
    text = format_memory_context(incidents, max_chars=4500)
    n = len(incidents)
    if n == 0:
        print("Loaded past incidents: 0 (store empty or no overlap)", flush=True)
    else:
        preview = []
        for i, inc in enumerate(incidents[:3]):
            if isinstance(inc, dict):
                ts = str(inc.get("timestamp", ""))[:19]
                rc = str(inc.get("root_cause", "")).replace("\n", " ")[:80]
                preview.append(
                    f"  [{i + 1}] {ts} — {rc}…" if len(str(inc.get("root_cause", ""))) > 80 else f"  [{i + 1}] {ts} — {rc}"
                )
        extra = f" (+{n - 3} more)" if n > 3 else ""
        lines = "\n".join(preview)
        print(
            f"Loaded past incidents: {n}{extra}\n{lines}",
            flush=True,
        )
    return {"memory_relevant_incidents": incidents, "memory_context": text}

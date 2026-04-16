"""Tool: extract timestamps, PIDs, and IDs from raw log text for grounded recommendations (no LLM)."""

from __future__ import annotations

import re
from typing import Any


_TS_PATTERNS = [
    re.compile(
        r"\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\b"
    ),
    re.compile(r"\b(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\b"),
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"),
]

_PID_PATTERNS = [
    re.compile(r"(?i)\b(?:pid|process\s*id)\s*[:=]\s*(\d+)\b"),
    re.compile(r"(?i)\bprocess(?:\s+|\[)(\d{2,8})\b"),
    re.compile(r"\[(\d+)\]\s*$"),
]

_ID_PATTERNS = [
    re.compile(r"(?i)(?:trace[_-]?id|request[_-]?id|correlation[_-]?id|x-request-id)\s*[:=]\s*([\w\-]{8,64})", re.I),
    re.compile(r"(?i)\b(?:req[_-]?id|txn[_-]?id)\s*[:=]\s*([\w\-]{6,48})", re.I),
]


def _dedupe_cap(seq: list[str], cap: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= cap:
            break
    return out


def extract_log_context_anchors_tool(raw_log: str) -> dict[str, Any]:
    """Return short lists of timestamps, process IDs, and correlation IDs for LLM grounding."""
    text = raw_log or ""
    if not text.strip():
        return {
            "timestamps": [],
            "process_ids": [],
            "correlation_ids": [],
            "line_count": 0,
        }

    ts: list[str] = []
    for pat in _TS_PATTERNS:
        for m in pat.finditer(text):
            ts.append(m.group(1) if m.lastindex else m.group(0))
    pids: list[str] = []
    for pat in _PID_PATTERNS:
        for m in pat.finditer(text):
            g = m.group(1) if m.lastindex else m.group(0)
            if g.isdigit():
                pids.append(g)
    cids: list[str] = []
    for pat in _ID_PATTERNS:
        for m in pat.finditer(text):
            cids.append(m.group(1))

    return {
        "timestamps": _dedupe_cap(ts, 8),
        "process_ids": _dedupe_cap(pids, 6),
        "correlation_ids": _dedupe_cap(cids, 6),
        "line_count": len(text.splitlines()),
    }

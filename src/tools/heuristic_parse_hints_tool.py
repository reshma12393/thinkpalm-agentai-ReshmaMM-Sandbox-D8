"""Tool: deterministic hints for log parsing (exception names, file hints) — no LLM."""

from __future__ import annotations

import re
from typing import Any

from utils.log_preprocess import scan_log_text


def heuristic_parse_hints_tool(raw_log: str) -> dict[str, Any]:
    """Extract keyword/exception/path hints from heuristic scan for the parse LLM to refine."""
    raw = (raw_log or "").strip()
    if not raw:
        return {
            "keywords": [],
            "exception_hints": [],
            "file_path_hints": [],
            "error_sample_lines": [],
        }

    scan = scan_log_text(raw)
    keywords: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = s.strip()
        if s and len(s) >= 2 and s not in seen:
            seen.add(s)
            keywords.append(s[:200])

    exc_re = re.compile(
        r"\b([A-Z][a-zA-Z0-9]*(?:Error|Exception|Failure|Timeout|Death))\b"
    )
    file_re = re.compile(r'File "([^"]+)"')
    for line in scan.error_hits[:16]:
        for m in exc_re.finditer(line):
            add(m.group(1))
        for m in file_re.finditer(line):
            path = m.group(1)
            leaf = path.replace("\\", "/").split("/")[-1]
            if leaf:
                add(leaf)

    exceptions = [k for k in keywords if k.endswith(("Error", "Exception", "Failure", "Timeout"))]
    return {
        "keywords": keywords[:24],
        "exception_hints": exceptions[:12],
        "error_sample_lines": [str(x)[:300] for x in scan.error_hits[:6]],
    }

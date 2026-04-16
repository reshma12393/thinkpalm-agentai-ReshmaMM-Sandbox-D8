"""Rank past incidents by overlap with the current log parse and format prompts."""

from __future__ import annotations

import re
from typing import Any

from memory.store import load_history

_STOP = frozenset(
    "a an the and or for to of in on at is are was were be been being as by it its this that with from "
    "have has had not no but if when into than then also only same such".split()
)


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_./:@-]{3,}", (text or "").lower())
    return {w for w in words if w not in _STOP}


def _score_match(current: dict[str, Any], rec: dict[str, Any]) -> float:
    cur_parts = [
        str(current.get("raw_log", "")),
        str(current.get("log_errors_analysis", "")),
        str(current.get("log_warnings_analysis", "")),
    ]
    cur_toks = _tokenize(" ".join(cur_parts))
    pl = current.get("parsed_log")
    if isinstance(pl, dict):
        for key in ("error_type", "module"):
            v = str(pl.get(key) or "").strip()
            if len(v) >= 2:
                cur_toks.add(v.lower())
        for w in pl.get("keywords") or []:
            s = str(w).strip().lower()
            if len(s) >= 2:
                cur_toks.add(s)
    if not cur_toks:
        return 0.0

    past_parts = [
        str(rec.get("raw_log", "")),
        str(rec.get("root_cause", "")),
        str(rec.get("recommendation", "")),
    ]
    for p in rec.get("detected_patterns") or []:
        past_parts.append(str(p))
    pl_p = rec.get("parsed_log")
    if isinstance(pl_p, dict):
        for key in ("error_type", "module"):
            v = str(pl_p.get(key) or "").strip()
            if v:
                past_parts.append(v)
        for w in pl_p.get("keywords") or []:
            past_parts.append(str(w))
    past_toks = _tokenize(" ".join(past_parts))
    if not past_toks:
        return 0.0

    inter = cur_toks & past_toks
    union = cur_toks | past_toks
    jacc = len(inter) / len(union) if union else 0.0

    bonus = 0.0
    if isinstance(pl, dict) and isinstance(pl_p, dict):
        cm = str(pl.get("module") or "").strip().lower()
        pm = str(pl_p.get("module") or "").strip().lower()
        if cm and pm and cm == pm:
            bonus += 0.12
        ce = str(pl.get("error_type") or "").strip().lower()
        pe = str(pl_p.get("error_type") or "").strip().lower()
        if ce and pe and ce == pe:
            bonus += 0.1

    return min(1.0, jacc + bonus)


def select_relevant_incidents(
    current: dict[str, Any],
    *,
    limit: int = 5,
    min_score: float = 0.03,
) -> list[dict[str, Any]]:
    """Return highest-scoring past incidents (excluding exact same raw_log as current)."""
    history = load_history()
    if not history:
        return []

    cur_raw = str(current.get("raw_log", "") or "").strip()
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for rec in history:
        if not isinstance(rec, dict):
            continue
        pr = str(rec.get("raw_log", "") or "").strip()
        if cur_raw and pr == cur_raw:
            continue
        s = _score_match(current, rec)
        ts = str(rec.get("timestamp") or "")
        scored.append((s, ts, rec))

    scored.sort(key=lambda x: (-x[0], x[1]))
    strong = [rec for s, _ts, rec in scored if s >= min_score][:limit]
    if strong:
        return strong
    weak = [rec for s, _ts, rec in scored if s > 0][: min(2, limit)]
    return weak


def format_memory_context(incidents: list[dict[str, Any]], *, max_chars: int = 4500) -> str:
    """Compact text block for LLM prompts."""
    if not incidents:
        return ""
    blocks: list[str] = []
    for i, inc in enumerate(incidents, 1):
        ts = str(inc.get("timestamp") or "")[:32]
        rc = str(inc.get("root_cause") or "").strip()
        rec = str(inc.get("recommendation") or "").strip()
        patterns = inc.get("detected_patterns")
        pat_s = ""
        if isinstance(patterns, list) and patterns:
            pat_s = "Patterns: " + ", ".join(str(p) for p in patterns[:8] if str(p).strip())
        snippet = str(inc.get("raw_log") or "").strip().replace("\n", " ")[:400]
        parts = [f"[{i}] t={ts}"]
        if pat_s:
            parts.append(pat_s)
        if rc:
            parts.append("Root cause: " + rc[:900])
        if rec:
            parts.append("Recommendation: " + rec[:700])
        if snippet:
            parts.append("Log snippet: " + snippet + ("…" if len(str(inc.get("raw_log", ""))) > 400 else ""))
        blocks.append("\n".join(parts))
    text = "\n\n---\n\n".join(blocks)
    return text[:max_chars]


def pattern_hints_from_incidents(incidents: list[dict[str, Any]], *, cap: int = 16) -> list[str]:
    """Deduped pattern labels from stored incidents for merging with rule/LLM output."""
    seen: set[str] = set()
    out: list[str] = []
    for inc in incidents:
        if not isinstance(inc, dict):
            continue
        for p in inc.get("detected_patterns") or []:
            s = str(p).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= cap:
                return out
    return out

"""Heuristic log RCA text when LLM steps fail JSON parsing or return empty strings."""

from __future__ import annotations

import json
from typing import Any

from utils.log_preprocess import summarize_hit_themes
from utils.rca_text_format import normalize_to_n_lines


def _lines_from_hits(key: str, state: dict[str, Any], *, cap: int = 48) -> list[str]:
    raw = state.get(key)
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if s:
            out.append(s)
        if len(out) >= cap:
            break
    return out


def preprocess_error_warning_blocks(state: dict[str, Any]) -> tuple[str, str]:
    """Brief prescan **summaries** from heuristic hits (no raw listing of every line)."""
    pe = _lines_from_hits("preprocess_error_hits", state, cap=16)
    pw = _lines_from_hits("preprocess_warning_hits", state, cap=16)
    err = ""
    warn = ""
    if pe:
        err = (
            f"Pre-scan summary (error-class): {len(pe)} sample line(s) stored in state; "
            f"aggregate themes from tags: {summarize_hit_themes(pe)}. "
            "(Verbatim samples live in preprocess_error_hits.)"
        )
    if pw:
        warn = (
            f"Pre-scan summary (warning-class): {len(pw)} sample line(s) stored; "
            f"themes: {summarize_hit_themes(pw)}."
        )
    return err, warn


def heuristic_root_cause(state: dict[str, Any]) -> tuple[str, float]:
    """Three-line diagnosis from structured fields and narratives when the LLM fails."""
    pl = state.get("parsed_log")
    parsed = pl if isinstance(pl, dict) else {}
    et = str(parsed.get("error_type") or "").strip()
    mod = str(parsed.get("module") or "").strip()
    kw_list = parsed.get("keywords") if isinstance(parsed.get("keywords"), list) else []
    kw_s = ", ".join(str(x).strip() for x in kw_list if str(x).strip())[:500]

    le = (state.get("log_errors_analysis") or "").strip()
    lw = (state.get("log_warnings_analysis") or "").strip()
    lai = (state.get("log_analysis_insight") or "").strip()
    pat_list = [str(x).strip() for x in (state.get("detected_patterns") or []) if str(x).strip()][:8]

    if et or mod or kw_s or le or lw or lai or pat_list:
        if et or mod:
            line1 = f"Structured classification: error_type «{et or '—'}», module «{mod or '—'}»."
        elif pat_list:
            line1 = (
                "Extracted patterns (no parser error_type/module): "
                + "; ".join(pat_list[:5])
                + ("…" if len(pat_list) > 5 else "")
            )
        elif le or lw:
            line1 = "Extracted error/warning narratives (parser classification sparse); see condensed text below."
        else:
            line1 = f"Structured classification: error_type «{et or '—'}», module «{mod or '—'}»."
        line2 = f"Parser keywords: {kw_s or '—'}."
        if le:
            line3 = "Error narrative (condensed): " + le[:450] + ("…" if len(le) > 450 else "")
        elif lw:
            line3 = "Warning narrative (condensed): " + lw[:450] + ("…" if len(lw) > 450 else "")
        elif lai:
            line3 = "Analyser note (condensed): " + lai[:450] + ("…" if len(lai) > 450 else "")
        elif pat_list:
            line3 = "Pattern-only view: triage using the labels in line 1 and similar lines in the raw log."
        else:
            line3 = "No long-form error or warning narrative in state beyond parser fields."
        return normalize_to_n_lines("\n".join([line1, line2, line3]), 3), 0.52

    e, w = preprocess_error_warning_blocks(state)
    parts = [p for p in (e, w) if p]
    if parts:
        return normalize_to_n_lines("\n\n".join(parts), 3), 0.5

    raw = (state.get("raw_log") or "").strip()
    if raw:
        excerpt = raw[:2000] + ("…" if len(raw) > 2000 else "")
        return normalize_to_n_lines(
            "Parsing produced limited structure; inspect the raw log for failure context.\n"
            f"Excerpt (truncated): {excerpt}",
            3,
        ), 0.35
    return normalize_to_n_lines("No log text or pre-scan signals were available.", 3), 0.2


def heuristic_recommendation(state: dict[str, Any], root_cause: str) -> str:
    """Actionable stub when the recommendation LLM returns nothing."""
    pat_raw = state.get("detected_patterns")
    pats = [str(x).strip() for x in (pat_raw or []) if str(x).strip()][:12]
    pat_suffix = ""
    if pats:
        pat_suffix = " Detected pattern labels to verify against: " + "; ".join(pats) + "."

    rc = (root_cause or "").strip()
    if rc:
        return (
            "Confirm the failing component and environment, reproduce with the same inputs, apply a minimal fix "
            "(config, code, or infrastructure), then re-run the workflow or health checks. "
            "Use the pre-scan lines in the diagnosis if timestamps or line numbers are available."
        ) + pat_suffix
    e, w = preprocess_error_warning_blocks(state)
    if e or w:
        parts = [p for p in (e, w) if p]
        return "Focus on these pre-scan signals first:\n\n" + "\n\n".join(parts)
    raw = (state.get("raw_log") or "").strip()
    if raw:
        return (
            "Reproduce the failure, isolate the first ERROR/exception block, and fix that subsystem "
            f"(excerpt: {raw[:600]}{'…' if len(raw) > 600 else ''})."
        )
    return "Provide a fuller log (timestamps, stack traces) and re-run analysis."


def heuristic_findings_summary(
    state: dict[str, Any],
    *,
    log_anchors: dict[str, Any] | None = None,
) -> str:
    """Short findings paragraph when the recommendation LLM does not return findings_summary."""
    le = (state.get("log_errors_analysis") or "").strip()
    lw = (state.get("log_warnings_analysis") or "").strip()
    rc = (state.get("root_cause") or "").strip()
    parts: list[str] = ["Summary of findings (heuristic):"]
    if le:
        parts.append("Error-class signals: " + le[:800] + ("…" if len(le) > 800 else ""))
    if lw:
        parts.append("Warning-class signals: " + lw[:600] + ("…" if len(lw) > 600 else ""))
    if not le and not lw:
        e, w = preprocess_error_warning_blocks(state)
        if e or w:
            parts.append("Pre-scan flagged error/warning-like lines.")
    anchor_bits: list[str] = []
    if isinstance(log_anchors, dict):
        ts = log_anchors.get("timestamps") or []
        pids = log_anchors.get("process_ids") or []
        cids = log_anchors.get("correlation_ids") or []
        if ts:
            anchor_bits.append(f"timestamps seen in log (sample): {', '.join(str(t) for t in ts[:3])}")
        if pids:
            anchor_bits.append(f"process IDs referenced: {', '.join(str(p) for p in pids[:4])}")
        if cids:
            anchor_bits.append(f"correlation/request ids: {', '.join(str(c) for c in cids[:3])}")
    if anchor_bits:
        parts.append("Context anchors: " + "; ".join(anchor_bits) + ".")
    if rc:
        parts.append(f"Root cause line: {rc[:500]}")
    return " ".join(parts).strip()


def heuristic_extracted_signals_summary(state: dict[str, Any]) -> str:
    """Non-LLM summary stitched from first-level extractions (for log_findings_summary fallback)."""
    chunks: list[str] = []
    pe = _lines_from_hits("preprocess_error_hits", state, cap=12)
    pw = _lines_from_hits("preprocess_warning_hits", state, cap=12)
    if pe:
        chunks.append(
            "Pre-scan error/failure-like samples (" + str(len(pe)) + "): " + " | ".join(x[:200] for x in pe[:8])
        )
    if pw:
        chunks.append(
            "Pre-scan warning-like samples (" + str(len(pw)) + "): " + " | ".join(x[:200] for x in pw[:8])
        )
    ps = str(state.get("preprocess_summary") or "").strip()
    if ps:
        chunks.append("Pre-scan overview: " + ps[:1500])
    le = str(state.get("log_errors_analysis") or "").strip()
    lw = str(state.get("log_warnings_analysis") or "").strip()
    if le:
        chunks.append("Error narrative (LLM split): " + le[:2500])
    if lw:
        chunks.append("Warning narrative (LLM split): " + lw[:2500])
    lai = str(state.get("log_analysis_insight") or "").strip()
    lap = str(state.get("log_analysis_plan_failure") or "").strip()
    if lai:
        chunks.append("Log analyser insight (Sonnet): " + lai[:2000])
    if lap:
        chunks.append("Plan-generation angle (log analyser): " + lap[:1500])
    pl = state.get("parsed_log")
    if isinstance(pl, dict) and (pl.get("error_type") or pl.get("module") or pl.get("keywords")):
        chunks.append(
            "Parsed fields: "
            + json.dumps(
                {
                    "error_type": pl.get("error_type", ""),
                    "module": pl.get("module", ""),
                    "keywords": (pl.get("keywords") or [])[:12],
                },
                ensure_ascii=False,
            )
        )
    pat = state.get("detected_patterns")
    if isinstance(pat, list) and pat:
        chunks.append("Detected patterns: " + ", ".join(str(x) for x in pat[:12]))
    mem = str(state.get("memory_context") or "").strip()
    if mem:
        chunks.append("Similar past incidents (preview): " + mem[:1200])
    if not chunks:
        return (
            "No structured extractions were available to summarize.\n"
            "Confirm the log pipeline completed (preprocess, parser, error/warning split).\n"
            "Re-submit log text and run analysis again."
        )
    blob = "\n\n".join(chunks)[:8000]
    return normalize_to_n_lines(blob, 3)

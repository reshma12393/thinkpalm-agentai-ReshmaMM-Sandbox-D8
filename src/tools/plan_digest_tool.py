"""Tool: build plan assessment digest (structural + extracted JSON signals) — no LLM."""

from __future__ import annotations

from typing import Any

from utils.plan_preprocess import format_plan_digest_for_llm, resolve_plan_skeleton_outline


def plan_assessment_digest_tool(state: dict[str, Any]) -> str:
    """Produce the same digest string previously built inline before plan LLM."""
    pe = [str(x).strip() for x in (state.get("plan_preprocess_errors") or []) if str(x).strip()]
    pw = [str(x).strip() for x in (state.get("plan_preprocess_warnings") or []) if str(x).strip()]
    extracted = {
        "warnings": list(state.get("plan_extracted_warnings") or []),
        "errors": list(state.get("plan_extracted_errors") or []),
        "messages": list(state.get("plan_extracted_messages") or []),
    }
    sk = resolve_plan_skeleton_outline(state)
    sk5 = str(state.get("plan_skeleton_summary") or "").strip()
    return format_plan_digest_for_llm(
        structural_errors=pe,
        structural_warnings=pw,
        extracted=extracted,
        skeleton_outline=sk,
        include_skeleton=bool(sk),
        skeleton_five_sentences=sk5,
    )

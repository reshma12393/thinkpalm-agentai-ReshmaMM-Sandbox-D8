"""Agent: structural plan JSON preprocess (no LLM)."""

from __future__ import annotations

import json
from typing import Any

from graphs.state import GraphState
from utils.plan_preprocess import (
    analyze_plan_structure,
    compact_plan_outline,
    extract_plan_issue_signals,
    normalize_plan_for_llm,
    summarize_plan_skeleton_five_sentences,
)


def plan_preprocess_agent(state: GraphState) -> dict[str, Any]:
    raw = (state.get("plan_json") or "").strip()
    if not raw:
        return {
            "plan_preprocess_errors": [],
            "plan_preprocess_warnings": [],
            "plan_extracted_warnings": [],
            "plan_extracted_errors": [],
            "plan_extracted_messages": [],
            "plan_skeleton_outline": "",
            "plan_skeleton_summary": "",
            "history": ["plan_preprocess: empty plan_json"],
        }

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "plan_preprocess_errors": [f"Invalid JSON in plan: {e}"],
            "plan_preprocess_warnings": [],
            "plan_extracted_warnings": [],
            "plan_extracted_errors": [],
            "plan_extracted_messages": [],
            "plan_skeleton_outline": "",
            "plan_skeleton_summary": summarize_plan_skeleton_five_sentences(None, invalid_json=True),
            "history": ["plan_preprocess: json decode error"],
        }

    sig = extract_plan_issue_signals(obj)
    errs, warns = analyze_plan_structure(obj)
    normalized = normalize_plan_for_llm(obj)
    total_hits = len(sig["warnings"]) + len(sig["errors"]) + len(sig["messages"])
    use_skeleton = total_hits < 12 or (total_hits == 0 and not errs and not warns)
    skeleton = compact_plan_outline(obj) if use_skeleton else ""
    sk_summary = summarize_plan_skeleton_five_sentences(obj, skeleton_outline=skeleton)
    hist = [
        "plan_preprocess: "
        f"struct_errors={len(errs)} struct_warnings={len(warns)}; "
        f"issue_key_hits={total_hits} skeleton={'yes' if use_skeleton else 'no'}"
    ]
    return {
        "plan_json": normalized,
        "plan_preprocess_errors": errs,
        "plan_preprocess_warnings": warns,
        "plan_extracted_warnings": sig["warnings"],
        "plan_extracted_errors": sig["errors"],
        "plan_extracted_messages": sig["messages"],
        "plan_skeleton_outline": skeleton,
        "plan_skeleton_summary": sk_summary,
        "history": hist,
    }

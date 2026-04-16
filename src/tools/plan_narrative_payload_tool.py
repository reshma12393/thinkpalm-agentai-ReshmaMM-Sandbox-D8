"""Bundle plan feasibility, structural signals, and merged assessment for the plan narrative LLM."""

from __future__ import annotations

from typing import Any

_MAX_LIST_ITEMS = 64
_MAX_ITEM_CHARS = 600
_MAX_ANALYSIS = 24_000
_MAX_SKELETON = 12_000
_MAX_SKEL_SUMMARY = 4_000
_MAX_JSON_EXCERPT = 16_000

_SKEL_OUT_HEAD = "Plan skeleton (compact outline):"
# Legacy headings (no parenthetical); match ``plan_analysis_agent`` output.
_SKEL_SUM_HEADS = (
    "Plan skeleton summary:",
    "Plan skeleton summary :",  # legacy stray space
)


def _plan_deep_analysis_without_duplicate_skeleton_block(full: str) -> str:
    """Remove embedded skeleton summary from plan_deep_analysis; ``plan_skeleton_summary`` is a separate payload key."""
    text = (full or "").strip()
    for head in _SKEL_SUM_HEADS:
        if head not in text:
            continue
        i = text.find(head)
        before = text[:i].rstrip()
        tail = text[i + len(head) :].lstrip()
        j = tail.find(_SKEL_OUT_HEAD)
        if j != -1:
            from_outline = tail[j:].strip()
            if before and from_outline:
                return f"{before}\n\n{from_outline}"
            return from_outline or before
        return before
    return text


def _trim_str_list(raw: Any, *, cap_items: int, cap_each: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if not s:
            continue
        if len(s) > cap_each:
            s = s[: cap_each - 1] + "…"
        out.append(s)
        if len(out) >= cap_items:
            break
    return out


def build_plan_narrative_payload(state: dict[str, Any]) -> dict[str, Any]:
    """JSON-oriented input for natural-language plan summary + recommendation."""
    feasible = state.get("plan_feasible")
    analysis = _plan_deep_analysis_without_duplicate_skeleton_block(str(state.get("plan_deep_analysis") or ""))
    if len(analysis) > _MAX_ANALYSIS:
        analysis = analysis[: _MAX_ANALYSIS - 1] + "…"

    sk = str(state.get("plan_skeleton_outline") or "")
    if len(sk) > _MAX_SKELETON:
        sk = sk[: _MAX_SKELETON - 1] + "…"

    sk_sum = str(state.get("plan_skeleton_summary") or "")
    if len(sk_sum) > _MAX_SKEL_SUMMARY:
        sk_sum = sk_sum[: _MAX_SKEL_SUMMARY - 1] + "…"

    pj = str(state.get("plan_json") or "")
    if len(pj) > _MAX_JSON_EXCERPT:
        pj = pj[: _MAX_JSON_EXCERPT - 1] + "…"

    return {
        "plan_feasible": feasible,
        "merged_warnings": _trim_str_list(state.get("plan_warnings_list"), cap_items=_MAX_LIST_ITEMS, cap_each=_MAX_ITEM_CHARS),
        "merged_errors": _trim_str_list(state.get("plan_errors_list"), cap_items=_MAX_LIST_ITEMS, cap_each=_MAX_ITEM_CHARS),
        "merged_messages": _trim_str_list(state.get("plan_messages_list"), cap_items=_MAX_LIST_ITEMS, cap_each=_MAX_ITEM_CHARS),
        "structural_preprocess_errors": _trim_str_list(
            state.get("plan_preprocess_errors"), cap_items=_MAX_LIST_ITEMS, cap_each=_MAX_ITEM_CHARS
        ),
        "structural_preprocess_warnings": _trim_str_list(
            state.get("plan_preprocess_warnings"), cap_items=_MAX_LIST_ITEMS, cap_each=_MAX_ITEM_CHARS
        ),
        "extracted_issue_errors": _trim_str_list(
            state.get("plan_extracted_errors"), cap_items=_MAX_LIST_ITEMS, cap_each=_MAX_ITEM_CHARS
        ),
        "extracted_issue_warnings": _trim_str_list(
            state.get("plan_extracted_warnings"), cap_items=_MAX_LIST_ITEMS, cap_each=_MAX_ITEM_CHARS
        ),
        "extracted_issue_messages": _trim_str_list(
            state.get("plan_extracted_messages"), cap_items=_MAX_LIST_ITEMS, cap_each=_MAX_ITEM_CHARS
        ),
        "plan_assessment_analysis": analysis,
        "plan_skeleton_summary": sk_sum,
        "plan_skeleton_outline": sk,
        "plan_json_excerpt": pj,
    }

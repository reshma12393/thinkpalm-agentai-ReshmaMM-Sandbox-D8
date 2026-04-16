"""Deterministic plan → RCA fields when the plan narrative LLM fails or returns empty."""

from __future__ import annotations

from typing import Any, Literal

from utils.rca_text_format import normalize_to_n_lines


def _severity(
    feasible: bool | None,
    errors: list[str],
    warnings: list[str],
) -> Literal["High", "Medium", "Low"]:
    if feasible is False or len(errors) > 0:
        return "High"
    if len(warnings) > 2:
        return "Medium"
    if len(warnings) > 0:
        return "Medium"
    return "Low"


def _confidence(feasible: bool | None, analysis: str) -> float:
    if feasible is None:
        return 0.35
    if not (analysis or "").strip():
        return 0.45
    return 0.82


def _join_bullets(items: list[str], *, cap: int = 24) -> str:
    lines = [f"- {str(x).strip()}" for x in items[:cap] if str(x).strip()]
    return "\n".join(lines)


def state_specific_recommendation_summary(state: dict[str, Any]) -> str:
    """Synthesize feasibility plus merged error/warning/message text from prior steps (three lines)."""
    feasible = state.get("plan_feasible")
    errors = [str(x).strip() for x in (state.get("plan_errors_list") or []) if str(x).strip()]
    warnings = [str(x).strip() for x in (state.get("plan_warnings_list") or []) if str(x).strip()]
    messages = [str(x).strip() for x in (state.get("plan_messages_list") or []) if str(x).strip()]
    analysis = (state.get("plan_deep_analysis") or "").strip()

    if feasible is True:
        line1 = "Prior assessment marked this plan as feasible as written."
    elif feasible is False:
        line1 = "Prior assessment marked this plan as infeasible or not executable as stated."
    else:
        line1 = "Feasibility was not established from the supplied plan input."

    err_ex = "; ".join(errors[:8])
    warn_ex = "; ".join(warnings[:8])
    if err_ex and warn_ex:
        line2 = f"Merged error signals include: {err_ex}. Merged warnings include: {warn_ex}."
    elif err_ex:
        line2 = f"Merged error signals include: {err_ex}."
    elif warn_ex:
        line2 = f"Merged warning signals include: {warn_ex}."
    else:
        line2 = "No error or warning strings were present in the merged lists from prior steps."

    msg_ex = "; ".join(messages[:8])
    if msg_ex and analysis:
        line3 = f"Messages: {msg_ex}. Assessor notes: {analysis[:900]}"
    elif msg_ex:
        line3 = f"Messages from prior steps: {msg_ex}."
    elif analysis:
        line3 = f"Assessor notes from prior step: {analysis[:1200]}"
    else:
        line3 = "No merged messages or assessor analysis text beyond the lists above."

    return normalize_to_n_lines("\n".join([line1, line2, line3]), 3)


def deterministic_plan_rca(state: dict[str, Any]) -> dict[str, Any]:
    """Map plan feasibility, merged lists, and assessment text to RCA output (no LLM)."""
    feasible = state.get("plan_feasible")
    warnings = list(state.get("plan_warnings_list") or [])
    errors = list(state.get("plan_errors_list") or [])
    messages = list(state.get("plan_messages_list") or [])
    analysis = (state.get("plan_deep_analysis") or "").strip()
    has_issues = bool(warnings or errors)

    findings_summary: str
    root: str
    recommendation: str

    if feasible is False:
        cause_bits: list[str] = []
        if errors:
            cause_bits.append("Reported errors and structural issues indicate the plan cannot run as written.")
        if warnings and not errors:
            cause_bits.append("Warnings and assessment suggest missing or conflicting steps.")
        if analysis:
            cause_bits.append(analysis[:1200])
        findings_summary = (
            "Summary of findings: The JSON plan was assessed as infeasible (not executable as stated). "
            + ("Key issues include: " + "; ".join(errors[:6]) + ". " if errors else "")
            + ("Further warnings: " + "; ".join(warnings[:4]) + ". " if warnings else "")
            + (f"Assessment notes: {analysis[:800]}" if analysis else "")
        ).strip()

        root = "Cause for infeasibility may include: " + (
            "; ".join(errors[:8]) if errors else "contradictions, missing prerequisites, or ordering that cannot succeed."
        )
        if analysis and len(root) < 400:
            root = f"{root} Additional detail: {analysis[:500]}"

        rec_parts: list[str] = []
        rec_parts.append(
            "Recommendations: revise the plan to remove blockers—address each error above, "
            "validate dependencies and ordering, then re-run assessment."
        )
        if messages:
            rec_parts.append("Clarifications to pursue:\n" + _join_bullets(messages))
        if warnings:
            rec_parts.append("Before retrying, also resolve or accept these warnings:\n" + _join_bullets(warnings))
        recommendation = "\n\n".join(rec_parts)

    elif feasible is True:
        if has_issues:
            findings_summary = (
                "Summary of findings: The plan is feasible, but warnings and/or errors were still flagged "
                "(from structure, extracted keys, or assessment). "
                f"Warnings ({len(warnings)}): " + ("; ".join(warnings[:8]) if warnings else "none") + ". "
                f"Errors ({len(errors)}): " + ("; ".join(errors[:8]) if errors else "none") + ". "
                + (f"Analysis: {analysis[:600]}" if analysis else "")
            ).strip()
            root = (
                "The plan can likely be executed; residual warnings or errors should be reviewed before production use."
            )
            recommendation = (
                "Recommendations: considering these warnings and errors, you may obtain a more optimized or safer plan by "
                "tightening unclear steps, adding rollback or validation gates, and resolving duplicate or conflicting items. "
                + (f"Optimization ideas from assessment: {analysis}" if analysis else "")
                + ("\n\nDetails:\n" + _join_bullets(warnings + errors) if (warnings or errors) else "")
            )
        else:
            findings_summary = (
                "Summary of findings: The plan is feasible and no blocking warnings or errors were listed. "
                + (f"Assessment highlights: {analysis[:900]}" if analysis else "No extra issues surfaced in assessment.")
            ).strip()
            root = "The submitted JSON plan appears feasible; no blocking issues were flagged in the merged assessment."
            recommendation = (
                (f"Recommendations: {analysis}" if analysis else "Recommendations: keep monitoring assumptions and dependencies as you execute; add tests or checkpoints where the plan touches external systems.")
            )
        if messages:
            recommendation = recommendation + "\n\nAdditional notes:\n" + _join_bullets(messages)

    else:
        findings_summary = "Summary of findings: Plan assessment was incomplete (empty or missing JSON input)."
        root = "Plan assessment did not complete; provide valid JSON to assess feasibility."
        recommendation = "Recommendations: submit a well-formed JSON plan object or array, then re-run analysis."

    sev = _severity(feasible, errors, warnings)
    conf = _confidence(feasible, analysis)

    return {
        "parsed_log": {},
        "detected_patterns": [],
        "root_cause": root,
        "confidence": conf,
        "severity": sev,
        "findings_summary": findings_summary,
        "recommendation": recommendation,
        "recommendation_summary": state_specific_recommendation_summary(state),
        "log_errors_analysis": "",
        "log_warnings_analysis": "",
    }


def state_specific_findings_summary(state: dict[str, Any]) -> str:
    """Compact, case-varying summary from actual lists (when narrative LLM omits findings_summary only)."""
    feasible = state.get("plan_feasible")
    warnings = [str(x).strip() for x in (state.get("plan_warnings_list") or []) if str(x).strip()]
    errors = [str(x).strip() for x in (state.get("plan_errors_list") or []) if str(x).strip()]
    messages = [str(x).strip() for x in (state.get("plan_messages_list") or []) if str(x).strip()]
    analysis = (state.get("plan_deep_analysis") or "").strip()
    pre_e = [str(x).strip() for x in (state.get("plan_preprocess_errors") or []) if str(x).strip()]
    pre_w = [str(x).strip() for x in (state.get("plan_preprocess_warnings") or []) if str(x).strip()]

    bits: list[str] = []
    if feasible is True:
        bits.append("Feasibility is assessed as yes.")
    elif feasible is False:
        bits.append("Feasibility is assessed as no.")
    else:
        bits.append("Feasibility could not be determined (missing or incomplete plan input).")

    if errors:
        bits.append("Errors: " + "; ".join(errors[:20]))
    if warnings:
        bits.append("Warnings: " + "; ".join(warnings[:20]))
    if messages:
        bits.append("Notes: " + "; ".join(messages[:16]))
    if pre_e:
        bits.append("Structural problems: " + "; ".join(pre_e[:12]))
    if pre_w:
        bits.append("Structural cautions: " + "; ".join(pre_w[:12]))
    if analysis:
        bits.append("Assessment text: " + analysis[:2500])

    return "\n".join(bits).strip() or "No structured plan signals were available to summarize."


def state_specific_root_cause(state: dict[str, Any]) -> str:
    """Short diagnosis from concrete list items (partial LLM fallback)."""
    feasible = state.get("plan_feasible")
    errors = [str(x).strip() for x in (state.get("plan_errors_list") or []) if str(x).strip()]
    warnings = [str(x).strip() for x in (state.get("plan_warnings_list") or []) if str(x).strip()]
    analysis = (state.get("plan_deep_analysis") or "").strip()

    if feasible is False:
        if errors:
            return "Blocking items point to: " + "; ".join(errors[:10])
        return "The plan cannot be executed as structured; review dependencies and invalid steps."
    if feasible is True and warnings and not errors:
        return "No hard blockers were listed; residual risk comes from: " + "; ".join(warnings[:10])
    if feasible is True and (errors or warnings):
        return "Mixed signals: " + (
            ("Errors: " + "; ".join(errors[:8]) + ". ") if errors else ""
        ) + (("Warnings: " + "; ".join(warnings[:8]) + ".") if warnings else "")
    if analysis:
        return "Assessment emphasis: " + analysis[:1200]
    return "Insufficient detail to name a single cause; see listed errors and warnings."


def state_specific_recommendation(state: dict[str, Any]) -> str:
    """Action hints tied to actual items (partial LLM fallback)."""
    feasible = state.get("plan_feasible")
    errors = [str(x).strip() for x in (state.get("plan_errors_list") or []) if str(x).strip()]
    warnings = [str(x).strip() for x in (state.get("plan_warnings_list") or []) if str(x).strip()]
    analysis = (state.get("plan_deep_analysis") or "").strip()

    if feasible is False:
        parts = [
            "Address these first, then re-assess: " + ("; ".join(errors[:12]) if errors else "(see structural issues in preprocess output).")
        ]
        if warnings:
            parts.append("Then review warnings: " + "; ".join(warnings[:10]))
        return "\n".join(parts)

    if feasible is True and (warnings or errors):
        w = "Refine or validate: " + "; ".join((warnings + errors)[:14])
        return w + (("\n\n" + analysis[:2000]) if analysis else "")

    if analysis:
        return "Follow the assessor’s guidance: " + analysis[:3000]

    return "Re-run assessment after any plan edits; add tests or checkpoints on critical steps."

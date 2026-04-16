"""LLM: natural-language findings_summary, root_cause, recommendation from plan feasibility + signals."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from graphs.state import GraphState
from tools.plan_narrative_payload_tool import build_plan_narrative_payload
from utils.agent_debug import log_input_state, log_llm_response, log_parsed_output
from utils.llm import get_llm, invoke_llm_chain
from utils.parser import extract_llm_json_object
from utils.plan_rca_fallback import (
    deterministic_plan_rca,
    state_specific_findings_summary,
    state_specific_recommendation,
    state_specific_recommendation_summary,
    state_specific_root_cause,
)
from utils.rca_text_format import normalize_to_n_lines
from utils.tool_logging import log_tool_run


class _PlanNarrativeSchema(BaseModel):
    findings_summary: str = ""
    recommendation: str = ""
    recommendation_summary: str = ""
    root_cause: str = ""
    severity: Literal["High", "Medium", "Low"] = "Medium"


_PLAN_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You turn structured plan assessment data (JSON payload) into natural language for stakeholders.\n\n"
            "The payload includes:\n"
            "- plan_feasible: true / false / null (feasible vs infeasible from prior assessment step)\n"
            "- merged_errors, merged_warnings, merged_messages: strings merged from preprocess + extraction + digest LLM\n"
            "- structural_preprocess_* and extracted_issue_*: the same signals split by origin (use for specificity)\n"
            "- plan_assessment_analysis: narrative from the digest step (must align with the lists, not contradict them)\n"
            "- plan_skeleton_summary and plan_skeleton_outline; plan_json_excerpt\n\n"
            "**Grounding rule:** `findings_summary`, `root_cause`, `recommendation_summary`, and `recommendation` MUST reflect "
            "only what appears in this payload—especially merged_errors, merged_warnings, merged_messages, plan_feasible, and "
            "plan_assessment_analysis. Do not invent issues, products, or failures not supported by those fields.\n\n"
            "Requirements:\n"
            "- **findings_summary**: exactly **three lines** (newline-separated in the JSON string). Line 1: feasibility and primary "
            "blockers or confirmation. Line 2: main error themes (or state none if lists empty). Line 3: warning/message themes or "
            "overall risk. Paraphrase actual list items; vary wording across runs.\n"
            "- **root_cause**: exactly **three lines**: why the plan is infeasible or what drives residual risk, using list content.\n"
            "- **recommendation_summary**: exactly **three lines**: a tight reply to those findings—same themes as errors/warnings/messages "
            "and feasibility—before action items.\n"
            "- **recommendation**: exactly **three lines** of concrete next steps that **follow from** recommendation_summary and the "
            "same merged strings (fix ordering, resolve named blockers, clarify messages), not generic IT advice.\n"
            "- Do **not** paste section headings such as `Plan skeleton summary:` or repeat the "
            "`plan_skeleton_summary` string verbatim inside recommendation or recommendation_summary—paraphrase in your own words.\n"
            "- **severity**: High / Medium / Low from real severity of issues.\n\n"
            "Output only one JSON object with keys: findings_summary, root_cause, recommendation_summary, recommendation, severity.\n"
            "No markdown fences, no text outside JSON.",
        ),
        ("human", "{payload}"),
    ]
)


def _strip_plan_skeleton_heading_echo(text: str) -> str:
    """Remove lines that echo the digest section heading (LLM sometimes copies it into prose fields)."""
    t = (text or "").strip()
    if not t:
        return ""
    lines = [
        ln
        for ln in t.replace("\r\n", "\n").split("\n")
        if not ln.strip().lower().startswith("plan skeleton summary")
    ]
    return "\n".join(lines).strip()


def _normalize_severity(raw: Any, fallback: str) -> Literal["High", "Medium", "Low"]:
    s = str(raw or "").strip().lower()
    if s == "high":
        return "High"
    if s == "medium":
        return "Medium"
    if s == "low":
        return "Low"
    if fallback in ("High", "Medium", "Low"):
        return fallback  # type: ignore[return-value]
    return "Medium"


def _coalesce_narrative_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Map common LLM key variants to canonical names."""
    lk = {
        str(k).strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in raw.items()
    }

    def pick(*names: str) -> str:
        for n in names:
            if n in lk and lk[n] is not None:
                s = str(lk[n]).strip()
                if s:
                    return s
        return ""

    return {
        "findings_summary": pick(
            "findings_summary",
            "finding_summary",
            "summary",
            "summary_of_findings",
            "findings",
            "assessment_summary",
        ),
        "recommendation": pick("recommendation", "recommendations", "rec", "remediation", "next_steps"),
        "recommendation_summary": pick(
            "recommendation_summary",
            "rec_summary",
            "summary_for_recommendation",
            "action_summary",
        ),
        "root_cause": pick("root_cause", "rootcause", "cause", "diagnosis", "primary_issue"),
        "severity": pick("severity", "risk", "impact"),
    }


def plan_narrative_agent(state: GraphState) -> dict[str, Any]:
    st = dict(state)
    det = deterministic_plan_rca(st)
    payload_dict = build_plan_narrative_payload(st)
    log_tool_run("plan_narrative", "build_plan_narrative_payload", payload_dict)
    payload = json.dumps(payload_dict, ensure_ascii=False)
    log_input_state(
        "plan_narrative",
        {
            "payload_chars": len(payload),
            "plan_feasible": payload_dict.get("plan_feasible"),
            "merged_errors_n": len(payload_dict.get("merged_errors") or []),
            "merged_warnings_n": len(payload_dict.get("merged_warnings") or []),
        },
    )

    chain = _PLAN_NARRATIVE_PROMPT | get_llm()
    content = invoke_llm_chain(chain, {"payload": payload[:260_000]})
    log_llm_response("plan_narrative", content)

    json_obj = extract_llm_json_object(content)
    if json_obj is None:
        # List-led fallbacks (vary with case); keep severity/confidence from deterministic rules.
        hist = [
            "plan_narrative: no_valid_JSON — state_specific_findings/rec/root (not rigid template)",
            "plan_narrative: severity_confidence_from deterministic_plan_rca",
        ]
        fs0 = state_specific_findings_summary(st)
        rc0 = state_specific_root_cause(st)
        rs0 = state_specific_recommendation_summary(st)
        rec0 = state_specific_recommendation(st)
        result = {
            "findings_summary": normalize_to_n_lines(fs0, 3),
            "recommendation": normalize_to_n_lines(rec0, 3),
            "recommendation_summary": normalize_to_n_lines(rs0, 3),
            "root_cause": normalize_to_n_lines(rc0, 3),
            "severity": str(det["severity"]),
            "confidence": float(det["confidence"]),
            "parsed_log": {},
            "detected_patterns": [],
            "log_errors_analysis": "",
            "log_warnings_analysis": "",
            "history": hist,
        }
        log_parsed_output("plan_narrative", {**result, "json_fallback": True})
        return result

    coalesced = _coalesce_narrative_keys(json_obj)
    try:
        validated = _PlanNarrativeSchema.model_validate(
            {
                "findings_summary": coalesced.get("findings_summary") or "",
                "recommendation": coalesced.get("recommendation") or "",
                "recommendation_summary": coalesced.get("recommendation_summary") or "",
                "root_cause": coalesced.get("root_cause") or "",
                "severity": coalesced.get("severity") or det.get("severity", "Medium"),
            }
        )
    except ValidationError:
        validated = _PlanNarrativeSchema.model_validate(
            {
                "findings_summary": "",
                "recommendation": "",
                "recommendation_summary": "",
                "root_cause": "",
                "severity": det.get("severity", "Medium"),
            }
        )

    llm_fs = (validated.findings_summary or "").strip()
    llm_rec = (validated.recommendation or "").strip()
    llm_rs = (validated.recommendation_summary or "").strip()
    llm_rc = (validated.root_cause or "").strip()

    fs = _strip_plan_skeleton_heading_echo(llm_fs)
    rec = _strip_plan_skeleton_heading_echo(llm_rec)
    rs = _strip_plan_skeleton_heading_echo(llm_rs)
    rc = _strip_plan_skeleton_heading_echo(llm_rc)
    state_fill_used = not fs or not rec or not rc or not rs
    fs = fs or state_specific_findings_summary(st)
    rec = rec or state_specific_recommendation(st)
    rs = rs or state_specific_recommendation_summary(st)
    rc = rc or state_specific_root_cause(st)
    fs = normalize_to_n_lines(fs, 3)
    rc = normalize_to_n_lines(rc, 3)
    rs = normalize_to_n_lines(rs, 3)
    rec = normalize_to_n_lines(rec, 3)
    sev = _normalize_severity(validated.severity, str(det.get("severity") or "Medium"))

    hist = ["plan_narrative: LLM JSON parsed"]
    if state_fill_used:
        hist.append("plan_narrative: state_specific_fill_for_empty_fields (not rigid template)")

    result = {
        "findings_summary": fs,
        "recommendation": rec,
        "recommendation_summary": rs,
        "root_cause": rc,
        "severity": sev,
        "confidence": float(det["confidence"]),
        "parsed_log": {},
        "detected_patterns": [],
        "log_errors_analysis": "",
        "log_warnings_analysis": "",
        "history": hist,
    }
    log_parsed_output("plan_narrative", {**result, "json_fallback": False})
    return result

"""Agent: LLM plan assessment from digest (merge structural + extracted + model)."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from graphs.state import GraphState
from tools.plan_digest_tool import plan_assessment_digest_tool
from utils.agent_debug import log_input_state, log_llm_response, log_parsed_output
from utils.llm import get_llm, invoke_llm_chain
from utils.parser import extract_llm_json_object, parse_llm_json
from utils.plan_preprocess import resolve_plan_skeleton_outline
from utils.tool_logging import log_tool_run

_MAX_SKEL_IN_DEEP_ANALYSIS = 12_000


def _compose_plan_deep_analysis(
    *,
    plan_feasible: bool,
    merged_errors: list[str],
    merged_warnings: list[str],
    merged_messages: list[str],
    skeleton: str,
) -> str:
    """Feasibility line plus compact JSON outline. Five-sentence skeleton stays in ``plan_skeleton_summary`` only."""
    flag = "true" if plan_feasible else "false"
    header = (
        f"Feasibility (digest assessment): plan_feasible={flag}. "
        f"Merged signals: {len(merged_errors)} error(s), {len(merged_warnings)} warning(s), "
        f"{len(merged_messages)} message(s)."
    )
    parts: list[str] = [header, ""]
    sk = (skeleton or "").strip()
    if sk:
        if len(sk) > _MAX_SKEL_IN_DEEP_ANALYSIS:
            sk = sk[: _MAX_SKEL_IN_DEEP_ANALYSIS - 1] + "…"
        parts.extend(["Plan skeleton (compact outline):", sk])
    return "\n".join(parts).strip()


class PlanAnalysisSchema(BaseModel):
    plan_feasible: bool = True
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


_PLAN_LLM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You assess a plan using only the **plan_digest** string below. It was produced by a deterministic "
            "Python tool (structural checks + extracted keys)—not by you. "
            "You do **not** have the full raw plan—do not invent fields that are not implied by the digest.\n\n"
            "Tasks:\n"
            "1. Treat **Structural checks** and **issue-like key** sections as primary evidence.\n"
            "2. Add any **implicit** warnings (risks, missing assumptions) or **implicit** errors "
            "(contradictions, infeasible ordering) suggested by the digest.\n"
            "3. Set **plan_feasible** to true if the plan can likely be executed; false if infeasible or broken.\n"
            "4. **messages**: optional short bullets for clarifications that are not duplicates of warnings/errors.\n\n"
            "Output only one JSON object with exactly these keys:\n"
            '- "plan_feasible": boolean\n'
            '- "warnings": array of short strings\n'
            '- "errors": array of short strings\n'
            '- "messages": array of short strings\n\n'
            "Rules: no markdown, no code fences, JSON only.",
        ),
        ("human", "{plan_digest}"),
    ]
)


def _coerce_plan_lists(d: dict[str, Any]) -> dict[str, Any]:
    def _as_str_list(key: str) -> list[str]:
        raw = d.get(key, [])
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for x in raw:
            s = str(x).strip()
            if s:
                out.append(s[:500])
        return out[:48]

    feasible = d.get("plan_feasible", True)
    if not isinstance(feasible, bool):
        feasible = str(feasible).lower() in ("true", "1", "yes")

    return {
        "plan_feasible": feasible,
        "warnings": _as_str_list("warnings"),
        "errors": _as_str_list("errors"),
        "messages": _as_str_list("messages"),
    }


def _plan_fallback_empty() -> dict[str, Any]:
    return {
        "plan_feasible": True,
        "warnings": [],
        "errors": [],
        "messages": [],
    }


def _dedupe_strs(seq: list[str], *, cap: int = 96) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s[:500])
        if len(out) >= cap:
            break
    return out


def plan_analysis_agent(state: GraphState) -> dict[str, Any]:
    plan_json = (state.get("plan_json") or "").strip()
    if not plan_json:
        log_input_state("plan_analysis", {"plan_json": "", "note": "empty plan_json"})
        log_llm_response("plan_analysis", "[skipped — empty plan_json]")
        out = {
            "plan_feasible": None,
            "plan_warnings_list": [],
            "plan_errors_list": [],
            "plan_messages_list": [],
            "plan_deep_analysis": "",
            "history": ["plan_analysis: skipped empty plan_json"],
        }
        log_parsed_output("plan_analysis", out)
        return out

    plan_digest = plan_assessment_digest_tool(dict(state))
    log_tool_run(
        "plan_analysis",
        "plan_assessment_digest_tool",
        {"digest_chars": len(plan_digest), "digest_preview": plan_digest[:1500]},
    )
    log_input_state(
        "plan_analysis",
        {
            "plan_digest_preview": plan_digest[:2500] + ("..." if len(plan_digest) > 2500 else ""),
            "plan_digest_chars": len(plan_digest),
            "note": "LLM input is digest from plan_assessment_digest_tool only",
        },
    )
    chain = _PLAN_LLM_PROMPT | get_llm()
    content = invoke_llm_chain(chain, {"plan_digest": plan_digest[:55_000]})
    log_llm_response("plan_analysis", content)

    used_fallback = extract_llm_json_object(content) is None
    data = parse_llm_json(content, default=_plan_fallback_empty())
    data = _coerce_plan_lists(data)

    try:
        validated = PlanAnalysisSchema.model_validate(
            {
                "plan_feasible": data["plan_feasible"],
                "warnings": data["warnings"],
                "errors": data["errors"],
                "messages": data["messages"],
            }
        )
    except ValidationError:
        used_fallback = True
        validated = PlanAnalysisSchema.model_validate(_plan_fallback_empty())

    pre_e = [str(x).strip() for x in (state.get("plan_preprocess_errors") or []) if str(x).strip()]
    pre_w = [str(x).strip() for x in (state.get("plan_preprocess_warnings") or []) if str(x).strip()]
    ext_e = [str(x).strip() for x in (state.get("plan_extracted_errors") or []) if str(x).strip()]
    ext_w = [str(x).strip() for x in (state.get("plan_extracted_warnings") or []) if str(x).strip()]
    ext_m = [str(x).strip() for x in (state.get("plan_extracted_messages") or []) if str(x).strip()]

    merged_errors = _dedupe_strs(pre_e + ext_e + list(validated.errors))
    merged_warnings = _dedupe_strs(pre_w + ext_w + list(validated.warnings))
    merged_messages = _dedupe_strs(ext_m + list(validated.messages))

    hist = ["plan_analysis: digest-only LLM"]
    if used_fallback:
        hist.append("plan_analysis: json_fallback")
    hist.append(
        f"plan_analysis: merged struct+extracted+llm "
        f"err={len(merged_errors)} warn={len(merged_warnings)} msg={len(merged_messages)}"
    )

    skeleton = resolve_plan_skeleton_outline(dict(state))
    deep = _compose_plan_deep_analysis(
        plan_feasible=validated.plan_feasible,
        merged_errors=merged_errors,
        merged_warnings=merged_warnings,
        merged_messages=merged_messages,
        skeleton=skeleton,
    )

    result = {
        "plan_feasible": validated.plan_feasible,
        "plan_warnings_list": merged_warnings,
        "plan_errors_list": merged_errors,
        "plan_messages_list": merged_messages,
        "plan_deep_analysis": deep,
        "history": hist,
    }
    log_parsed_output("plan_analysis", {**result, "json_fallback": used_fallback})
    return result

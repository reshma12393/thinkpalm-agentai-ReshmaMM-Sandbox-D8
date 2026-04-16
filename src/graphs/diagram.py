"""Mermaid diagrams for RCA graph topology and per-run execution path (easy to read in UI)."""

from __future__ import annotations

from typing import Any

# Short labels for graph node ids (shown inside boxes)
NODE_LABELS: dict[str, str] = {
    "__start__": "START",
    "classify_input": "1 · Classify (plan vs log)",
    "plan_preprocess": "2 · Plan preprocess (structural)",
    "plan_analysis": "3 · Plan analysis (LLM digest)",
    "plan_narrative": "4 · Plan narrative (LLM summary + recommendation)",
    "log_signal_pipeline": "2 · Log signals (preprocess → parse → ERROR/WARN → memory → patterns)",
    "log_analyser": "3 · Log analyser (Claude Sonnet on extractions)",
    "log_findings_summary": "4 · Findings summary (LLM on extractions)",
    "root_cause": "5 · Root cause (LLM)",
    "recommendation": "6 · Recommendation (LLM)",
    "__end__": "END",
}


def _label(node_id: str) -> str:
    return NODE_LABELS.get(node_id, node_id.replace("_", " ").title())


def mermaid_reference_topology(*, input_kind: str) -> str:
    """Static flowchart for the branch that applies to this input (plan vs log)."""
    if input_kind == "plan":
        return """flowchart TD
  %% Plan branch — JSON
  S([START]) --> CI["{_ci}"]
  CI --> PP["{_pp}"]
  PP --> PA["{_pa}"]
  PA --> PN["{_pn}"]
  PN --> E([END])
  PP -.->|preprocess only| E
""".format(
            _ci=_label("classify_input"),
            _pp=_label("plan_preprocess"),
            _pa=_label("plan_analysis"),
            _pn=_label("plan_narrative"),
        )

    return """flowchart TD
  %% Log branch — plain text
  S([START]) --> CI["{_ci}"]
  CI --> LS["{_ls}"]
  LS --> LFS["{_lfs}"]
  LFS --> RC["{_rc}"]
  RC --> REC["{_rec}"]
  REC --> E([END])
  LS -.->|preprocess only| E
""".format(
        _ci=_label("classify_input"),
        _ls=_label("log_signal_pipeline"),
        _lfs=_label("log_findings_summary"),
        _rc=_label("root_cause"),
        _rec=_label("recommendation"),
    )


def mermaid_execution_path(ordered_nodes: list[str]) -> str:
    """Linear flowchart of nodes actually executed this run (from trace order)."""
    if not ordered_nodes:
        return (
            "flowchart LR\n"
            "classDef empty fill:#fff3e0,stroke:#ef6c00\n"
            '  empty["No nodes in trace"]:::empty\n'
        )

    n = len(ordered_nodes)
    lines = [
        "%% Execution path (this run)",
        "flowchart LR",
        "classDef done fill:#e3f2fd,stroke:#1565c0,stroke-width:2px",
        "classDef term fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px",
    ]
    for i, raw in enumerate(ordered_nodes):
        lab = _label(raw)
        # escape quotes in label for mermaid
        safe = lab.replace('"', "'")
        lines.append(f'  step{i}["{i + 1}. {safe}"]:::done')
    lines.append("  S([START]):::term --> step0")
    for i in range(n - 1):
        lines.append(f"  step{i} --> step{i + 1}")
    lines.append(f"  step{n - 1} --> E([END]):::term")

    return "\n".join(lines)


def ordered_nodes_from_trace(trace_steps: list[dict]) -> list[str]:
    """Node names in execution order (keeps repeats if LangGraph emits them)."""
    out: list[str] = []
    for step in trace_steps:
        if not isinstance(step, dict):
            continue
        n = step.get("node")
        if n is not None and str(n).strip():
            out.append(str(n).strip())
    return out


def ordered_nodes_from_step_models(steps: list[Any]) -> list[str]:
    """Same as :func:`ordered_nodes_from_trace` but for objects with a ``node`` attribute."""
    out: list[str] = []
    for step in steps:
        n = getattr(step, "node", None)
        if n is not None and str(n).strip():
            out.append(str(n).strip())
    return out

"""Streamlit UI for RCA (expects FastAPI at RCA_API_URL, default http://127.0.0.1:8000)."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _ensure_rca_package_root_on_path() -> None:
    """Ensure top-level packages (``graphs``, ``agents``, …) import when Streamlit runs without ``pip install``.

    Walks upward from this file until ``src/graphs/workflow.py`` exists, then prepends ``src`` to ``sys.path``.
    """
    try:
        import graphs.workflow  # noqa: F401
    except ImportError:
        pass
    else:
        return
    here = Path(__file__).resolve()
    for d in [here, *here.parents]:
        marker = d / "src" / "graphs" / "workflow.py"
        if marker.is_file():
            src_root = str(d / "src")
            if src_root not in sys.path:
                sys.path.insert(0, src_root)
            return


_ensure_rca_package_root_on_path()

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from graphs.diagram import (
    mermaid_execution_path,
    mermaid_reference_topology,
    ordered_nodes_from_trace,
)

# Legacy ``plan_analysis`` strings may still contain this marker before hidden digest prose (strip in UI/report).
RCA_PLAN_NARR_SPLIT_MARKER = "__RCA_NARR_SPLIT__"

_DEFAULT_API = "http://127.0.0.1:8000"


def _post_analyze(
    base_url: str,
    log: str,
    *,
    include_trace: bool = False,
    preprocess_only: bool = False,
) -> dict:
    params: list[str] = []
    if include_trace:
        params.append("include_trace=true")
    if preprocess_only:
        params.append("preprocess_only=true")
    qs = "?" + "&".join(params) if params else ""
    url = f"{base_url.rstrip('/')}/analyze{qs}"
    payload = json.dumps({"log": log}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def _confidence_label(conf: float) -> str:
    c = float(conf)
    if c <= 1.0 and c >= 0.0:
        return f"{c * 100:.1f}%"
    return f"{c:.2f}"


def _format_plan_skeleton_summary_display(text: str) -> str:
    """Show five-sentence plan skeleton as separate paragraphs (API often joins with spaces)."""
    t = (text or "").strip()
    if not t:
        return "—"
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return "\n\n".join(parts)
    return t


def _plan_analysis_without_assessor_narrative(plan_analysis: str) -> str:
    """Drop digest LLM prose after split marker (legacy payloads only; new API omits it)."""
    pa = (plan_analysis or "").strip()
    if not pa:
        return pa
    if RCA_PLAN_NARR_SPLIT_MARKER in pa:
        return pa.split(RCA_PLAN_NARR_SPLIT_MARKER, 1)[0].strip()
    if "Assessor narrative:" in pa:
        return pa.split("Assessor narrative:", 1)[0].strip()
    return pa


def _render_plan_analysis_collapsible(plan_analysis: str) -> None:
    """Show feasibility text in the open; compact outline and long blocks behind expanders."""
    pa = _plan_analysis_without_assessor_narrative((plan_analysis or "").strip())
    if not pa:
        st.markdown("—")
        return
    key_outline = "Plan skeleton (compact outline):"
    if key_outline not in pa:
        st.markdown(pa)
        return
    i = pa.index(key_outline)
    head = pa[:i].rstrip()
    rest = pa[i + len(key_outline) :].lstrip()
    if RCA_PLAN_NARR_SPLIT_MARKER in rest:
        outline, _, _legacy_tail = rest.partition(RCA_PLAN_NARR_SPLIT_MARKER)
        outline = outline.rstrip()
    elif "Assessor narrative:" in rest:
        j = rest.index("Assessor narrative:")
        outline = rest[:j].rstrip()
    else:
        outline = rest
    if head:
        st.markdown(head)
    with st.expander("Plan skeleton (compact outline)", expanded=False):
        st.code(outline or "—", language=None)


def _plan_analysis_without_duplicate_skeleton(plan_analysis: str, skeleton_summary: str) -> str:
    """Remove the embedded skeleton summary block when plan_skeleton_summary is shown elsewhere."""
    pa = (plan_analysis or "").strip()
    sk = (skeleton_summary or "").strip()
    if not sk:
        return pa
    heads = (
        "Plan skeleton summary:",
    )
    i = -1
    head_len = 0
    for h in heads:
        if h in pa:
            i = pa.find(h)
            head_len = len(h)
            break
    if i < 0:
        return pa
    for nxt in ("Plan skeleton (compact outline):", RCA_PLAN_NARR_SPLIT_MARKER, "Assessor narrative:"):
        j = pa.find(nxt, i + head_len)
        if j != -1:
            part1 = pa[:i].rstrip()
            part2 = pa[j:].lstrip()
            if part1 and part2:
                return f"{part1}\n\n{part2}"
            return part2 or part1
    return pa


def _format_report_text(data: dict, source_name: str) -> str:
    """Human-readable report for .txt download."""
    lines = [
        "RCA Analysis Report",
        "=" * 48,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Source file: {source_name}",
        "",
        "--- Summary ---",
        f"Mode:           {'Preprocess only (no LLM)' if data.get('preprocess_only') else 'Full analysis'}",
        f"Input kind:     {data.get('input_kind', '')}",
        f"Confidence:     {_confidence_label(float(data.get('confidence', 0.0)))}",
        f"Severity:       {data.get('severity', '')}",
        "",
    ]
    if data.get("input_kind") != "plan" and (
        data.get("preprocess_summary") or data.get("preprocess_error_hits") or data.get("preprocess_warning_hits")
    ):
        lines.extend(
            [
                "--- Pre-scan (heuristic, before LLM) ---",
                str(data.get("preprocess_summary", "")).strip() or "(none)",
                "",
            ]
        )
        peh = data.get("preprocess_error_hits") or []
        pwh = data.get("preprocess_warning_hits") or []
        if peh:
            lines.append("Error-pattern sample lines:")
            for h in peh[:32]:
                lines.append(f"  {h}")
            lines.append("")
        if pwh:
            lines.append("Warning-pattern sample lines:")
            for h in pwh[:32]:
                lines.append(f"  {h}")
            lines.append("")
        dig = str(data.get("preprocess_llm_digest", "")).strip()
        if dig:
            lines.extend(
                [
                    "--- LLM digest (compact, same as sent to early model steps) ---",
                    dig,
                    "",
                ]
            )
    fs = str(data.get("findings_summary", "")).strip()
    lines.extend(
        [
            "--- Findings summary ---",
            fs or "(none)",
            "",
            "--- Root cause ---",
            str(data.get("root_cause", "")).strip() or "(none)",
            "",
            "--- Recommendation ---",
            str(data.get("recommendation", "")).strip() or "(none)",
            "",
        ]
    )
    if data.get("input_kind") == "plan":
        ppe = data.get("plan_preprocess_errors") or []
        ppw = data.get("plan_preprocess_warnings") or []
        if ppe or ppw:
            lines.extend(["--- Plan structure (pre-LLM) ---", "Errors:"])
            for err in ppe:
                lines.append(f"  • {err}")
            lines.extend(["", "Warnings:"])
            for w in ppw:
                lines.append(f"  • {w}")
            lines.append("")
        sk_rep = str(data.get("plan_skeleton_summary", "")).strip()
        pa_rep = _plan_analysis_without_assessor_narrative(
            _plan_analysis_without_duplicate_skeleton(str(data.get("plan_analysis", "")), sk_rep)
        )
        lines.extend(["", "--- Plan analysis ---", pa_rep])
        lines.extend(
            [
                "",
                "--- Plan skeleton summary (five sentences) ---",
                sk_rep or "(none)",
            ]
        )
    else:
        lines.extend(
            [
                "--- Log analysis (errors) ---",
                str(data.get("log_errors_analysis", "")).strip() or "(none)",
                "",
                "--- Log analysis (warnings) ---",
                str(data.get("log_warnings_analysis", "")).strip() or "(none)",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_mermaid_chart(mermaid_source: str, *, height: int = 400) -> None:
    """Render Mermaid using the CDN build (needs network once for the script)."""
    text = (mermaid_source or "").strip()
    if not text:
        st.caption("(empty diagram)")
        return
    # Server-generated Mermaid only; avoid closing HTML/script tags in edge-case labels
    safe = text.replace("</script", "<\\/script").replace("</pre", "<\\/pre")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body style="margin:0;padding:12px;background:#f8f9fa;font-family:system-ui,sans-serif;">
<pre class="mermaid" style="margin:0;">{safe}</pre>
<script>
  mermaid.initialize({{ startOnLoad: true, theme: "neutral", securityLevel: "loose",
    flowchart: {{ useMaxWidth: true, htmlLabels: true }} }});
</script>
</body></html>"""
    # ``key=`` is not supported on ``components.html`` in all Streamlit versions.
    components.html(html, height=height, scrolling=True)


def _render_trace_panel(trace_payload: dict) -> None:
    """Show graph diagrams, path legend, and per-node trace table."""
    st.subheader("Execution trace")
    legend = trace_payload.get("trace_legend") or ""
    if legend:
        st.markdown(legend)

    ik = str(trace_payload.get("input_kind", "log"))
    steps = trace_payload.get("trace") or []

    ref_src = trace_payload.get("trace_mermaid_reference") or mermaid_reference_topology(
        input_kind=ik
    )
    run_src = trace_payload.get("trace_mermaid_execution")
    if run_src is None and steps:
        run_src = mermaid_execution_path(ordered_nodes_from_trace(list(steps)))

    st.markdown("##### Reference topology")
    st.caption("Typical nodes for this input type (plan vs log). Dashed line: preprocess-only early exit.")
    _render_mermaid_chart(ref_src, height=360)

    st.markdown("##### This run")
    st.caption("Order of nodes executed for this request.")
    if run_src:
        _render_mermaid_chart(run_src, height=min(120 + 56 * max(1, len(steps)), 520))
    else:
        st.info("No trace steps — diagram unavailable.")

    if not steps:
        st.info("No trace steps were returned.")
        return
    rows = []
    for s in steps:
        keys = s.get("keys") or []
        rows.append(
            {
                "Step": int(s.get("step", 0)) + 1,
                "Node": s.get("node", ""),
                "State keys updated": ", ".join(keys) if keys else "—",
                "Update summary": s.get("summary", "") or "—",
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Step": st.column_config.NumberColumn("Step", width="small", format="%d"),
            "Node": st.column_config.TextColumn("Graph node", width="medium"),
            "State keys updated": st.column_config.TextColumn("Keys", width="medium"),
            "Update summary": st.column_config.TextColumn("Summary", width="large"),
        },
    )
    trace_json = json.dumps({"trace": steps, "trace_legend": legend}, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download trace (.json)",
        data=trace_json.encode("utf-8"),
        file_name="rca_execution_trace.json",
        mime="application/json",
        key="dl_trace_json",
    )


def _render_results(data: dict, uploaded_name: str) -> None:
    """Display analysis in aligned tables / metrics and offer downloads."""
    conf = float(data.get("confidence", 0.0))
    pre_only = bool(data.get("preprocess_only"))

    st.subheader("Analysis report")

    if pre_only:
        st.info(
            "**Preprocessing only** — Ollama / LLM steps were not run. "
            "Review the output below, then use **Continue to LLM analysis** to run the full pipeline."
        )

    st.markdown("##### Summary")
    summary_rows = [
        {"Metric": "Mode", "Value": "Preprocess only (no LLM)" if pre_only else "Full analysis"},
        {"Metric": "Input kind", "Value": str(data.get("input_kind", "—"))},
        {"Metric": "Confidence", "Value": _confidence_label(conf)},
        {"Metric": "Severity", "Value": str(data.get("severity", "—"))},
    ]
    st.dataframe(
        pd.DataFrame(summary_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="small"),
            "Value": st.column_config.TextColumn("Details", width="large"),
        },
    )

    if data.get("input_kind") != "plan" and (
        data.get("preprocess_summary") or data.get("preprocess_error_hits") or data.get("preprocess_warning_hits")
    ):
        st.markdown("##### Pre-scan (regex and structural checks)")
        ps = str(data.get("preprocess_summary", "")).strip()
        if ps:
            with st.container():
                st.markdown(ps)
        peh = data.get("preprocess_error_hits") or []
        pwh = data.get("preprocess_warning_hits") or []
        if peh:
            st.markdown("**Error / failure pattern samples**")
            st.dataframe(
                pd.DataFrame({"#": range(1, len(peh) + 1), "Hit": peh}),
                width="stretch",
                hide_index=True,
            )
        if pwh:
            st.markdown("**Warning pattern samples**")
            st.dataframe(
                pd.DataFrame({"#": range(1, len(pwh) + 1), "Hit": pwh}),
                width="stretch",
                hide_index=True,
            )
        dig = str(data.get("preprocess_llm_digest", "")).strip()
        if dig:
            with st.expander("LLM digest (compact — for debugging)", expanded=False):
                st.code(dig, language=None)

    if data.get("input_kind") == "plan" and (
        data.get("plan_preprocess_errors") or data.get("plan_preprocess_warnings")
    ):
        st.markdown("##### Plan structure (pre-LLM)")
        ppe = data.get("plan_preprocess_errors") or []
        ppw = data.get("plan_preprocess_warnings") or []
        if ppe:
            st.markdown("**Structural errors**")
            st.dataframe(
                pd.DataFrame({"#": range(1, len(ppe) + 1), "Detail": ppe}),
                width="stretch",
                hide_index=True,
            )
        if ppw:
            st.markdown("**Structural warnings**")
            st.dataframe(
                pd.DataFrame({"#": range(1, len(ppw) + 1), "Detail": ppw}),
                width="stretch",
                hide_index=True,
            )

    with st.container():
        st.markdown("**Findings summary**")
        if pre_only:
            st.caption("(Not generated — run full analysis.)")
        st.markdown(str(data.get("findings_summary", "")) or "—")

    with st.container():
        st.markdown("**Root cause**")
        if pre_only:
            st.caption("(Not generated — run full analysis.)")
        st.markdown(str(data.get("root_cause", "")) or "—")

    with st.container():
        st.markdown("**Recommendation**")
        if pre_only:
            st.caption("(Not generated — run full analysis.)")
        st.markdown(str(data.get("recommendation", "")) or "—")

    if data.get("input_kind") == "plan":
        with st.container():
            sk_txt = str(data.get("plan_skeleton_summary", "")).strip()
            st.markdown("**Plan analysis**")
            if pre_only:
                st.caption("(Not generated — run full analysis.)")
            pa_display = _plan_analysis_without_duplicate_skeleton(
                str(data.get("plan_analysis", "")),
                sk_txt,
            )
            _render_plan_analysis_collapsible(pa_display)
            with st.expander("Plan skeleton summary", expanded=False):
                st.markdown(_format_plan_skeleton_summary_display(sk_txt))
    else:
        st.markdown("##### Log signal analysis")
        err_a = str(data.get("log_errors_analysis", "")).strip()
        warn_a = str(data.get("log_warnings_analysis", "")).strip()
        if pre_only:
            st.caption("LLM log signal analysis was not run. Continue to full analysis to populate this section.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Category": "Error-class signals", "Analysis": err_a or "—"},
                    {"Category": "Warning-class signals", "Analysis": warn_a or "—"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    report_txt = _format_report_text(data, uploaded_name)
    report_json = json.dumps(data, indent=2, ensure_ascii=False)
    safe_base = "".join(c if c.isalnum() or c in "._-" else "_" for c in uploaded_name)[:80]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download report (.txt)",
            data=report_txt.encode("utf-8"),
            file_name=f"rca_report_{safe_base}_{ts}.txt",
            mime="text/plain",
            width="stretch",
        )
    with d2:
        st.download_button(
            label="Download report (.json)",
            data=report_json.encode("utf-8"),
            file_name=f"rca_report_{safe_base}_{ts}.json",
            mime="application/json",
            width="stretch",
        )


def main() -> None:
    st.set_page_config(page_title="RCA Agent", layout="wide")
    api_base = os.environ.get("RCA_API_URL", _DEFAULT_API).strip()

    if "trace_ui_open" not in st.session_state:
        st.session_state.trace_ui_open = False
    if "cached_trace" not in st.session_state:
        st.session_state.cached_trace = None
    if "last_analyze_body" not in st.session_state:
        st.session_state.last_analyze_body = None
    if "last_upload_name" not in st.session_state:
        st.session_state.last_upload_name = "upload"
    if "awaiting_llm_continue" not in st.session_state:
        st.session_state.awaiting_llm_continue = False
    if "last_run_preprocess_only" not in st.session_state:
        st.session_state.last_run_preprocess_only = False
    if "last_report" not in st.session_state:
        st.session_state.last_report = None

    st.title("RCA Agent")
    st.markdown(
        "A multi-agent system for automated root cause analysis (RCA) of logs and plans from the cargo planning assist application.<br>"
        "<br>"
        " This tool ingests unstructured logs or execution plans, maps them to known failure patterns, infers likely root causes, and generates actionable recommendations. "
        "It leverages a graph of agents (built with <b>LangGraph</b> and <b>LangChain</b>) working on a shared state to provide deep diagnostics and decision support.",
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload a <b>log</b> (<code>.txt</code>, <code>.log</code>, <code>.out</code>) or a <b>plan</b> file (<code>.json</code>). "
        "JSON is classified as a plan.",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a log or plan file",
        type=["txt", "json", "log", "out"],
        help=".json → plan; .txt / .log / .out → log text",
    )

    col_pre, col_full = st.columns(2)
    with col_pre:
        run_preprocess = st.button(
            "1. Preprocess only (review first)",
            help="Regex / structural checks only — no LLM. Then continue when ready.",
            width="stretch",
        )
    with col_full:
        run_full = st.button(
            "Full LLM analysis",
            help="Run the complete pipeline in one step.",
            width="stretch",
        )

    if run_preprocess or run_full:
        if uploaded is None:
            st.warning("Upload a file first.")
        else:
            body = uploaded.getvalue().decode("utf-8", errors="replace")
            if not body.strip():
                st.warning("The uploaded file is empty.")
            else:
                name = getattr(uploaded, "name", "upload") or "upload"
                preprocess_only = run_preprocess and not run_full
                try:
                    with st.spinner(
                        "Running preprocessing…"
                        if preprocess_only
                        else "Running full analysis (may take a while)…"
                    ):
                        data = _post_analyze(
                            api_base,
                            body,
                            preprocess_only=preprocess_only,
                        )
                except urllib.error.HTTPError as e:
                    st.error(f"API error ({e.code}): {e.read().decode('utf-8', errors='replace')[:500]}")
                except urllib.error.URLError as e:
                    st.error(f"Cannot reach API at {api_base}: {e.reason}")
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
                else:
                    st.session_state.last_analyze_body = body
                    st.session_state.last_upload_name = name
                    st.session_state.awaiting_llm_continue = bool(data.get("preprocess_only"))
                    st.session_state.last_run_preprocess_only = bool(data.get("preprocess_only"))
                    st.session_state.cached_trace = None
                    st.session_state.trace_ui_open = False
                    st.session_state.last_report = (data, name)

    if st.session_state.awaiting_llm_continue and st.session_state.last_analyze_body:
        st.divider()
        st.markdown("##### Next step")
        if st.button(
            "Continue to LLM analysis",
            type="primary",
            help="Uses the same file content as your last preprocess-only run.",
        ):
            try:
                with st.spinner("Running full analysis (may take a while)…"):
                    data = _post_analyze(
                        api_base,
                        st.session_state.last_analyze_body,
                        preprocess_only=False,
                    )
            except urllib.error.HTTPError as e:
                st.error(f"API error ({e.code}): {e.read().decode('utf-8', errors='replace')[:500]}")
            except urllib.error.URLError as e:
                st.error(f"Cannot reach API at {api_base}: {e.reason}")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
            else:
                st.session_state.awaiting_llm_continue = False
                st.session_state.last_run_preprocess_only = False
                st.session_state.cached_trace = None
                st.session_state.trace_ui_open = False
                st.session_state.last_report = (data, st.session_state.last_upload_name)

    if st.session_state.last_report is not None:
        _render_results(st.session_state.last_report[0], st.session_state.last_report[1])

    if st.session_state.last_analyze_body:
        st.divider()
        if st.button("Show execution trace", help="Loads node-by-node graph steps (re-runs analysis with tracing)."):
            st.session_state.trace_ui_open = True
            st.session_state.cached_trace = None

        if st.session_state.trace_ui_open:
            if st.session_state.cached_trace is None:
                try:
                    with st.spinner("Loading graph trace (this re-runs the analysis with tracing enabled)…"):
                        st.session_state.cached_trace = _post_analyze(
                            api_base,
                            st.session_state.last_analyze_body,
                            include_trace=True,
                            preprocess_only=st.session_state.get("last_run_preprocess_only", False),
                        )
                except urllib.error.HTTPError as e:
                    st.session_state.trace_ui_open = False
                    st.error(
                        f"Trace API error ({e.code}): "
                        f"{e.read().decode('utf-8', errors='replace')[:500]}"
                    )
                except urllib.error.URLError as e:
                    st.session_state.trace_ui_open = False
                    st.error(f"Cannot reach API: {e.reason}")
                except Exception as e:  # noqa: BLE001
                    st.session_state.trace_ui_open = False
                    st.error(str(e))
            if st.session_state.cached_trace is not None:
                with st.expander("Graph / node trace", expanded=True):
                    _render_trace_panel(st.session_state.cached_trace)


main()

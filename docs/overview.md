# RCA Agent — project overview
AI Root Cause Analysis Pipeline
A multi-agent system for automated root cause analysis (RCA) of logs and plans from the cargo planning assist application. This tool ingests unstructured logs or execution plans, maps them to known failure patterns, infers likely root causes, and generates actionable recommendations. It leverages a graph of agents (built with LangGraph and LangChain) working on a shared state to provide deep diagnostics and decision support.

Note:
This RCA agent is tailored for use with the CPDSS (Cargo Planning Decision Support System) project, which produces cargo allocation plans and operational logs as part of its workflow.

High-level description of the pipeline, branches, and repository layout. For setup, configuration (Claude API / Ollama), and run commands, see the root **`README.md`**.

## Purpose

The RCA agent classifies input as **plan JSON** or **log text**, runs the matching **LangGraph** branch, and returns analysis, root cause, severity, and recommendations where applicable. The HTTP API is under `src/apis/`; optional **Streamlit** UI lives in `src/ui/`.

## Branches

- **Plan:** preprocess → plan analysis (LLM) → plan narrative (LLM) → end.
- **Log:** signal pipeline → log analyser (LLM) → findings summary → root cause → recommendation → end.

**Preprocess-only** stops after the preprocess step (no downstream LLM nodes); use the API flag or Streamlit button.

## Repository map

| Path | Role |
|------|------|
| `src/graphs/` | State and workflow |
| `src/agents/` | Graph nodes |
| `src/apis/` | FastAPI |
| `src/utils/` | Config and LLM helpers |
| `src/memory/` | Checkpointer and optional JSON store |
| `src/ui/` | CLI and Streamlit |
| `tests/` | Test / sample data |
| `screenshots/` | UI screenshots |
| `docs/` | This overview and other docs |

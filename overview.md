# RCA Agent — project overview

## Purpose

**RCA Agent** is an AI-assisted **root cause analysis** pipeline built on **LangGraph**. It ingests a single text upload (or API body), **classifies** it as either **structured plan JSON** or **plain log text**, then executes a **branch-specific** graph. Outputs include structured state, narrative findings, root cause, severity, and recommendations where the branch defines them.

The stack includes **FastAPI** (`POST`/`PUT` `/analyze`), optional **execution traces**, and a **Streamlit** UI that calls the API. LLM calls are configured centrally (`src/utils/config.py`, typically **Ollama**).

---

## Inputs and branches

| Kind | Detection | Typical use |
|------|-----------|-------------|
| **Plan** | Classifier treats input as JSON plan | CI/build or deployment “plan” documents with steps, errors, warnings |
| **Log** | Everything else as free-form log text | Application or system logs for incident triage |

Routing: `classify_input` → `plan_preprocess` **or** `log_signal_pipeline`.

---

## Preprocess only

**Intent:** Run **classification plus preprocess** only — **no downstream LLM nodes** — so you can review structure and extracted signals before spending tokens or running full analysis.

**How it is triggered**

- **API:** `POST /analyze?preprocess_only=true` (or `PUT`), with `stop_after_preprocess` set via initial state from that flag.
- **Streamlit:** button **“1. Preprocess only (review first)”**.

**Plan branch**

- Nodes: `classify_input` → `plan_preprocess` → **END**
- `plan_preprocess` parses/validates JSON, runs structural checks, extracts warnings/errors/messages, and can produce skeleton outline/summary text **without** `plan_analysis` or `plan_narrative`.

**Log branch**

- Nodes: `classify_input` → `log_signal_pipeline` → **END**
- `log_signal_pipeline` performs heuristic scanning (e.g. ERROR/WARNING-style lines), builds preprocess summaries, digests, and sample hits for optional later LLM steps.

**Response flag:** `preprocess_only: true` when the run stopped after preprocess.

---

## Full LLM analysis

**Intent:** Run the **complete** branch pipeline with all LLM steps.

**How it is triggered**

- **API:** `/analyze` with **`preprocess_only=false`** (default).
- **Streamlit:** **“Full LLM analysis”**.

**Plan branch (JSON)**

1. `plan_preprocess` — structural prep and signal extraction  
2. `plan_analysis` — LLM digest of the plan  
3. `plan_narrative` — LLM narrative (findings, feasibility-oriented reasoning, recommendation-style output merged with plan context)  
4. **END**

**Log branch (text)**

1. `log_signal_pipeline` — preprocess and signal extraction  
2. `log_analyser` — LLM on extractions  
3. `log_findings_summary` — LLM summary of findings  
4. `root_cause` — LLM root cause + confidence  
5. `recommendation` — LLM recommendation + severity  
6. **END**

**Optional trace:** `include_trace=true` returns per-node steps plus Mermaid for **reference topology** vs **this run’s execution path**.

---

## Repository map (high level)

| Area | Role |
|------|------|
| `src/graphs/` | `GraphState`, workflow (`workflow.py`), diagrams for traces |
| `src/agents/` | One node per graph step (classify, plan, log, root cause, recommendation) |
| `src/apis/` | FastAPI app and request/response models |
| `src/utils/` | Config, LLM helpers, plan/log preprocess utilities |
| `src/memory/` | Checkpointer and optional JSON incident store |
| `src/ui/` | Streamlit front-end |

---

## Further reading

Install, Ollama configuration, CLI/API/Streamlit commands, and layout details: see **`README.md`** in this directory.

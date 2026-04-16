# AI Root Cause Analysis Agent

A multi-agent system for automated root cause analysis (RCA) of logs and plans from the cargo planning assist application. This tool ingests unstructured logs or execution plans, maps them to known failure patterns, infers likely root causes, and generates actionable recommendations. It leverages a graph of agents (built with LangGraph and LangChain) working on a shared state to provide deep diagnostics and decision support.

> **Note:**  
This RCA agent is tailored for use with the **CPDSS (Cargo Planning Decision Support System)** project, which produces cargo allocation plans and operational logs as part of its workflow.  
 **Sample input logs and plans** for testing or demonstration are modelled after typical output from CPDSS, ensuring realistic scenarios for root cause analysis.  
 The system is designed to understand and reason about failures or inefficiencies specifically within the context of vessel stowage, cargo allocation, and related decision support as found in CPDSS outputs.


## Input / Output Expectations

**Input:**  
- Plans (both feasible and infeasible) or logs from the Cargo Planning Assist application.

**Output:**  
- **Failure analysis** derived from the provided failure logs, including clear recommendations to address each identified failure.
- **Error/warning summary** for the analyzed logs, highlighting potential issues or inefficiencies detected during the planning process.
- **Actionable recommendations** to improve and optimize plan generation, enabling the creation of more robust cargo plans and reducing the likelihood of repeated failures.


---

## Architecture

| Layer | Role |
|--------|------|
| **Graphs** (`src/graphs/`) | `StateGraph` over `GraphState`: linear pipeline with `START` / `END`, optional thread checkpointing for multi-turn runs. |
| **Agents** (`src/agents/`) | Four nodes: parse logs, detect patterns, infer root cause, produce recommendation + severity. |
| **Tools** (`src/tools/`) | Pure helpers: `lookup_pattern` (keyword → pattern labels), `calculate_severity` (error_type → baseline severity). |
| **Utils** (`src/utils/`) | **`config.py`** + **`get_config()`**: LLM provider (`anthropic` or `ollama`), timeouts, retries, debug. **`get_llm()`** returns LangChain **`ChatAnthropic`** (Claude API) or **`ChatOllama`** depending on `RCA_LLM_PROVIDER`. |
| **Memory** (`src/memory/`) | Short-term: LangGraph `MemorySaver` checkpointer. Long-term: append-only JSON history in `memory/store.json` (optional use via `save_incident` / `load_history`). |
| **APIs** (`src/apis/`) | FastAPI `POST /analyze` or `PUT /analyze` for programmatic access. |
| **UI** (`src/ui/`) | CLI (`main.py`), Streamlit front-end calling the API. |

**Pipeline (data flow):**

```text
raw_log → log_parser → pattern_detection → root_cause → recommendation → final state
```

Shared state includes `raw_log`, `parsed_log`, `detected_patterns`, `root_cause`, `confidence`, `severity`, `recommendation`, and append-only `history`.

---

## Configuration

LLM behavior is **centralized in `src/utils/config.py`**. The function **`get_config()`** returns provider-specific settings (see below). **No `.env` file is required** — you can export environment variables, use **`rca-agent/.env`** (loaded automatically when `python-dotenv` is installed), **or** edit `config.py` directly.

### Anthropic Claude API (default provider)

The default LLM provider is **`anthropic`** (Claude via the official API), as set by `_DEFAULT_LLM_PROVIDER` in `src/utils/config.py`. To call Claude you need an API key from the [Anthropic Console](https://console.anthropic.com/).

| Variable | Purpose |
|----------|---------|
| **`RCA_LLM_PROVIDER`** or **`LLM_PROVIDER`** | Set to **`anthropic`** (default) to use the Claude API, or **`ollama`** for a local Ollama server. |
| **`ANTHROPIC_API_KEY`** or **`RCA_ANTHROPIC_API_KEY`** | **Required** for Claude. Used by LangChain `ChatAnthropic`. |
| **`ANTHROPIC_MODEL`** or **`RCA_ANTHROPIC_MODEL`** | Default chat model for most agents (code default: **`claude-opus-4-6`** — see `_DEFAULT_ANTHROPIC_MODEL` in `config.py`). |
| **`ANTHROPIC_API_URL`** or **`RCA_ANTHROPIC_API_URL`** | Optional. Custom API base URL (e.g. proxy or VPC endpoint). If unset, the Anthropic client uses its default endpoint. |
| **`ANTHROPIC_TEMPERATURE`** | Optional. Overrides sampling temperature when set (otherwise `OLLAMA_TEMPERATURE` / `RCA_OLLAMA_TEMPERATURE` / built-in default **`0`**). |
| **`RCA_LOG_ANALYSER_MODEL`** / **`ANTHROPIC_LOG_ANALYSER_MODEL`** | Log-analysis step only: defaults to **`claude-3-5-sonnet-20241022`** when provider is Anthropic (see `get_log_analyser_llm()` in `src/utils/llm.py`). |
| **`RCA_RECOMMENDATION_MODEL`** / **`ANTHROPIC_RECOMMENDATION_MODEL`** | Recommendation step only: defaults to **`claude-opus-4-6`** when provider is Anthropic (see `get_recommendation_llm()` in `src/utils/llm.py`). |
| **`OLLAMA_TIMEOUT`** | HTTP timeout (seconds) for **both** providers’ LLM clients (default **120**). |
| **`OLLAMA_RETRY_COUNT`** | Extra retries after the first LLM call (default **2** → 3 attempts total). |
| **`RCA_STRICT_ANTHROPIC`** | If **`1`** / **`true`** / **`yes`** and the Anthropic provider is selected but **`ANTHROPIC_API_KEY`** is empty, the API process **exits on startup** (see `src/apis/app.py` lifespan). |

**Example (shell):**

```bash
export RCA_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="sk-ant-api03-..."
# optional:
export ANTHROPIC_MODEL=claude-opus-4-6
export RCA_LOG_ANALYSER_MODEL=claude-3-5-sonnet-20241022
```

**Example (`rca-agent/.env`):**

```env
RCA_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Changing the Ollama server URL (local models)

Switch to local models by setting **`RCA_LLM_PROVIDER=ollama`** (or **`LLM_PROVIDER=ollama`**) and configuring Ollama.

To point the app at a specific Ollama host (e.g. on your LAN):

1. Open **`src/utils/config.py`**.
2. Set the module-level variable **`OLLAMA_BASE_URL`** (see the commented example in that file), or leave it **`None`** and use environment variables only:

   ```python
   OLLAMA_BASE_URL = "http://Your.Ollama.Remote.IP:11434"
   ```

3. If **`OLLAMA_BASE_URL`** is **`None`** in code, the URL is taken from **`OLLAMA_BASE_URL`** env, then **`RCA_OLLAMA_BASE_URL`**, then **`_DEFAULT_BASE_URL`** (`http://localhost:11434`).

**Precedence for the Ollama server URL:** non-empty **`OLLAMA_BASE_URL` in `config.py`** → **`OLLAMA_BASE_URL` env** → **`RCA_OLLAMA_BASE_URL` env** → **`_DEFAULT_BASE_URL`** in `config.py`.

**Ollama model name:** **`OLLAMA_MODEL`** / **`RCA_OLLAMA_MODEL`** (code default **`mistral`** unless overridden in `config.py`).

### Troubleshooting connectivity

| Issue | What to check |
|--------|----------------|
| **Claude API errors / 401** | **`ANTHROPIC_API_KEY`** is set and valid. Create or rotate keys in the [Anthropic Console](https://console.anthropic.com/). Ensure **`RCA_LLM_PROVIDER`** is **`anthropic`** (or unset, if your `config.py` default is Anthropic). |
| **Cannot reach Ollama** | Only when **`RCA_LLM_PROVIDER=ollama`**: Ollama is running (`ollama serve` on that host). From the app host, run `curl -s http://<host>:11434/api/tags` using the **`base_url`** from `get_config()`. |
| **Wrong Ollama IP** | Set **`OLLAMA_BASE_URL`** in **`src/utils/config.py`** or via env to the correct LAN IP (e.g. `http://192.168.1.50:11434`). |
| **Port blocked / timeout** | Ollama uses port **11434** by default. For Claude, increase **`OLLAMA_TIMEOUT`** if requests to the Anthropic API are slow. |

---

## Agents

| Agent | Responsibility |
|--------|----------------|
| **log_parser** | Uses an LLM with structured output to extract `error_type`, `module`, and `keywords` from `raw_log`. |
| **pattern_detection** | Reads `parsed_log`, calls **`lookup_pattern`** on keywords (and related fields) to populate `detected_patterns`. |
| **root_cause** | Uses an LLM with structured output, combining `parsed_log` and `detected_patterns` into `root_cause` and `confidence` (0–1). |
| **recommendation** | Calls **`calculate_severity`** on `error_type`, then an LLM for a concrete `recommendation` and final `severity` (`High` / `Medium` / `Low`). |

Most LLM agents use **`utils.llm.get_llm()`** and **`utils.llm.invoke_llm_chain`** (see [Configuration](#configuration)). The log pipeline’s **log analyser** and **recommendation** steps can use **`get_log_analyser_llm()`** and **`get_recommendation_llm()`** for dedicated Claude models when **`RCA_LLM_PROVIDER=anthropic`**. Retries follow **`OLLAMA_RETRY_COUNT`** (default **2** extra → **3** attempts total); then per-agent JSON fallbacks apply.

| Variable | Purpose |
|----------|---------|
| `RCA_LLM_PROVIDER` / `LLM_PROVIDER` | **`anthropic`** (default) or **`ollama`**. |
| `ANTHROPIC_API_KEY` / `RCA_ANTHROPIC_API_KEY` | Claude API key when using Anthropic. |
| `ANTHROPIC_MODEL` / `RCA_ANTHROPIC_MODEL` | Primary Claude model id (see [Anthropic Claude API](#anthropic-claude-api-default-provider)). |
| `OLLAMA_BASE_URL` | *(Ollama only; optional in env.)* Used when **`OLLAMA_BASE_URL`** in **`src/utils/config.py`** is **`None`**. Default **`http://localhost:11434`**. Legacy: `RCA_OLLAMA_BASE_URL`. |
| `OLLAMA_MODEL` / `RCA_OLLAMA_MODEL` | Ollama model name when **`RCA_LLM_PROVIDER=ollama`**. |
| `OLLAMA_TIMEOUT` | HTTP timeout in seconds for LLM clients (default **`120`**). |
| `OLLAMA_RETRY_COUNT` | Extra LLM invoke attempts after the first (default **`2`** → 3 tries total for JSON chains). |
| `OLLAMA_TEMPERATURE` / `RCA_OLLAMA_TEMPERATURE` / `ANTHROPIC_TEMPERATURE` | Sampling temperature (default **`0`**). |
| `DEBUG` or `RCA_DEBUG` | Truthy → ``get_config()['debug']`` is **true**: **`rca.llm`** logs (LLM request start/end, base URL, model, retries/errors) and per-agent debug prints. When **false**, those logs are suppressed. |

---

## Ollama setup (optional — only if `RCA_LLM_PROVIDER=ollama`)

1. **Install Ollama** — download and install from [ollama.com](https://ollama.com).

2. **Pull a model** (example; set **`OLLAMA_MODEL`** to match what you pull):

   ```bash
   ollama pull llama3
   ```

3. **Run the Ollama server** (if it is not already running as a background service):

   ```bash
   ollama serve
   ```

4. **Verify** the API is up — open [http://localhost:11434](http://localhost:11434) in a browser or:

   ```bash
   curl -s http://localhost:11434/api/tags
   ```

5. **Run the project** — install [Python dependencies](#python-setup) first, then from the `rca-agent` directory:

   ```bash
   uvicorn main:app --reload
   ```

   Default URL: **http://127.0.0.1:8000** (same as `uvicorn apis.app:app --reload` after [installing the package](#python-setup)).

---

## Python setup

- **Python** 3.10 or newer.

```bash
cd rca-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The editable install registers the packages under **`src/`** (`agents`, `apis`, `graphs`, …) so `main.py` and Uvicorn resolve imports.

---

## How to run

### CLI (graph + checkpointer)

```bash
cd rca-agent
python main.py "Your log line or multi-line text here"
python main.py --thread-id my-thread-1 "Another run"
```

Prints full state fields including `parsed_log`, patterns, root cause, confidence, severity, recommendation, and `history`.

### HTTP API (listen on all interfaces)

```bash
cd rca-agent
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level info
```

On startup, behavior depends on **`get_config()['llm_provider']`**:

- **`anthropic`:** the app logs whether **`ANTHROPIC_API_KEY`** is set. If the key is missing, set **`RCA_STRICT_ANTHROPIC=1`** to **exit** on startup instead of continuing with warnings.
- **`ollama`:** the app probes Ollama at **`get_config()['base_url']`** (`GET /api/tags`). By default it **warns** if Ollama is unreachable; set **`RCA_STRICT_OLLAMA=1`** to **exit** on failure.

Messages go to the **`rca.api`** logger.

### Streamlit UI (calls the API)

Start the API on port **8000**, then:

```bash
cd rca-agent
streamlit run src/ui/streamlit_app.py
```

Override the API base URL if needed:

```bash
export RCA_API_URL=http://127.0.0.1:8000
```

---

## Example: API input / output

**Request**

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"log":"ERROR: database connection timeout after 30s to primary replica"}'
```

**Response** (shape; values depend on models and logs)

```json
{
  "root_cause": "The database client could not complete a connection within the configured timeout, indicating the primary replica was unreachable or overloaded.",
  "confidence": 0.72,
  "severity": "High",
  "recommendation": "Check primary DB health and network path; verify connection pool limits and timeout settings for the affected module; review recent failover or load events."
}
```

---

## Tests, documentation, and UI assets

| Location | Contents |
|----------|----------|
| **`tests/`** | **Test data** — sample logs, plan JSON, and other inputs for manual or scripted testing (Streamlit upload, CLI `--file`, or `POST /analyze`). |
| **`screenshots/`** | **UI screenshots** — captures of the Streamlit app and related views for documentation and demos. |
| **`docs/`** | **Project overview** and supplemental docs — start with **`docs/overview.md`** for a concise pipeline and repository map. |

---

## Project layout

```text
rca-agent/
  src/
    agents/        # LangGraph node functions
    apis/          # FastAPI app
    graphs/        # State schema + workflow
    memory/        # Checkpointer + JSON incident store
    tools/         # Helpers
    utils/         # config, LLM, parsers
    ui/            # CLI + Streamlit
  tests/           # Sample / test inputs (see table above)
  screenshots/     # UI screenshots (see table above)
  docs/            # Overview and extra documentation (see table above)
  main.py          # CLI entry + ASGI app export for uvicorn
  pyproject.toml
  requirements.txt
```
## Future Scope

As this system is modeled based on a real project, there is scope for extending its use to CPDSS and similar projects or operational workflows.

- **Generalization:** With refinement, the pipeline can be adapted for broader root cause analysis in other planning or logistics domains, not just CPDSS outputs.
- **Agent Orchestration Improvements:**  
  Fine-tuning the agent orchestration with more representative samples (diverse plan types and real production logs) will improve the system's robustness, accuracy, and scalability.
- **Learning from Feedback:**  
  Integrating feedback loops or automated history analysis could enable the agents to learn from prior failures, making the recommendations increasingly reliable over time.
- **Advanced Integration:**  
  Additional integration points—such as with real-time monitoring systems or alerting frameworks—could turn this pipeline into a proactive diagnostic component in complex operational environments.
  
---
# How to run (rca-agent)

Short steps to run the **AI Root Cause Analysis** pipeline. All commands assume your shell’s current directory is the **`rca-agent`** project folder.

---

## 1. Prerequisites

- **Python** 3.10+
- **Ollama** installed on a machine reachable from this app ([ollama.com](https://ollama.com))
- A model pulled on that Ollama host (default in config is **`llama3`**):

  ```bash
  ollama pull llama3
  ```

---

## 2. Point the app at your Ollama server

Connection settings live in **`src/utils/config.py`**.

- Set **`OLLAMA_BASE_URL`** there (for example a LAN IP and port **11434**).  
- A **`.env` file is not required**; you can also use environment variables if you prefer.

After editing, the app uses whatever **`get_config()`** resolves (see that file for precedence).

---

## 3. Install Python dependencies

```bash
cd rca-agent
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

The second line performs an editable install so packages under **`src/`** are importable (`agents`, `apis`, `graphs`, … — needed for `main.py`, Uvicorn, and Streamlit).

---

## 4. Ensure Ollama is running

On the machine where Ollama is installed, the service should be up (often automatic). You can verify from a machine that can reach the server:

```bash
curl -s http://<ollama-host>:11434/api/tags
```

Use the same host/port as in **`OLLAMA_BASE_URL`** in **`src/utils/config.py`**.

---

## 5. Run the application (pick one)

### A. Command-line (full graph + checkpointer)

```bash
cd rca-agent
source .venv/bin/activate
python main.py "Paste your log text here"
```

Optional thread id for checkpointing:

```bash
python main.py --thread-id my-thread "Another log"
```

### B. HTTP API (FastAPI)

```bash
cd rca-agent
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level info
```

- App: **http://127.0.0.1:8000**
- Analyze: **`POST /analyze`** (or **`PUT /analyze`**) with JSON `{"log": "..."}`

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"log":"ERROR: connection timeout"}'
```

### C. Streamlit UI (needs the API)

1. Start the API (section B) on port **8000**.
2. In another terminal:

   ```bash
   cd rca-agent
   source .venv/bin/activate
   streamlit run src/ui/streamlit_app.py
   ```

You can upload logs as **`.txt`**, **`.log`**, or **`.out`** (e.g. build/test stdout); use **`.json`** for a plan.

If the API is not on the default URL, set:

```bash
export RCA_API_URL=http://127.0.0.1:8000
```

---

## Optional: strict startup if Ollama is down

When starting the API, you can force exit if Ollama is unreachable:

```bash
RCA_STRICT_OLLAMA=1 uvicorn main:app --reload --port 8000
```

---

## More detail

See **`README.md`** in this folder for architecture, configuration tables, and troubleshooting.

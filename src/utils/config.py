"""Centralized LLM settings (Ollama or Anthropic Claude). Edit this file or set environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _rca_project_root() -> Path:
    """The ``rca-agent`` checkout: contains ``requirements.txt`` and ``src/graphs/workflow.py``."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "requirements.txt").is_file() and (parent / "src" / "graphs" / "workflow.py").is_file():
            return parent
    return here.parents[2]


def _load_dotenv() -> None:
    """Load ``.env`` files into the process environment when present.

    Tries, in order (first wins per key; shell-exported vars still win over all):

    1. ``rca-agent/.env`` (next to this package)
    2. Parent folder ``MiniProject/.env`` (repo root above ``rca-agent``)
    3. ``./.env`` in the current working directory (where you run uvicorn)

    Uses ``python-dotenv`` when installed; otherwise no-op.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    rca_root = _rca_project_root()
    candidates = [
        rca_root / ".env",
        rca_root.parent / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()
    for env_file in candidates:
        try:
            resolved = env_file.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        load_dotenv(resolved, override=False)


_load_dotenv()

# Default provider is Anthropic (see ``_DEFAULT_LLM_PROVIDER``). Set ``ANTHROPIC_API_KEY``
# in the environment or in ``rca-agent/.env`` (see ``.env.example``). Use local Ollama:
# ``RCA_LLM_PROVIDER=ollama`` in ``.env`` or export it (and configure ``OLLAMA_*``).

# --- Optional: pin the Ollama server here when using ``RCA_LLM_PROVIDER=ollama``.
OLLAMA_BASE_URL: str | None = "http://192.168.2.106:11434"

# Defaults when neither the override above nor environment variables are set.
_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "mistral"
_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
_DEFAULT_TIMEOUT = 120.0
_DEFAULT_RETRY_COUNT = 2
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_DEBUG = False
_DEFAULT_LLM_PROVIDER = "anthropic"
# _DEFAULT_LLM_PROVIDER = "ollama"

def _debug_from_env() -> bool:
    """True if ``DEBUG`` or ``RCA_DEBUG`` is set to a truthy value (parsed only here)."""
    for key in ("DEBUG", "RCA_DEBUG"):
        v = os.getenv(key)
        if v is not None and str(v).strip().lower() in ("1", "true", "yes", "on"):
            return True
    return _DEFAULT_DEBUG


def _llm_provider() -> str:
    """``ollama`` (local) or ``anthropic`` (Claude API)."""
    v = (os.getenv("RCA_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or _DEFAULT_LLM_PROVIDER).strip().lower()
    if v in ("ollama", "anthropic"):
        return v
    return _DEFAULT_LLM_PROVIDER


def get_config() -> dict[str, Any]:
    """Return LLM connection and generation settings.

    **Provider:** ``RCA_LLM_PROVIDER`` / ``LLM_PROVIDER`` → ``anthropic`` (default) or ``ollama``.

    **Ollama — precedence for ``base_url``:** module-level ``OLLAMA_BASE_URL`` (if set) →
    environment ``OLLAMA_BASE_URL`` / ``RCA_OLLAMA_BASE_URL`` → ``_DEFAULT_BASE_URL``.
    Model: ``OLLAMA_MODEL`` / ``RCA_OLLAMA_MODEL``.

    **Anthropic:** ``ANTHROPIC_API_KEY`` or ``RCA_ANTHROPIC_API_KEY`` (required to call the API).
    Model: ``ANTHROPIC_MODEL`` / ``RCA_ANTHROPIC_MODEL`` (default from module).
    Log analyser node: ``RCA_LOG_ANALYSER_MODEL`` / ``ANTHROPIC_LOG_ANALYSER_MODEL`` (default Claude Sonnet 3.5).
    Recommendation node: ``RCA_RECOMMENDATION_MODEL`` / ``ANTHROPIC_RECOMMENDATION_MODEL`` (default ``claude-opus-4-6``).
    Optional: ``ANTHROPIC_API_URL`` for proxies / VPC endpoints.

    Returns a dict including: ``llm_provider``, ``base_url``, ``model``, ``anthropic_api_key``,
    ``anthropic_api_url``, ``timeout``, ``retry_count``, ``temperature``, ``debug``.
    """
    provider = _llm_provider()
    timeout = float(os.getenv("OLLAMA_TIMEOUT", str(_DEFAULT_TIMEOUT)))
    retry_count = int(os.getenv("OLLAMA_RETRY_COUNT", str(_DEFAULT_RETRY_COUNT)))
    temperature = float(
        os.getenv("OLLAMA_TEMPERATURE")
        or os.getenv("RCA_OLLAMA_TEMPERATURE")
        or os.getenv("ANTHROPIC_TEMPERATURE")
        or str(_DEFAULT_TEMPERATURE)
    )

    if provider == "anthropic":
        model = (
            os.getenv("ANTHROPIC_MODEL")
            or os.getenv("RCA_ANTHROPIC_MODEL")
            or _DEFAULT_ANTHROPIC_MODEL
        ).strip()
        api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("RCA_ANTHROPIC_API_KEY") or "").strip()
        api_url = (os.getenv("ANTHROPIC_API_URL") or os.getenv("RCA_ANTHROPIC_API_URL") or "").strip() or None
        return {
            "llm_provider": provider,
            "base_url": "",
            "model": model,
            "anthropic_api_key": api_key,
            "anthropic_api_url": api_url,
            "timeout": timeout,
            "retry_count": retry_count,
            "temperature": temperature,
            "debug": _debug_from_env(),
        }

    if OLLAMA_BASE_URL:
        base = str(OLLAMA_BASE_URL).rstrip("/")
    else:
        base = (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("RCA_OLLAMA_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
    model = (
        os.getenv("OLLAMA_MODEL")
        or os.getenv("RCA_OLLAMA_MODEL")
        or _DEFAULT_MODEL
    ).strip()
    return {
        "llm_provider": provider,
        "base_url": base,
        "model": model,
        "anthropic_api_key": "",
        "anthropic_api_url": None,
        "timeout": timeout,
        "retry_count": retry_count,
        "temperature": temperature,
        "debug": _debug_from_env(),
    }

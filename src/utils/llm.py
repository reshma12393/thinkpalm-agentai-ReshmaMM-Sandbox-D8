"""Shared chat model and LLM invoke helpers (Ollama or Anthropic Claude via :mod:`utils.config`)."""

from __future__ import annotations

import inspect
import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from utils.config import get_config
from utils.llm_logging import llm_debug
from utils.llm_retry import invoke_llm_chain  # noqa: F401 — re-exported for callers

# Cache one instance per unique client settings from :func:`get_config` and overrides.
_cache: dict[tuple[Any, ...], BaseChatModel] = {}


def _build_chat_ollama(
    *,
    model: str,
    base_url: str,
    temperature: float,
    timeout: float,
) -> ChatOllama:
    """Construct ``ChatOllama`` with HTTP timeouts on the underlying Ollama client."""
    params = inspect.signature(ChatOllama.__init__).parameters
    kwargs: dict[str, object] = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
    }
    if "timeout" in params:
        kwargs["timeout"] = timeout
    ck: dict[str, float] = {"timeout": timeout}
    kwargs["client_kwargs"] = ck
    kwargs["async_client_kwargs"] = ck
    return ChatOllama(**kwargs)


def _build_chat_anthropic(
    *,
    model: str,
    api_key: str,
    api_url: str | None,
    temperature: float,
    timeout: float,
) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    kwargs: dict[str, Any] = {
        "model": model,
        "anthropic_api_key": api_key,
        "temperature": temperature,
        "default_request_timeout": timeout,
    }
    if api_url:
        kwargs["anthropic_api_url"] = api_url
    return ChatAnthropic(**kwargs)


def get_llm(model: str | None = None, *, temperature: float | None = None) -> BaseChatModel:
    """Return a shared chat model from configuration.

    **Ollama** (default): ``RCA_LLM_PROVIDER=ollama`` — uses ``OLLAMA_*`` / module overrides.

    **Anthropic Claude API:** set ``RCA_LLM_PROVIDER=anthropic`` and ``ANTHROPIC_API_KEY``.
    Optional: ``ANTHROPIC_MODEL``, ``ANTHROPIC_API_URL``.

    - **Retries:** Use :func:`invoke_llm_chain` to run ``prompt | get_llm()`` with the shared retry policy.
    """
    cfg = get_config()
    provider = str(cfg.get("llm_provider") or "ollama")
    temp = float(cfg["temperature"]) if temperature is None else float(temperature)
    timeout = float(cfg["timeout"])
    name = (model or cfg["model"]).strip()

    if provider == "anthropic":
        api_key = str(cfg.get("anthropic_api_key") or "").strip()
        if not api_key:
            raise ValueError(
                "Anthropic is selected (RCA_LLM_PROVIDER=anthropic) but ANTHROPIC_API_KEY "
                "(or RCA_ANTHROPIC_API_KEY) is not set."
            )
        api_url = cfg.get("anthropic_api_url")
        key = ("anthropic", name, temp, timeout, hash(api_key), api_url or "")
        if key not in _cache:
            _cache[key] = _build_chat_anthropic(
                model=name,
                api_key=api_key,
                api_url=api_url if isinstance(api_url, str) else None,
                temperature=temp,
                timeout=timeout,
            )
            llm_debug(
                "LLM client created: provider=anthropic model=%s temperature=%s timeout=%s",
                name,
                temp,
                timeout,
            )
        return _cache[key]

    base_url = str(cfg["base_url"]).rstrip("/")
    key = ("ollama", name, temp, base_url, timeout)
    if key not in _cache:
        _cache[key] = _build_chat_ollama(
            model=name,
            base_url=base_url,
            temperature=temp,
            timeout=timeout,
        )
        llm_debug(
            "LLM client created: provider=ollama base_url=%s model=%s temperature=%s timeout=%s",
            base_url,
            name,
            temp,
            timeout,
        )
    return _cache[key]


def get_log_analyser_llm(*, temperature: float | None = 0.2) -> BaseChatModel:
    """LLM for :mod:`agents.log_analyser_agent` — **Claude Sonnet** when ``RCA_LLM_PROVIDER=anthropic``.

    Model: ``RCA_LOG_ANALYSER_MODEL`` / ``ANTHROPIC_LOG_ANALYSER_MODEL`` (default ``claude-3-5-sonnet-20241022``).
    For Ollama, uses the configured Ollama model (Sonnet not available locally).
    """
    cfg = get_config()
    temp = float(cfg["temperature"]) if temperature is None else float(temperature)
    if str(cfg.get("llm_provider") or "") == "anthropic":
        name = (
            os.getenv("RCA_LOG_ANALYSER_MODEL")
            or os.getenv("ANTHROPIC_LOG_ANALYSER_MODEL")
            or "claude-3-5-sonnet-20241022"
        ).strip()
        return get_llm(model=name, temperature=temp)
    return get_llm(temperature=temp)


def get_recommendation_llm(*, temperature: float | None = 0.2) -> BaseChatModel:
    """LLM for :mod:`agents.recommendation_agent` — **Claude Opus** when ``RCA_LLM_PROVIDER=anthropic``.

    Model: ``RCA_RECOMMENDATION_MODEL`` / ``ANTHROPIC_RECOMMENDATION_MODEL`` (default ``claude-opus-4-6``).
    For Ollama, uses the configured Ollama model.
    """
    cfg = get_config()
    temp = float(cfg["temperature"]) if temperature is None else float(temperature)
    if str(cfg.get("llm_provider") or "") == "anthropic":
        name = (
            os.getenv("RCA_RECOMMENDATION_MODEL")
            or os.getenv("ANTHROPIC_RECOMMENDATION_MODEL")
            or "claude-opus-4-6"
        ).strip()
        return get_llm(model=name, temperature=temp)
    return get_llm(temperature=temp)

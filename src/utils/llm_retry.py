"""Retry LLM chain invokes for empty output, malformed JSON, and transient failures."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import Runnable

from utils.config import get_config
from utils.llm_logging import llm_debug
from utils.parser import extract_llm_json_object

_logger = logging.getLogger("rca.llm")


def invoke_llm_chain(
    chain: Runnable,
    invoke_input: dict[str, Any],
    *,
    max_retries: int | None = None,
    require_json_object: bool = True,
) -> str:
    """Run ``chain.invoke`` with retries before the caller applies fallbacks.

    Default ``max_retries`` comes from :func:`utils.config.get_config` (``retry_count``).

    - **Attempts:** ``1 + max_retries``.
    - **Retries when:** ``invoke`` raises; response is empty/whitespace; or
      ``require_json_object`` and no JSON object can be extracted (malformed).
    - **Returns:** Last non-empty string if JSON never parsed but text exists,
      else ``""`` after exhausting attempts.

    When ``get_config()['debug']`` is true, logs request lifecycle, config snapshot, errors, and retries.
    """
    cfg = get_config()
    if max_retries is None:
        max_retries = int(cfg["retry_count"])
    attempts = max_retries + 1

    provider = str(cfg.get("llm_provider") or "ollama")
    endpoint = (
        cfg["base_url"]
        if provider == "ollama"
        else (cfg.get("anthropic_api_url") or "https://api.anthropic.com")
    )
    llm_debug(
        "LLM request start: provider=%s endpoint=%s model=%s max_retries=%s attempts=%s require_json_object=%s invoke_keys=%s",
        provider,
        endpoint,
        cfg["model"],
        max_retries,
        attempts,
        require_json_object,
        list(invoke_input.keys()),
    )
    _logger.info(
        "LLM request in flight (%s, timeout=%ss, %s, model=%s) — graph will continue after the model responds.",
        provider,
        cfg["timeout"],
        endpoint,
        cfg["model"],
    )

    last_content = ""
    for attempt in range(1, attempts + 1):
        try:
            msg = chain.invoke(invoke_input)
            raw = getattr(msg, "content", None)
            if raw is None:
                raw = str(msg)
            content = str(raw)
        except Exception as exc:
            llm_debug(
                "LLM invoke attempt %s/%s error: %s",
                attempt,
                attempts,
                exc,
                exc_info=True,
            )
            continue
        last_content = content
        stripped = content.strip()
        if not stripped:
            llm_debug(
                "LLM invoke attempt %s/%s: empty response, retrying",
                attempt,
                attempts,
            )
            continue
        if require_json_object:
            if extract_llm_json_object(content) is not None:
                llm_debug(
                    "LLM request end: success on attempt %s/%s (JSON ok, len=%s)",
                    attempt,
                    attempts,
                    len(content),
                )
                return content
            llm_debug(
                "LLM invoke attempt %s/%s: malformed JSON, retrying",
                attempt,
                attempts,
            )
            continue
        llm_debug(
            "LLM request end: success on attempt %s/%s (len=%s)",
            attempt,
            attempts,
            len(content),
        )
        return content

    llm_debug(
        "LLM request end: exhausted after %s attempts (returning last content len=%s)",
        attempts,
        len(last_content or ""),
    )
    return last_content or ""

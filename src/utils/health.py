"""Ollama connectivity check using :func:`utils.config.get_config` only (no direct env reads)."""

from __future__ import annotations

import urllib.error
import urllib.request

from utils.config import get_config


def check_ollama_connection(*, silent: bool = False) -> bool:
    """GET ``/api/tags`` on the configured Ollama base URL.

    Uses ``base_url`` and ``timeout`` from :func:`utils.config.get_config`.

    Returns:
        ``True`` if the HTTP response status is 200, else ``False``.

    On failure, unless ``silent`` is ``True``, prints
    ``Failed to connect to Ollama at <base_url>``.
    """
    cfg = get_config()
    base_url = str(cfg["base_url"]).rstrip("/")
    timeout = float(cfg["timeout"])
    url = f"{base_url}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode() == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        if not silent:
            print(f"Failed to connect to Ollama at {base_url}")
        return False

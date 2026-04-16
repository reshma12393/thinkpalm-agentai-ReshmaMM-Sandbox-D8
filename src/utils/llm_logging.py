"""LLM debug logging gated solely by :func:`utils.config.get_config` ``[\"debug\"]``."""

from __future__ import annotations

import logging
from typing import Any

from utils.config import get_config

_logger = logging.getLogger("rca.llm")


def llm_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Emit INFO log only when ``get_config()['debug']`` is true."""
    if not get_config()["debug"]:
        return
    _logger.info(msg, *args, **kwargs)

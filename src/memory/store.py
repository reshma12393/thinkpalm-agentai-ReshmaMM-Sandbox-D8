"""Append-only RCA incident history in a local JSON file."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_STORE_PATH = Path(__file__).resolve().parent / "store.json"


def save_incident(data: dict) -> None:
    """Append one incident. Ensures raw_log, root_cause, recommendation, timestamp.

    Optional keys (for retrieval / similarity): ``detected_patterns``, compact ``parsed_log``.
    """
    print("Saving incident to memory...", flush=True)
    ts = data.get("timestamp")
    if not ts:
        ts = datetime.now(timezone.utc).isoformat()
    record: dict[str, object] = {
        "raw_log": str(data.get("raw_log", "")),
        "root_cause": str(data.get("root_cause", "")),
        "recommendation": str(data.get("recommendation", "")),
        "timestamp": str(ts),
    }
    dp = data.get("detected_patterns")
    if isinstance(dp, list):
        record["detected_patterns"] = [str(x) for x in dp if str(x).strip()][:32]
    pl = data.get("parsed_log")
    if isinstance(pl, dict):
        kw = pl.get("keywords")
        keywords: list[str] = []
        if isinstance(kw, list):
            keywords = [str(x) for x in kw if str(x).strip()][:24]
        record["parsed_log"] = {
            "error_type": str(pl.get("error_type", "") or ""),
            "module": str(pl.get("module", "") or ""),
            "keywords": keywords,
        }
    history = load_history()
    history.append(record)
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_history() -> list:
    """Return all saved incidents, oldest first. Missing or invalid file yields []."""
    if not _STORE_PATH.is_file():
        return []
    try:
        text = _STORE_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return []
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []

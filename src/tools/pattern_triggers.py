"""Shared log substring triggers (from ``pattern_lookup``) for preprocess + rule lookup.

Each key is matched case-insensitively as a substring on a line; value is the RCA label.
"""

from __future__ import annotations

# Needle (matched in lowercased line) -> human-readable failure / signal label
LOG_SUBSTRING_TRIGGERS: dict[str, str] = {
    "density mismatch": "Input data inconsistency",
    "constraint violation": "Invalid optimization constraints",
    "timeout": "Performance bottleneck",
}

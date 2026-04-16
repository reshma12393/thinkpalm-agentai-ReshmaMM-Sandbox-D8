"""Normalize multi-line RCA text (e.g. exactly three lines for display/API)."""

from __future__ import annotations

import re


def normalize_to_n_lines(text: str, n: int = 3) -> str:
    """Return exactly ``n`` non-empty lines, newline-separated.

    Splits on existing newlines first; if fewer than ``n`` lines, splits the remainder on
    sentence boundaries; if still short, extends by repeating the last segment (last resort).
    Does not inject domain-specific example phrases.
    """
    text = (text or "").strip()
    if not text:
        return "\n".join(["—"] * n)

    raw_lines = [x.strip() for x in text.replace("\r\n", "\n").split("\n") if x.strip()]
    if len(raw_lines) >= n:
        return "\n".join(raw_lines[:n])

    blob = " ".join(raw_lines) if raw_lines else text
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", blob) if p.strip()]
    out: list[str] = []
    for p in parts:
        if len(out) >= n:
            break
        out.append(p)
    if len(out) < n and len(parts) == 1 and parts[0]:
        words = parts[0].split()
        if len(words) >= n:
            base, rem = divmod(len(words), n)
            out_w: list[str] = []
            idx = 0
            for line_i in range(n):
                take = base + (1 if line_i < rem else 0)
                out_w.append(" ".join(words[idx : idx + take]))
                idx += take
            return "\n".join(out_w)
    if not out:
        step = max(1, len(blob) // n)
        for i in range(0, min(len(blob), n * step), step):
            seg = blob[i : i + step].strip()
            if seg:
                out.append(seg)
            if len(out) >= n:
                break
    while len(out) < n:
        out.append(out[-1] if out else blob[: min(240, len(blob))])
    return "\n".join(out[:n])

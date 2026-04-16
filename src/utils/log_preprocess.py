"""Heuristic error/warning line scan before LLM calls (no Ollama).

Covers typical **log file** shapes: log levels (ERROR/WARN), bracket/pipe delimiters,
Loguru-style ``| ERROR |`` columns, Uvicorn ``ERROR:`` / ASGI lines, asyncio task
lines, SQLAlchemy/psycopg2 exception tails, ``level=`` / ``severity=`` structured
logs, stack traces, and domain triggers from ``pattern_triggers``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from tools.pattern_triggers import LOG_SUBSTRING_TRIGGERS


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# Ordered: first match wins per line (put specific stack / format patterns before broad ``ERROR``).
_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Traceback \(most recent call last\)"), "python_traceback"),
    (re.compile(r'(?i)^\s*File "[^"]+", line \d+(?:, in \w+)?'), "python_stack_frame"),
    (re.compile(r"(?i)Caused by:\s*$"), "java_caused_by"),
    (re.compile(r"(?i)Exception in thread"), "java_exception_thread"),
    (re.compile(r"(?i)^\s+at [\w$.]+\((?:[\w.]+\.java|\w+\.scala):\d+\)\s*$"), "java_scala_frame"),
    (re.compile(r"(?i)uncaught exception|unhandled exception|unhandled rejection"), "uncaught"),
    (re.compile(r"(?i)(?:stack trace|stacktrace)\s*:"), "stacktrace_label"),
    # Uvicorn / asyncio (common in nohup.out)
    (re.compile(r"(?i)exception in asgi application"), "asgi_exception_line"),
    (re.compile(r"(?i)task exception was never retrieved"), "async_task_lost"),
    (re.compile(r"(?i)future:\s*<task[^>]*\bexception\s*="), "asyncio_future_exc"),
    # DB exception lines (SQLAlchemy / PostgreSQL)
    (re.compile(r"(?i)sqlalchemy\.exc\.\w+"), "sqlalchemy_exc"),
    (re.compile(r"(?i)psycopg2\.errors\.\w+"), "psycopg2_error"),
    (re.compile(r"(?i)duplicate key value violates unique constraint"), "pg_unique_violation"),
    (re.compile(r"(?i)^\s*(?:Error|ERROR):\s"), "error_prefix_colon"),
    (re.compile(r"(?i)\b(?:FATAL|CRITICAL|SEVERE)\b"), "level_fatal"),
    # Bracket / pipe log levels (log4j, nginx-style, many app loggers)
    (re.compile(r"(?i)\[(?:ERROR|ERR|FATAL|CRITICAL|SEVERE)\]"), "bracket_error"),
    # Loguru: " | ERROR    | " (spaces); compact "|ERROR|" still matches below
    (re.compile(r"(?i)\|\s*(?:ERROR|ERR|FATAL|CRITICAL)\s+\|"), "loguru_error_pipe"),
    (re.compile(r"(?i)\|(?:ERROR|ERR|FATAL|CRITICAL)\|"), "pipe_error"),
    (re.compile(r"(?i)(?:^|\s)(?:ERROR|FATAL|CRITICAL)\s*>\s"), "arrow_error"),
    # key=value / JSON-ish severity (structlog, zap, winston-json)
    (re.compile(r"(?i)(?:^|[\s,{\"])level[\"']?\s*[=:]\s*[\"']?(?:error|fatal|critical|err)\b"), "kv_level"),
    (re.compile(r"(?i)(?:^|[\s,{\"])severity[\"']?\s*[=:]\s*[\"']?(?:ERROR|FATAL|CRITICAL)\b"), "kv_severity"),
    (re.compile(r"(?i)\b(?:log\s+)?level\s*=\s*(?:error|fatal|critical)\b"), "equals_level"),
    (re.compile(r"(?i)\bERROR\b"), "level_error"),
    (re.compile(r"(?i)\bERR\b"), "token_err"),
    (re.compile(r"(?i)\berror\b.*(?:exception|failed|failure)"), "error_failure"),
    (re.compile(r"(?i)\b(?:panic|fatal error)\b"), "panic_or_fatal"),
    (re.compile(r"(?i)java\.lang\.[\w]+(?:Exception|Error)\b"), "java_exception_type"),
    (re.compile(
        r"(?i)\b(?:RuntimeException|NullPointerException|IOException|SQLException|"
        r"OSError|KeyError|AttributeError|TypeError|ValueError|ImportError|"
        r"ReferenceError|SyntaxError|AssertionError|ValidationError|"
        r"ResourceClosedError|InvalidRequestError|UniqueViolation)(?:\s*:|\s|$)"
    ),
     "named_exception"),
    (re.compile(r"(?i)(?:nested exception|root cause)\s*:"), "nested_exception"),
    (re.compile(r"(?i)ECONNREFUSED|ETIMEDOUT|Connection refused|connection reset"), "network_hard"),
    (re.compile(r"(?i)ssl handshake|certificate (?:verify failed|error)|TLSV\d_ALERT"), "tls_error"),
    (re.compile(r"(?i)out of memory|\boom\b|killed process|SIGKILL|SIGSEGV|segmentation fault"), "oom_signal"),
    (re.compile(r"(?i)exit (?:code|status)\s*[1-9]\d*"), "nonzero_exit"),
    (re.compile(r"(?i)non-zero exit|failed with exit"), "exit_failed"),
    (re.compile(r"(?i)\b(?:FAILED|FAILURE)\b\s*:"), "failed_label"),
]

_WARNING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Often logged at INFO but signals HTTP/API failure path (sample nohup.out)
    (re.compile(r"(?i)\bHTTP Exception\b"), "http_exception_phrase"),
    (re.compile(r"(?i)\[(?:WARN|WARNING|WRN)\]"), "bracket_warn"),
    (re.compile(r"(?i)\|\s*WARN(?:ING)?\s+\|"), "loguru_warn_pipe"),
    (re.compile(r"(?i)\|(?:WARN|WARNING|WRN)\|"), "pipe_warn"),
    (re.compile(r"(?i)(?:^|[\s,{\"])level[\"']?\s*[=:]\s*[\"']?(?:warn|warning)\b"), "kv_level_warn"),
    (re.compile(r"(?i)(?:^|[\s,{\"])severity[\"']?\s*[=:]\s*[\"']?WARN(?:ING)?\b"), "kv_severity_warn"),
    (re.compile(r"(?i)\bWARN(?:ING)?\b"), "level_warn"),
    (re.compile(r"(?i)\bWRN\b"), "token_wrn"),
    (re.compile(r"(?i)\bDEPRECATED\b|\bdeprecated\b"), "deprecated"),
    (re.compile(r"(?i)\bNOTICE\b"), "notice"),
    (re.compile(r"(?i)\bretri(?:ed|ing)\b.*\b(?:success|ok)\b"), "retry_recovered"),
]


@dataclass
class PreprocessScan:
    error_hits: list[str] = field(default_factory=list)
    warning_hits: list[str] = field(default_factory=list)
    summary: str = ""
    error_line_total: int = 0
    warning_line_total: int = 0


def summarize_hit_themes(hits: list[str], *, top: int = 6) -> str:
    """Aggregate bracket/tag labels from pre-scan hit lines (not raw log text)."""
    themes: list[str] = []
    for h in hits[:64]:
        m = re.search(r"\[([^\]]+)\]", h)
        if not m:
            continue
        inner = m.group(1)
        if inner.startswith("trigger:"):
            inner = inner.split("|", 1)[0].replace("trigger:", "").strip()
        themes.append(inner.strip()[:72])
    if not themes:
        return "none tagged in samples (or unlabeled lines)"
    cnt = Counter(themes)
    return ", ".join(f"{k} ({v})" for k, v in cnt.most_common(top))


def _prescan_summary_prose(
    err_total: int,
    warn_total: int,
    err_hits: list[str],
    warn_hits: list[str],
    max_stored: int,
) -> str:
    """Short prescan narrative—counts and themes only, not a dump of sample lines."""
    parts: list[str] = [
        f"A heuristic pass over the log classified **{err_total}** line(s) as error/failure-like and "
        f"**{warn_total}** as warning-like (up to **{max_stored}** samples per kind are stored in state for tools; "
        "they are not listed here)."
    ]
    if err_total:
        parts.append(f"Error-side themes seen in the sample set: {summarize_hit_themes(err_hits)}.")
    else:
        parts.append("No strong error/failure line patterns were detected in the pre-scan.")
    if warn_total:
        parts.append(f"Warning-side themes in the sample set: {summarize_hit_themes(warn_hits)}.")
    else:
        parts.append("No strong warning line patterns were detected in the pre-scan.")
    return "\n".join(parts)


def _truncate_line(s: str, max_len: int = 220) -> str:
    s = strip_ansi(s).rstrip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _match_substring_trigger(lowercase_line: str) -> tuple[str, str] | None:
    """First domain trigger from ``LOG_SUBSTRING_TRIGGERS`` that matches; else None."""
    for needle, label in LOG_SUBSTRING_TRIGGERS.items():
        if needle in lowercase_line:
            return needle, label
    return None


def _analyze_line(line: str) -> tuple[str | None, str]:
    """Return (classification, display_fragment) for one line.

    classification is ``error``, ``warning``, or None.
    """
    s = _truncate_line(line)
    if not s.strip():
        return None, ""

    lower = s.lower()

    for rx, label in _ERROR_PATTERNS:
        if rx.search(s):
            return "error", f"[{label}] {s}"

    trig = _match_substring_trigger(lower)
    if trig is not None:
        needle, plabel = trig
        return "error", f"[trigger:{plabel}|{needle!r}] {s}"

    for rx, label in _WARNING_PATTERNS:
        if rx.search(s):
            return "warning", f"[{label}] {s}"

    return None, ""


def scan_log_text(
    raw: str,
    *,
    max_hits_per_kind: int = 64,
    max_lines: int = 100_000,
) -> PreprocessScan:
    """Scan line-by-line for error- vs warning-like patterns (regex + ``pattern_triggers``)."""
    lines = raw.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    err: list[str] = []
    warn: list[str] = []
    err_total = 0
    warn_total = 0

    for i, line in enumerate(lines, start=1):
        cls, fragment = _analyze_line(line)
        if cls == "error":
            err_total += 1
            if len(err) < max_hits_per_kind:
                err.append(f"line {i}: {fragment}")
        elif cls == "warning":
            warn_total += 1
            if len(warn) < max_hits_per_kind:
                warn.append(f"line {i}: {fragment}")

    summary = _prescan_summary_prose(err_total, warn_total, err, warn, max_hits_per_kind)
    return PreprocessScan(
        error_hits=err,
        warning_hits=warn,
        summary=summary,
        error_line_total=err_total,
        warning_line_total=warn_total,
    )


def build_preprocess_llm_digest(scan: PreprocessScan, *, max_chars: int = 6000) -> str:
    """Short digest for downstream LLMs: **summary themes and counts**, not raw line dumps."""
    err_h = scan.error_hits
    warn_h = scan.warning_hits
    stack_n = sum(1 for h in err_h if "[python_stack_frame]" in h)

    lines: list[str] = [
        "### Pre-scan digest (summary — regex/heuristics, no LLM)",
        f"- **Totals in file:** {scan.error_line_total} error-classified lines, "
        f"{scan.warning_line_total} warning-classified lines.",
        f"- **Sample store:** {len(err_h)} error samples, {len(warn_h)} warning samples (capped during scan).",
        f"- **Error themes (tag aggregates from samples):** {summarize_hit_themes(err_h)}.",
        f"- **Warning themes (tag aggregates from samples):** {summarize_hit_themes(warn_h)}.",
    ]
    if stack_n:
        lines.append(
            f"- **Note:** {stack_n} stored error samples are stack-frame lines (often repetitive); "
            "full lines are in state, not listed here."
        )
    lines.append(
        "- Use the raw log excerpt in the same prompt for verbatim detail; this block is only a high-level prescan summary."
    )

    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 40] + "\n... [digest truncated]"
    return text


def excerpt_log_for_llm(raw: str, *, max_total: int = 24_000) -> str:
    """Prefer head + tail when the log is huge so the model still sees both ends."""
    if len(raw) <= max_total:
        return raw
    half = (max_total - 120) // 2
    mid = len(raw) - 2 * half
    return (
        raw[:half]
        + f"\n\n... [{mid} characters omitted from middle of log] ...\n\n"
        + raw[-half:]
    )

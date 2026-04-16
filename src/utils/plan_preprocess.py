"""Structural plan / API-response checks before plan LLM (no Ollama).

Supports shapes like ``[{ "Form_data_logs": [...], "status": "success" }, 200]`` —
payload + HTTP code tuple — and plain JSON objects.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


def _unwrap_api_tuple(obj: Any) -> tuple[Any, int | None]:
    """If value is ``[body, http_code]``, return ``(body, code)``; else ``(obj, None)``."""
    if not isinstance(obj, list) or len(obj) != 2:
        return obj, None
    second = obj[1]
    if isinstance(second, int):
        return obj[0], second
    if isinstance(second, str):
        s = second.strip()
        if s.isdigit():
            return obj[0], int(s)
    return obj, None


_SUCCESS_STATUS = frozenset({"success", "ok", "completed", "complete", "succeeded"})
_FAILURE_STATUS = frozenset({"error", "failed", "failure", "fail"})


def analyze_plan_structure(obj: Any) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` from deterministic rules (no LLM)."""
    errors: list[str] = []
    warnings: list[str] = []

    body, http = _unwrap_api_tuple(obj)
    if http is not None:
        if http >= 500:
            errors.append(f"HTTP status {http} indicates a server error class response.")
        elif http >= 400:
            errors.append(f"HTTP status {http} indicates a client error class response.")
        elif http != 200:
            warnings.append(f"HTTP status is {http} (not 200 OK).")

    if isinstance(body, dict):
        st = body.get("status")
        if isinstance(st, str):
            sl = st.strip().lower()
            if sl in _FAILURE_STATUS:
                errors.append(f"Payload field status is {st!r} (failure).")
            elif sl not in _SUCCESS_STATUS and sl:
                warnings.append(f"Payload field status is {st!r} (verify expected success semantics).")

        fdl = body.get("Form_data_logs")
        if fdl is not None:
            if not isinstance(fdl, list):
                errors.append("Form_data_logs is present but not a JSON array.")
            elif len(fdl) == 0:
                warnings.append("Form_data_logs is empty.")
            else:
                form_ids: list[str] = []
                for i, entry in enumerate(fdl):
                    if not isinstance(entry, dict):
                        warnings.append(f"Form_data_logs[{i}] is not an object.")
                        continue
                    fid = entry.get("form_id")
                    if fid is not None:
                        form_ids.append(str(fid))
                    if not entry.get("form_name") and not entry.get("form_data"):
                        warnings.append(f"Form_data_logs[{i}] has minimal fields (check completeness).")
                dup = [k for k, n in Counter(form_ids).items() if n > 1]
                for k in dup:
                    warnings.append(f"form_id {k!r} appears multiple times (possible duplicate entries).")

    elif isinstance(body, list) and not http:
        warnings.append("Plan root is a JSON array (not the [payload, http_code] two-element form); structure may be ambiguous.")

    return errors, warnings


def normalize_plan_for_llm(obj: Any) -> str:
    """Pretty JSON for the model; unwrap ``[body, code]`` to a clear envelope."""
    body, http = _unwrap_api_tuple(obj)
    if http is not None:
        envelope = {
            "_interpretation": "Unwrapped API-style array [payload, http_status_code].",
            "http_status_code": http,
            "payload": body,
        }
        return json.dumps(envelope, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)


_WARNING_KEY_NAMES = frozenset(
    {
        "warning",
        "warnings",
        "alert",
        "alerts",
        "caution",
        "cautions",
        "caveat",
        "caveats",
    }
)
_ERROR_KEY_NAMES = frozenset(
    {
        "error",
        "errors",
        "err",
        "exception",
        "exceptions",
        "failure",
        "failures",
        "fatal",
        "fault",
        "faults",
    }
)
_MESSAGE_KEY_NAMES = frozenset(
    {
        "message",
        "messages",
        "msg",
        "msgs",
        "note",
        "notes",
        "detail",
        "details",
        "description",
        "descriptions",
        "info",
        "reason",
        "reasons",
        "remark",
        "remarks",
        "issues",
        "issue",
        "comment",
        "comments",
        "log",
        "logs",
        "body",
        "text",
        "summary",
        "summaries",
    }
)


def _bucket_for_key(key: Any) -> str | None:
    n = str(key).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in n:
        n = n.replace("__", "_")
    if n in _WARNING_KEY_NAMES:
        return "warnings"
    if n in _ERROR_KEY_NAMES:
        return "errors"
    if n in _MESSAGE_KEY_NAMES:
        return "messages"
    return None


def _flatten_issue_value(bucket: str, value: Any, path: str, out: dict[str, list[str]], cap: int) -> None:
    seen: set[str] = set(out[bucket])

    def add(text: str) -> None:
        s = str(text).strip()
        if not s or len(out[bucket]) >= cap:
            return
        s = s[:1200]
        if s in seen:
            return
        seen.add(s)
        out[bucket].append(s)

    if isinstance(value, str):
        add(f"{path}: {value}" if path != "$" else value)
        return
    if value is None:
        return
    if isinstance(value, (int, float, bool)):
        add(f"{path}: {value!r}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _flatten_issue_value(bucket, item, f"{path}[{i}]", out, cap)
        return
    if isinstance(value, dict):
        for sk in ("message", "text", "content", "detail", "description", "reason", "msg"):
            if sk in value and isinstance(value[sk], str) and value[sk].strip():
                add(f"{path}.{sk}: {value[sk].strip()}")
        if not any(
            isinstance(value.get(sk), str) and str(value.get(sk, "")).strip()
            for sk in ("message", "text", "content", "detail", "description", "reason", "msg")
        ):
            try:
                add(f"{path}: {json.dumps(value, ensure_ascii=False)[:900]}")
            except (TypeError, ValueError):
                add(f"{path}: {str(value)[:900]}")
        return
    add(f"{path}: {str(value)[:900]}")


def _walk_for_issue_keys(obj: Any, path: str, out: dict[str, list[str]], cap: int) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            b = _bucket_for_key(k)
            sub = f"{path}.{k}" if path != "$" else f"$.{k}"
            if b is not None:
                _flatten_issue_value(b, v, sub, out, cap)
            _walk_for_issue_keys(v, sub, out, cap)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk_for_issue_keys(item, f"{path}[{i}]", out, cap)


def extract_plan_issue_signals(obj: Any, *, max_per_bucket: int = 64) -> dict[str, list[str]]:
    """Walk JSON and collect values under warning-, error-, and message-like keys."""
    out: dict[str, list[str]] = {"warnings": [], "errors": [], "messages": []}
    _walk_for_issue_keys(obj, "$", out, max_per_bucket)
    return out


def _finalize_five_sentences(parts: list[str]) -> str:
    """Join into exactly five sentences (each ends with a period)."""
    cleaned: list[str] = []
    for p in parts:
        t = str(p).strip()
        if not t:
            continue
        if not t.endswith("."):
            t += "."
        cleaned.append(t)
    pad = (
        "Further detail is available in the compact outline and merged assessment lists when present."
    )
    while len(cleaned) < 5:
        cleaned.append(pad)
    return " ".join(cleaned[:5])


def summarize_plan_skeleton_five_sentences(
    obj: Any | None,
    *,
    skeleton_outline: str = "",
    invalid_json: bool = False,
) -> str:
    """Prose summary of plan shape in exactly five sentences (no LLM)."""
    if invalid_json:
        return _finalize_five_sentences(
            [
                "The plan input could not be parsed as JSON, so no object or array shape was extracted.",
                "Structural checks and issue-like key scans were skipped until the payload is valid JSON.",
                "Fix encoding, brackets, and quoting, then resubmit for a full skeleton summary.",
                "Feasibility and merged signals may still reflect preprocessor error strings if any were recorded.",
                "A valid plan document is required for step-level outline and optimization review.",
            ]
        )

    if obj is None:
        return _finalize_five_sentences(
            [
                "No parsed plan object was available when building the skeleton summary.",
                "Submit a JSON object or array so top-level fields and nesting can be described.",
                "Until then, rely on preprocess messages and merged lists for any recorded issues.",
                "The compact outline section may be empty when the root payload is missing.",
                "Re-run analysis after providing structured plan content.",
            ]
        )

    sk = (skeleton_outline or "").strip()
    sig = extract_plan_issue_signals(obj)
    ne, nw, nm = len(sig["errors"]), len(sig["warnings"]), len(sig["messages"])

    if isinstance(obj, dict):
        keys = list(obj.keys())
        nk = len(keys)
        if nk == 0:
            return _finalize_five_sentences(
                [
                    "The plan root is an empty JSON object with no top-level keys.",
                    "No named fields are present to describe steps, phases, or metadata at the root.",
                    f"Issue-like scans still recorded {ne} error-like, {nw} warning-like, and {nm} message-like values if present deeper in the tree.",
                    "An empty object may indicate a placeholder or incomplete upload.",
                    "Populate the plan or use merged lists and feasibility output for assessor context.",
                ]
            )
        preview = ", ".join(str(k) for k in keys[:10])
        if nk > 10:
            preview += ", …"
        s1 = f"The plan root is a JSON object with {nk} top-level key(s): {preview}"
        parts2: list[str] = []
        for k in keys[:6]:
            v = obj[k]
            if isinstance(v, dict):
                tag = "nested object"
            elif isinstance(v, list):
                tag = f"array ({len(v)} item(s))"
            elif isinstance(v, bool):
                tag = "boolean"
            elif isinstance(v, (int, float)):
                tag = "number"
            elif v is None:
                tag = "null"
            else:
                tag = "text"
            parts2.append(f"{k} → {tag}")
        s2 = "Sample field shapes: " + "; ".join(parts2) if parts2 else "Top-level values use mixed JSON types."
        s3 = (
            f"A key scan found {ne} error-like value(s), {nw} warning-like value(s), and {nm} message-like value(s) "
            "under conventional field names."
        )
        if sk:
            s4 = (
                f"A compact JSON outline for reviewers is about {len(sk)} characters and may be truncated for display."
            )
        else:
            s4 = (
                "No separate compact outline string was stored, often because many issue-like hits left the digest lean."
            )
        s5 = "This structural snapshot supports feasibility checks together with merged errors, warnings, and assessor notes."
        return _finalize_five_sentences([s1, s2, s3, s4, s5])

    if isinstance(obj, list):
        n = len(obj)
        s1 = f"The plan root is a JSON array with {n} element(s)."
        if n == 0:
            s2 = "The array is empty, so no ordered steps are encoded at the root."
        else:
            first = obj[0]
            if isinstance(first, dict):
                s2 = f"The first element is an object with {len(first)} key(s)."
            elif isinstance(first, list):
                s2 = f"The first element is a nested array with {len(first)} nested item(s)."
            else:
                s2 = f"The first element is a {type(first).__name__.lower()} scalar."
        s3 = (
            f"Issue-like values in the tree: {ne} error-like, {nw} warning-like, {nm} message-like."
        )
        s4 = (
            f"Compact outline length is about {len(sk)} characters." if sk else "No stored compact outline was attached for this array root."
        )
        s5 = "Array-shaped plans are summarized here for shape review before deeper feasibility analysis."
        return _finalize_five_sentences([s1, s2, s3, s4, s5])

    s1 = f"The plan root is a single JSON value ({type(obj).__name__}), not a nested object or array."
    s2 = "Downstream tooling typically expects an object or array to represent steps and metadata."
    s3 = f"Issue-like scans on this value yielded {ne}, {nw}, and {nm} hits in those categories."
    s4 = "Consider wrapping content in an object if the workflow expects named fields."
    s5 = "Structural review still applies when interpreting feasibility and merged assessment output."
    return _finalize_five_sentences([s1, s2, s3, s4, s5])


def compact_plan_outline(obj: Any, *, max_chars: int = 8000) -> str:
    """Compact JSON when issue-key extraction is sparse (for LLM context only)."""
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        s = str(obj)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n... [truncated]"


def resolve_plan_skeleton_outline(state: dict[str, Any], *, max_chars: int = 8000) -> str:
    """Prefer preprocess skeleton; otherwise build a compact outline from ``plan_json`` (digest / analysis text)."""
    sk = (state.get("plan_skeleton_outline") or "").strip()
    if sk:
        if len(sk) > max_chars:
            return sk[: max_chars - 1] + "…"
        return sk
    raw = (state.get("plan_json") or "").strip()
    if not raw:
        return ""
    try:
        return compact_plan_outline(json.loads(raw), max_chars=max_chars).strip()
    except (json.JSONDecodeError, TypeError):
        return ""


def format_plan_digest_for_llm(
    *,
    structural_errors: list[str],
    structural_warnings: list[str],
    extracted: dict[str, list[str]],
    skeleton_outline: str,
    include_skeleton: bool,
    skeleton_five_sentences: str = "",
) -> str:
    """Human-readable block passed to plan LLM (not the full raw plan JSON)."""
    parts: list[str] = [
        "# Plan digest (pre-parsed)",
        "",
        "The following was derived by scanning JSON keys (warnings/errors/messages-like fields) "
        "and deterministic structure rules. The full raw plan is NOT attached—reason from this digest.",
        "",
    ]
    sk5 = (skeleton_five_sentences or "").strip()
    # if sk5:
    #     parts.extend(["## Plan skeleton summary (five sentences)", "", sk5, ""])
    parts.append("## Structural checks (deterministic)")
    parts.append("")
    if structural_errors:
        parts.append("Errors:")
        parts.extend(f"- {e}" for e in structural_errors)
    else:
        parts.append("Errors: (none)")
    parts.append("")
    if structural_warnings:
        parts.append("Warnings:")
        parts.extend(f"- {w}" for w in structural_warnings)
    else:
        parts.append("Warnings: (none)")
    parts.extend(["", "## Values under issue-like keys", ""])

    ew, ee, em = extracted.get("warnings", []), extracted.get("errors", []), extracted.get("messages", [])
    parts.append("### warnings / alerts (from keys)")
    if ew:
        parts.extend(f"- {x}" for x in ew)
    else:
        parts.append("- (none extracted)")
    parts.extend(["", "### errors / failures (from keys)"])
    if ee:
        parts.extend(f"- {x}" for x in ee)
    else:
        parts.append("- (none extracted)")
    parts.extend(["", "### messages / notes / details (from keys)"])
    if em:
        parts.extend(f"- {x}" for x in em)
    else:
        parts.append("- (none extracted)")

    if include_skeleton and skeleton_outline.strip():
        parts.extend(
            [
                "",
                "## Compact plan outline (shape reference; may be truncated)",
                "",
                skeleton_outline.strip(),
            ]
        )

    return "\n".join(parts)

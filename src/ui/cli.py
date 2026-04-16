"""Minimal CLI to run the graph with optional thread id for checkpointing."""

from __future__ import annotations

import argparse
import json
import logging

from graphs.workflow import get_initial_state, get_runnable_graph

logging.getLogger("rca.tools").setLevel(logging.INFO)
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RCA multi-agent (LangGraph)")
    parser.add_argument(
        "message",
        nargs="?",
        default="Error: connection reset by peer\nTimeout after 30s",
        help="Raw log text to analyze (default is a short sample)",
    )
    parser.add_argument(
        "--thread-id",
        default="default",
        help="Conversation thread id for checkpointing",
    )
    parser.add_argument(
        "--file",
        "-f",
        metavar="PATH",
        help="Read input from a file (.json = plan; .txt / .log / .out = log text)",
    )
    args = parser.parse_args(argv)

    message = args.message
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            message = f.read()

    graph = get_runnable_graph(with_checkpointer=True)
    config = {"configurable": {"thread_id": args.thread_id}}
    result = graph.invoke(get_initial_state(message), config=config)

    print("input_kind:", result.get("input_kind", ""))
    print("parsed_log:", json.dumps(result["parsed_log"], indent=2))
    print("detected_patterns:", result["detected_patterns"])
    print("log_errors_analysis:", result.get("log_errors_analysis", ""))
    print("log_warnings_analysis:", result.get("log_warnings_analysis", ""))
    print("plan_feasible:", result.get("plan_feasible"))
    print("plan_warnings_list:", result.get("plan_warnings_list", []))
    print("plan_errors_list:", result.get("plan_errors_list", []))
    print("plan_messages_list:", result.get("plan_messages_list", []))
    print("plan_deep_analysis:", result.get("plan_deep_analysis", ""))
    print("findings_summary:", result.get("findings_summary", ""))
    print("root_cause:", result["root_cause"])
    print("confidence:", result["confidence"])
    print("severity:", result["severity"])
    print("recommendation:", result["recommendation"])
    print("history:")
    for line in result["history"]:
        print(f"  - {line}")
    return 0

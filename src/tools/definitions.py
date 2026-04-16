"""Register LangChain-compatible tools here."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def echo(message: str) -> str:
    """Echo the message (placeholder for real tools)."""
    return message


def get_tools():
    """Return tools to bind to an agent or expose to the graph."""
    return [echo]

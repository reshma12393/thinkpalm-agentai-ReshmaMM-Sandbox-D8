"""Thread-scoped checkpointing for LangGraph (conversation state)."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer():
    """In-memory saver suitable for dev/tests. Swap for Postgres/SQLite in prod."""
    return MemorySaver()

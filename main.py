"""Application entrypoint for the RCA multi-agent system."""

from __future__ import annotations

import sys

# Exposes ASGI app for: uvicorn main:app --reload (run from rca-agent/ after pip install -e .)
from apis.app import app  # noqa: F401

from ui.cli import main


if __name__ == "__main__":
    sys.exit(main())

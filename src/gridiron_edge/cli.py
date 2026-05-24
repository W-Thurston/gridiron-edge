# src/gridiron_edge/cli.py
"""Gridiron Edge CLI entrypoint shim.

Delegates to gridiron_edge.cli.main so that the pyproject.toml entry
point (gridiron_edge.cli:main) continues to work unchanged.
"""

from gridiron_edge.cli.main import app, main

__all__: list[str] = ["app", "main"]

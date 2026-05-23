# src/gridiron_edge/cli/__init__.py
"""Gridiron Edge CLI package.

Re-exports ``app`` and ``main`` so that existing imports from
``gridiron_edge.cli`` continue to work after the package split.
"""

from gridiron_edge.cli.main import app, main

__all__ = ["app", "main"]

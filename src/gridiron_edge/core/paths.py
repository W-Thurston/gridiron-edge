# src/gridiron_edge/core/paths.py

from pathlib import Path


def repo_root() -> Path:
    """Return the absolute path to the repository root.

    Resolves upward from this file's location:
    ``core/`` → ``gridiron_edge/`` → ``src/`` → repo root.

    Returns:
        Absolute path to the repository root directory.
    """
    return Path(__file__).resolve().parents[3]

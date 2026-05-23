# src/gridiron_edge/cli/_shared.py
"""Shared utilities for CLI sub-modules."""

from __future__ import annotations

import os

import typer


def get_owm_api_key(owm_api_key: str | None) -> str:
    """Resolve OpenWeather API key from flag or environment variable."""
    key: str | None = owm_api_key or os.environ.get("OWM_API_KEY")
    if not key:
        raise typer.BadParameter(
            "Missing OpenWeather API key. Provide --owm-api-key or set env var OWM_API_KEY.",
        )
    return key

# src/gridiron_edge/api/_prop_id.py

"""Prop ID encoding/decoding helpers.

The composite prop_id format ``{game_id}__{player_id}__{stat_type}``
is used by both ``/props/{prop_id}`` and ``/compare/player/{prop_id}``.
This module centralizes the decode logic so both routes stay in sync.
"""

from __future__ import annotations

from fastapi import HTTPException

from gridiron_edge.models.catalog import PROP_STAT_FAMILIES


def decode_prop_id(prop_id: str) -> tuple[str, str, str]:
    """Decode a composite prop_id into (game_id, player_id, stat_type).

    Format: ``{game_id}__{player_id}__{stat_type}``. Double-underscore
    separator so single-underscore game_ids (like "2026_01_KC_LAC")
    aren't ambiguous.

    Raises:
        HTTPException: 404 with actionable message on parse or family
            validation failure.
    """
    parts: list[str] = prop_id.split("__")
    if len(parts) != 3:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Malformed prop_id: {prop_id!r}. "
                f"Expected format: {{game_id}}__{{player_id}}__{{stat_type}}."
            ),
        )
    game_id, player_id, stat_type = parts
    if stat_type not in PROP_STAT_FAMILIES:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown stat_type in prop_id: {stat_type!r}. "
                f"Registered families: {sorted(PROP_STAT_FAMILIES)}."
            ),
        )
    return game_id, player_id, stat_type

# src/gridiron_edge/api/serializers/players.py

"""Serializer for /players/{player_id}/history."""

from __future__ import annotations

from gridiron_edge.api.schemas.players import (
    PlayerGameRow,
    PlayerHistoryResponse,
)


def serialize_player_history(payload: dict) -> PlayerHistoryResponse:
    """Convert a load_player_history payload into the response schema.

    Args:
        payload: Dict from ``load_player_history`` with keys player_id,
            player_name, stat, season, rows.

    Returns:
        PlayerHistoryResponse with per-game items. No field_status —
        the series either exists (populated) or the route 404s upstream.
    """
    rows = payload.get("rows", [])
    items = [
        PlayerGameRow(
            week=int(r["week"]),
            value=r.get("value"),
            opponent=str(r["opponent"]),
            game_id=str(r["game_id"]),
            is_home=bool(r["is_home"]),
        )
        for r in rows
    ]

    return PlayerHistoryResponse(
        player_id=str(payload["player_id"]),
        player_name=str(payload.get("player_name", "")),
        stat=str(payload["stat"]),
        season=payload.get("season"),
        items=items,
        total=len(items),
    )

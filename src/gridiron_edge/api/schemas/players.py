# src/gridiron_edge/api/schemas/players.py

"""Schemas for /players/{player_id}/history."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse


class PlayerGameRow(BaseModel):
    """One game's value for a player's chosen stat.

    Sufficient for a per-game bar chart or sparkline: the week, the stat
    value that game, the opponent, and home/away context.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    week: int
    value: float | None = None
    opponent: str
    game_id: str
    is_home: bool


class PlayerHistoryResponse(BaseListResponse[PlayerGameRow]):
    """Response for GET /players/{player_id}/history.

    Per-game series for one (player, stat, season). Bars don't vary by
    any cohort/split — this is the player's raw game-by-game production.
    """

    player_id: str
    player_name: str = Field(
        default="",
        description="Display name; empty if player_id not found.",
    )
    stat: str = Field(
        description="Requested stat key, e.g. 'rush_yards'.",
    )
    season: int | None = Field(
        default=None,
        description="Season the series covers (int year, e.g. 2024).",
    )

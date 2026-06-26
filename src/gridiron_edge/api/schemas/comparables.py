# src/gridiron_edge/api/schemas/comparables.py
"""Schemas for per-game comparables endpoints (/games/{game_id}/comparables).

Currently blocked on comparables retrieval; responses return null shapes
with structured `_meta.field_status` entries. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class ComparableGame(BaseModel):
    """A historical game similar to the current matchup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date_label: str | None = Field(default=None, description="e.g. '2024 · Wk 11'.")
    favorite: str | None = Field(default=None)
    underdog: str | None = Field(default=None)
    line: str | None = Field(default=None)
    final_score: str | None = Field(default=None)
    favorite_won: bool | None = Field(default=None)
    favorite_covered: bool | None = Field(default=None)
    note: str | None = Field(default=None)


class GameComparables(BaseResponse):
    """Response for GET /games/{game_id}/comparables."""

    game_id: str
    comparables: list[ComparableGame] | None = Field(default=None)
    sample_size: int | None = Field(default=None)
    favorite_win_rate: float | None = Field(default=None)
    favorite_cover_rate: float | None = Field(default=None)

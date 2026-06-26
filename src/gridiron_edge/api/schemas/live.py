# src/gridiron_edge/api/schemas/live.py
"""Schemas for live game endpoints (/live and /live/{game_id}).

Currently blocked on live state ingest; responses return null shapes
with structured `_meta.field_status` entries. See ROADMAP §9.5.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class LiveScore(BaseModel):
    """Current score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    home: int | None = Field(default=None)
    away: int | None = Field(default=None)


class DrivePoint(BaseModel):
    """A single drive entry in the drive chart."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    team: str | None = Field(default=None)
    quarter: str | None = Field(default=None)
    summary: str | None = Field(default=None, description="e.g. '12 plays · 75 yds'.")
    result: str | None = Field(default=None, description="'TD', 'FG', 'Punt', 'Active'.")
    wp_change: float | None = Field(
        default=None, description="Win-prob delta in percentage points."
    )


class LiveOdds(BaseModel):
    """Live odds for a market."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    market: str | None = Field(default=None)
    home_line: str | None = Field(default=None)
    away_line: str | None = Field(default=None)
    fair_home: str | None = Field(default=None)
    fair_away: str | None = Field(default=None)
    edge_pct: float | None = Field(default=None)


class LiveGameSummary(BaseModel):
    """A row in the /live list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: str | None = Field(default=None)
    status: str | None = Field(default=None, description="e.g. 'Q3 · 8:42'.")
    score: LiveScore | None = Field(default=None)
    home_win_prob: float | None = Field(default=None)


class LiveGame(BaseResponse):
    """Response for GET /live/{game_id}."""

    game_id: str
    status: str | None = Field(default=None)
    score: LiveScore | None = Field(default=None)
    clock: str | None = Field(default=None)
    possession: str | None = Field(default=None)
    down_distance: str | None = Field(default=None)
    yard_line: str | None = Field(default=None)
    live_win_prob: float | None = Field(default=None)
    drives: list[DrivePoint] | None = Field(default=None)
    odds: list[LiveOdds] | None = Field(default=None)

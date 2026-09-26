# src/gridiron_edge/api/schemas/comparables.py
"""Schemas for per-game comparables endpoints (/games/{game_id}/comparables).

Serializes persisted `ComparableGamesBatch` evidence (ROADMAP.md Tier 3
#10, U13; `DECISIONS.md` D54) when it exists for the game's selected Win
forecast event; otherwise null with a `_meta.field_status` entry. Fields
mirror the persisted `ComparableGameMatch`/`ComparableFeatureContribution`
shapes directly rather than pre-formatted display strings (no `date_label`/
`line`/`final_score`/`note`-style text) — formatting real data into
human-readable labels is a frontend concern, matching the precedent set by
`/explain`'s `ExplainFactor` (D52).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class ComparableFactor(BaseModel):
    """One feature's contribution to the squared distance between two games."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_name: str
    query_value: float
    candidate_value: float
    squared_difference: float


class ComparableGame(BaseModel):
    """A historical game similar to the current matchup, with its real outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: str
    rank: int
    distance: float
    season: str
    week: int
    game_date: str
    away_team: str
    home_team: str
    away_score: int
    home_score: int
    favorite_team: str | None = Field(
        default=None, description="Long team name of the historical favorite; null if no line."
    )
    spread_magnitude: float | None = Field(
        default=None, description="Absolute historical closing spread; null if no line."
    )
    favorite_won: bool | None = Field(default=None)
    favorite_covered: bool | None = Field(default=None)
    top_contributing_features: list[ComparableFactor] = Field(default_factory=list)


class GameComparables(BaseResponse):
    """Response for GET /games/{game_id}/comparables."""

    game_id: str
    comparables: list[ComparableGame] | None = Field(default=None)
    sample_size: int | None = Field(default=None)
    favorite_win_rate: float | None = Field(default=None)
    favorite_cover_rate: float | None = Field(default=None)

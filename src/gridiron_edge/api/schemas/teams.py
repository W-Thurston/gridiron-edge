# src/gridiron_edge/api/schemas/teams.py

"""Schemas for /teams and /teams/{abbr}."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse


class TeamRecord(BaseModel):
    """Win/loss/tie record for a team within a season."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    wins: int = 0
    losses: int = 0
    ties: int = 0


class TeamRankingRow(BaseModel):
    """A single row in the /teams power rankings list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    abbr: str
    name: str
    city: str | None = None
    conference: str | None = None
    division: str | None = None
    primary_color: str | None = None
    secondary_color: str | None = None
    rating: float | None = None
    rank: int | None = None
    record: TeamRecord | None = None
    trend: float | None = None
    off_rating: float | None = None
    def_rating: float | None = None
    rating_pct: float | None = Field(
        default=None,
        description="Percentile rank of Elo rating within the league (0-1).",
    )
    avg_wins_pct: float | None = Field(
        default=None,
        description="Percentile rank of projected average wins (0-1).",
    )
    make_playoffs_pct: float | None = Field(
        default=None,
        description="Percentile rank of playoff probability (0-1).",
    )
    win_sb_pct: float | None = Field(
        default=None,
        description="Percentile rank of Super Bowl win probability (0-1).",
    )


class TeamRankingsList(BaseListResponse[TeamRankingRow]):
    """Response for GET /teams."""

    season: str | None = Field(default=None)
    as_of_week: int | None = Field(default=None)


class RatingHistoryPoint(BaseModel):
    """A single (week, rating) point in the team's Elo trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    week: int
    rating: float


class RecentResult(BaseModel):
    """A single completed game in the team's recent history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    week: int
    date: str | None = None
    opponent: str | None = None
    is_home: bool | None = None
    result: str | None = Field(default=None, description="'W', 'L', or 'T'.")
    score_for: int | None = None
    score_against: int | None = None


class TeamProfile(BaseResponse):
    """Response for GET /teams/{abbr}."""

    abbr: str
    name: str
    city: str | None = None
    conference: str | None = None
    division: str | None = None
    primary_color: str | None = None
    secondary_color: str | None = None
    season: str | None = None
    as_of_week: int | None = None
    rating: float | None = None
    rank: int | None = None
    record: TeamRecord | None = None
    trend: float | None = None
    off_rating: float | None = None
    def_rating: float | None = None
    rating_pct: float | None = Field(
        default=None,
        description="Percentile rank of Elo rating within the league (0-1).",
    )
    avg_wins_pct: float | None = Field(
        default=None,
        description="Percentile rank of projected average wins (0-1).",
    )
    make_playoffs_pct: float | None = Field(
        default=None,
        description="Percentile rank of playoff probability (0-1).",
    )
    win_sb_pct: float | None = Field(
        default=None,
        description="Percentile rank of Super Bowl win probability (0-1).",
    )
    rating_history: list[RatingHistoryPoint] | None = None
    recent_results: list[RecentResult] | None = None
    schedule_difficulty: float | None = None
    playoff_probability: float | None = None
    top_players: list[dict] | None = Field(
        default=None,
        description="Top players by WAR — blocked pending WAR computation.",
    )
    cohort_splits: dict | None = Field(
        default=None,
        description=(
            "Per-team cohort splits: {cohort_name: {metric: value, "
            "'rank_metric': int, 'sample_size': int}}. Cohorts include "
            "season, l4, home, away. Populated from team_cohort_splits.parquet."
        ),
    )

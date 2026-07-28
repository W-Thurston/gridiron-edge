# src/gridiron_edge/api/schemas/projections.py

"""Schemas for /projections."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse


class TeamProjectionRow(BaseModel):
    """A single team's postseason projection row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    abbr: str
    name: str
    avg_wins: float | None = None
    make_playoffs: float | None = Field(
        default=None,
        description="P(makes playoffs)",
    )
    reach_div: float | None = Field(
        default=None,
        description="P(reaches divisional round)",
    )
    reach_conf: float | None = Field(
        default=None,
        description="P(reaches conference championship)",
    )
    reach_sb: float | None = Field(
        default=None,
        description="P(reaches Super Bowl)",
    )
    win_sb: float | None = Field(
        default=None,
        description="P(wins Super Bowl)",
    )
    elo_delta: float | None = Field(
        default=None,
        description="Change in Elo rating from the prior week in the same season.",
    )
    clinched: bool | None = None
    eliminated: bool | None = None


class ProjectionsList(BaseListResponse[TeamProjectionRow]):
    """Response for GET /projections."""

    season: str | None = Field(default=None)
    computed_at: str | None = Field(
        default=None,
        description="ISO timestamp of when the projections CSV was last written.",
    )
    n_simulations: int | None = Field(
        default=None,
        description="Number of Monte Carlo simulations run.",
    )


class ProjectionGridWeek(BaseModel):
    """One team's schedule and outcome context for one regular-season week."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    week: int = Field(ge=1, le=18)
    state: Literal[
        "played",
        "projected",
        "bye",
        "unavailable",
    ]
    opponent: str | None = None
    is_home: bool | None = None
    game_id: str | None = None
    game_date: str | None = None
    game_time: str | None = None
    win_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Team win probability for the week. For played games, the "
            "current simulation artifact contains the fixed outcome rather "
            "than an archived pregame forecast."
        ),
    )
    actual_result: Literal["W", "L", "T"] | None = None


class ProjectionGridTeam(BaseModel):
    """One team's complete regular-season weekly projection row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    abbr: str
    name: str
    weeks: list[ProjectionGridWeek]


class ProjectionGridResponse(
    BaseListResponse[ProjectionGridTeam],
):
    """Response for GET /projections/grid."""

    season: str | None = None
    completed_through_week: int = Field(
        default=0,
        ge=0,
        le=18,
        description=(
            "Last completed regular-season week. Zero means no "
            "regular-season games have been completed."
        ),
    )
    regular_season_weeks: int = Field(default=18, ge=1)

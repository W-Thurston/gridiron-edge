# src/gridiron_edge/api/schemas/projections.py

"""Schemas for /projections."""

from __future__ import annotations

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

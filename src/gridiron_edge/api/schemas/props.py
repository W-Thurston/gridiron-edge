# src/gridiron_edge/api/schemas/props.py

"""Schemas for /props and /props/{prop_id}."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse


class ProjectionBlock(BaseModel):
    """Champion-model projection for a prop.

    All fields populate from the archive's champion row. Defensive
    nullability: 100% of the 1,433 archived rows populate these today,
    but the archive is not yet fully backfilled — future algorithms
    may have different NaN patterns.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    predicted_mean: float | None = None
    predicted_std: float | None = Field(
        default=None,
        description="Composite std: sqrt(model_rmse² + player_L3_std²).",
    )
    lo_90: float | None = Field(
        default=None,
        description="Lower bound of 90% uncertainty band.",
    )
    hi_90: float | None = Field(
        default=None,
        description="Upper bound of 90% uncertainty band.",
    )


class LineBlock(BaseModel):
    """Odds-derived context for a prop.

    All fields are null in T2 — the odds-join at prediction time is not
    yet implemented. Serializer marks with field_status: pending.
    Populates when the odds-join lands as a future backend addition.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    line: float | None = None
    p_over: float | None = Field(
        default=None,
        description="Model-implied P(over the line).",
    )
    lean: str | None = Field(
        default=None,
        description="'Over', 'Under', or 'No Edge'.",
    )
    confidence_tier: str | None = Field(
        default=None,
        description="'Low', 'Moderate', or 'High' — derived from p_over.",
    )


class PropSummary(BaseModel):
    """A single row in the /props list response.

    Sufficient for the prop shop screen's per-prop card. Detail-only
    context (historical vs opponent, situational splits, reasoning)
    lives on PropDetail.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prop_id: str = Field(
        description="Composite: {game_id}__{player_id}__{stat_type}.",
    )
    game_id: str
    season: str | None = None
    week: int | None = None

    player_id: str
    player_name: str
    position: str
    team: str

    stat_type: str
    model_key: str = Field(
        description="Composite {model_name}_{model_type} identifier.",
    )

    projection: ProjectionBlock | None = None
    line_context: LineBlock | None = None


class PropList(BaseListResponse[PropSummary]):
    """Response for GET /props."""

    season: str | None = Field(default=None)
    week: int | None = Field(default=None)
    stat_type: str | None = Field(default=None)
    position: str | None = Field(default=None)


class PropDetail(BaseResponse):
    """Response for GET /props/{prop_id}."""

    prop_id: str
    game_id: str
    season: str | None = None
    week: int | None = None

    player_id: str
    player_name: str
    position: str
    team: str

    stat_type: str
    model_key: str

    projection: ProjectionBlock | None = None
    line_context: LineBlock | None = None

    # Scaffolded fields. Populated when the corresponding backend
    # workstreams land.
    historical_vs_opponent: list[dict] | None = Field(
        default=None,
        description=(
            "Past matchups vs this opponent's defense. Pending: "
            "opponent-adjusted matchup aggregation (ROADMAP §9 Tier 3)."
        ),
    )
    situational_splits: dict | None = Field(
        default=None,
        description=(
            "Home/away, dome/outdoor, favored/underdog splits. Pending: "
            "prop cohort splits (ROADMAP §9 Tier 3)."
        ),
    )
    prop_reasoning: dict | None = Field(
        default=None,
        description=(
            "Feature-attribution explanation for this projection. "
            "Blocked on feature-attribution workstream."
        ),
    )
    injury_status: dict | None = Field(
        default=None,
        description=("Player injury status. Blocked on injury data source (ROADMAP §5.3)."),
    )
    recent_form: list[dict] | None = Field(
        default=None,
        description="Last-N-games aggregation. Pending backend work.",
    )
    multi_book_shopping: dict | None = Field(
        default=None,
        description=("Per-book line and odds comparison. Blocked on multi-book odds ingest (W7)."),
    )

# src/gridiron_edge/api/schemas/games.py
"""Schedule-complete schemas for /games and /games/{game_id}."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse


class WeatherBlock(BaseModel):
    """Weather snapshot for a game."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temp_f: float | None = None
    wind_mph: float | None = None
    conditions: str | None = None
    precip_pct: float | None = None


class WinPredictionBlock(BaseModel):
    """Persisted Win component and its independent forecast provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    selection_status: str | None = None
    away_win_prob: float | None = None
    home_win_prob: float | None = None
    model_name: str | None = None
    model_type: str | None = None
    event_id: str | None = None
    run_id: str | None = None
    generated_at: str | None = None
    role: str | None = None


class SpreadPredictionBlock(BaseModel):
    """Persisted spread component derived from its selected Win source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    model_spread: float | None = Field(
        default=None,
        description=(
            "NFL home-line convention: negative means the Home team is favored; "
            "positive means the Away team is favored. The value equals "
            "projected Away score minus projected Home score."
        ),
    )
    uncertainty: float | None = None
    source_event_id: str | None = None
    model_name: str | None = None
    model_type: str | None = None
    calibration_key: str | None = None
    calibration_updated_at: str | None = None


class TotalPredictionBlock(BaseModel):
    """Persisted Total component and its independent forecast provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    selection_status: str | None = None
    model_total: float | None = None
    uncertainty: float | None = None
    model_name: str | None = None
    model_type: str | None = None
    event_id: str | None = None
    run_id: str | None = None
    generated_at: str | None = None
    role: str | None = None
    uncertainty_trained_at: str | None = None


class ProjectedScoreBlock(BaseModel):
    """Persisted projected-score availability and score values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    home: float | None = None
    away: float | None = None


class StartingQbSide(BaseModel):
    """One side's assumed starting quarterback, as persisted with the forecast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_id: str
    name: str | None = None
    source: str = Field(
        description="depth_chart_report, nflverse_schedule or carried_forward.",
    )
    projection_as_of: str | None = Field(
        default=None,
        description="Depth chart snapshot time (UTC); only for a depth_chart_report source.",
    )
    nflverse_player_id: str | None = Field(
        default=None,
        description="The nflverse schedule's own value when the forecast was composed.",
    )
    carried_forward_player_id: str | None = None
    differs_from_carry_forward: bool | None = None


class StartingQbBlock(BaseModel):
    """The quarterbacks the forecast assumed, or why they were not recorded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(
        description=(
            "available, starter_unresolved or sidecar_unavailable as recorded with the "
            "product; not_recorded for a product composed before starters were recorded."
        ),
    )
    away: StartingQbSide | None = None
    home: StartingQbSide | None = None


class GameSummary(BaseModel):
    """One scheduled row in the /games list response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: str
    game_date: str | None = None
    week: int | None = None
    season: str | None = None
    away_team: str
    home_team: str
    win: WinPredictionBlock
    spread: SpreadPredictionBlock
    total: TotalPredictionBlock
    projected_score: ProjectedScoreBlock
    starting_qb: StartingQbBlock


class GameList(BaseListResponse[GameSummary]):
    """Response for GET /games."""

    season: str | None = Field(default=None)
    week: int | None = Field(default=None)


class GameDetail(BaseResponse):
    """Response for GET /games/{game_id}."""

    game_id: str
    game_date: str | None = None
    week: int | None = None
    season: str | None = None
    day_of_week: str | None = None
    kick: str | None = Field(default=None, description="Persisted local kickoff time.")
    venue: str | None = None
    away_team: str
    home_team: str
    win: WinPredictionBlock
    spread: SpreadPredictionBlock
    total: TotalPredictionBlock
    projected_score: ProjectedScoreBlock
    starting_qb: StartingQbBlock
    weather: WeatherBlock | None = None
    team_comparison: dict | None = None
    swing_factors: list[dict] | None = None
    injuries: list[dict] | None = None
    top_prop_edges: list[dict] | None = None

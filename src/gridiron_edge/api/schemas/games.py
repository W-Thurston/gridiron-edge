# src/gridiron_edge/api/schemas/games.py

"""Schemas for /games and /games/{game_id}."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse


class WeatherBlock(BaseModel):
    """Weather snapshot for a game.

    Fields populate from the schedule/weather join. When the join produces
    no data (e.g., domed venue, missing feed), all fields are None and
    the serializer marks the block as UNAVAILABLE_NO_WEATHER.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temp_f: float | None = None
    wind_mph: float | None = None
    conditions: str | None = None
    precip_pct: float | None = None


class PredictionBlock(BaseModel):
    """Champion-model prediction for a game.

    All fields populate from the win_prob champion's archived row. When
    the champion manifest is missing, the entire block is None and the
    serializer marks CHAMPION_NOT_WRITTEN in field_status.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    home_win_prob: float | None = None
    away_win_prob: float | None = None
    home_win_lo: float | None = Field(
        default=None,
        description="Lower bound of 90% uncertainty band on home_win_prob.",
    )
    home_win_hi: float | None = Field(
        default=None,
        description="Upper bound of 90% uncertainty band on home_win_prob.",
    )
    confidence_tier: str | None = Field(
        default=None,
        description="'Low', 'Moderate', or 'High'.",
    )
    model_spread: float | None = Field(
        default=None,
        description="Predicted point spread (home - away).",
    )
    model_total: float | None = None
    projected_home_score: float | None = None
    projected_away_score: float | None = None


class GameSummary(BaseModel):
    """A single row in the /games list response.

    Slimmer than GameDetail — omits weather, team comparison, swing
    factors, injuries, and prop edges. Sufficient for the games-list
    screen's per-game card.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: str
    game_date: str | None = None
    week: int | None = None
    season: str | None = None
    away_team: str
    home_team: str
    prediction: PredictionBlock | None = None


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
    kick: str | None = Field(
        default=None,
        description="Local kickoff time, e.g. '1:00 PM ET'.",
    )
    venue: str | None = None
    away_team: str
    home_team: str
    weather: WeatherBlock | None = None
    prediction: PredictionBlock | None = None

    # Optional fields awaiting their required derived datasets.
    team_comparison: dict | None = Field(
        default=None,
        description=(
            "Opponent-adjusted percentile stats table. Pending: no "
            "backend for percentile computation."
        ),
    )
    swing_factors: list[dict] | None = Field(
        default=None,
        description=("Per-game swing factors. Blocked on feature-attribution workstream."),
    )
    injuries: list[dict] | None = Field(
        default=None,
        description=(
            "Injury reports for both teams. Blocked on injury data source (ROADMAP §5.3)."
        ),
    )
    top_prop_edges: list[dict] | None = Field(
        default=None,
        description=("Top prop edges for this game. Pending the required edges data source."),
    )

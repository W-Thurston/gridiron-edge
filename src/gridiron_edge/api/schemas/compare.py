# src/gridiron_edge/api/schemas/compare.py

"""Schemas for /compare endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class StatRow(BaseModel):
    """A single row in a comparison table.

    ``team_a_value`` and ``team_b_value`` are loose ``float | int | str``
    unions because different stat rows carry different value types (Elo
    rating is float, record is a formatted string, rank is int). Frontend
    formats via ``unit``.

    Rows for blocked/pending stats emit ``team_a_value: None`` and
    ``team_b_value: None`` with a matching entry in the response's
    ``_meta.field_status`` keyed on the row's ``key``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(
        description="Stable identifier for field_status lookup and row ordering.",
    )
    label: str = Field(
        description="Display label for the row, e.g. 'Rating' or 'Record'.",
    )
    unit: str | None = Field(
        default=None,
        description="Formatting hint: 'elo', 'rank', 'record', 'pct', 'raw'.",
    )
    team_a_value: float | int | str | None = None
    team_b_value: float | int | str | None = None
    team_a_pct: float | None = Field(
        default=None,
        description="Team A's league-wide percentile (0-1) on this stat.",
    )
    team_b_pct: float | None = Field(
        default=None,
        description="Team B's league-wide percentile (0-1) on this stat.",
    )


class CompareTeamsResponse(BaseResponse):
    """Response for GET /compare/teams."""

    season: str | None = None
    team_a: str
    team_b: str
    stats: list[StatRow] = Field(default_factory=list)
    cohort_splits: dict[str, dict] | None = Field(
        default=None,
        description=(
            "Per-team cohort splits: {team_abbr: {cohort: {metric: value, "
            "'rank_metric': int, 'sample_size': int}}}. Populated from "
            "team_cohort_splits.parquet."
        ),
    )


class PlayerVsDefenseRow(BaseModel):
    """A single row in the player-vs-defense comparison table.

    ``projection_value`` is populated from the champion model's archive
    row. ``defense_value`` is null in T2 — blocked pending
    opponent-allowed-by-position aggregation (ROADMAP §9 Tier 3).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(
        description="Stable identifier for field_status lookup and row ordering.",
    )
    label: str = Field(
        description="Display label for the row, e.g. 'Projected Mean' or 'Avg Allowed'.",
    )
    unit: str | None = Field(
        default=None,
        description="Formatting hint: 'yards', 'attempts', 'pct', 'rank', 'raw'.",
    )
    projection_value: float | int | str | None = None
    defense_value: float | int | str | None = None


class ComparePlayerResponse(BaseResponse):
    """Response for GET /compare/player/{prop_id}."""

    prop_id: str
    game_id: str
    player_id: str
    player_name: str
    position: str
    team: str
    stat_type: str
    model_key: str
    season: str | None = None
    week: int | None = None
    stats: list[PlayerVsDefenseRow] = Field(default_factory=list)

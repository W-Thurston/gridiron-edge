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


class CompareTeamsResponse(BaseResponse):
    """Response for GET /compare/teams."""

    season: str | None = None
    team_a: str
    team_b: str
    stats: list[StatRow] = Field(default_factory=list)

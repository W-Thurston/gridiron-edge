# src/gridiron_edge/api/schemas/defense.py

"""Schema for /defense/{team}/allowed."""

from __future__ import annotations

from pydantic import Field

from gridiron_edge.api.schemas._base import BaseResponse


class DefenseAllowedResponse(BaseResponse):
    """Per-defense allowed aggregates for one (team, position, stat_type).

    Returns all cohorts (season/l4/home/away) so the frontend split
    switcher can toggle without refetching. Each cohort maps to
    {avg_allowed, sample_size, rank_against_position}.
    """

    team: str
    position: str = Field(
        default="",
        description="Position the stat_type applies to; empty if unknown.",
    )
    stat_type: str
    cohorts: dict | None = Field(
        default=None,
        description=(
            "Nested {cohort: {avg_allowed, sample_size, "
            "rank_against_position}}. None if no data for this "
            "(team, position, stat_type)."
        ),
    )

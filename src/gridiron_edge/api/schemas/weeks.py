# src/gridiron_edge/api/schemas/weeks.py

"""Schemas for /weeks/current."""

from __future__ import annotations

from pydantic import Field

from gridiron_edge.api.schemas._base import BaseResponse


class CurrentWeek(BaseResponse):
    """The current NFL season + week, derived from the schedule."""

    season: int = Field(description="NFL season year, e.g. 2025 for the 2025-26 season.")
    week: int = Field(description="Week number within the season.")
    source: str | None = Field(
        default=None,
        description="How the week was resolved: 'schedule' or 'fallback'.",
    )

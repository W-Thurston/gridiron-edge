# src/gridiron_edge/api/serializers/weeks.py

"""Serializer for /weeks/current."""

from __future__ import annotations

from gridiron_edge.api.schemas.weeks import CurrentWeek


def serialize_current_week(season: int, week: int, source: str) -> CurrentWeek:
    """Wrap the resolved current-week tuple in the response model."""
    return CurrentWeek(season=season, week=week, source=source)

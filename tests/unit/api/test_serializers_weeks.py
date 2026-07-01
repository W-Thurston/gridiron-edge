# tests/unit/api/test_serializers_weeks.py

"""Unit tests for weeks serializer."""

from __future__ import annotations

from gridiron_edge.api.serializers.weeks import serialize_current_week


class TestSerializeCurrentWeek:
    def test_maps_fields(self) -> None:
        result = serialize_current_week(season=2025, week=10, source="schedule")
        assert result.season == 2025
        assert result.week == 10
        assert result.source == "schedule"

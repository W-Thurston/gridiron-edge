# tests/unit/api/test_schemas_weeks.py

"""Unit tests for weeks schema."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.weeks import CurrentWeek


class TestCurrentWeek:
    def test_construction(self) -> None:
        cw = CurrentWeek(season=2025, week=10)
        assert cw.season == 2025
        assert cw.week == 10
        assert cw.source is None

    def test_with_source(self) -> None:
        cw = CurrentWeek(season=2025, week=10, source="schedule")
        assert cw.source == "schedule"

    def test_is_frozen(self) -> None:
        cw = CurrentWeek(season=2025, week=10)
        with pytest.raises(ValidationError):
            cw.season = 2026

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            CurrentWeek(season=2025, week=10, unexpected="x")

    def test_wire_shape(self) -> None:
        cw = CurrentWeek(season=2025, week=10, source="schedule")
        dumped = cw.model_dump(by_alias=True, exclude_none=True)
        assert dumped == {"season": 2025, "week": 10, "source": "schedule"}

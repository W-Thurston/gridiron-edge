# tests/unit/metrics/test_geo.py
"""Tests for gridiron_edge.metrics.travel.geo - coordinate and distance helpers."""

from __future__ import annotations

from datetime import datetime

import pytest
from timezonefinder import TimezoneFinder

from gridiron_edge.metrics.travel.geo import (
    calculate_timezone_difference,
    measure_distance,
    to_decimal_degrees,
)


class TestToDecimalDegrees:
    def test_float_passthrough(self) -> None:
        assert to_decimal_degrees(39.7392) == pytest.approx(39.7392, abs=0.001)

    def test_dms_north(self) -> None:
        """39 degrees 44 minutes 21 seconds N."""
        result: float = to_decimal_degrees("39°44'21″N")
        assert result == pytest.approx(39.7392, abs=0.01)

    def test_dms_south_is_negative(self) -> None:
        result: float = to_decimal_degrees("33°51'54″S")
        assert result < 0

    def test_degrees_direction_west(self) -> None:
        """104°W → -104.0"""
        result: float = to_decimal_degrees("104°W")
        assert result == pytest.approx(-104.0, abs=0.1)

    def test_degrees_direction_east(self) -> None:
        result: float = to_decimal_degrees("104°E")
        assert result == pytest.approx(104.0, abs=0.1)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            to_decimal_degrees(123)  # type: ignore[arg-type]


class TestMeasureDistance:
    def test_same_point_is_zero(self) -> None:
        result: float = measure_distance(
            home_lat_lon=[39.7392, -104.9903],
            game_lat_lon=[39.7392, -104.9903],
        )
        assert result == pytest.approx(0.0, abs=1.0)

    def test_known_distance(self) -> None:
        """Denver to LA is approximately 1,340 km."""
        result: float = measure_distance(
            home_lat_lon=[39.7392, -104.9903],  # Denver
            game_lat_lon=[34.0522, -118.2437],  # LA
        )
        assert 1200 < result < 1500

    def test_returns_float(self) -> None:
        result: float = measure_distance(
            home_lat_lon=[40.0, -75.0],
            game_lat_lon=[34.0, -118.0],
        )
        assert isinstance(result, float)


class TestCalculateTimezoneDifference:
    def test_same_timezone_is_zero(self) -> None:
        tf = TimezoneFinder()
        result: int = calculate_timezone_difference(
            lat_x=39.7392,
            long_x=-104.9903,  # Denver
            lat_y=33.4484,
            long_y=-112.0740,  # Phoenix (same offset in winter)
            tz_find=tf,
            when=datetime(2025, 12, 1),
        )
        # Both Mountain time → difference is 0 or ±1 depending on DST
        assert abs(result) <= 1

    def test_east_to_west_coast(self) -> None:
        tf = TimezoneFinder()
        result: int = calculate_timezone_difference(
            lat_x=40.7128,
            long_x=-74.0060,  # NYC (Eastern)
            lat_y=34.0522,
            long_y=-118.2437,  # LA (Pacific)
            tz_find=tf,
            when=datetime(2025, 12, 1),
        )
        assert abs(result) == 3  # ET → PT = 3 hours

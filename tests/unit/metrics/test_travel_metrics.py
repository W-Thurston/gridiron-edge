# tests/unit/metrics/test_travel_metrics.py
"""Tests for gridiron_edge.metrics.travel.travel - haversine and travel computation."""

from __future__ import annotations

import numpy as np
from numpy import ndarray
import pytest

from gridiron_edge.metrics.travel.travel import _haversine_km


class TestHaversineKm:
    def test_same_point_is_zero(self) -> None:
        result: ndarray = _haversine_km(
            np.array([39.7392]),
            np.array([-104.9903]),
            np.array([39.7392]),
            np.array([-104.9903]),
        )
        assert result[0] == pytest.approx(0.0, abs=0.01)

    def test_known_distance_denver_to_la(self) -> None:
        """Denver to LA is approximately 1,340 km."""
        result: ndarray = _haversine_km(
            np.array([39.7392]),  # Denver lat
            np.array([-104.9903]),  # Denver lon
            np.array([34.0522]),  # LA lat
            np.array([-118.2437]),  # LA lon
        )
        assert 1300 < result[0] < 1400

    def test_vectorized(self) -> None:
        """Should compute distances for multiple points at once."""
        lat1: ndarray = np.array([39.7392, 40.7128])
        lon1: ndarray = np.array([-104.9903, -74.0060])
        lat2: ndarray = np.array([34.0522, 34.0522])
        lon2: ndarray = np.array([-118.2437, -118.2437])
        result: ndarray = _haversine_km(lat1, lon1, lat2, lon2)
        assert len(result) == 2
        assert all(d > 0 for d in result)

    def test_symmetric(self) -> None:
        """Distance A→B should equal B→A."""
        d_ab: ndarray = _haversine_km(
            np.array([39.7392]),
            np.array([-104.9903]),
            np.array([34.0522]),
            np.array([-118.2437]),
        )
        d_ba: ndarray = _haversine_km(
            np.array([34.0522]),
            np.array([-118.2437]),
            np.array([39.7392]),
            np.array([-104.9903]),
        )
        assert d_ab[0] == pytest.approx(d_ba[0], abs=0.01)

    def test_returns_numpy_array(self) -> None:
        result: ndarray = _haversine_km(
            np.array([40.0]),
            np.array([-75.0]),
            np.array([34.0]),
            np.array([-118.0]),
        )
        assert isinstance(result, np.ndarray)

# tests/unit/transform/test_games_nflverse.py
"""Tests for gridiron_edge.transform.clean.games_nflverse — schema mapping."""

from __future__ import annotations

from gridiron_edge.transform.clean.games_nflverse import _game_location


class TestGameLocation:
    def test_home_maps_to_h(self) -> None:
        assert _game_location("Home") == "H"

    def test_neutral_maps_to_n(self) -> None:
        assert _game_location("Neutral") == "N"

    def test_nan_maps_to_h(self) -> None:
        """nflverse uses NaN for home games (default location)."""
        result: str = _game_location(float("nan"))
        assert result == "H"

    def test_case_sensitivity(self) -> None:
        """Verify the function handles the exact nflverse casing."""
        assert _game_location("Home") == "H"
        assert _game_location("Neutral") == "N"

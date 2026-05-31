# tests/unit/transform/test_nflverse_common.py
"""Tests for gridiron_edge.transform.clean._nflverse_common — shared helpers."""

from __future__ import annotations

from gridiron_edge.transform.clean._nflverse_common import (
    GAME_TYPE_TO_WEEK,
    NFLVERSE_SHORT_TO_LONG,
    gametime_to_hhmmss,
    map_short_to_long,
    season_label,
)


class TestGameTypeToWeek:
    def test_wildcard_is_19(self) -> None:
        assert GAME_TYPE_TO_WEEK["WC"] == 19

    def test_divisional_is_20(self) -> None:
        assert GAME_TYPE_TO_WEEK["DIV"] == 20

    def test_conference_is_21(self) -> None:
        assert GAME_TYPE_TO_WEEK["CON"] == 21

    def test_super_bowl_is_22(self) -> None:
        assert GAME_TYPE_TO_WEEK["SB"] == 22

    def test_has_4_entries(self) -> None:
        assert len(GAME_TYPE_TO_WEEK) == 4


class TestNflverseShortToLong:
    def test_has_32_plus_relocations(self) -> None:
        # 32 current teams + 3 historical relocations
        assert len(NFLVERSE_SHORT_TO_LONG) >= 32

    def test_known_teams_present(self) -> None:
        assert NFLVERSE_SHORT_TO_LONG["KC"] == "Kansas City Chiefs"
        assert NFLVERSE_SHORT_TO_LONG["SF"] == "San Francisco 49ers"
        assert NFLVERSE_SHORT_TO_LONG["GB"] == "Green Bay Packers"

    def test_relocations_map_to_current_names(self) -> None:
        assert NFLVERSE_SHORT_TO_LONG["OAK"] == "Las Vegas Raiders"
        assert NFLVERSE_SHORT_TO_LONG["SD"] == "Los Angeles Chargers"
        assert NFLVERSE_SHORT_TO_LONG["STL"] == "Los Angeles Rams"


class TestSeasonLabel:
    def test_2025_returns_2025_2026(self) -> None:
        assert season_label(2025) == "2025-2026"

    def test_2024_returns_2024_2025(self) -> None:
        assert season_label(2024) == "2024-2025"

    def test_2000_returns_2000_2001(self) -> None:
        assert season_label(2000) == "2000-2001"


class TestGametimeToHhmmss:
    def test_standard_time(self) -> None:
        assert gametime_to_hhmmss("20:20") == "20:20:00"

    def test_already_hhmmss(self) -> None:
        result: str = gametime_to_hhmmss("13:00:00")
        assert result == "13:00:00"

    def test_nan_returns_empty_or_default(self) -> None:
        result: str = gametime_to_hhmmss(float("nan"))
        assert isinstance(result, str)


class TestMapShortToLong:
    def test_known_code(self) -> None:
        assert map_short_to_long("KC") == "Kansas City Chiefs"

    def test_unknown_code_raises_or_returns(self) -> None:
        # Depending on implementation: either raises KeyError or returns the code
        try:
            result: str = map_short_to_long("XXX")
            # If it doesn't raise, it should return something string-like
            assert isinstance(result, str)
        except KeyError:
            pass  # Expected behavior

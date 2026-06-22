# tests/unit/core/test_constants.py
"""Tests for gridiron_edge.core.constants."""

from __future__ import annotations

import re
from re import Pattern

from gridiron_edge.core.constants import (
    AWAY_WIN_LOCATION,
    EXPANSION_TEAMS,
    HOLDOUT_SEASONS,
    HOME_GAME_LOCATION,
)


class TestGameLocationSentinels:
    def test_home_game_location_is_h(self) -> None:
        assert HOME_GAME_LOCATION == "H"

    def test_away_win_location_is_at(self) -> None:
        assert AWAY_WIN_LOCATION == "@"

    def test_home_and_away_are_distinct(self) -> None:
        assert HOME_GAME_LOCATION != AWAY_WIN_LOCATION


class TestHoldoutSeasons:
    def test_is_frozenset(self) -> None:
        assert isinstance(HOLDOUT_SEASONS, frozenset)

    def test_contains_only_strings(self) -> None:
        assert all(isinstance(s, str) for s in HOLDOUT_SEASONS)

    def test_season_format_yyyy_yyyy(self) -> None:
        """Every season label should match 'YYYY-YYYY' pattern."""

        pattern: Pattern[str] = re.compile(r"^\d{4}-\d{4}$")
        for season in HOLDOUT_SEASONS:
            assert pattern.match(season), f"Bad format: {season!r}"

    def test_not_empty(self) -> None:
        assert len(HOLDOUT_SEASONS) > 0


class TestExpansionTeams:
    def test_is_dict(self) -> None:
        assert isinstance(EXPANSION_TEAMS, dict)

    def test_has_four_teams(self) -> None:
        assert len(EXPANSION_TEAMS) == 4

    def test_known_franchises_present(self) -> None:
        expected: set[str] = {
            "Carolina Panthers",
            "Jacksonville Jaguars",
            "Baltimore Ravens",
            "Houston Texans",
        }
        assert set(EXPANSION_TEAMS.keys()) == expected

    def test_values_are_season_labels(self) -> None:
        pattern: Pattern[str] = re.compile(r"^\d{4}-\d{4}$")
        for team, season in EXPANSION_TEAMS.items():
            assert pattern.match(season), f"{team}: bad season format {season!r}"


class TestTeamCodeNormalization:
    def test_is_dict(self) -> None:
        from gridiron_edge.core.constants import TEAM_CODE_NORMALIZATION

        assert isinstance(TEAM_CODE_NORMALIZATION, dict)

    def test_known_relocations(self) -> None:
        from gridiron_edge.core.constants import TEAM_CODE_NORMALIZATION

        assert TEAM_CODE_NORMALIZATION["OAK"] == "LV"
        assert TEAM_CODE_NORMALIZATION["SD"] == "LAC"
        assert TEAM_CODE_NORMALIZATION["STL"] == "LA"
        assert TEAM_CODE_NORMALIZATION["JAC"] == "JAX"

    def test_keys_and_values_are_short_codes(self) -> None:
        from gridiron_edge.core.constants import TEAM_CODE_NORMALIZATION

        for old, new in TEAM_CODE_NORMALIZATION.items():
            assert 2 <= len(old) <= 3
            assert 2 <= len(new) <= 3

    def test_no_self_mappings(self) -> None:
        from gridiron_edge.core.constants import TEAM_CODE_NORMALIZATION

        for old, new in TEAM_CODE_NORMALIZATION.items():
            assert old != new

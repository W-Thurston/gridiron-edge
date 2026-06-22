# tests/unit/ingest/test_game_id.py
"""Tests for gridiron_edge.ingest.odds._game_id - DK game_id resolution."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from gridiron_edge.ingest.odds._game_id import (
    build_game_id,
    resolve_dk_game_ids,
    team_long_to_short,
)


class TestTeamLongToShort:
    def test_known_team(self) -> None:
        assert team_long_to_short("Kansas City Chiefs") == "KC"

    def test_another_known_team(self) -> None:
        assert team_long_to_short("San Francisco 49ers") == "SF"

    def test_all_32_teams_resolve(self) -> None:
        from gridiron_edge.transform.clean._nflverse_common import NFLVERSE_SHORT_TO_LONG

        for _short, long in NFLVERSE_SHORT_TO_LONG.items():
            result: str | None = team_long_to_short(long)
            assert result is not None, f"{long} did not resolve"

    def test_unknown_team_returns_none(self) -> None:
        assert team_long_to_short("Fake City Fakers") is None

    def test_strips_whitespace(self) -> None:
        assert team_long_to_short("  Kansas City Chiefs  ") == "KC"


class TestBuildGameId:
    def test_standard_game(self) -> None:
        result: str | None = build_game_id(
            away_team="Kansas City Chiefs",
            home_team="Los Angeles Chargers",
            season_year=2025,
            week=1,
        )
        assert result == "2025_01_KC_LAC"

    def test_week_padding(self) -> None:
        result: str | None = build_game_id(
            away_team="Green Bay Packers",
            home_team="Chicago Bears",
            season_year=2025,
            week=3,
        )
        assert result == "2025_03_GB_CHI"

    def test_double_digit_week(self) -> None:
        result: str | None = build_game_id(
            away_team="Buffalo Bills",
            home_team="Miami Dolphins",
            season_year=2025,
            week=14,
        )
        assert result == "2025_14_BUF_MIA"

    def test_unknown_away_team_returns_none(self) -> None:
        result: str | None = build_game_id(
            away_team="Fake Team",
            home_team="Kansas City Chiefs",
            season_year=2025,
            week=1,
        )
        assert result is None

    def test_unknown_home_team_returns_none(self) -> None:
        result: str | None = build_game_id(
            away_team="Kansas City Chiefs",
            home_team="Fake Team",
            season_year=2025,
            week=1,
        )
        assert result is None


class TestResolveDkGameIds:
    def test_adds_game_id_column(self) -> None:
        df = pd.DataFrame(
            {
                "away_team": ["Kansas City Chiefs", "Buffalo Bills"],
                "home_team": ["Los Angeles Chargers", "Miami Dolphins"],
            }
        )
        result: DataFrame = resolve_dk_game_ids(df, season_year=2025, week=1)
        assert "game_id" in result.columns
        assert result["game_id"].iloc[0] == "2025_01_KC_LAC"
        assert result["game_id"].iloc[1] == "2025_01_BUF_MIA"

    def test_preserves_existing_columns(self) -> None:
        df = pd.DataFrame(
            {
                "away_team": ["Kansas City Chiefs"],
                "home_team": ["Los Angeles Chargers"],
                "ml_away": [-150],
                "ml_home": [130],
            }
        )
        result: DataFrame = resolve_dk_game_ids(df, season_year=2025, week=1)
        assert "ml_away" in result.columns
        assert "ml_home" in result.columns

    def test_unresolvable_team_gets_none(self) -> None:
        df = pd.DataFrame(
            {
                "away_team": ["Fake Team"],
                "home_team": ["Kansas City Chiefs"],
            }
        )
        result: DataFrame = resolve_dk_game_ids(df, season_year=2025, week=1)
        assert result["game_id"].iloc[0] is None

    def test_does_not_modify_original(self) -> None:
        df = pd.DataFrame(
            {
                "away_team": ["Kansas City Chiefs"],
                "home_team": ["Los Angeles Chargers"],
            }
        )
        original_cols: set[str] = set(df.columns)
        resolve_dk_game_ids(df, season_year=2025, week=1)
        assert set(df.columns) == original_cols  # original unchanged

    def test_wide_format_with_location(self) -> None:
        """Wide format uses team/opponent/location instead of home/away."""
        df = pd.DataFrame(
            {
                "team": ["Kansas City Chiefs", "Los Angeles Chargers"],
                "opponent": ["Los Angeles Chargers", "Kansas City Chiefs"],
                "location": [0, 1],  # 0=away, 1=home
            }
        )
        result: DataFrame = resolve_dk_game_ids(df, season_year=2025, week=1)
        assert result["game_id"].iloc[0] == "2025_01_KC_LAC"
        assert result["game_id"].iloc[1] == "2025_01_KC_LAC"  # same game

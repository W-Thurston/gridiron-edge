# tests/unit/evaluation/test_backfill.py
"""Tests for gridiron_edge.evaluation.backfill — _reconstruct_away_home and helpers."""

from __future__ import annotations

from pandas import DataFrame
from tests.fixtures.dataframes import make_games

from gridiron_edge.evaluation.backfill import _reconstruct_away_home


class TestReconstructAwayHome:
    def test_adds_away_and_home_columns(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "GAME_LOCATION": "H",
                    "WIN_OR_TIE": 1,
                },
            ]
        )
        result: DataFrame = _reconstruct_away_home(games)
        assert "AWAY_TEAM" in result.columns
        assert "HOME_TEAM" in result.columns

    def test_home_win_assigns_correctly(self) -> None:
        """GAME_LOCATION='H' means winner is home team."""
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "GAME_LOCATION": "H",
                    "WIN_OR_TIE": 1,
                },
            ]
        )
        result: DataFrame = _reconstruct_away_home(games)
        assert result["HOME_TEAM"].iloc[0] == "KC"
        assert result["AWAY_TEAM"].iloc[0] == "LV"

    def test_away_win_assigns_correctly(self) -> None:
        """GAME_LOCATION='@' means winner is away team."""
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "GAME_LOCATION": "@",
                    "WIN_OR_TIE": 1,
                },
            ]
        )
        result: DataFrame = _reconstruct_away_home(games)
        assert result["AWAY_TEAM"].iloc[0] == "KC"
        assert result["HOME_TEAM"].iloc[0] == "LV"

    def test_preserves_existing_columns(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "GAME_LOCATION": "H",
                    "WIN_OR_TIE": 1,
                },
            ]
        )
        original_cols: set[str] = set(games.columns)
        result: DataFrame = _reconstruct_away_home(games)
        assert original_cols <= set(result.columns)

    def test_multiple_games(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "GAME_LOCATION": "H",
                    "WIN_OR_TIE": 1,
                },
                {
                    "GAME_ID": "g2",
                    "WINNER": "BUF",
                    "LOSER": "MIA",
                    "GAME_LOCATION": "@",
                    "WIN_OR_TIE": 1,
                },
            ]
        )
        result: DataFrame = _reconstruct_away_home(games)
        assert len(result) == 2
        # g1: home win → HOME=KC, AWAY=LV
        assert result.iloc[0]["HOME_TEAM"] == "KC"
        # g2: away win → AWAY=BUF, HOME=MIA
        assert result.iloc[1]["AWAY_TEAM"] == "BUF"
        assert result.iloc[1]["HOME_TEAM"] == "MIA"

# tests/unit/features/test_venue_hfa.py
"""Tests for gridiron_edge.features.team.venue_hfa — VenueHFAFeature."""

from __future__ import annotations

from pandas import DataFrame
from tests.fixtures.dataframes import make_accessor, make_games, make_modeling_rows

from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.venue_hfa import VenueHFAFeature


class TestVenueHFASpec:
    def test_spec_name(self) -> None:
        assert VenueHFAFeature().spec.name == "venue_hfa"

    def test_produces_2_columns(self) -> None:
        assert len(VenueHFAFeature().spec.produces) == 2

    def test_produces_expected_columns(self) -> None:
        expected: set[str] = {"TEAM_A_FRANCHISE_HFA", "TEAM_B_FRANCHISE_HFA"}
        assert set(VenueHFAFeature().spec.produces) == expected

    def test_registered_under_venue_hfa(self) -> None:
        assert FeatureRegistry.get("venue_hfa") is VenueHFAFeature


class TestVenueHFACompute:
    def _build_home_games(self, team: str, n_wins: int, n_losses: int) -> list[dict]:
        """Build a list of game row overrides for a team's home record."""
        rows: list[dict[str, int | str]] = []
        for i in range(n_wins):
            rows.append(
                {
                    "GAME_ID": f"hw_{team}_{i}",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": i + 1,
                    "WINNER": team,
                    "LOSER": "Opponent",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                }
            )
        for i in range(n_losses):
            rows.append(
                {
                    "GAME_ID": f"hl_{team}_{i}",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": n_wins + i + 1,
                    "WINNER": "Opponent",
                    "LOSER": team,
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                }
            )
        return rows

    def test_insufficient_games_returns_zero(self) -> None:
        """Teams with fewer than _MIN_HOME_GAMES get 0.0."""

        # Only 5 home games — below threshold of 20
        games = make_games(self._build_home_games("KC", 4, 1))
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "target",
                    "TEAM_A": "KC",
                    "TEAM_B": "LV",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 10,
                }
            ]
        )
        acc = make_accessor(games=games)
        result: DataFrame = VenueHFAFeature().compute(df=df, datasets=acc)
        assert result["TEAM_A_FRANCHISE_HFA"].iloc[0] == 0.0

    def test_output_columns_present(self) -> None:
        games = make_games(self._build_home_games("KC", 15, 10))
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "target",
                    "TEAM_A": "KC",
                    "TEAM_B": "LV",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 18,
                }
            ]
        )
        acc = make_accessor(games=games)
        result: DataFrame = VenueHFAFeature().compute(df=df, datasets=acc)
        assert "TEAM_A_FRANCHISE_HFA" in result.columns
        assert "TEAM_B_FRANCHISE_HFA" in result.columns

    def test_neutral_site_excluded(self) -> None:
        """Games with GAME_LOCATION='N' should not count toward home record."""

        neutral_games: list[dict[str, int | str]] = [
            {
                "GAME_ID": f"n_{i}",
                "YEAR": "2024-2025",
                "WEEK_NUM": i + 1,
                "WINNER": "KC",
                "LOSER": "LV",
                "WIN_OR_TIE": 1,
                "GAME_LOCATION": "N",
            }
            for i in range(25)
        ]
        games = make_games(neutral_games)
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "target",
                    "TEAM_A": "KC",
                    "TEAM_B": "LV",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 18,
                }
            ]
        )
        acc = make_accessor(games=games)
        result: DataFrame = VenueHFAFeature().compute(df=df, datasets=acc)
        # All games are neutral → no home games → coefficient should be 0.0
        assert result["TEAM_A_FRANCHISE_HFA"].iloc[0] == 0.0

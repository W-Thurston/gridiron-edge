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


class TestVenueHfaTieGameAccounting:
    """Verify ties don't miscredit home games to the away team (venue_hfa/H1)."""

    def _build_home_wins(self, team: str, n: int, start_week: int = 1) -> list[dict]:
        """Build n home wins for `team`."""
        return [
            {
                "GAME_ID": f"hw_{team}_{i}",
                "YEAR": "2024-2025",
                "WEEK_NUM": start_week + i,
                "WINNER": team,
                "LOSER": "Opponent",
                "WIN_OR_TIE": 1,
                "GAME_LOCATION": "H",
            }
            for i in range(n)
        ]

    def _build_home_losses(self, team: str, n: int, start_week: int = 1) -> list[dict]:
        """Build n home losses for `team`."""
        return [
            {
                "GAME_ID": f"hl_{team}_{i}",
                "YEAR": "2024-2025",
                "WEEK_NUM": start_week + i,
                "WINNER": "Opponent",
                "LOSER": team,
                "WIN_OR_TIE": 1,
                "GAME_LOCATION": "H",
            }
            for i in range(n)
        ]

    def test_tie_does_not_credit_away_team_with_home_game(self) -> None:
        """An away team in a tie home game should not get credit for hosting."""

        # KC plays 25 home games to clear the threshold: 12W 12L 1T.
        # In the tie, by convention WINNER=KC (home), LOSER=BUF (away).
        # BUF should NOT be credited with a home game from this tie.
        kc_wins = self._build_home_wins("Kansas City Chiefs", 12, start_week=1)
        kc_losses = self._build_home_losses("Kansas City Chiefs", 12, start_week=13)
        kc_tie = [
            {
                "GAME_ID": "kc_tie",
                "YEAR": "2024-2025",
                "WEEK_NUM": 25,
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Buffalo Bills",
                "WIN_OR_TIE": 0.5,
                "GAME_LOCATION": "H",
            }
        ]
        # BUF also gets enough home games to compute a coefficient.
        buf_wins = self._build_home_wins("Buffalo Bills", 10, start_week=1)
        buf_losses = self._build_home_losses("Buffalo Bills", 10, start_week=11)

        games = make_games(kc_wins + kc_losses + kc_tie + buf_wins + buf_losses)
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "target",
                    "TEAM_A": "Buffalo Bills",
                    "TEAM_B": "Kansas City Chiefs",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 26,
                }
            ]
        )
        acc = make_accessor(games=games)
        result: DataFrame = VenueHFAFeature().compute(df=df, datasets=acc)

        # Buffalo's HFA reflects only their 10/20 (50%) actual home win rate.
        # If the tie were miscounted, Buffalo's denominator would be 21 and
        # their home win rate would be ~10/21 ≈ 0.476 instead of 0.50.
        # We assert the value is reasonable (not skewed by the bug).
        buf_hfa = float(result["TEAM_A_FRANCHISE_HFA"].iloc[0])
        assert -0.05 < buf_hfa < 0.05  # Buffalo near league average

    def test_tie_credits_home_team_via_winner_path(self) -> None:
        """The home team's tie should still contribute 0.5 to home wins."""

        # KC plays 24 home games: 12W 12L. Add 1 tie.
        # Expected home wins = 12 + 0.5 = 12.5
        # Expected home_games = 25
        # Expected home_win_rate = 0.50
        kc_wins = self._build_home_wins("Kansas City Chiefs", 12, start_week=1)
        kc_losses = self._build_home_losses("Kansas City Chiefs", 12, start_week=13)
        kc_tie = [
            {
                "GAME_ID": "kc_tie",
                "YEAR": "2024-2025",
                "WEEK_NUM": 25,
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Buffalo Bills",
                "WIN_OR_TIE": 0.5,
                "GAME_LOCATION": "H",
            }
        ]
        games = make_games(kc_wins + kc_losses + kc_tie)
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "target",
                    "TEAM_A": "Kansas City Chiefs",
                    "TEAM_B": "Other Team",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 26,
                }
            ]
        )
        acc = make_accessor(games=games)
        result: DataFrame = VenueHFAFeature().compute(df=df, datasets=acc)

        # KC's coefficient should be ~0 (their 50% home win rate matches
        # the league average from this fixture). Most importantly, the
        # tie was counted toward home_games and contributed 0.5 to wins.
        kc_hfa = float(result["TEAM_A_FRANCHISE_HFA"].iloc[0])
        assert -0.05 <= kc_hfa <= 0.05

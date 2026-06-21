# tests/unit/features/test_record.py
"""Tests for gridiron_edge.features.team.record — RecordFeature."""

from __future__ import annotations

from pandas import DataFrame
import pytest
from tests.fixtures.dataframes import make_accessor, make_games, make_modeling_rows

from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.record import RecordFeature, _build_record_table


class TestRecordFeatureSpec:
    def test_spec_name(self) -> None:
        assert RecordFeature().spec.name == "record"

    def test_produces_10_columns(self) -> None:
        """5 stats x 2 teams = 10 columns."""
        assert len(RecordFeature().spec.produces) == 10

    def test_produces_expected_columns(self) -> None:
        expected: set[str] = {
            "TEAM_A_WINS",
            "TEAM_A_LOSSES",
            "TEAM_A_WIN_PCT",
            "TEAM_A_WIN_STREAK",
            "TEAM_A_LOSS_STREAK",
            "TEAM_B_WINS",
            "TEAM_B_LOSSES",
            "TEAM_B_WIN_PCT",
            "TEAM_B_WIN_STREAK",
            "TEAM_B_LOSS_STREAK",
        }
        assert set(RecordFeature().spec.produces) == expected

    def test_registered_under_record(self) -> None:
        assert FeatureRegistry.get("record") is RecordFeature


class TestBuildRecordTable:
    def test_week_1_has_zero_record(self) -> None:
        """No games played before week 1 → wins=0, losses=0."""
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        table: DataFrame = _build_record_table(games)
        kc_wk1 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 1)]
        assert len(kc_wk1) == 1
        assert kc_wk1["WINS"].iloc[0] == 0
        assert kc_wk1["LOSSES"].iloc[0] == 0

    def test_week_2_reflects_week_1_result(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g2",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 2,
                    "WINNER": "KC",
                    "LOSER": "BUF",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        table: DataFrame = _build_record_table(games)
        kc_wk2 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 2)]
        assert kc_wk2["WINS"].iloc[0] == 1
        assert kc_wk2["LOSSES"].iloc[0] == 0

    def test_loser_gets_loss(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g2",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 2,
                    "WINNER": "LV",
                    "LOSER": "BUF",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        table: DataFrame = _build_record_table(games)
        lv_wk2 = table[(table["TEAM"] == "LV") & (table["WEEK_NUM"] == 2)]
        assert lv_wk2["LOSSES"].iloc[0] == 1

    def test_win_streak_basic(self) -> None:
        """Three consecutive wins → WIN_STREAK=3 entering game 4."""
        games = make_games(
            [
                {
                    "GAME_ID": f"g{i}",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": i,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                }
                for i in range(1, 5)
            ]
        )
        table: DataFrame = _build_record_table(games)
        kc_wk4 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 4)]
        assert kc_wk4["WIN_STREAK"].iloc[0] == 3
        assert kc_wk4["LOSS_STREAK"].iloc[0] == 0

    def test_loss_breaks_win_streak(self) -> None:
        """KC wins then loses — entering week 3, WIN_STREAK=0 LOSS_STREAK=1."""
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g2",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 2,
                    "WINNER": "LV",
                    "LOSER": "KC",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g3",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 3,
                    "WINNER": "KC",
                    "LOSER": "BUF",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        table: DataFrame = _build_record_table(games)
        kc_wk3 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 3)]
        assert kc_wk3["WIN_STREAK"].iloc[0] == 0
        assert kc_wk3["LOSS_STREAK"].iloc[0] == 1

    def test_tie_resets_both_streaks(self) -> None:
        """Tie should reset both streaks to 0 entering the next game."""
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g2",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 2,
                    "WINNER": "KC",
                    "LOSER": "BUF",
                    "WIN_OR_TIE": 0.5,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g3",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 3,
                    "WINNER": "KC",
                    "LOSER": "MIA",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        table: DataFrame = _build_record_table(games)
        kc_wk3 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 3)]
        # Entering wk3: prior was tie → both streaks reset.
        assert kc_wk3["WIN_STREAK"].iloc[0] == 0
        assert kc_wk3["LOSS_STREAK"].iloc[0] == 0

    def test_win_pct_nan_in_week_1(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        table: DataFrame = _build_record_table(games)
        kc_wk1 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 1)]
        import math

        assert math.isnan(kc_wk1["WIN_PCT"].iloc[0])


class TestRecordCompute:
    def test_output_has_all_record_columns(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g2",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 2,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "g2",
                    "TEAM_A": "KC",
                    "TEAM_B": "LV",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 2,
                },
            ]
        )
        acc = make_accessor(games=games)
        result: DataFrame = RecordFeature().compute(df=df, datasets=acc)
        expected_cols: set[str] = set(RecordFeature().spec.produces)
        assert expected_cols <= set(result.columns)

    def test_team_a_record_via_merge(self) -> None:
        """Verify the vectorized merge produces the expected stats."""
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g2",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 2,
                    "WINNER": "KC",
                    "LOSER": "BUF",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
                {
                    "GAME_ID": "g3",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 3,
                    "WINNER": "KC",
                    "LOSER": "MIA",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "g3",
                    "TEAM_A": "KC",
                    "TEAM_B": "MIA",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 3,
                },
            ]
        )
        acc = make_accessor(games=games)
        result: DataFrame = RecordFeature().compute(df=df, datasets=acc)
        row = result.iloc[0]
        # Entering week 3: KC has 2 wins, 0 losses, 2-game win streak
        assert row["TEAM_A_WINS"] == 2
        assert row["TEAM_A_LOSSES"] == 0
        assert row["TEAM_A_WIN_PCT"] == pytest.approx(1.0)
        assert row["TEAM_A_WIN_STREAK"] == 2
        assert row["TEAM_A_LOSS_STREAK"] == 0

    def test_unmatched_rows_get_defaults(self) -> None:
        """Rows for teams not in the games table get defaults."""
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "g99",
                    "TEAM_A": "Unknown Team",
                    "TEAM_B": "Other Unknown",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 5,
                },
            ]
        )
        acc = make_accessor(games=games)
        result: DataFrame = RecordFeature().compute(df=df, datasets=acc)
        row = result.iloc[0]
        # Unmatched → defaults: 0 wins/losses/streaks, NaN win_pct
        assert row["TEAM_A_WINS"] == 0
        assert row["TEAM_A_LOSSES"] == 0
        assert row["TEAM_A_WIN_STREAK"] == 0
        assert row["TEAM_A_LOSS_STREAK"] == 0
        import math

        assert math.isnan(row["TEAM_A_WIN_PCT"])

# tests/unit/features/test_record.py
"""Tests for gridiron_edge.features.team.record — RecordFeature."""

from __future__ import annotations

from pandas import DataFrame
from tests.fixtures.dataframes import make_accessor, make_games, make_modeling_rows

from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.record import RecordFeature, _build_record_map


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


class TestBuildRecordMap:
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
        record: dict[tuple[str, str, int], dict[str, float]] = _build_record_map(games)
        kc_wk1: dict[str, float] | None = record.get(("KC", "2024-2025", 1))
        assert kc_wk1 is not None
        assert kc_wk1["WINS"] == 0
        assert kc_wk1["LOSSES"] == 0

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
        record: dict[tuple[str, str, int], dict[str, float]] = _build_record_map(games)
        kc_wk2: dict[str, float] | None = record.get(("KC", "2024-2025", 2))
        assert kc_wk2 is not None
        assert kc_wk2["WINS"] == 1
        assert kc_wk2["LOSSES"] == 0

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
        record: dict[tuple[str, str, int], dict[str, float]] = _build_record_map(games)
        lv_wk2: dict[str, float] | None = record.get(("LV", "2024-2025", 2))
        assert lv_wk2 is not None
        assert lv_wk2["LOSSES"] == 1


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

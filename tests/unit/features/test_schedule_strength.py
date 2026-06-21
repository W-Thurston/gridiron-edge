# tests/unit/features/test_schedule_strength.py
"""Tests for gridiron_edge.features.team.schedule_strength — ScheduleStrengthFeature."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.dataframes import (
    make_accessor,
    make_games,
    make_modeling_rows,
)

from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.schedule_strength import (
    ScheduleStrengthFeature,
    _build_sos_sov_table,
)


class TestScheduleStrengthSpec:
    def test_spec_name(self) -> None:
        assert ScheduleStrengthFeature().spec.name == "schedule_strength"

    def test_produces_4_columns(self) -> None:
        assert len(ScheduleStrengthFeature().spec.produces) == 4

    def test_produces_expected_columns(self) -> None:
        expected: set[str] = {"TEAM_A_SOS", "TEAM_A_SOV", "TEAM_B_SOS", "TEAM_B_SOV"}
        assert set(ScheduleStrengthFeature().spec.produces) == expected

    def test_depends_on_elo(self) -> None:
        """SOS/SOV requires Elo state, so it should depend on 'team_elo'."""
        assert "team_elo" in ScheduleStrengthFeature().spec.depends_on

    def test_registered_under_schedule_strength(self) -> None:
        assert FeatureRegistry.get("schedule_strength") is ScheduleStrengthFeature


class TestBuildSosSovTable:
    def test_week_1_has_nan_sos(self) -> None:
        """No prior opponents in week 1 → SOS should be NaN."""
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
        elo = pd.DataFrame(
            [
                {"NFL_TEAM": "KC", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1520.0},
                {"NFL_TEAM": "LV", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1480.0},
            ]
        )
        table: DataFrame = _build_sos_sov_table(games, elo)
        kc_wk1 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 1)]
        assert len(kc_wk1) == 1
        assert pd.isna(kc_wk1["SOS"].iloc[0])
        assert pd.isna(kc_wk1["SOV"].iloc[0])

    def test_week_2_sos_equals_opponent_elo(self) -> None:
        """After 1 game, SOS = opponent's pre-game Elo."""
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
        elo = pd.DataFrame(
            [
                {"NFL_TEAM": "KC", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1520.0},
                {"NFL_TEAM": "LV", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1480.0},
                {"NFL_TEAM": "KC", "NFL_YEAR": "2024-2025", "NFL_WEEK": 2, "ELO": 1530.0},
                {"NFL_TEAM": "BUF", "NFL_YEAR": "2024-2025", "NFL_WEEK": 2, "ELO": 1510.0},
            ]
        )
        table: DataFrame = _build_sos_sov_table(games, elo)
        kc_wk2 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 2)]
        assert kc_wk2["SOS"].iloc[0] == pytest.approx(1480.0)
        # KC won → SOV should also equal LV's week-1 Elo.
        assert kc_wk2["SOV"].iloc[0] == pytest.approx(1480.0)

    def test_sov_excludes_losses(self) -> None:
        """SOV should average only opponents the team beat, not those it lost to."""
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
                    "WINNER": "BUF",
                    "LOSER": "KC",
                    "WIN_OR_TIE": 1,
                    "GAME_LOCATION": "H",
                },
            ]
        )
        elo = pd.DataFrame(
            [
                {"NFL_TEAM": "LV", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1480.0},
                {"NFL_TEAM": "BUF", "NFL_YEAR": "2024-2025", "NFL_WEEK": 2, "ELO": 1600.0},
            ]
        )
        table: DataFrame = _build_sos_sov_table(games, elo)
        # KC's record entering a hypothetical week 3: beat LV (Elo 1480),
        # lost to BUF (Elo 1600).
        # SOS = mean(1480, 1600) = 1540
        # SOV = mean(1480) = 1480  (BUF excluded because KC lost)
        # We need a week 3 game to inspect KC's standing-after-week-2 record.
        games2 = make_games(
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
                    "WINNER": "BUF",
                    "LOSER": "KC",
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
        table = _build_sos_sov_table(games2, elo)
        kc_wk3 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 3)]
        assert kc_wk3["SOS"].iloc[0] == pytest.approx(1540.0)
        assert kc_wk3["SOV"].iloc[0] == pytest.approx(1480.0)

    def test_ties_excluded_from_sov(self) -> None:
        """Tie games count toward SOS but not SOV."""
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "WINNER": "KC",
                    "LOSER": "LV",
                    "WIN_OR_TIE": 0.5,
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
        elo = pd.DataFrame(
            [
                {"NFL_TEAM": "LV", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1480.0},
                {"NFL_TEAM": "BUF", "NFL_YEAR": "2024-2025", "NFL_WEEK": 2, "ELO": 1510.0},
            ]
        )
        table: DataFrame = _build_sos_sov_table(games, elo)
        # KC entering week 2: tied vs LV → SOS counts LV but SOV does not.
        kc_wk2 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 2)]
        assert kc_wk2["SOS"].iloc[0] == pytest.approx(1480.0)
        # No outright wins yet → SOV NaN.
        assert pd.isna(kc_wk2["SOV"].iloc[0])

    def test_missing_opponent_elo_excludes_game(self) -> None:
        """Games with no opponent Elo in the state table are skipped."""
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
        # Elo state only has LV; BUF is missing.
        elo = pd.DataFrame(
            [
                {"NFL_TEAM": "LV", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1480.0},
            ]
        )
        games3 = make_games(
            [
                *games.to_dict(orient="records"),
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
        table: DataFrame = _build_sos_sov_table(games3, elo)
        kc_wk3 = table[(table["TEAM"] == "KC") & (table["WEEK_NUM"] == 3)]
        # Only LV's Elo is known; BUF's missing.
        # SOS should be mean of just LV's Elo.
        assert kc_wk3["SOS"].iloc[0] == pytest.approx(1480.0)
        assert kc_wk3["SOV"].iloc[0] == pytest.approx(1480.0)


class TestScheduleStrengthCompute:
    def test_output_has_all_columns(self) -> None:
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
                    "LOSER": "MIA",
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
        elo = pd.DataFrame(
            [
                {"NFL_TEAM": "KC", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1520.0},
                {"NFL_TEAM": "LV", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1480.0},
                {"NFL_TEAM": "KC", "NFL_YEAR": "2024-2025", "NFL_WEEK": 2, "ELO": 1530.0},
                {"NFL_TEAM": "LV", "NFL_YEAR": "2024-2025", "NFL_WEEK": 2, "ELO": 1470.0},
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
        acc = make_accessor(games=games, elo_state=elo)
        result: DataFrame = ScheduleStrengthFeature().compute(df=df, datasets=acc)
        expected_cols: set[str] = set(ScheduleStrengthFeature().spec.produces)
        assert expected_cols <= set(result.columns)

    def test_team_a_sos_via_merge(self) -> None:
        """Verify the vectorized merge produces the expected SOS value."""
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
                    "LOSER": "MIA",
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
        elo = pd.DataFrame(
            [
                {"NFL_TEAM": "LV", "NFL_YEAR": "2024-2025", "NFL_WEEK": 1, "ELO": 1480.0},
                {"NFL_TEAM": "MIA", "NFL_YEAR": "2024-2025", "NFL_WEEK": 2, "ELO": 1500.0},
            ]
        )
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "g3",
                    "TEAM_A": "KC",
                    "TEAM_B": "BUF",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 3,
                },
            ]
        )
        acc = make_accessor(games=games, elo_state=elo)
        result: DataFrame = ScheduleStrengthFeature().compute(df=df, datasets=acc)
        row = result.iloc[0]
        # KC entering week 3: beat LV (Elo 1480) and MIA (Elo 1500).
        # SOS = (1480 + 1500) / 2 = 1490
        # SOV = same (both wins).
        assert row["TEAM_A_SOS"] == pytest.approx(1490.0)
        assert row["TEAM_A_SOV"] == pytest.approx(1490.0)

    def test_empty_games_yields_nan(self) -> None:
        df = make_modeling_rows(
            [
                {
                    "GAME_ID": "g1",
                    "TEAM_A": "KC",
                    "TEAM_B": "LV",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                },
            ]
        )
        acc = make_accessor(
            games=pd.DataFrame(),
            elo_state=pd.DataFrame(),
        )
        result: DataFrame = ScheduleStrengthFeature().compute(df=df, datasets=acc)
        assert pd.isna(result["TEAM_A_SOS"].iloc[0])
        assert pd.isna(result["TEAM_A_SOV"].iloc[0])

# tests/unit/features/test_schedule_strength.py
"""Tests for gridiron_edge.features.team.schedule_strength — ScheduleStrengthFeature."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.fixtures.dataframes import (
    make_games,
)

from gridiron_edge.features.team.schedule_strength import (
    ScheduleStrengthFeature,
    _build_sos_sov_map,
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
        from gridiron_edge.features.registry import FeatureRegistry

        assert FeatureRegistry.get("schedule_strength") is ScheduleStrengthFeature


class TestBuildSosSovMap:
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
        sos_map: dict[tuple[str, str, int], dict[str, float]] = _build_sos_sov_map(games, elo)
        kc_wk1: dict[str, float] | None = sos_map.get(("KC", "2024-2025", 1))
        assert kc_wk1 is not None
        assert pd.isna(kc_wk1["SOS"])

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
        sos_map: dict[tuple[str, str, int], dict[str, float]] = _build_sos_sov_map(games, elo)
        kc_wk2: dict[str, float] | None = sos_map.get(("KC", "2024-2025", 2))
        assert kc_wk2 is not None
        # KC played LV in week 1 → SOS = LV's week-1 Elo = 1480
        assert kc_wk2["SOS"] == pytest.approx(1480.0)

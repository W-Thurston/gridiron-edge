# tests/unit/features/test_epa.py
"""Tests for gridiron_edge.features.team.epa — TeamEpaFeature."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pandas import DataFrame
from tests.fixtures.dataframes import make_epa_by_game

from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.epa import (
    DEFAULT_ROLLING_WINDOW,
    EPA_COLS,
    TeamEpaFeature,
    _build_rolling_epa,
)


class TestEpaConstants:
    def test_default_rolling_window_is_4(self) -> None:
        assert DEFAULT_ROLLING_WINDOW == 4

    def test_epa_cols_has_8_metrics(self) -> None:
        assert len(EPA_COLS) == 8

    def test_epa_cols_contains_expected_metrics(self) -> None:
        expected: set[str] = {
            "off_epa_per_play",
            "off_pass_epa",
            "off_rush_epa",
            "off_success_rate",
            "def_epa_per_play",
            "def_pass_epa",
            "def_rush_epa",
            "def_success_rate",
        }
        assert set(EPA_COLS) == expected


class TestTeamEpaFeatureSpec:
    def test_spec_name(self) -> None:
        assert TeamEpaFeature().spec.name == "epa"

    def test_produces_16_columns(self) -> None:
        """8 EPA metrics x 2 teams = 16 columns."""

        assert len(TeamEpaFeature().spec.produces) == 16

    def test_produces_team_a_and_team_b_prefixes(self) -> None:
        produces: Sequence[str] = TeamEpaFeature().spec.produces
        team_a: list[str] = [c for c in produces if c.startswith("TEAM_A_")]
        team_b: list[str] = [c for c in produces if c.startswith("TEAM_B_")]
        assert len(team_a) == 8
        assert len(team_b) == 8

    def test_registered_under_epa(self) -> None:
        assert FeatureRegistry.get("epa") is TeamEpaFeature


class TestBuildRollingEpa:
    def test_returns_dataframe(self) -> None:
        epa = make_epa_by_game(teams=["KC", "SF"], seasons=[2024], weeks_per_season=6)
        result: DataFrame = _build_rolling_epa(epa, window=4)
        assert isinstance(result, pd.DataFrame)

    def test_no_lookahead_in_week_1(self) -> None:
        """Week 1 of the first season should have NaN rolling values."""

        epa = make_epa_by_game(teams=["KC"], seasons=[2024], weeks_per_season=6)
        result: DataFrame = _build_rolling_epa(epa, window=4)
        week1 = result.loc[(result["season"] == 2024) & (result["week"] == 1), :]
        assert week1["rolling_off_epa_per_play"].isna().all()

    def test_later_weeks_have_values(self) -> None:
        """After enough games, rolling values should not be NaN."""

        epa = make_epa_by_game(teams=["KC"], seasons=[2024], weeks_per_season=10)
        result: DataFrame = _build_rolling_epa(epa, window=4)
        week6 = result.loc[(result["season"] == 2024) & (result["week"] == 6), :]
        assert not week6["rolling_off_epa_per_play"].isna().any()

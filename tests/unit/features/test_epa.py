# tests/unit/features/test_epa.py
"""Tests for canonical EPA definitions and rolling-window calculation."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.dataframes import make_epa_by_game

from gridiron_edge.features.team.epa import (
    DEFAULT_ROLLING_WINDOW,
    EPA_COLS,
    _apply_cpoe_availability_policy,
    _build_rolling_epa,
    _build_target_rolling_epa,
)


class TestEpaConstants:
    def test_default_rolling_window_is_4(self) -> None:
        assert DEFAULT_ROLLING_WINDOW == 4

    def test_epa_cols_has_36_metrics(self) -> None:
        assert len(EPA_COLS) == 36

    def test_epa_cols_contains_expected_metrics(self) -> None:
        expected: set[str] = {
            "off_epa_per_play",
            "off_pass_epa",
            "off_rush_epa",
            "off_success_rate",
            "off_pass_success_rate",
            "off_rush_success_rate",
            "off_explosive_rate",
            "off_third_down_pct",
            "off_redzone_td_pct",
            "off_turnover_rate",
            "off_sack_rate",
            "off_plays",
            "off_yards_per_play",
            "off_redzone_attempts",
            "off_int_rate",
            "off_penalty_rate",
            "off_avg_score_diff",
            "off_close_game_pct",
            "def_epa_per_play",
            "def_pass_epa",
            "def_rush_epa",
            "def_success_rate",
            "def_pass_success_rate",
            "def_rush_success_rate",
            "def_explosive_rate",
            "def_third_down_pct",
            "def_redzone_td_pct",
            "def_turnover_rate",
            "def_sack_rate",
            "def_plays",
            "def_yards_per_play",
            "def_redzone_attempts",
            "def_int_rate",
            "def_penalty_rate",
            "def_avg_score_diff",
            "def_close_game_pct",
        }
        assert set(EPA_COLS) == expected
        # Columns screened and removed from the canonical set (DECISIONS.md D72).
        removed = {
            "off_fumble_rate",
            "off_cpoe",
            "off_cpoe_available",
            "def_cpoe",
            "def_cpoe_available",
            "st_epa_per_play",
            "punt_epa_per_play",
        }
        assert removed.isdisjoint(EPA_COLS)
        assert "def_fumble_rate" not in EPA_COLS


class TestApplyCpoeAvailabilityPolicy:
    """DECISIONS.md D60: fill missing CPOE with a paired availability flag."""

    def _epa_raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": ["1999_01_a", "2024_01_a"],
                "season": [1999, 2024],
                "week": [1, 1],
                "team": ["KC", "KC"],
                "off_cpoe": [float("nan"), 3.5],
                "def_cpoe": [float("nan"), -1.2],
            }
        )

    def test_fills_missing_cpoe_with_zero(self) -> None:
        result = _apply_cpoe_availability_policy(self._epa_raw())
        assert result["off_cpoe"].tolist() == [0.0, 3.5]
        assert result["def_cpoe"].tolist() == [0.0, -1.2]

    def test_marks_filled_rows_unavailable(self) -> None:
        result = _apply_cpoe_availability_policy(self._epa_raw())
        assert result["off_cpoe_available"].tolist() == [0, 1]
        assert result["def_cpoe_available"].tolist() == [0, 1]

    def test_does_not_mutate_input(self) -> None:
        source = self._epa_raw()
        _apply_cpoe_availability_policy(source)
        assert source["off_cpoe"].isna().iloc[0]
        assert "off_cpoe_available" not in source.columns

    def test_tolerates_missing_cpoe_columns(self) -> None:
        """A source lacking off_cpoe/def_cpoe (e.g. a stale on-disk parquet
        from before D60) is passed through rather than raising.
        """
        source = pd.DataFrame({"game_id": ["1999_01_a"], "team": ["KC"]})
        result = _apply_cpoe_availability_policy(source)
        assert "off_cpoe_available" not in result.columns
        assert "def_cpoe_available" not in result.columns


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


class TestLaterColumnsUseExistingRollingBoundary:
    """Later-added EPA_COLS entries ride the same generic rolling machinery as
    every other entry -- confirm no separate leakage surface for them."""

    def _epa_with_new_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [f"2024_{week:02d}_KC" for week in range(1, 5)],
                "season": [2024] * 4,
                "week": [1, 2, 3, 4],
                "team": ["KC"] * 4,
                "off_yards_per_play": [1.0, 2.0, 3.0, 4.0],
                "off_sack_rate": [0.10, 0.20, 0.30, 0.40],
            }
        )

    def test_week_one_has_no_self_inclusion(self) -> None:
        rolled = _build_rolling_epa(self._epa_with_new_metrics(), window=4)
        week1 = rolled.loc[rolled["week"] == 1, :]
        assert week1["rolling_off_yards_per_play"].isna().all()
        assert week1["rolling_off_sack_rate"].isna().all()

    def test_later_week_uses_strictly_prior_games_only(self) -> None:
        rolled = _build_rolling_epa(self._epa_with_new_metrics(), window=4)
        week4 = rolled.loc[rolled["week"] == 4, :]
        assert week4["rolling_off_yards_per_play"].iloc[0] == pytest.approx((1.0 + 2.0 + 3.0) / 3)
        assert week4["rolling_off_sack_rate"].iloc[0] == pytest.approx((0.10 + 0.20 + 0.30) / 3)


class TestBuildRollingEpaExcludePlayoffs:
    """Verify optional playoff exclusion in rolling EPA history."""

    def _make_history_with_playoffs(self) -> pd.DataFrame:
        """Build epa_by_game spanning regular and playoff weeks for one team.

        2024 weeks 17, 18 are regular-season; weeks 19, 20 are playoff
        rounds. 2025 week 1 is the start of the next regular season.
        """
        return pd.DataFrame(
            {
                "game_id": [
                    "2024_17_a",
                    "2024_18_a",
                    "2024_19_a",
                    "2024_20_a",
                    "2025_01_a",
                ],
                "season": [2024, 2024, 2024, 2024, 2025],
                "week": [17, 18, 19, 20, 1],
                "team": ["KC", "KC", "KC", "KC", "KC"],
                "off_epa_per_play": [0.10, 0.20, 0.30, 0.40, 0.50],
            }
        )

    def test_excludes_playoff_games_by_default(self) -> None:
        epa = self._make_history_with_playoffs()
        rolled: DataFrame = _build_rolling_epa(epa, window=2)

        wk1 = rolled.loc[(rolled["season"] == 2025) & (rolled["week"] == 1), :]
        assert len(wk1) == 1
        # 2025 week 1 rolling should be the mean of weeks 17 and 18.
        assert wk1["rolling_off_epa_per_play"].iloc[0] == pytest.approx((0.10 + 0.20) / 2)

    def test_includes_playoff_games_when_flag_false(self) -> None:
        epa = self._make_history_with_playoffs()
        rolled: DataFrame = _build_rolling_epa(
            epa,
            window=2,
            exclude_playoffs=False,
        )

        wk1 = rolled.loc[(rolled["season"] == 2025) & (rolled["week"] == 1), :]
        assert len(wk1) == 1
        # 2025 week 1 rolling should be the mean of weeks 19 and 20.
        assert wk1["rolling_off_epa_per_play"].iloc[0] == pytest.approx((0.30 + 0.40) / 2)

    def test_playoff_weeks_dropped_from_output(self) -> None:
        """When exclude_playoffs=True, no rolling rows exist for playoff weeks."""
        epa = self._make_history_with_playoffs()
        rolled: DataFrame = _build_rolling_epa(epa, window=2)

        playoff_rows = rolled.loc[rolled["week"] > 18, :]
        assert playoff_rows.empty


def test_retired_epa_registration_is_absent() -> None:
    from gridiron_edge.features.registry import (
        FeatureRegistry,
    )

    with pytest.raises(
        KeyError,
        match="Feature 'epa' is not registered",
    ):
        FeatureRegistry.get("epa")


class TestBuildTargetRollingEpa:
    """Verify explicit future identities use completed history only."""

    def test_prior_season_seeds_week_one(self) -> None:
        epa = pd.DataFrame(
            {
                "game_id": ["2025_17_KC", "2025_18_KC"],
                "season": [2025, 2025],
                "week": [17, 18],
                "team": ["KC", "KC"],
                "off_epa_per_play": [0.10, 0.30],
            }
        )
        targets = pd.DataFrame(
            {
                "game_id": ["2026_01_KC"],
                "season": [2026],
                "week": [1],
                "team": ["KC"],
            }
        )

        result = _build_target_rolling_epa(epa, targets, window=4)

        assert result["rolling_off_epa_per_play"].iloc[0] == pytest.approx(0.20)

    def test_future_targets_do_not_become_observations(self) -> None:
        epa = pd.DataFrame(
            {
                "game_id": ["2025_18_KC"],
                "season": [2025],
                "week": [18],
                "team": ["KC"],
                "off_epa_per_play": [0.25],
            }
        )
        targets = pd.DataFrame(
            {
                "game_id": ["2026_01_KC", "2026_02_KC"],
                "season": [2026, 2026],
                "week": [1, 2],
                "team": ["KC", "KC"],
            }
        )

        result = _build_target_rolling_epa(epa, targets, window=4)

        assert result["rolling_off_epa_per_play"].tolist() == pytest.approx([0.25, 0.25])

    def test_completed_current_season_game_seeds_later_target(self) -> None:
        epa = pd.DataFrame(
            {
                "game_id": ["2025_18_KC", "2026_01_KC"],
                "season": [2025, 2026],
                "week": [18, 1],
                "team": ["KC", "KC"],
                "off_epa_per_play": [0.10, 0.50],
            }
        )
        targets = pd.DataFrame(
            {
                "game_id": ["2026_02_KC"],
                "season": [2026],
                "week": [2],
                "team": ["KC"],
            }
        )

        result = _build_target_rolling_epa(epa, targets, window=2)

        assert result["rolling_off_epa_per_play"].iloc[0] == pytest.approx(0.30)

    def test_target_matches_historical_shifted_rolling_value(self) -> None:
        epa = make_epa_by_game(teams=["KC"], seasons=[2024], weeks_per_season=4)
        target = epa.loc[
            epa["week"] == 4,
            ["game_id", "season", "week", "team"],
        ]

        historical = _build_rolling_epa(epa, window=4)
        explicit = _build_target_rolling_epa(epa, target, window=4)

        expected = historical.loc[
            historical["week"] == 4,
            "rolling_off_epa_per_play",
        ].iloc[0]
        assert explicit["rolling_off_epa_per_play"].iloc[0] == pytest.approx(expected)

    def test_duplicate_target_identity_is_computed_once(self) -> None:
        epa = pd.DataFrame(
            {
                "game_id": ["2025_18_KC"],
                "season": [2025],
                "week": [18],
                "team": ["KC"],
                "off_epa_per_play": [0.25],
            }
        )
        targets = pd.DataFrame(
            {
                "game_id": ["target-a", "target-b"],
                "season": [2026, 2026],
                "week": [1, 1],
                "team": ["KC", "KC"],
            }
        )

        result = _build_target_rolling_epa(
            epa,
            targets,
            window=4,
        )

        assert len(result) == 1
        assert result["rolling_off_epa_per_play"].iloc[0] == pytest.approx(0.25)

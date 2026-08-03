# tests/unit/evaluation/test_tune.py
"""Tests for gridiron_edge.evaluation.tune - grid constants, helpers, and tiny grid search."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.tune import (
    _WEEK_18,
    _WEEKS_EARLY,
    _WEEKS_MID,
    _WEEKS_POST,
    DIVISOR_VALUES,
    K_EARLY_VALUES,
    K_MID_VALUES,
    K_POST_VALUES,
    K_VALUES,
    K_WEEK18_VALUES,
    REGRESS_VALUES,
    TuneResult,
    _brier,
    _prepare_games,
    run_grid_search,
)
from gridiron_edge.ratings.elo.simulator import _k_for_week


class TestGridConstants:
    """Verify grid parameter lists are well-formed."""

    def test_k_values_has_5(self) -> None:
        assert len(K_VALUES) == 5

    def test_divisor_values_has_5(self) -> None:
        assert len(DIVISOR_VALUES) == 5

    def test_regress_values_has_4(self) -> None:
        assert len(REGRESS_VALUES) == 4

    def test_v2_grid_size_is_100(self) -> None:
        assert len(K_VALUES) * len(DIVISOR_VALUES) * len(REGRESS_VALUES) == 100

    def test_k_early_values_has_6(self) -> None:
        assert len(K_EARLY_VALUES) == 6

    def test_k_week18_values_has_7(self) -> None:
        assert len(K_WEEK18_VALUES) == 7

    def test_all_k_values_positive_or_zero(self) -> None:
        for vals in (K_VALUES, K_EARLY_VALUES, K_MID_VALUES, K_POST_VALUES, K_WEEK18_VALUES):
            assert all(v >= 0 for v in vals)

    def test_all_divisor_values_positive(self) -> None:
        assert all(v > 0 for v in DIVISOR_VALUES)

    def test_all_regress_values_between_0_and_1(self) -> None:
        assert all(0 < v < 1 for v in REGRESS_VALUES)


class TestWeekZones:
    """Verify week zone boundaries are correct and exhaustive."""

    def test_early_is_weeks_1_to_4(self) -> None:
        assert frozenset(range(1, 5)) == _WEEKS_EARLY

    def test_mid_is_weeks_5_to_17(self) -> None:
        assert frozenset(range(5, 18)) == _WEEKS_MID

    def test_week_18_is_just_18(self) -> None:
        assert frozenset([18]) == _WEEK_18

    def test_post_is_weeks_19_to_22(self) -> None:
        assert frozenset(range(19, 23)) == _WEEKS_POST

    def test_zones_cover_all_22_weeks(self) -> None:
        all_weeks: frozenset[int] = _WEEKS_EARLY | _WEEKS_MID | _WEEK_18 | _WEEKS_POST
        assert all_weeks == frozenset(range(1, 23))

    def test_zones_dont_overlap(self) -> None:
        zones: list[frozenset[int]] = [_WEEKS_EARLY, _WEEKS_MID, _WEEK_18, _WEEKS_POST]
        for i, a in enumerate(zones):
            for b in zones[i + 1 :]:
                assert a.isdisjoint(b)


class TestKForWeek:
    def test_week_1_returns_k_early(self) -> None:
        assert _k_for_week(1, k_early=10, k_mid=20, k_week18=5, k_post=30) == 10

    def test_week_10_returns_k_mid(self) -> None:
        assert _k_for_week(10, k_early=10, k_mid=20, k_week18=5, k_post=30) == 20

    def test_week_18_returns_k_week18(self) -> None:
        assert _k_for_week(18, k_early=10, k_mid=20, k_week18=5, k_post=30) == 5

    def test_week_19_returns_k_post(self) -> None:
        assert _k_for_week(19, k_early=10, k_mid=20, k_week18=5, k_post=30) == 30

    def test_week_22_returns_k_post(self) -> None:
        assert _k_for_week(22, k_early=10, k_mid=20, k_week18=5, k_post=30) == 30


class TestBrier:
    def test_perfect_predictions(self) -> None:
        assert _brier([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == pytest.approx(0.0)

    def test_worst_predictions(self) -> None:
        assert _brier([0.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_coin_flip(self) -> None:
        assert _brier([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)

    def test_empty_returns_nan(self) -> None:
        import math

        assert math.isnan(_brier([], []))


class TestTuneResultDataclass:
    def test_is_frozen(self) -> None:
        tr = TuneResult(
            k=20.0,
            divisor=400.0,
            regress_frac=0.33,
            train_brier=0.24,
            holdout_brier=0.23,
            overfit_gap=0.01,
            train_games=1000,
            holdout_games=200,
            elapsed_s=1.5,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            tr.k = 30.0  # type: ignore[misc]


class TestRunGridSearchTiny:
    """Run grid search with a single-point grid to verify the pipeline works."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        result: DataFrame = run_grid_search(
            repo=None,
            k_values=[20.0],
            divisor_values=[400.0],
            regress_values=[0.33],
            save_path=None,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_result_has_brier_columns(self, tmp_path: Path) -> None:
        result: DataFrame = run_grid_search(
            repo=None,
            k_values=[20.0],
            divisor_values=[400.0],
            regress_values=[0.33],
            save_path=None,
        )
        assert "holdout_brier" in result.columns
        assert "train_brier" in result.columns


def test_prepare_games_uses_canonical_team_identity() -> None:
    games = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "YEAR": [
                "2024-2025",
                "2024-2025",
            ],
            "AWAY_TEAM": ["Bills", "Chiefs"],
            "HOME_TEAM": ["Dolphins", "Ravens"],
            "AWAY_SCORE": [24, 20],
            "HOME_SCORE": [17, 27],
        }
    )

    prepared, seasons, teams_by_year = _prepare_games(games)

    assert len(prepared) == 2
    assert seasons == ["2024-2025"]
    assert teams_by_year == {
        "2024-2025": {
            "Bills",
            "Chiefs",
            "Dolphins",
            "Ravens",
        }
    }


def test_prepare_games_excludes_unplayed_games() -> None:
    games = pd.DataFrame(
        {
            "YEAR": [
                "2024-2025",
                "2024-2025",
            ],
            "AWAY_TEAM": ["Bills", "Chiefs"],
            "HOME_TEAM": ["Dolphins", "Ravens"],
            "AWAY_SCORE": [24, pd.NA],
            "HOME_SCORE": [17, pd.NA],
        }
    )

    prepared, _seasons, teams_by_year = _prepare_games(games)

    assert len(prepared) == 1
    assert teams_by_year["2024-2025"] == {
        "Bills",
        "Dolphins",
    }


@pytest.mark.parametrize(
    (
        "away_score",
        "home_score",
    ),
    [
        (
            24,
            pd.NA,
        ),
        (
            pd.NA,
            17,
        ),
    ],
)
def test_prepare_games_excludes_partial_scores(
    away_score: object,
    home_score: object,
) -> None:
    games = pd.DataFrame(
        {
            "YEAR": ["2024-2025"],
            "AWAY_TEAM": ["Bills"],
            "HOME_TEAM": ["Dolphins"],
            "AWAY_SCORE": [away_score],
            "HOME_SCORE": [home_score],
        }
    )

    prepared, seasons, teams_by_year = _prepare_games(games)

    assert prepared.empty
    assert seasons == []
    assert teams_by_year == {}


def test_prepare_games_rejects_missing_canonical_identity() -> None:
    games = pd.DataFrame(
        {
            "YEAR": ["2024-2025"],
            "AWAY_SCORE": [24],
            "HOME_SCORE": [17],
        }
    )

    with pytest.raises(
        ValueError,
        match="AWAY_TEAM",
    ):
        _prepare_games(games)


def test_prepare_games_rejects_empty_team_identity() -> None:
    games = pd.DataFrame(
        {
            "YEAR": ["2024-2025"],
            "AWAY_TEAM": [""],
            "HOME_TEAM": ["Dolphins"],
            "AWAY_SCORE": [24],
            "HOME_SCORE": [17],
        }
    )

    with pytest.raises(
        ValueError,
        match="empty team identities",
    ):
        _prepare_games(games)


def test_prepare_games_does_not_mutate_input() -> None:
    games = pd.DataFrame(
        {
            "YEAR": ["2024-2025"],
            "AWAY_TEAM": [" Bills "],
            "HOME_TEAM": ["Dolphins"],
            "AWAY_SCORE": [24],
            "HOME_SCORE": [17],
        }
    )
    expected = games.copy(deep=True)

    _prepare_games(games)

    pd.testing.assert_frame_equal(
        games,
        expected,
    )


def test_prepare_games_uses_score_completion_contract() -> None:
    import inspect

    source = inspect.getsource(_prepare_games)

    assert "WINNER" not in source
    assert "LOSER" not in source
    assert "GAME_LOCATION" not in source
    assert "WIN_OR_TIE" not in source
    assert 'games["AWAY_SCORE"].notna()' in source
    assert 'games["HOME_SCORE"].notna()' in source

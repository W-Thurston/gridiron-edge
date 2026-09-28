# tests/unit/features/test_rating_ensemble.py

"""Tests for the rating-ensemble features (Glicko-2, linear ratings, QB rating)."""

from __future__ import annotations

from typing import cast

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.rating_history import GameSpec, make_rating_history

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.rating_ensemble import (
    GLICKO2_METRICS,
    LINEAR_RATING_METRICS,
    QB_ELO_METRICS,
    RATING_ENSEMBLE_METRICS,
    HomeAwayGlicko2Feature,
    HomeAwayLinearRatingsFeature,
    HomeAwayQbEloFeature,
)

_FEATURES = (HomeAwayGlicko2Feature, HomeAwayLinearRatingsFeature, HomeAwayQbEloFeature)
_PLAYED: list[GameSpec] = [
    (2026, 1, "X", "Y", 30, 10),
    (2026, 2, "Z", "X", 17, 20),
    (2026, 2, "Y", "W", 13, 27),
]


class _FakeDatasets:
    def __init__(self, games: DataFrame, sidecar: DataFrame) -> None:
        self._games = games
        self._sidecar = sidecar

    def games(self) -> DataFrame:
        return self._games

    def qb_coach_sidecar(self) -> DataFrame:
        return self._sidecar


def _sidecar(games: DataFrame, upcoming: DataFrame) -> DataFrame:
    """Sidecar with played starters ``qb_<team>`` and null starters for upcoming."""
    played = pd.DataFrame(
        {
            "GAME_ID": games["GAME_ID"],
            "YEAR": games["YEAR"],
            "WEEK_NUM": games["WEEK_NUM"],
            "AWAY_TEAM": games["AWAY_TEAM"],
            "HOME_TEAM": games["HOME_TEAM"],
            "AWAY_QB_ID": "qb_" + games["AWAY_TEAM"],
            "HOME_QB_ID": "qb_" + games["HOME_TEAM"],
        }
    )
    unplayed = upcoming.assign(AWAY_QB_ID=None, HOME_QB_ID=None)
    return pd.concat([played, unplayed], ignore_index=True)


def _upcoming(weeks: tuple[int, ...] = (3, 6, 18)) -> DataFrame:
    rows = [
        {
            "GAME_ID": f"2026_{week:02d}_X_Y",
            "YEAR": "2026-2027",
            "WEEK_NUM": week,
            "AWAY_TEAM": "X",
            "HOME_TEAM": "Y",
        }
        for week in weeks
    ]
    return pd.DataFrame(rows)


def _world(
    played: list[GameSpec] | None = None,
) -> tuple[DataFrame, _FakeDatasets, DataFrame]:
    games = make_rating_history(_PLAYED if played is None else played)
    upcoming = _upcoming()
    datasets = _FakeDatasets(games, _sidecar(games, upcoming))
    frame = pd.concat(
        [games[["GAME_ID", "YEAR", "WEEK_NUM", "AWAY_TEAM", "HOME_TEAM"]], upcoming],
        ignore_index=True,
    )
    return frame, datasets, games


def _run(frame: DataFrame, datasets: _FakeDatasets) -> DataFrame:
    result = frame
    for feature_cls in _FEATURES:
        result = feature_cls().compute(df=result, datasets=cast(DatasetAccessor, datasets))
    return result


def _metric_columns() -> list[str]:
    return [f"{side}_{m.upper()}" for m in RATING_ENSEMBLE_METRICS for side in ("AWAY", "HOME")]


def test_features_are_registered_and_declare_every_column() -> None:
    produced: list[str] = []
    for name, feature_cls in (
        ("home_away_glicko2", HomeAwayGlicko2Feature),
        ("home_away_linear_ratings", HomeAwayLinearRatingsFeature),
        ("home_away_qb_elo", HomeAwayQbEloFeature),
    ):
        assert FeatureRegistry.get(name) is feature_cls
        produced.extend(feature_cls.spec.produces)
    assert sorted(produced) == sorted(_metric_columns())
    assert len(RATING_ENSEMBLE_METRICS) == len(
        {*GLICKO2_METRICS, *LINEAR_RATING_METRICS, *QB_ELO_METRICS}
    )


def test_volatility_is_not_emitted() -> None:
    assert not any("VOLATILITY" in column for column in _metric_columns())


def test_rows_order_and_existing_columns_are_preserved() -> None:
    frame, datasets, _ = _world()
    frame = frame.assign(SENTINEL=range(len(frame)))
    result = _run(frame, datasets)
    assert len(result) == len(frame)
    assert result["GAME_ID"].tolist() == frame["GAME_ID"].tolist()
    assert result["SENTINEL"].tolist() == frame["SENTINEL"].tolist()
    assert set(_metric_columns()) <= set(result.columns)


def test_recompute_replaces_previously_produced_columns() -> None:
    frame, datasets, _ = _world()
    once = _run(frame, datasets)
    twice = _run(once, datasets)
    pd.testing.assert_frame_equal(once, twice)


def test_no_nulls_for_any_row_including_first_week_and_lookahead() -> None:
    frame, datasets, _ = _world()
    result = _run(frame, datasets)
    assert not result[_metric_columns()].isna().any().any()
    assert (result["YEAR"] == "1999-2000").any()
    assert (result["WEEK_NUM"] == 18).any()


def test_lookahead_horizons_equal_the_next_week_value() -> None:
    frame, datasets, _ = _world()
    result = _run(frame, datasets)
    upcoming = result.loc[result["GAME_ID"].str.startswith("2026_") & (result["WEEK_NUM"] >= 3)]
    reference = upcoming.iloc[0][_metric_columns()]
    for _, row in upcoming.iterrows():
        pd.testing.assert_series_equal(row[_metric_columns()], reference, check_names=False)


def test_orientation_stronger_side_has_higher_rating() -> None:
    frame, datasets, _ = _world()
    result = _run(frame, datasets)
    row = result.loc[result["GAME_ID"] == "2026_06_X_Y"].iloc[0]  # X won and Y lost earlier
    for metric in ("GLICKO2_RATING", "MASSEY_RATING", "COLLEY_RATING", "KEENER_RATING", "QB_ELO"):
        assert row[f"AWAY_{metric}"] > row[f"HOME_{metric}"], metric


def test_a_games_own_result_never_enters_its_features() -> None:
    frame, datasets, games = _world()
    baseline = _run(frame, datasets)
    for score in (0, 99):
        mutated = games.copy()
        mutated.loc[mutated["GAME_ID"] == "2026_02_Z_X", ["AWAY_SCORE", "HOME_SCORE"]] = [score, 7]
        altered = _run(frame, _FakeDatasets(mutated, _sidecar(mutated, _upcoming())))
        target = baseline["GAME_ID"] == "2026_02_Z_X"
        pd.testing.assert_frame_equal(
            baseline.loc[target].reset_index(drop=True),
            altered.loc[target].reset_index(drop=True),
        )


def test_later_results_never_change_earlier_rows() -> None:
    frame, datasets, games = _world()
    baseline = _run(frame, datasets)
    mutated = games.copy()
    late = mutated["GAME_ID"] == "2026_02_Y_W"
    mutated.loc[late, ["AWAY_SCORE", "HOME_SCORE"]] = [0, 60]
    altered = _run(frame, _FakeDatasets(mutated, _sidecar(mutated, _upcoming())))
    earlier = baseline["GAME_ID"].isin(
        games.loc[(games["YEAR"] != "2026-2027") | (games["WEEK_NUM"] < 2), "GAME_ID"]
    )
    pd.testing.assert_frame_equal(
        baseline.loc[earlier].reset_index(drop=True),
        altered.loc[earlier].reset_index(drop=True),
    )
    assert not baseline.loc[~earlier, _metric_columns()].equals(
        altered.loc[~earlier, _metric_columns()]
    )


class TestQbElo:
    def test_unplayed_game_uses_carried_forward_starter(self) -> None:
        frame, datasets, _ = _world()
        result = _run(frame, datasets)
        row = result.loc[result["GAME_ID"] == "2026_03_X_Y"].iloc[0]
        history_row = result.loc[result["GAME_ID"] == "2026_02_Z_X"].iloc[0]
        # qb_X started weeks 1 and 2, so after week 2 they have two prior starts.
        assert row["AWAY_QB_ELO_STARTS"] == 2
        assert row["HOME_QB_ELO_STARTS"] == 2  # qb_Y started weeks 1 and 2
        assert history_row["HOME_QB_ELO_STARTS"] == 1

    def test_debut_qb_has_zero_rating_and_starts(self) -> None:
        frame, datasets, _ = _world()
        result = _run(frame, datasets)
        row = result.loc[result["GAME_ID"] == "2026_01_X_Y"].iloc[0]
        assert row["AWAY_QB_ELO"] == 0.0
        assert row["AWAY_QB_ELO_STARTS"] == 0

    def test_empty_sidecar_yields_null_qb_columns_only(self) -> None:
        frame, _, games = _world()
        empty = _FakeDatasets(games, pd.DataFrame())
        result = _run(frame, empty)
        assert result[_columns_for(QB_ELO_METRICS)].isna().all().all()
        assert (
            not result[_columns_for((*GLICKO2_METRICS, *LINEAR_RATING_METRICS))].isna().any().any()
        )

    def test_completed_game_without_starter_raises(self) -> None:
        frame, _, games = _world()
        sidecar = _sidecar(games, _upcoming())
        sidecar = sidecar.loc[sidecar["GAME_ID"] != "2026_01_X_Y"]
        with pytest.raises(ValueError, match="no recorded starting QB"):
            HomeAwayQbEloFeature().compute(
                df=frame, datasets=cast(DatasetAccessor, _FakeDatasets(games, sidecar))
            )


def _columns_for(metrics: tuple[str, ...]) -> list[str]:
    return [f"{side}_{m.upper()}" for m in metrics for side in ("AWAY", "HOME")]


@pytest.mark.parametrize("feature_cls", _FEATURES)
def test_truncated_history_raises(feature_cls: type) -> None:
    frame, _, games = _world()
    truncated = games.loc[games["YEAR"] != "1999-2000"]
    datasets = _FakeDatasets(truncated, _sidecar(truncated, _upcoming()))
    with pytest.raises(ValueError, match="must begin with season"):
        feature_cls().compute(df=frame, datasets=cast(DatasetAccessor, datasets))


@pytest.mark.parametrize("feature_cls", _FEATURES)
def test_missing_game_columns_raise(feature_cls: type) -> None:
    frame, datasets, _ = _world()
    with pytest.raises(ValueError, match="missing required columns"):
        feature_cls().compute(
            df=frame.drop(columns=["HOME_TEAM"]), datasets=cast(DatasetAccessor, datasets)
        )

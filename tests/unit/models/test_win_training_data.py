# tests/unit/models/test_win_training_data.py

"""Tests for canonical Win classification data preparation."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.core.constants import (
    HOLDOUT_SEASONS,
)
from gridiron_edge.models.game_prediction._features import (
    _chronological_split_masks,
    _prepare_data,
)
from gridiron_edge.models.game_prediction.game_schema import (
    HOME_WIN_TARGET,
)
from gridiron_edge.models.game_prediction.win_prob import (
    WinProbTrainer,
)


def _season_pair() -> tuple[str, str]:
    """Return one training season and one configured holdout season."""
    holdout = sorted(HOLDOUT_SEASONS)[0]
    training = "2000-2001"

    if training in HOLDOUT_SEASONS:
        training = "1999-2000"

    return training, holdout


def _canonical_rows() -> DataFrame:
    """Return intentionally unordered canonical Win rows."""
    training, holdout = _season_pair()

    return DataFrame(
        {
            "GAME_ID": [
                "holdout-win",
                "training-tie",
                "training-away-win",
                "training-home-win",
                "training-null-feature",
            ],
            "YEAR": [
                holdout,
                training,
                training,
                training,
                training,
            ],
            "WEEK_NUM": [
                1,
                3,
                2,
                1,
                4,
            ],
            "GAME_DATE": [
                f"{holdout[:4]}-09-01",
                f"{training[:4]}-09-21",
                f"{training[:4]}-09-14",
                f"{training[:4]}-09-07",
                f"{training[:4]}-09-28",
            ],
            HOME_WIN_TARGET: [
                1,
                pd.NA,
                0,
                1,
                1,
            ],
            "MODEL_FEATURE": [
                50.0,
                30.0,
                20.0,
                10.0,
                float("nan"),
            ],
        }
    )


def _feature_fn(frame: DataFrame) -> DataFrame:
    """Select one controlled model feature."""
    return frame.loc[
        :,
        ["MODEL_FEATURE"],
    ].copy()


def test_win_spec_uses_canonical_target() -> None:
    assert WinProbTrainer().spec.target_col == HOME_WIN_TARGET


def test_prepare_data_does_not_require_result() -> None:
    frame = _canonical_rows()

    assert "RESULT" not in frame.columns

    _prepare_data(
        frame,
        _feature_fn,
    )


def test_prepare_data_excludes_ties_and_null_features() -> None:
    (
        x_train,
        y_train,
        x_hold,
        y_hold,
        _train_seasons,
        _holdout_seasons,
        _year_train,
        _year_hold,
    ) = _prepare_data(
        _canonical_rows(),
        _feature_fn,
    )

    assert len(x_train) == 2
    assert len(y_train) == 2
    assert len(x_hold) == 1
    assert len(y_hold) == 1

    assert y_train.tolist() == [1, 0]
    assert y_hold.tolist() == [1]


def test_prepare_data_preserves_home_win_class_meaning() -> None:
    (
        _x_train,
        y_train,
        _x_hold,
        y_hold,
        _train_seasons,
        _holdout_seasons,
        _year_train,
        _year_hold,
    ) = _prepare_data(
        _canonical_rows(),
        _feature_fn,
    )

    assert set(y_train.tolist()) == {0, 1}
    assert set(y_hold.tolist()) == {1}


def test_prepare_data_sorts_chronologically() -> None:
    (
        x_train,
        y_train,
        _x_hold,
        _y_hold,
        _train_seasons,
        _holdout_seasons,
        _year_train,
        _year_hold,
    ) = _prepare_data(
        _canonical_rows(),
        _feature_fn,
    )

    assert x_train["MODEL_FEATURE"].tolist() == [
        10.0,
        20.0,
    ]
    assert y_train.tolist() == [1, 0]


def test_prepare_data_splits_configured_holdout_seasons() -> None:
    training, holdout = _season_pair()

    (
        _x_train,
        _y_train,
        _x_hold,
        _y_hold,
        train_seasons,
        holdout_seasons,
        _year_train,
        _year_hold,
    ) = _prepare_data(
        _canonical_rows(),
        _feature_fn,
    )

    assert train_seasons == [training]
    assert holdout_seasons == [holdout]


def test_prepare_data_does_not_mutate_input() -> None:
    frame = _canonical_rows()
    expected = frame.copy(deep=True)

    _prepare_data(
        frame,
        _feature_fn,
    )

    pd.testing.assert_frame_equal(
        frame,
        expected,
    )


def test_missing_home_win_target_is_rejected() -> None:
    frame = _canonical_rows().drop(columns=[HOME_WIN_TARGET])

    with pytest.raises(
        ValueError,
        match=("Canonical Win modeling data is missing required target column: HOME_WIN"),
    ):
        _prepare_data(
            frame,
            _feature_fn,
        )


# ---------------------------------------------------------------------------
# Chronological training/holdout boundary (D4)
# ---------------------------------------------------------------------------


class TestChronologicalSplitMasks:
    """Training must never include a season more recent than the earliest
    configured holdout season - not just "any season not explicitly listed
    as holdout." An in-progress current season is chronologically after a
    fixed historical holdout window and must be excluded from both pools,
    rather than silently joining training.
    """

    def test_season_after_holdout_window_is_excluded_from_training(self) -> None:
        year = pd.Series(["2022-2023", "2023-2024", "2026-2027"])
        holdout_seasons = frozenset({"2023-2024"})

        train_mask, hold_mask = _chronological_split_masks(year, holdout_seasons)

        assert train_mask.tolist() == [True, False, False]
        assert hold_mask.tolist() == [False, True, False]

    def test_season_before_holdout_window_is_included_in_training(self) -> None:
        year = pd.Series(["1999-2000", "2023-2024"])
        holdout_seasons = frozenset({"2023-2024"})

        train_mask, hold_mask = _chronological_split_masks(year, holdout_seasons)

        assert train_mask.tolist() == [True, False]
        assert hold_mask.tolist() == [False, True]


def test_prepare_data_excludes_season_after_holdout_window() -> None:
    """End-to-end: a completed game from a season chronologically after the
    holdout window (e.g. an in-progress current season) must not appear in
    either the training or the holdout split."""
    training, holdout = _season_pair()
    after_holdout = f"{int(holdout[:4]) + 3}-{int(holdout[:4]) + 4}"

    frame = _canonical_rows()
    extra = pd.DataFrame(
        {
            "GAME_ID": ["after-holdout-game"],
            "YEAR": [after_holdout],
            "WEEK_NUM": [1],
            "GAME_DATE": [f"{after_holdout[:4]}-09-10"],
            HOME_WIN_TARGET: [1],
            "MODEL_FEATURE": [99.0],
        }
    )
    frame = pd.concat([frame, extra], ignore_index=True)

    (
        x_train,
        _y_train,
        x_hold,
        _y_hold,
        train_seasons,
        holdout_seasons,
        _year_train,
        _year_hold,
    ) = _prepare_data(frame, _feature_fn)

    assert 99.0 not in x_train["MODEL_FEATURE"].tolist()
    assert 99.0 not in x_hold["MODEL_FEATURE"].tolist()
    assert after_holdout not in train_seasons
    assert after_holdout not in holdout_seasons
    assert train_seasons == [training]
    assert holdout_seasons == [holdout]

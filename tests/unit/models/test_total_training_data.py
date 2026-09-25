# tests/unit/models/test_total_training_data.py

"""Tests for canonical Total regression data preparation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.core.constants import (
    HOLDOUT_SEASONS,
)
from gridiron_edge.models.game_prediction._columns import (
    _EXPANDED_FEATURES,
)
from gridiron_edge.models.game_prediction.game_schema import (
    ACTUAL_TOTAL_TARGET,
)
from gridiron_edge.models.game_prediction.total import (
    TotalTrainer,
    _prepare_total_data,
)


def _season_pair() -> tuple[str, str]:
    """Return one training and one configured holdout season."""
    holdout = sorted(HOLDOUT_SEASONS)[0]
    training = "2000-2001"

    if training in HOLDOUT_SEASONS:
        training = "1999-2000"

    return training, holdout


def _canonical_modeling_frame() -> DataFrame:
    """Return canonical rows for controlled Total preparation."""
    training, holdout = _season_pair()

    rows = DataFrame(
        {
            "GAME_ID": [
                "holdout",
                "training-tie",
                "training-later",
                "training-earlier",
                "missing-target",
                "missing-feature",
            ],
            "YEAR": [
                holdout,
                training,
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
                5,
            ],
            "GAME_DATE": [
                f"{holdout[:4]}-09-01",
                f"{training[:4]}-09-21",
                f"{training[:4]}-09-14",
                f"{training[:4]}-09-07",
                f"{training[:4]}-09-28",
                f"{training[:4]}-10-05",
            ],
            "AWAY_TEAM": [
                "Away",
                "Away",
                "Away",
                "Away",
                "Away",
                "Away",
            ],
            "HOME_TEAM": [
                "Home",
                "Home",
                "Home",
                "Home",
                "Home",
                "Home",
            ],
            ACTUAL_TOTAL_TARGET: [
                50.0,
                42.0,
                47.0,
                41.0,
                float("nan"),
                44.0,
            ],
        }
    )

    feature_columns: list[str] = [
        column for column in _EXPANDED_FEATURES if column not in rows.columns
    ]
    feature_values = DataFrame(
        1.0,
        index=rows.index,
        columns=feature_columns,
    )
    rows: DataFrame = pd.concat(
        [
            rows,
            feature_values,
        ],
        axis=1,
    )
    assert rows.columns.is_unique

    rows.loc[
        rows["GAME_ID"] == "missing-feature",
        "AWAY_ELO",
    ] = float("nan")

    return rows


def test_total_spec_uses_canonical_target() -> None:
    assert TotalTrainer().spec.target_col == ACTUAL_TOTAL_TARGET


def test_prepare_total_data_uses_persisted_target(
    tmp_path: Path,
) -> None:
    frame = _canonical_modeling_frame()

    assert "actual_total" not in frame.columns
    assert "PTS_WINNER" not in frame.columns
    assert "PTS_LOSER" not in frame.columns

    with patch(
        "gridiron_edge.datasets.loaders.load_modeling_file",
        return_value=frame,
    ):
        _prepare_total_data(tmp_path)


def test_prepare_total_data_retains_tied_games(
    tmp_path: Path,
) -> None:
    frame = _canonical_modeling_frame()

    with patch(
        "gridiron_edge.datasets.loaders.load_modeling_file",
        return_value=frame,
    ):
        (
            _x_train,
            y_train,
            _x_holdout,
            _y_holdout,
            _train_seasons,
            _holdout_seasons,
            _year_train,
            _year_holdout,
        ) = _prepare_total_data(tmp_path)

    assert 42.0 in y_train.tolist()


def test_prepare_total_data_excludes_unavailable_rows(
    tmp_path: Path,
) -> None:
    frame = _canonical_modeling_frame()

    with patch(
        "gridiron_edge.datasets.loaders.load_modeling_file",
        return_value=frame,
    ):
        (
            x_train,
            y_train,
            x_holdout,
            y_holdout,
            _train_seasons,
            _holdout_seasons,
            _year_train,
            _year_holdout,
        ) = _prepare_total_data(tmp_path)

    assert len(x_train) == 3
    assert len(y_train) == 3
    assert len(x_holdout) == 1
    assert len(y_holdout) == 1


def test_prepare_total_data_is_chronological(
    tmp_path: Path,
) -> None:
    frame = _canonical_modeling_frame()

    marker = "AWAY_ELO"
    frame.loc[
        frame["GAME_ID"] == "training-earlier",
        marker,
    ] = 10.0
    frame.loc[
        frame["GAME_ID"] == "training-later",
        marker,
    ] = 20.0
    frame.loc[
        frame["GAME_ID"] == "training-tie",
        marker,
    ] = 30.0

    with patch(
        "gridiron_edge.datasets.loaders.load_modeling_file",
        return_value=frame,
    ):
        (
            x_train,
            y_train,
            _x_holdout,
            _y_holdout,
            _train_seasons,
            _holdout_seasons,
            _year_train,
            _year_holdout,
        ) = _prepare_total_data(tmp_path)

    assert x_train[marker].tolist() == [
        10.0,
        20.0,
        30.0,
    ]
    assert y_train.tolist() == [
        41.0,
        47.0,
        42.0,
    ]


def test_prepare_total_data_reports_seasons(
    tmp_path: Path,
) -> None:
    training, holdout = _season_pair()

    with patch(
        "gridiron_edge.datasets.loaders.load_modeling_file",
        return_value=_canonical_modeling_frame(),
    ):
        (
            _x_train,
            _y_train,
            _x_holdout,
            _y_holdout,
            train_seasons,
            holdout_seasons,
            _year_train,
            _year_holdout,
        ) = _prepare_total_data(tmp_path)

    assert train_seasons == [training]
    assert holdout_seasons == [holdout]


def test_prepare_total_data_does_not_mutate_input(
    tmp_path: Path,
) -> None:
    frame = _canonical_modeling_frame()
    expected = frame.copy(deep=True)

    with patch(
        "gridiron_edge.datasets.loaders.load_modeling_file",
        return_value=frame,
    ):
        _prepare_total_data(tmp_path)

    pd.testing.assert_frame_equal(
        frame,
        expected,
    )


def test_prepare_total_data_excludes_season_after_holdout_window(
    tmp_path: Path,
) -> None:
    """A completed game from a season chronologically after the holdout
    window (e.g. an in-progress current season) must not appear in either
    the training or the holdout split (D4)."""
    training, holdout = _season_pair()
    after_holdout = f"{int(holdout[:4]) + 3}-{int(holdout[:4]) + 4}"

    frame = _canonical_modeling_frame()
    extra = frame.loc[frame["GAME_ID"] == "training-earlier"].copy()
    extra["GAME_ID"] = "after-holdout-game"
    extra["YEAR"] = after_holdout
    extra["GAME_DATE"] = f"{after_holdout[:4]}-09-10"
    extra[ACTUAL_TOTAL_TARGET] = 77.0
    frame = pd.concat([frame, extra], ignore_index=True)

    with patch(
        "gridiron_edge.datasets.loaders.load_modeling_file",
        return_value=frame,
    ):
        (
            x_train,
            _y_train,
            x_holdout,
            _y_holdout,
            train_seasons,
            holdout_seasons,
            _year_train,
            _year_holdout,
        ) = _prepare_total_data(tmp_path)

    assert after_holdout not in train_seasons
    assert after_holdout not in holdout_seasons
    assert train_seasons == [training]
    assert holdout_seasons == [holdout]
    total_rows_seen = len(x_train) + len(x_holdout)
    assert total_rows_seen == 4  # unchanged from the base fixture's 3 train + 1 holdout


def test_missing_actual_total_is_rejected(
    tmp_path: Path,
) -> None:
    frame = _canonical_modeling_frame().drop(columns=[ACTUAL_TOTAL_TARGET])

    with (
        patch(
            "gridiron_edge.datasets.loaders.load_modeling_file",
            return_value=frame,
        ),
        pytest.raises(
            ValueError,
            match=("Canonical Total modeling data is missing required target column: ACTUAL_TOTAL"),
        ),
    ):
        _prepare_total_data(tmp_path)

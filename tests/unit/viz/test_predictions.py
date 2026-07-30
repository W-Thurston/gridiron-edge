# tests/unit/viz/test_predictions.py

"""Tests for visualization prediction-frame assembly."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
from pandas import DataFrame

from gridiron_edge.ratings.elo.predict import (
    EloPredictionStatus,
)
from gridiron_edge.viz.predictions import (
    build_predictions_df,
)


def _domain_predictions() -> DataFrame:
    """Create ready and missing-Elo domain prediction rows."""
    return DataFrame(
        {
            "WEEK_NUM": [1, 1],
            "GAME_DAY_OF_WEEK": [
                "Sunday",
                "Sunday",
            ],
            "GAME_DATE": [
                "2026-09-06",
                "2026-09-06",
            ],
            "AWAY_TEAM": [
                "Kansas City Chiefs",
                "Baltimore Ravens",
            ],
            "HOME_TEAM": [
                "Los Angeles Chargers",
                "Buffalo Bills",
            ],
            "GAMETIME": [
                "20:20:00",
                "13:00:00",
            ],
            "YEAR": [
                "2026-2027",
                "2026-2027",
            ],
            "GAME_ID": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
            ],
            "GAME_LOCATION": [
                "H",
                "N",
            ],
            "AWAY_TEAM_ELO": [
                1520.0,
                1510.0,
            ],
            "HOME_TEAM_ELO": [
                1480.0,
                pd.NA,
            ],
            "AWAY_WIN_PROB": pd.Series(
                [
                    0.547835,
                    pd.NA,
                ],
                dtype="Float64",
            ),
            "HOME_WIN_PROB": pd.Series(
                [
                    0.452165,
                    pd.NA,
                ],
                dtype="Float64",
            ),
            "PREDICTION_STATUS": pd.Series(
                [
                    EloPredictionStatus.READY.value,
                    (EloPredictionStatus.MISSING_HOME_ELO.value),
                ],
                dtype="string",
            ),
        }
    )


@patch("gridiron_edge.viz.predictions.predict_elo_for_week")
def test_builder_delegates_to_domain_prediction(
    mock_predict: MagicMock,
) -> None:
    mock_predict.return_value = _domain_predictions()

    result = build_predictions_df(
        year="2026-2027",
        week=1,
    )

    mock_predict.assert_called_once_with(
        year="2026-2027",
        week=1,
        repo=None,
    )

    assert len(result) == 2


@patch("gridiron_edge.viz.predictions.predict_elo_for_week")
def test_builder_preserves_numeric_probabilities(
    mock_predict: MagicMock,
) -> None:
    mock_predict.return_value = _domain_predictions()

    result = build_predictions_df(
        year="2026-2027",
        week=1,
    )

    ready = result.loc[result["GAME_ID"] == "2026_01_KC_LAC"].iloc[0]

    assert float(ready["AWAY_WIN_PROB"]) == 0.547835
    assert float(ready["HOME_WIN_PROB"]) == 0.452165
    assert (float(ready["AWAY_WIN_PROB"]) + float(ready["HOME_WIN_PROB"])) == 1.0


@patch("gridiron_edge.viz.predictions.predict_elo_for_week")
def test_builder_adds_formatted_display_probabilities(
    mock_predict: MagicMock,
) -> None:
    mock_predict.return_value = _domain_predictions()

    result = build_predictions_df(
        year="2026-2027",
        week=1,
    )

    ready = result.loc[result["GAME_ID"] == "2026_01_KC_LAC"].iloc[0]

    assert ready["AWAY_TEAM_WIN_PROB"] == "54.8 %"
    assert ready["HOME_TEAM_WIN_PROB"] == "45.2 %"


@patch("gridiron_edge.viz.predictions.predict_elo_for_week")
def test_missing_elo_row_remains_visible(
    mock_predict: MagicMock,
) -> None:
    mock_predict.return_value = _domain_predictions()

    result = build_predictions_df(
        year="2026-2027",
        week=1,
    )

    missing = result.loc[result["GAME_ID"] == "2026_01_BAL_BUF"].iloc[0]

    assert len(result) == 2
    assert missing["PREDICTION_STATUS"] == (EloPredictionStatus.MISSING_HOME_ELO.value)
    assert pd.isna(missing["HOME_TEAM_ELO"])
    assert pd.isna(missing["AWAY_WIN_PROB"])
    assert pd.isna(missing["HOME_WIN_PROB"])
    assert pd.isna(missing["AWAY_TEAM_WIN_PROB"])
    assert pd.isna(missing["HOME_TEAM_WIN_PROB"])


@patch("gridiron_edge.viz.predictions.predict_elo_for_week")
def test_neutral_site_identity_is_preserved(
    mock_predict: MagicMock,
) -> None:
    mock_predict.return_value = _domain_predictions()

    result = build_predictions_df(
        year="2026-2027",
        week=1,
    )

    neutral = result.loc[result["GAME_ID"] == "2026_01_BAL_BUF"].iloc[0]

    assert neutral["GAME_LOCATION"] == "N"
    assert neutral["AWAY_TEAM"] == "Baltimore Ravens"
    assert neutral["HOME_TEAM"] == "Buffalo Bills"


@patch("gridiron_edge.viz.predictions.predict_elo_for_week")
def test_empty_domain_result_remains_empty(
    mock_predict: MagicMock,
) -> None:
    expected = _domain_predictions().iloc[0:0].copy()
    mock_predict.return_value = expected

    result = build_predictions_df(
        year="2026-2027",
        week=1,
    )

    assert result.empty
    pd.testing.assert_frame_equal(
        result,
        expected,
    )


@patch("gridiron_edge.viz.predictions.predict_elo_for_week")
def test_builder_does_not_mutate_domain_result(
    mock_predict: MagicMock,
) -> None:
    predictions = _domain_predictions()
    original = predictions.copy(deep=True)
    mock_predict.return_value = predictions

    build_predictions_df(
        year="2026-2027",
        week=1,
    )

    pd.testing.assert_frame_equal(
        predictions,
        original,
    )

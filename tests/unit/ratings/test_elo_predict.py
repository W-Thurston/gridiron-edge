# tests/unit/ratings/test_elo_predict.py

"""Tests for canonical schedule-to-Elo prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.writers import (
    write_csv,
)
from gridiron_edge.ratings.elo.predict import (
    EloPredictionStatus,
    format_elo_prediction_percentages,
    predict_elo_for_week,
    predict_schedule_with_elo,
)


def _schedule() -> DataFrame:
    """Create focused schedule rows with location context."""
    return DataFrame(
        {
            "WEEK_NUM": [1, 1, 2],
            "GAME_DAY_OF_WEEK": [
                "Sunday",
                "Sunday",
                "Thursday",
            ],
            "GAME_DATE": [
                "2026-09-06",
                "2026-09-06",
                "2026-09-10",
            ],
            "AWAY_TEAM": [
                "Kansas City Chiefs",
                "Baltimore Ravens",
                "Green Bay Packers",
            ],
            "HOME_TEAM": [
                "Los Angeles Chargers",
                "Buffalo Bills",
                "Chicago Bears",
            ],
            "GAMETIME": [
                "20:20:00",
                "13:00:00",
                "20:15:00",
            ],
            "YEAR": [
                "2026-2027",
                "2026-2027",
                "2026-2027",
            ],
            "GAME_ID": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
                "2026_02_GB_CHI",
            ],
            "GAME_LOCATION": [
                "H",
                "N",
                "H",
            ],
        }
    )


def _elo_state() -> DataFrame:
    """Create Elo state with one intentionally missing team."""
    return DataFrame(
        {
            "NFL_TEAM": [
                "Kansas City Chiefs",
                "Los Angeles Chargers",
                "Baltimore Ravens",
            ],
            "NFL_YEAR": [
                "2026-2027",
                "2026-2027",
                "2026-2027",
            ],
            "NFL_WEEK": [1, 1, 1],
            "ELO": [
                1520.0,
                1480.0,
                1510.0,
            ],
        }
    )


def test_ready_game_has_numeric_complementary_probabilities() -> None:
    result = predict_schedule_with_elo(
        _schedule(),
        _elo_state(),
        year="2026-2027",
        week=1,
    )

    ready = result.loc[result["GAME_ID"] == "2026_01_KC_LAC"].iloc[0]

    assert ready["PREDICTION_STATUS"] == (EloPredictionStatus.READY.value)
    assert pd.notna(ready["AWAY_WIN_PROB"])
    assert pd.notna(ready["HOME_WIN_PROB"])
    assert float(ready["AWAY_WIN_PROB"]) > 0.0
    assert float(ready["HOME_WIN_PROB"]) > 0.0
    assert isinstance(
        ready["HOME_WIN_PROB"],
        float,
    )
    assert (ready["AWAY_WIN_PROB"] + ready["HOME_WIN_PROB"]) == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_missing_home_elo_preserves_schedule_row() -> None:
    result = predict_schedule_with_elo(
        _schedule(),
        _elo_state(),
        year="2026-2027",
        week=1,
    )

    assert len(result) == 2

    missing = result.loc[result["GAME_ID"] == "2026_01_BAL_BUF"].iloc[0]

    assert missing["PREDICTION_STATUS"] == (EloPredictionStatus.MISSING_HOME_ELO.value)
    assert missing["AWAY_TEAM_ELO"] == 1510.0
    assert pd.isna(missing["HOME_TEAM_ELO"])
    assert pd.isna(missing["AWAY_WIN_PROB"])
    assert pd.isna(missing["HOME_WIN_PROB"])


def test_missing_both_elos_is_explicit() -> None:
    result = predict_schedule_with_elo(
        _schedule(),
        _elo_state().iloc[0:0].copy(),
        year="2026-2027",
        week=1,
    )

    assert len(result) == 2
    assert set(result["PREDICTION_STATUS"]) == {EloPredictionStatus.MISSING_BOTH_ELO.value}


def test_neutral_site_identity_is_preserved() -> None:
    result = predict_schedule_with_elo(
        _schedule(),
        _elo_state(),
        year="2026-2027",
        week=1,
    )

    neutral = result.loc[result["GAME_ID"] == "2026_01_BAL_BUF"].iloc[0]

    assert neutral["GAME_LOCATION"] == "N"
    assert neutral["AWAY_TEAM"] == "Baltimore Ravens"
    assert neutral["HOME_TEAM"] == "Buffalo Bills"


def test_only_requested_week_is_returned() -> None:
    result = predict_schedule_with_elo(
        _schedule(),
        _elo_state(),
        year="2026-2027",
        week=1,
    )

    assert set(result["WEEK_NUM"]) == {1}
    assert "2026_02_GB_CHI" not in set(result["GAME_ID"])


def test_schedule_order_is_preserved() -> None:
    schedule = _schedule().iloc[[1, 0, 2]].reset_index(drop=True)

    result = predict_schedule_with_elo(
        schedule,
        _elo_state(),
        year="2026-2027",
        week=1,
    )

    assert result["GAME_ID"].tolist() == [
        "2026_01_BAL_BUF",
        "2026_01_KC_LAC",
    ]


def test_inputs_are_not_mutated() -> None:
    schedule = _schedule()
    elo_state = _elo_state()

    schedule_original = schedule.copy(deep=True)
    elo_original = elo_state.copy(deep=True)

    predict_schedule_with_elo(
        schedule,
        elo_state,
        year="2026-2027",
        week=1,
    )

    pd.testing.assert_frame_equal(
        schedule,
        schedule_original,
    )
    pd.testing.assert_frame_equal(
        elo_state,
        elo_original,
    )


def test_rejects_duplicate_elo_identity() -> None:
    elo_state = pd.concat(
        [
            _elo_state(),
            _elo_state().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Elo state contains duplicate identities",
    ):
        predict_schedule_with_elo(
            _schedule(),
            elo_state,
            year="2026-2027",
            week=1,
        )


@pytest.mark.parametrize(
    ("input_name", "column", "message"),
    [
        (
            "schedule",
            "GAME_ID",
            "Schedule is missing required columns: GAME_ID",
        ),
        (
            "elo",
            "ELO",
            "Elo state is missing required columns: ELO",
        ),
    ],
)
def test_rejects_missing_required_columns(
    input_name: str,
    column: str,
    message: str,
) -> None:
    schedule = _schedule()
    elo_state = _elo_state()

    if input_name == "schedule":
        schedule = schedule.drop(columns=[column])
    else:
        elo_state = elo_state.drop(columns=[column])

    with pytest.raises(
        ValueError,
        match=message,
    ):
        predict_schedule_with_elo(
            schedule,
            elo_state,
            year="2026-2027",
            week=1,
        )


def test_file_loading_entrypoint_uses_registered_datasets(
    tmp_path: Path,
) -> None:
    write_csv(
        tmp_path,
        "schedule_upcoming",
        _schedule(),
    )
    write_csv(
        tmp_path,
        "elo_state",
        _elo_state(),
    )

    result = predict_elo_for_week(
        year="2026-2027",
        week=1,
        repo=tmp_path,
    )

    assert len(result) == 2
    assert "AWAY_WIN_PROB" in result.columns
    assert "PREDICTION_STATUS" in result.columns


def test_percentage_formatting_does_not_recalculate() -> None:
    predictions = predict_schedule_with_elo(
        _schedule(),
        _elo_state(),
        year="2026-2027",
        week=1,
    )
    original = predictions.copy(deep=True)

    formatted = format_elo_prediction_percentages(predictions)

    pd.testing.assert_series_equal(
        formatted["AWAY_WIN_PROB"],
        original["AWAY_WIN_PROB"],
    )
    pd.testing.assert_series_equal(
        formatted["HOME_WIN_PROB"],
        original["HOME_WIN_PROB"],
    )
    pd.testing.assert_frame_equal(
        predictions,
        original,
    )


def test_percentage_formatting_preserves_missing_values() -> None:
    predictions = predict_schedule_with_elo(
        _schedule(),
        _elo_state(),
        year="2026-2027",
        week=1,
    )

    formatted = format_elo_prediction_percentages(predictions)

    missing = formatted.loc[formatted["GAME_ID"] == "2026_01_BAL_BUF"].iloc[0]

    assert pd.isna(missing["AWAY_TEAM_WIN_PROB"])
    assert pd.isna(missing["HOME_TEAM_WIN_PROB"])

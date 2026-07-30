# tests / unit / models / game_prediction / test_weekly_spread_product.py

"""Tests for calibrated spread attachment to weekly win products."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.models.game_prediction.weekly_spread_product import (
    SpreadCalibration,
    WeeklySpreadStatus,
    attach_derived_spreads,
    parse_spread_calibration,
)
from gridiron_edge.models.game_prediction.weekly_win_product import WeeklyWinStatus


def _win_product() -> DataFrame:
    return DataFrame(
        {
            "game_id": ["game-1", "game-2", "game-3"],
            "away_team": ["Away One", "Away Two", "Away Three"],
            "home_team": ["Home One", "Home Two", "Home Three"],
            "neutral_site": [False, True, False],
            "win_status": [
                WeeklyWinStatus.AVAILABLE.value,
                WeeklyWinStatus.AVAILABLE.value,
                WeeklyWinStatus.FORECAST_MISSING.value,
            ],
            "home_win_prob": [0.75, 0.25, pd.NA],
            "win_model_name": ["win_prob", "win_prob", pd.NA],
            "win_model_type": ["elo", "xgboost", pd.NA],
            "win_event_id": ["event-1", "event-2", pd.NA],
        }
    )


def _calibration(
    model_type: str,
    *,
    sigma: float = 13.6,
    margin_std: float = 13.89,
) -> SpreadCalibration:
    return SpreadCalibration(
        model_name="win_prob",
        model_type=model_type,
        calibration_key=f"win_prob_{model_type}",
        sigma=sigma,
        margin_std=margin_std,
        updated_at="2026-07-30T12:00:00+00:00",
    )


def test_spread_sign_convention_is_home_line() -> None:
    product = attach_derived_spreads(
        _win_product(),
        {
            ("win_prob", "elo"): _calibration("elo"),
            ("win_prob", "xgboost"): _calibration("xgboost"),
        },
    )

    home_favored = product.loc[product["game_id"] == "game-1"].iloc[0]
    away_favored = product.loc[product["game_id"] == "game-2"].iloc[0]
    assert home_favored["model_spread"] < 0
    assert away_favored["model_spread"] > 0


def test_selected_win_model_uses_its_exact_calibration() -> None:
    product = attach_derived_spreads(
        _win_product(),
        {
            ("win_prob", "elo"): _calibration("elo", sigma=10.0),
            ("win_prob", "xgboost"): _calibration("xgboost", sigma=20.0),
        },
    )

    elo = product.loc[product["game_id"] == "game-1"].iloc[0]
    xgboost = product.loc[product["game_id"] == "game-2"].iloc[0]
    assert abs(float(xgboost["model_spread"])) == pytest.approx(2 * abs(float(elo["model_spread"])))
    assert elo["spread_model_type"] == "elo"
    assert xgboost["spread_model_type"] == "xgboost"


def test_missing_calibration_does_not_fabricate_spread() -> None:
    product = attach_derived_spreads(
        _win_product(),
        {("win_prob", "elo"): _calibration("elo")},
    )

    missing = product.loc[product["game_id"] == "game-2"].iloc[0]
    assert missing["spread_status"] == (WeeklySpreadStatus.CALIBRATION_UNAVAILABLE.value)
    assert pd.isna(missing["model_spread"])
    assert pd.isna(missing["spread_uncertainty"])


def test_spread_provenance_identifies_event_and_calibration() -> None:
    product = attach_derived_spreads(
        _win_product(),
        {("win_prob", "elo"): _calibration("elo")},
    )

    row = product.loc[product["game_id"] == "game-1"].iloc[0]
    assert row["spread_source_event_id"] == "event-1"
    assert row["spread_model_name"] == "win_prob"
    assert row["spread_model_type"] == "elo"
    assert row["spread_calibration_key"] == "win_prob_elo"
    assert row["spread_calibration_updated_at"] == ("2026-07-30T12:00:00+00:00")
    assert row["spread_uncertainty"] == pytest.approx(13.89)


def test_win_unavailable_row_retains_explicit_blocker() -> None:
    product = attach_derived_spreads(
        _win_product(),
        {("win_prob", "elo"): _calibration("elo")},
    )

    row = product.loc[product["game_id"] == "game-3"].iloc[0]
    assert row["spread_status"] == WeeklySpreadStatus.WIN_UNAVAILABLE.value
    assert pd.isna(row["model_spread"])


def test_schedule_completeness_and_order_are_preserved() -> None:
    source = _win_product()
    product = attach_derived_spreads(
        source,
        {("win_prob", "elo"): _calibration("elo")},
    )

    assert len(product) == len(source)
    assert product["game_id"].tolist() == source["game_id"].tolist()
    assert product["neutral_site"].tolist() == source["neutral_site"].tolist()
    pd.testing.assert_frame_equal(source, _win_product())


def test_calibration_identity_mismatch_is_rejected() -> None:
    wrong = SpreadCalibration(
        model_name="win_prob",
        model_type="random_forest",
        calibration_key="win_prob_random_forest",
        sigma=13.0,
        margin_std=13.5,
        updated_at="2026-07-30T12:00:00+00:00",
    )

    with pytest.raises(ValueError, match="model_type does not match"):
        attach_derived_spreads(
            _win_product(),
            {("win_prob", "elo"): wrong},
        )


def test_strict_parser_requires_complete_persisted_payload() -> None:
    registry = {
        "win_prob_elo": {
            "sigma": 13.6,
            "margin_std": 13.89,
            "updated_at": "2026-07-30T12:00:00+00:00",
        },
        "win_prob_xgboost": {
            "sigma": 11.4,
            "updated_at": "2026-07-30T12:00:00+00:00",
        },
    }

    elo = parse_spread_calibration(
        registry,
        model_name="win_prob",
        model_type="elo",
    )
    xgboost = parse_spread_calibration(
        registry,
        model_name="win_prob",
        model_type="xgboost",
    )

    assert elo is not None
    assert elo.sigma == pytest.approx(13.6)
    assert xgboost is None


def test_no_fallback_calibration_is_used() -> None:
    calibration = parse_spread_calibration(
        {},
        model_name="win_prob",
        model_type="elo",
    )
    assert calibration is None

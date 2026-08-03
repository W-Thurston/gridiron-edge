# tests/unit/models/game_prediction/test_weekly_game_product.py

"""Tests for projected scores and final weekly game-product validation."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.models.game_prediction.product_validation import (
    validate_weekly_game_product,
)
from gridiron_edge.models.game_prediction.weekly_game_product import (
    ProjectedScoreStatus,
    attach_projected_scores,
    build_weekly_game_product,
)
from gridiron_edge.models.game_prediction.weekly_spread_product import WeeklySpreadStatus
from gridiron_edge.models.game_prediction.weekly_total_product import WeeklyTotalStatus
from gridiron_edge.models.game_prediction.weekly_win_product import WeeklyWinStatus


def _product() -> DataFrame:
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027", "2026-2027"],
            "week": [8, 8, 8],
            "game_id": ["game-1", "game-2", "game-3"],
            "away_team": ["Away One", "Away Two", "Away Three"],
            "home_team": ["Home One", "Home Two", "Home Three"],
            "neutral_site": [False, True, False],
            "win_status": [WeeklyWinStatus.AVAILABLE.value] * 3,
            "win_selection_status": ["selected"] * 3,
            "away_win_prob": [0.40, 0.50, 0.55],
            "home_win_prob": [0.60, 0.50, 0.45],
            "win_model_name": ["win_prob"] * 3,
            "win_model_type": ["elo"] * 3,
            "win_event_id": ["win-1", "win-2", "win-3"],
            "win_run_id": ["run-1"] * 3,
            "win_generated_at": ["2026-10-20T12:00:00+00:00"] * 3,
            "win_role": ["live"] * 3,
            "spread_status": [
                WeeklySpreadStatus.AVAILABLE.value,
                WeeklySpreadStatus.CALIBRATION_UNAVAILABLE.value,
                WeeklySpreadStatus.AVAILABLE.value,
            ],
            "model_spread": [-3.0, pd.NA, 2.5],
            "spread_uncertainty": [13.5, pd.NA, 13.5],
            "spread_source_event_id": ["win-1", pd.NA, "win-3"],
            "spread_model_name": ["win_prob", pd.NA, "win_prob"],
            "spread_model_type": ["elo", pd.NA, "elo"],
            "spread_calibration_key": ["win_prob_elo", pd.NA, "win_prob_elo"],
            "spread_calibration_updated_at": ["2026-07-30", pd.NA, "2026-07-30"],
            "total_status": [
                WeeklyTotalStatus.AVAILABLE.value,
                WeeklyTotalStatus.AVAILABLE.value,
                WeeklyTotalStatus.FORECAST_MISSING.value,
            ],
            "model_total": [44.0, 46.0, pd.NA],
            "total_uncertainty": [12.8, 12.8, pd.NA],
            "total_model_name": ["total", "total", pd.NA],
            "total_model_type": ["xgboost", "xgboost", pd.NA],
            "total_event_id": ["total-1", "total-2", pd.NA],
            "total_run_id": ["run-1", "run-1", pd.NA],
            "total_generated_at": [
                "2026-10-20T12:00:00+00:00",
                "2026-10-20T12:00:00+00:00",
                pd.NaT,
            ],
            "total_role": ["live", "live", pd.NA],
            "total_selection_status": ["selected", "selected", "missing"],
            "total_uncertainty_trained_at": ["2026-07-01", "2026-07-01", pd.NA],
        }
    )


def test_projected_scores_reconcile_to_total_and_spread() -> None:
    product = build_weekly_game_product(_product())
    row = product.loc[product["game_id"] == "game-1"].iloc[0]

    assert row["projected_score_status"] == ProjectedScoreStatus.AVAILABLE.value
    assert row["projected_home_score"] == pytest.approx(23.5)
    assert row["projected_away_score"] == pytest.approx(20.5)
    assert row["projected_home_score"] + row["projected_away_score"] == pytest.approx(44.0)
    assert row["projected_away_score"] - row["projected_home_score"] == pytest.approx(-3.0)


def test_missing_inputs_produce_granular_status() -> None:
    product = attach_projected_scores(_product())

    spread_missing = product.loc[product["game_id"] == "game-2"].iloc[0]
    total_missing = product.loc[product["game_id"] == "game-3"].iloc[0]

    assert spread_missing["projected_score_status"] == (
        ProjectedScoreStatus.SPREAD_UNAVAILABLE.value
    )
    assert total_missing["projected_score_status"] == (ProjectedScoreStatus.TOTAL_UNAVAILABLE.value)
    assert pd.isna(spread_missing["projected_home_score"])
    assert pd.isna(total_missing["projected_away_score"])


def test_uncertainty_unavailable_total_still_projects_scores() -> None:
    source = _product().iloc[[0]].copy()
    source["total_status"] = WeeklyTotalStatus.UNCERTAINTY_UNAVAILABLE.value
    source["total_uncertainty"] = pd.NA
    source["total_uncertainty_trained_at"] = pd.NA

    product = build_weekly_game_product(source)

    assert product.iloc[0]["projected_score_status"] == (ProjectedScoreStatus.AVAILABLE.value)
    assert product.iloc[0]["projected_home_score"] == pytest.approx(23.5)


def test_complete_product_remains_one_row_per_game() -> None:
    source = _product()
    product = build_weekly_game_product(source)

    assert len(product) == len(source)
    assert product["game_id"].tolist() == source["game_id"].tolist()
    assert product["neutral_site"].tolist() == source["neutral_site"].tolist()


def test_validator_rejects_projected_total_mismatch() -> None:
    product = attach_projected_scores(_product())
    product.loc[0, "projected_home_score"] = 99.0

    with pytest.raises(ValueError, match="reconcile to model_total"):
        validate_weekly_game_product(product)


def test_validator_rejects_projected_spread_mismatch() -> None:
    product = attach_projected_scores(_product())
    product.loc[0, "projected_home_score"] = 21.0
    product.loc[0, "projected_away_score"] = 23.0

    with pytest.raises(ValueError, match="reconcile to model_spread"):
        validate_weekly_game_product(product)


def test_validator_rejects_scores_on_blocked_row() -> None:
    product = attach_projected_scores(_product())
    product.loc[1, "projected_home_score"] = 24.0

    with pytest.raises(ValueError, match="requires null fields"):
        validate_weekly_game_product(product)


def test_validator_rejects_spread_provenance_mismatch() -> None:
    product = attach_projected_scores(_product())
    product.loc[0, "spread_source_event_id"] = "different-win-event"

    with pytest.raises(ValueError, match="Spread source event"):
        validate_weekly_game_product(product)


def test_validator_rejects_invalid_win_probability_complements() -> None:
    product = attach_projected_scores(_product())
    product.loc[0, "away_win_prob"] = 0.50

    with pytest.raises(ValueError, match="must sum to 1"):
        validate_weekly_game_product(product)


def test_validator_rejects_total_uncertainty_status_conflict() -> None:
    product = attach_projected_scores(_product())
    product.loc[0, "total_status"] = WeeklyTotalStatus.UNCERTAINTY_UNAVAILABLE.value

    with pytest.raises(ValueError, match="requires null fields"):
        validate_weekly_game_product(product)


def test_validator_rejects_duplicate_game_ids() -> None:
    product = attach_projected_scores(_product())
    product.loc[1, "game_id"] = "game-1"

    with pytest.raises(ValueError, match="duplicate game IDs"):
        validate_weekly_game_product(product)


def test_both_missing_inputs_have_combined_status() -> None:
    source = _product().iloc[[2]].copy()
    source["spread_status"] = WeeklySpreadStatus.CALIBRATION_UNAVAILABLE.value
    source["model_spread"] = pd.NA
    source["spread_uncertainty"] = pd.NA

    product = attach_projected_scores(source)

    assert product.iloc[0]["projected_score_status"] == (
        ProjectedScoreStatus.SPREAD_AND_TOTAL_UNAVAILABLE.value
    )


def test_validator_rejects_stale_win_provenance_on_unavailable_row() -> None:
    product = attach_projected_scores(_product())
    product.loc[0, "win_status"] = WeeklyWinStatus.FORECAST_MISSING.value

    with pytest.raises(ValueError, match="Unavailable win requires null fields"):
        validate_weekly_game_product(product)


def test_validator_rejects_non_utc_total_generation_time() -> None:
    product = attach_projected_scores(_product())
    product.loc[0, "total_generated_at"] = "2026-10-20T12:00:00"

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        validate_weekly_game_product(product)

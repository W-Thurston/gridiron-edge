# tests/evaluation/test_metrics.py

"""Tests for evaluation metrics — Brier score, log loss, calibration."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.evaluation.metrics import (
    accuracy,
    brier_score,
    calibration_table,
    log_loss,
    summarise,
)

# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_score_perfect_predictor() -> None:
    """A perfect predictor scores 0.0."""
    probs = pd.Series([1.0, 1.0, 0.0, 0.0])
    outcomes = pd.Series([1.0, 1.0, 0.0, 0.0])
    assert brier_score(probs, outcomes) == pytest.approx(0.0)


def test_brier_score_always_wrong() -> None:
    """A predictor that is always maximally wrong scores 1.0."""
    probs = pd.Series([0.0, 0.0, 1.0, 1.0])
    outcomes = pd.Series([1.0, 1.0, 0.0, 0.0])
    assert brier_score(probs, outcomes) == pytest.approx(1.0)


def test_brier_score_random_predictor() -> None:
    """A predictor that always predicts 0.5 scores 0.25."""
    probs = pd.Series([0.5] * 100)
    outcomes = pd.Series([1.0] * 50 + [0.0] * 50)
    assert brier_score(probs, outcomes) == pytest.approx(0.25)


def test_brier_score_with_ties() -> None:
    """Ties (0.5 outcome) are included in Brier score calculation."""
    probs = pd.Series([0.5, 0.5])
    outcomes = pd.Series([0.5, 0.5])
    assert brier_score(probs, outcomes) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


def test_log_loss_perfect_predictor() -> None:
    """A confident correct predictor has near-zero log loss."""
    probs = pd.Series([0.999, 0.999, 0.001, 0.001])
    outcomes = pd.Series([1.0, 1.0, 0.0, 0.0])
    assert log_loss(probs, outcomes) == pytest.approx(0.0, abs=0.01)


def test_log_loss_excludes_ties() -> None:
    """Ties (outcome=0.5) are excluded from log loss."""
    probs = pd.Series([0.6, 0.5, 0.5])
    outcomes = pd.Series([1.0, 0.5, 0.5])
    # Only the first row is scored
    result = log_loss(probs, outcomes)
    expected = log_loss(pd.Series([0.6]), pd.Series([1.0]))
    assert result == pytest.approx(expected)


def test_log_loss_all_ties_returns_nan() -> None:
    """All ties returns NaN since log loss is undefined."""
    probs = pd.Series([0.5, 0.5])
    outcomes = pd.Series([0.5, 0.5])
    import math

    assert math.isnan(log_loss(probs, outcomes))


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def test_accuracy_all_correct() -> None:
    probs = pd.Series([0.9, 0.9, 0.1, 0.1])
    outcomes = pd.Series([1.0, 1.0, 0.0, 0.0])
    assert accuracy(probs, outcomes) == pytest.approx(1.0)


def test_accuracy_all_wrong() -> None:
    probs = pd.Series([0.9, 0.9, 0.1, 0.1])
    outcomes = pd.Series([0.0, 0.0, 1.0, 1.0])
    assert accuracy(probs, outcomes) == pytest.approx(0.0)


def test_accuracy_excludes_ties() -> None:
    probs = pd.Series([0.9, 0.5, 0.5])
    outcomes = pd.Series([1.0, 0.5, 0.5])
    result = accuracy(probs, outcomes)
    assert result == pytest.approx(1.0)


def test_accuracy_all_ties_returns_nan() -> None:
    import math

    probs = pd.Series([0.5, 0.5])
    outcomes = pd.Series([0.5, 0.5])
    assert math.isnan(accuracy(probs, outcomes))


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------


def _make_eval_df(
    probs: list[float],
    outcomes: list[float],
    model: str = "elo_v1",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_version": model,
            "season": "2025-2026",
            "week": 1,
            "game_id": [f"g{i}" for i in range(len(probs))],
            "away_team": "KC",
            "home_team": "LAC",
            "away_win_prob": probs,
            "home_win_prob": [1.0 - p for p in probs],
            "away_team_won": outcomes,
        }
    )


def test_calibration_table_bucket_assignment() -> None:
    """Predictions in [0.5, 0.6) land in the correct bucket."""
    df = _make_eval_df([0.55, 0.56, 0.57], [1.0, 1.0, 0.0])
    table = calibration_table(df, n_buckets=10)
    bucket = table[table["bucket_low"] == 0.5]
    assert len(bucket) == 1
    assert bucket.iloc[0]["n_games"] == 3


def test_calibration_table_excludes_ties() -> None:
    """Tie outcomes are excluded from calibration."""
    df = _make_eval_df([0.55, 0.55, 0.55], [1.0, 0.5, 0.0])
    table = calibration_table(df, n_buckets=10)
    bucket = table[table["bucket_low"] == 0.5]
    assert bucket.iloc[0]["n_games"] == 2  # tie excluded


def test_calibration_table_empty_returns_empty() -> None:
    df = _make_eval_df([], [])
    result = calibration_table(df)
    assert result.empty


def test_calibration_table_error_column() -> None:
    """error = actual_win_rate - mean_predicted_prob."""
    df = _make_eval_df([0.55, 0.55], [1.0, 0.0])
    table = calibration_table(df, n_buckets=10)
    bucket = table[table["bucket_low"] == 0.5].iloc[0]
    expected_error = bucket["actual_win_rate"] - bucket["mean_predicted_prob"]
    assert bucket["error"] == pytest.approx(expected_error, abs=1e-4)


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------


def test_summarise_overall() -> None:
    df = _make_eval_df(
        [0.9, 0.9, 0.1, 0.1],
        [1.0, 1.0, 0.0, 0.0],
    )
    result = summarise(df)
    assert len(result) == 1
    assert result.iloc[0]["n_games"] == 4
    assert result.iloc[0]["accuracy"] == pytest.approx(1.0)
    assert result.iloc[0]["brier_score"] == pytest.approx(
        brier_score(pd.Series([0.9, 0.9, 0.1, 0.1]), pd.Series([1.0, 1.0, 0.0, 0.0]))
    )


def test_summarise_group_by_season() -> None:
    df = pd.concat(
        [
            _make_eval_df([0.9, 0.1], [1.0, 0.0]).assign(season="2024-2025"),
            _make_eval_df([0.6, 0.4], [1.0, 0.0]).assign(season="2025-2026"),
        ],
        ignore_index=True,
    )
    result = summarise(df, group_by="season")
    assert len(result) == 2
    assert set(result["season"].tolist()) == {"2024-2025", "2025-2026"}


def test_summarise_empty_returns_empty() -> None:
    result = summarise(pd.DataFrame())
    assert result.empty

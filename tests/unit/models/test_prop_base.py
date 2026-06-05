# tests/unit/models/test_prop_base.py
"""Tests for gridiron_edge.models.prop_prediction.base."""

from __future__ import annotations

import numpy as np
from numpy import ndarray
import pytest

from gridiron_edge.models.prop_prediction.base import (
    _MIN_ATTEMPTS,
    UNIVERSAL_FEATURE_COLS,
    PropModelMetadata,
    PropModelSpec,
    PropPrediction,
    evaluate_props,
)


class TestPropModelSpec:
    def test_required_fields(self) -> None:
        spec = PropModelSpec(
            name="qb_pass_yards",
            target_col="passing_yards",
            position_filter=["QB"],
        )
        assert spec.name == "qb_pass_yards"
        assert spec.target_col == "passing_yards"
        assert spec.position_filter == ["QB"]

    def test_description_defaults_empty(self) -> None:
        spec = PropModelSpec(
            name="test",
            target_col="col",
            position_filter=["QB"],
        )
        assert spec.description == ""

    def test_frozen(self) -> None:
        spec = PropModelSpec(
            name="test",
            target_col="col",
            position_filter=["QB"],
        )
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore[misc]


class TestPropModelMetadata:
    def test_required_fields(self) -> None:
        meta = PropModelMetadata(
            model_name="qb_pass_yards",
            trained_at="2024-01-01T00:00:00Z",
            target_col="passing_yards",
            holdout_mae=25.0,
            holdout_rmse=32.0,
            holdout_r2=0.45,
        )
        assert meta.model_name == "qb_pass_yards"
        assert meta.holdout_mae == 25.0

    def test_defaults(self) -> None:
        meta = PropModelMetadata(
            model_name="test",
            trained_at="now",
            target_col="col",
            holdout_mae=0.0,
            holdout_rmse=0.0,
            holdout_r2=0.0,
        )
        assert meta.training_seasons == []
        assert meta.holdout_seasons == []
        assert meta.parameters == {}
        assert meta.feature_columns == []
        assert meta.n_train_rows == 0
        assert meta.notes == ""


class TestPropPrediction:
    def test_fields(self) -> None:
        pred = PropPrediction(
            player_id="00-001",
            player_name="P.Mahomes",
            game_id="2024_01_BAL_KC",
            season=2024,
            week=1,
            predicted=275.0,
            actual=291.0,
        )
        assert pred.predicted == 275.0
        assert pred.actual == 291.0

    def test_actual_defaults_none(self) -> None:
        pred = PropPrediction(
            player_id="00-001",
            player_name="P.Mahomes",
            game_id="2024_01_BAL_KC",
            season=2024,
            week=1,
            predicted=275.0,
        )
        assert pred.actual is None


class TestEvaluateProps:
    def test_perfect_prediction(self) -> None:
        y: ndarray = np.array([100.0, 200.0, 300.0])
        metrics: dict[str, float] = evaluate_props(y, y)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["median_ae"] == pytest.approx(0.0)

    def test_known_values(self) -> None:
        y_true: ndarray = np.array([250.0, 300.0, 200.0, 275.0, 320.0])
        y_pred: ndarray = np.array([240.0, 310.0, 190.0, 260.0, 305.0])
        metrics: dict[str, float] = evaluate_props(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(12.0)
        assert metrics["rmse"] == pytest.approx(12.247, abs=0.01)
        assert metrics["r2"] == pytest.approx(0.914, abs=0.01)

    def test_constant_prediction(self) -> None:
        """Predicting the mean should give R²=0."""
        y_true: ndarray = np.array([100.0, 200.0, 300.0])
        y_pred: ndarray = np.array([200.0, 200.0, 200.0])
        metrics: dict[str, float] = evaluate_props(y_true, y_pred)
        assert metrics["r2"] == pytest.approx(0.0)

    def test_worse_than_mean(self) -> None:
        """Terrible predictions should give negative R²."""
        y_true: ndarray = np.array([100.0, 200.0, 300.0])
        y_pred: ndarray = np.array([500.0, 500.0, 500.0])
        metrics: dict[str, float] = evaluate_props(y_true, y_pred)
        assert metrics["r2"] < 0.0

    def test_single_value(self) -> None:
        """Single observation — R² should be 0 (ss_tot = 0)."""
        metrics: dict[str, float] = evaluate_props(np.array([100.0]), np.array([110.0]))
        assert metrics["mae"] == pytest.approx(10.0)
        assert metrics["r2"] == pytest.approx(0.0)


class TestMinAttempts:
    def test_passing_yards_threshold(self) -> None:
        col, min_val = _MIN_ATTEMPTS["passing_yards"]
        assert col == "attempts"
        assert min_val == 10

    def test_rushing_yards_threshold(self) -> None:
        col, min_val = _MIN_ATTEMPTS["rushing_yards"]
        assert col == "carries"
        assert min_val == 5

    def test_receiving_yards_threshold(self) -> None:
        col, min_val = _MIN_ATTEMPTS["receiving_yards"]
        assert col == "targets"
        assert min_val == 2


class TestUniversalFeatures:
    def test_count(self) -> None:
        assert len(UNIVERSAL_FEATURE_COLS) == 132

    def test_has_rolling(self) -> None:
        rolling: list[str] = [c for c in UNIVERSAL_FEATURE_COLS if "_L3_" in c or "_L6_" in c]
        assert len(rolling) == 92

    def test_has_matchup(self) -> None:
        matchup: list[str] = [
            c for c in UNIVERSAL_FEATURE_COLS if c.startswith("opp_") and "allowed" in c
        ]
        assert len(matchup) == 28

    def test_has_context(self) -> None:
        assert "implied_team_total" in UNIVERSAL_FEATURE_COLS
        assert "spread_line" in UNIVERSAL_FEATURE_COLS
        assert "is_home" in UNIVERSAL_FEATURE_COLS
        assert "roof_dome" in UNIVERSAL_FEATURE_COLS
        assert "TEMP_F" in UNIVERSAL_FEATURE_COLS
        assert "WIND_SPEED_MPH" in UNIVERSAL_FEATURE_COLS
        assert "rest_days" in UNIVERSAL_FEATURE_COLS

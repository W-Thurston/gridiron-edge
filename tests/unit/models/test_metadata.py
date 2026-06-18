# tests/unit/models/test_metadata.py

"""Tests for shared model metadata dataclasses.

Covers:
    - BaseModelMetadata: construction, defaults, kw_only enforcement,
      field overrides.
    - GameModelMetadata: classification and regression construction paths,
      NaN defaults, BaseModelMetadata inheritance.
    - PropModelMetadata: construction, required prop fields enforced,
      BaseModelMetadata inheritance.
"""

from __future__ import annotations

import math

import pytest

from gridiron_edge.models.artifact import BaseModelMetadata
from gridiron_edge.models.game_prediction.base import GameModelMetadata
from gridiron_edge.models.prop_prediction.base import PropModelMetadata

# ---------------------------------------------------------------------------
# BaseModelMetadata
# ---------------------------------------------------------------------------


class TestBaseModelMetadata:
    """Construction, defaults, and kw_only enforcement on the shared base."""

    def test_construction_with_required_fields(self) -> None:
        meta = BaseModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-18T00:00:00+00:00",
        )
        assert meta.model_name == "win_prob"
        assert meta.model_type == "random_forest"
        assert meta.task == "classification"
        assert meta.trained_at == "2026-06-18T00:00:00+00:00"

    def test_default_values(self) -> None:
        meta = BaseModelMetadata(
            model_name="win_prob",
            model_type="logistic",
            task="classification",
            trained_at="2026-06-18T00:00:00+00:00",
        )
        assert meta.schema_version == 2
        assert meta.training_seasons == []
        assert meta.holdout_seasons == []
        assert meta.parameters == {}
        assert meta.feature_columns == []
        assert meta.n_train_rows == 0
        assert meta.n_holdout_rows == 0
        assert meta.notes == ""

    def test_kw_only_enforced(self) -> None:
        with pytest.raises(TypeError):
            BaseModelMetadata(  # type: ignore[misc]
                "win_prob",
                "random_forest",
                "classification",
                "2026-06-18T00:00:00+00:00",
            )

    def test_field_overrides(self) -> None:
        meta = BaseModelMetadata(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            task="regression",
            trained_at="2026-06-18T00:00:00+00:00",
            schema_version=2,
            training_seasons=["1999-2000", "2000-2001"],
            holdout_seasons=["2023-2024"],
            parameters={"alpha": 0.1, "l1_ratio": 0.5},
            feature_columns=["a", "b", "c"],
            n_train_rows=1000,
            n_holdout_rows=200,
            notes="initial run",
        )
        assert meta.training_seasons == ["1999-2000", "2000-2001"]
        assert meta.holdout_seasons == ["2023-2024"]
        assert meta.parameters == {"alpha": 0.1, "l1_ratio": 0.5}
        assert meta.feature_columns == ["a", "b", "c"]
        assert meta.n_train_rows == 1000
        assert meta.n_holdout_rows == 200
        assert meta.notes == "initial run"


# ---------------------------------------------------------------------------
# GameModelMetadata
# ---------------------------------------------------------------------------


class TestGameModelMetadata:
    """Game-side subclass: classification + regression metric fields."""

    def test_inherits_base(self) -> None:
        meta = GameModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-18T00:00:00+00:00",
        )
        assert isinstance(meta, BaseModelMetadata)

    def test_classification_construction(self) -> None:
        meta = GameModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-18T00:00:00+00:00",
            holdout_brier=0.220,
            holdout_ece=0.018,
            holdout_auc=0.762,
            holdout_log_loss=0.628,
            holdout_accuracy=0.681,
        )
        # Classification metrics populated
        assert meta.holdout_brier == pytest.approx(0.220)
        assert meta.holdout_ece == pytest.approx(0.018)
        assert meta.holdout_auc == pytest.approx(0.762)
        assert meta.holdout_log_loss == pytest.approx(0.628)
        assert meta.holdout_accuracy == pytest.approx(0.681)
        # Regression metrics remain NaN
        assert math.isnan(meta.holdout_mae)
        assert math.isnan(meta.holdout_rmse)
        assert math.isnan(meta.holdout_r2)

    def test_regression_construction(self) -> None:
        meta = GameModelMetadata(
            model_name="total",
            model_type="xgboost",
            task="regression",
            trained_at="2026-06-18T00:00:00+00:00",
            holdout_mae=8.2,
            holdout_rmse=10.5,
            holdout_r2=0.31,
        )
        # Regression metrics populated
        assert meta.holdout_mae == pytest.approx(8.2)
        assert meta.holdout_rmse == pytest.approx(10.5)
        assert meta.holdout_r2 == pytest.approx(0.31)
        # Classification metrics remain NaN
        assert math.isnan(meta.holdout_brier)
        assert math.isnan(meta.holdout_ece)
        assert math.isnan(meta.holdout_auc)
        assert math.isnan(meta.holdout_log_loss)
        assert math.isnan(meta.holdout_accuracy)

    def test_default_metrics_are_nan(self) -> None:
        meta = GameModelMetadata(
            model_name="win_prob",
            model_type="xgboost",
            task="classification",
            trained_at="2026-06-18T00:00:00+00:00",
        )
        for field_name in (
            "holdout_brier",
            "holdout_ece",
            "holdout_auc",
            "holdout_log_loss",
            "holdout_accuracy",
            "holdout_mae",
            "holdout_rmse",
            "holdout_r2",
        ):
            assert math.isnan(getattr(meta, field_name)), field_name


# ---------------------------------------------------------------------------
# PropModelMetadata
# ---------------------------------------------------------------------------


class TestPropModelMetadata:
    """Prop-side subclass: required regression metrics + target_col."""

    def test_inherits_base(self) -> None:
        meta = PropModelMetadata(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            task="regression",
            trained_at="2026-06-18T00:00:00+00:00",
            target_col="passing_yards",
            holdout_mae=58.0,
            holdout_rmse=72.6,
            holdout_r2=0.071,
        )
        assert isinstance(meta, BaseModelMetadata)

    def test_construction(self) -> None:
        meta = PropModelMetadata(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            task="regression",
            trained_at="2026-06-18T00:00:00+00:00",
            target_col="passing_yards",
            holdout_mae=58.0,
            holdout_rmse=72.6,
            holdout_r2=0.071,
            training_seasons=["1999-2000", "2000-2001"],
            holdout_seasons=["2023-2024"],
            parameters={"alpha": 0.1, "l1_ratio": 0.5},
            feature_columns=["a", "b"],
            n_train_rows=5706,
            n_holdout_rows=1367,
        )
        assert meta.model_name == "qb_pass_yards"
        assert meta.target_col == "passing_yards"
        assert meta.holdout_mae == pytest.approx(58.0)
        assert meta.holdout_rmse == pytest.approx(72.6)
        assert meta.holdout_r2 == pytest.approx(0.071)
        assert meta.n_train_rows == 5706
        assert meta.n_holdout_rows == 1367

    def test_required_prop_fields_enforced(self) -> None:
        """Missing target_col / holdout metrics should raise."""
        with pytest.raises(TypeError):
            PropModelMetadata(  # type: ignore[call-arg]
                model_name="qb_pass_yards",
                model_type="elasticnet",
                task="regression",
                trained_at="2026-06-18T00:00:00+00:00",
                # Missing target_col, holdout_mae, holdout_rmse, holdout_r2
            )

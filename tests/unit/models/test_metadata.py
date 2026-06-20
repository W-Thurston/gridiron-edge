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
        assert meta.schema_version == 3
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
            metrics={
                "brier": 0.220,
                "ece": 0.018,
                "auc": 0.762,
                "log_loss": 0.628,
                "accuracy": 0.681,
            },
        )

        assert meta.metrics["brier"] == pytest.approx(0.220)
        assert meta.metrics["accuracy"] == pytest.approx(0.681)
        assert "mae" not in meta.metrics

    def test_regression_construction(self) -> None:
        meta = GameModelMetadata(
            model_name="total",
            model_type="xgboost",
            task="regression",
            trained_at="2026-06-18T00:00:00+00:00",
            metrics={"mae": 8.2, "rmse": 10.5, "r2": 0.31},
        )
        assert meta.metrics["mae"] == pytest.approx(8.2)
        assert meta.metrics["r2"] == pytest.approx(0.31)
        assert "brier" not in meta.metrics

    def test_default_metrics_dict_is_empty(self) -> None:
        meta = GameModelMetadata(
            model_name="win_prob",
            model_type="xgboost",
            task="classification",
            trained_at="2026-06-18T00:00:00+00:00",
        )
        assert meta.metrics == {}


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
            metrics={"mae": 58.0, "rmse": 72.6, "r2": 0.071},
        )
        assert isinstance(meta, BaseModelMetadata)

    def test_construction(self) -> None:
        meta = PropModelMetadata(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            task="regression",
            trained_at="2026-06-18T00:00:00+00:00",
            target_col="passing_yards",
            metrics={"mae": 58.0, "rmse": 72.6, "r2": 0.071},
            training_seasons=["1999-2000", "2000-2001"],
            holdout_seasons=["2023-2024"],
            parameters={"alpha": 0.1, "l1_ratio": 0.5},
            feature_columns=["a", "b"],
            n_train_rows=5706,
            n_holdout_rows=1367,
        )
        assert meta.target_col == "passing_yards"
        assert meta.metrics["mae"] == pytest.approx(58.0)
        assert meta.metrics["rmse"] == pytest.approx(72.6)
        assert meta.metrics["r2"] == pytest.approx(0.071)

    def test_required_target_col_enforced(self) -> None:
        """Missing target_col should raise."""
        with pytest.raises(TypeError):
            PropModelMetadata(  # type: ignore[call-arg]
                model_name="qb_pass_yards",
                model_type="elasticnet",
                task="regression",
                trained_at="2026-06-18T00:00:00+00:00",
                metrics={"mae": 0.0, "rmse": 0.0, "r2": 0.0},
                # Missing target_col
            )

"""Tests for prop model factory, HP grids, and clip ranges."""

from __future__ import annotations

from typing import Any

from gridiron_edge.models.prop_prediction.base import (
    PropModelType,
    _create_model,
    _get_param_grid,
)


class TestPropModelTypeEnum:
    def test_three_values(self) -> None:
        assert len(PropModelType) == 3

    def test_elasticnet_value(self) -> None:
        assert PropModelType.ELASTICNET.value == "elasticnet"

    def test_random_forest_value(self) -> None:
        assert PropModelType.RANDOM_FOREST.value == "random_forest"

    def test_xgboost_value(self) -> None:
        assert PropModelType.XGBOOST.value == "xgboost"


class TestCreateModel:
    def test_elasticnet_returns_scaler(self) -> None:
        model, scaler = _create_model(PropModelType.ELASTICNET)
        assert type(model).__name__ == "ElasticNet"
        assert type(scaler).__name__ == "StandardScaler"

    def test_rf_returns_no_scaler(self) -> None:
        model, scaler = _create_model(PropModelType.RANDOM_FOREST)
        assert type(model).__name__ == "RandomForestRegressor"
        assert scaler is None

    def test_xgb_returns_no_scaler(self) -> None:
        model, scaler = _create_model(PropModelType.XGBOOST)
        assert type(model).__name__ == "XGBRegressor"
        assert scaler is None


class TestParamGrid:
    def test_elasticnet_25_combos(self) -> None:
        grid: list[dict[str, Any]] = _get_param_grid(PropModelType.ELASTICNET)
        assert len(grid) == 25

    def test_rf_36_combos(self) -> None:
        grid: list[dict[str, Any]] = _get_param_grid(PropModelType.RANDOM_FOREST)
        assert len(grid) == 36

    def test_xgb_54_combos(self) -> None:
        grid: list[dict[str, Any]] = _get_param_grid(PropModelType.XGBOOST)
        assert len(grid) == 54

    def test_elasticnet_keys(self) -> None:
        grid: list[dict[str, Any]] = _get_param_grid(PropModelType.ELASTICNET)
        assert set(grid[0].keys()) == {"alpha", "l1_ratio"}

    def test_rf_keys(self) -> None:
        grid: list[dict[str, Any]] = _get_param_grid(PropModelType.RANDOM_FOREST)
        assert set(grid[0].keys()) == {"n_estimators", "max_depth", "min_samples_leaf"}

    def test_xgb_keys(self) -> None:
        grid: list[dict[str, Any]] = _get_param_grid(PropModelType.XGBOOST)
        assert set(grid[0].keys()) == {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
        }


class TestClipRanges:
    """Verify predictions are clipped to spec bounds."""

    def test_clip_from_spec(self) -> None:
        from gridiron_edge.models.prop_prediction.qb_pass_yards import (
            QBPassYardsTrainer,
        )

        trainer = QBPassYardsTrainer()
        assert trainer.spec.clip_lo == 0.0
        assert trainer.spec.clip_hi == 600

    def test_all_specs_have_clip_hi(self) -> None:
        from gridiron_edge.models.prop_prediction.qb_pass_yards import QBPassYardsTrainer
        from gridiron_edge.models.prop_prediction.qb_rush_yards import QBRushYardsTrainer
        from gridiron_edge.models.prop_prediction.rb_rush_yards import RBRushYardsTrainer
        from gridiron_edge.models.prop_prediction.te_rec_yards import TERecYardsTrainer
        from gridiron_edge.models.prop_prediction.wr_rec_yards import WRRecYardsTrainer

        expected: dict[str, int] = {
            "qb_pass_yards": 600,
            "qb_rush_yards": 200,
            "rb_rush_yards": 250,
            "wr_rec_yards": 300,
            "te_rec_yards": 250,
        }
        trainers: list[
            QBPassYardsTrainer
            | QBRushYardsTrainer
            | RBRushYardsTrainer
            | TERecYardsTrainer
            | WRRecYardsTrainer
        ] = [
            QBPassYardsTrainer(),
            QBRushYardsTrainer(),
            RBRushYardsTrainer(),
            WRRecYardsTrainer(),
            TERecYardsTrainer(),
        ]
        for t in trainers:
            assert t.spec.clip_hi == expected[t.spec.name], f"{t.spec.name} clip_hi mismatch"
            assert t.spec.clip_lo == 0.0, f"{t.spec.name} clip_lo should be 0"

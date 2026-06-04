# tests/unit/models/test_rb_rush_yards.py
"""Tests for gridiron_edge.models.prop_prediction.rb_rush_yards."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.models.prop_prediction.rb_rush_yards import (
    _FEATURE_COLUMNS,
    RBRushYardsTrainer,
)


class TestRBRushYardsSpec:
    def test_name(self) -> None:
        assert RBRushYardsTrainer().spec.name == "rb_rush_yards"

    def test_target_col(self) -> None:
        assert RBRushYardsTrainer().spec.target_col == "rushing_yards"

    def test_position_filter(self) -> None:
        assert RBRushYardsTrainer().spec.position_filter == ["RB", "FB"]

    def test_not_fitted_raises(self) -> None:
        trainer = RBRushYardsTrainer()
        dummy = pd.DataFrame({col: [0.0] for col in _FEATURE_COLUMNS})
        with pytest.raises(RuntimeError, match="Model not fitted"):
            trainer._predict(dummy)


class TestFeatureColumns:
    def test_count(self) -> None:
        assert len(_FEATURE_COLUMNS) == 16

    def test_has_rolling_features(self) -> None:
        rolling: list[str] = [c for c in _FEATURE_COLUMNS if "_L3_" in c or "_L6_" in c]
        assert len(rolling) >= 8

    def test_has_matchup_features(self) -> None:
        matchup: list[str] = [c for c in _FEATURE_COLUMNS if c.startswith("opp_")]
        assert len(matchup) >= 4

    def test_key_features_present(self) -> None:
        assert "rushing_yards_L3_mean" in _FEATURE_COLUMNS
        assert "rushing_yards_L6_mean" in _FEATURE_COLUMNS
        assert "carries_L3_mean" in _FEATURE_COLUMNS
        assert "opp_rush_yards_allowed_L6" in _FEATURE_COLUMNS

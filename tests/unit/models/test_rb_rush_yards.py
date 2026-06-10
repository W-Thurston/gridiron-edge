# tests/unit/models/test_rb_rush_yards.py
"""Tests for gridiron_edge.models.prop_prediction.rb_rush_yards."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.features.player._columns import PROP_FEATURE_COLS
from gridiron_edge.models.prop_prediction.rb_rush_yards import (
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
        dummy = pd.DataFrame({col: [0.0] for col in PROP_FEATURE_COLS})
        with pytest.raises(RuntimeError, match="Model not fitted"):
            trainer._predict(dummy)

    def test_uses_prop_feature_cols(self) -> None:
        trainer = RBRushYardsTrainer()
        assert trainer._feature_columns() == list(PROP_FEATURE_COLS)

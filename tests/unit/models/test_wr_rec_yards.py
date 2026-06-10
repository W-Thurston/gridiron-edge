# tests/unit/models/test_wr_rec_yards.py
"""Tests for gridiron_edge.models.prop_prediction.wr_rec_yards."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.features.player._columns import PROP_FEATURE_COLS
from gridiron_edge.models.prop_prediction.wr_rec_yards import (
    WRRecYardsTrainer,
)


class TestWRRecYardsSpec:
    def test_name(self) -> None:
        assert WRRecYardsTrainer().spec.name == "wr_rec_yards"

    def test_target_col(self) -> None:
        assert WRRecYardsTrainer().spec.target_col == "receiving_yards"

    def test_position_filter(self) -> None:
        assert WRRecYardsTrainer().spec.position_filter == ["WR"]

    def test_not_fitted_raises(self) -> None:
        trainer = WRRecYardsTrainer()
        dummy = pd.DataFrame({col: [0.0] for col in PROP_FEATURE_COLS})
        with pytest.raises(RuntimeError, match="Model not fitted"):
            trainer._predict(dummy)

    def test_uses_prop_feature_cols(self) -> None:
        trainer = WRRecYardsTrainer()
        assert trainer._feature_columns() == list(PROP_FEATURE_COLS)

# tests/unit/models/test_qb_pass_yards.py
"""Tests for gridiron_edge.models.prop_prediction.qb_pass_yards."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.models.prop_prediction.base import UNIVERSAL_FEATURE_COLS
from gridiron_edge.models.prop_prediction.qb_pass_yards import (
    QBPassYardsTrainer,
)


class TestQBPassYardsSpec:
    def test_name(self) -> None:
        assert QBPassYardsTrainer().spec.name == "qb_pass_yards"

    def test_target_col(self) -> None:
        assert QBPassYardsTrainer().spec.target_col == "passing_yards"

    def test_position_filter(self) -> None:
        assert QBPassYardsTrainer().spec.position_filter == ["QB"]

    def test_not_fitted_raises(self) -> None:
        trainer = QBPassYardsTrainer()
        dummy = pd.DataFrame({col: [0.0] for col in UNIVERSAL_FEATURE_COLS})
        with pytest.raises(RuntimeError, match="Model not fitted"):
            trainer._predict(dummy)

    def test_uses_universal_features(self) -> None:
        trainer = QBPassYardsTrainer()
        assert trainer._feature_columns() is UNIVERSAL_FEATURE_COLS

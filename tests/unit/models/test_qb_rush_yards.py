# tests/unit/models/test_qb_rush_yards.py
"""Tests for gridiron_edge.models.prop_prediction.qb_rush_yards."""

from __future__ import annotations

from gridiron_edge.features.player._columns import PROP_FEATURE_COLS
from gridiron_edge.models.prop_prediction.qb_rush_yards import QBRushYardsTrainer


class TestQBRushYardsSpec:
    """Verify model specification."""

    def test_name(self) -> None:
        assert QBRushYardsTrainer().spec.name == "qb_rush_yards"

    def test_target_col(self) -> None:
        assert QBRushYardsTrainer().spec.target_col == "rushing_yards"

    def test_position_filter(self) -> None:
        assert QBRushYardsTrainer().spec.position_filter == ["QB"]

    def test_uses_prop_feature_cols(self) -> None:
        trainer = QBRushYardsTrainer()
        assert trainer._feature_columns() == list(PROP_FEATURE_COLS)

    def test_description(self) -> None:
        assert "rushing" in QBRushYardsTrainer().spec.description.lower()

# tests/unit/models/test_qb_pass_yards.py
"""Tests for gridiron_edge.models.prop_prediction.qb_pass_yards."""

from __future__ import annotations

from gridiron_edge.features.player._columns import PROP_FEATURE_COLS
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

    def test_clip_hi(self) -> None:
        assert QBPassYardsTrainer().spec.clip_hi == 600

    def test_clip_lo(self) -> None:
        assert QBPassYardsTrainer().spec.clip_lo == 0.0

    def test_uses_prop_feature_cols(self) -> None:
        trainer = QBPassYardsTrainer()
        assert trainer._feature_columns() == list(PROP_FEATURE_COLS)

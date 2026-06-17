# tests/unit/models/test_te_rec_yards.py
"""Tests for gridiron_edge.models.prop_prediction.te_rec_yards."""

from __future__ import annotations

from gridiron_edge.features.player._columns import PROP_FEATURE_COLS
from gridiron_edge.models.prop_prediction.te_rec_yards import (
    TERecYardsTrainer,
)


class TestTERecYardsSpec:
    def test_name(self) -> None:
        assert TERecYardsTrainer().spec.name == "te_rec_yards"

    def test_target_col(self) -> None:
        assert TERecYardsTrainer().spec.target_col == "receiving_yards"

    def test_position_filter(self) -> None:
        assert TERecYardsTrainer().spec.position_filter == ["TE"]

    def test_clip_hi(self) -> None:
        assert TERecYardsTrainer().spec.clip_hi == 250

    def test_clip_lo(self) -> None:
        assert TERecYardsTrainer().spec.clip_lo == 0.0

    def test_uses_prop_feature_cols(self) -> None:
        trainer = TERecYardsTrainer()
        assert trainer._feature_columns() == list(PROP_FEATURE_COLS)

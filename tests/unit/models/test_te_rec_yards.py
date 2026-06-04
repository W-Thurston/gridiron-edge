# tests/unit/models/test_te_rec_yards.py
"""Tests for gridiron_edge.models.prop_prediction.te_rec_yards."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.models.prop_prediction.te_rec_yards import (
    _FEATURE_COLUMNS,
    TERecYardsTrainer,
)


class TestTERecYardsSpec:
    def test_name(self) -> None:
        assert TERecYardsTrainer().spec.name == "te_rec_yards"

    def test_target_col(self) -> None:
        assert TERecYardsTrainer().spec.target_col == "receiving_yards"

    def test_position_filter(self) -> None:
        assert TERecYardsTrainer().spec.position_filter == ["TE"]

    def test_not_fitted_raises(self) -> None:
        trainer = TERecYardsTrainer()
        dummy = pd.DataFrame({col: [0.0] for col in _FEATURE_COLUMNS})
        with pytest.raises(RuntimeError, match="Model not fitted"):
            trainer._predict(dummy)


class TestFeatureColumns:
    def test_count(self) -> None:
        assert len(_FEATURE_COLUMNS) == 20

    def test_has_rolling_features(self) -> None:
        rolling: list[str] = [c for c in _FEATURE_COLUMNS if "_L3_" in c or "_L6_" in c]
        assert len(rolling) >= 10

    def test_has_te_specific_matchup_features(self) -> None:
        """TE model should use TE-specific matchup features, not WR."""
        te_matchup: list[str] = [c for c in _FEATURE_COLUMNS if "opp_te_" in c]
        wr_matchup: list[str] = [c for c in _FEATURE_COLUMNS if "opp_wr_" in c]
        assert len(te_matchup) >= 4
        assert len(wr_matchup) == 0

    def test_key_features_present(self) -> None:
        assert "receiving_yards_L3_mean" in _FEATURE_COLUMNS
        assert "receiving_yards_L6_mean" in _FEATURE_COLUMNS
        assert "target_share_L3_mean" in _FEATURE_COLUMNS
        assert "opp_te_rec_yards_allowed_L6" in _FEATURE_COLUMNS

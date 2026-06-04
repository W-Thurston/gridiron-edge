# tests/unit/models/test_wr_rec_yards.py
"""Tests for gridiron_edge.models.prop_prediction.wr_rec_yards."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.models.prop_prediction.wr_rec_yards import (
    _FEATURE_COLUMNS,
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
        dummy = pd.DataFrame({col: [0.0] for col in _FEATURE_COLUMNS})
        with pytest.raises(RuntimeError, match="Model not fitted"):
            trainer._predict(dummy)


class TestFeatureColumns:
    def test_count(self) -> None:
        assert len(_FEATURE_COLUMNS) == 21

    def test_has_rolling_features(self) -> None:
        rolling: list[str] = [c for c in _FEATURE_COLUMNS if "_L3_" in c or "_L6_" in c]
        assert len(rolling) >= 12

    def test_has_matchup_features(self) -> None:
        matchup: list[str] = [c for c in _FEATURE_COLUMNS if c.startswith("opp_")]
        assert len(matchup) >= 4

    def test_has_usage_features(self) -> None:
        """WR model should include target share and air yards share."""
        assert "target_share_L3_mean" in _FEATURE_COLUMNS
        assert "air_yards_share_L3_mean" in _FEATURE_COLUMNS

    def test_key_features_present(self) -> None:
        assert "receiving_yards_L3_mean" in _FEATURE_COLUMNS
        assert "receiving_yards_L6_mean" in _FEATURE_COLUMNS
        assert "opp_wr_rec_yards_allowed_L6" in _FEATURE_COLUMNS

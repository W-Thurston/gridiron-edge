# tests/unit/models/test_qb_pass_yards.py
"""Tests for gridiron_edge.models.prop_prediction.qb_pass_yards."""

from __future__ import annotations

from gridiron_edge.models.prop_prediction.qb_pass_yards import (
    _FEATURE_COLUMNS,
    QBPassYardsTrainer,
)


class TestQBPassYardsSpec:
    def test_name(self) -> None:
        trainer = QBPassYardsTrainer()
        assert trainer.spec.name == "qb_pass_yards"

    def test_target_col(self) -> None:
        trainer = QBPassYardsTrainer()
        assert trainer.spec.target_col == "passing_yards"

    def test_position_filter(self) -> None:
        trainer = QBPassYardsTrainer()
        assert trainer.spec.position_filter == ["QB"]

    def test_not_fitted_raises(self) -> None:
        """Calling _predict before train should raise RuntimeError."""
        import pandas as pd
        import pytest

        trainer = QBPassYardsTrainer()
        dummy = pd.DataFrame({col: [0.0] for col in _FEATURE_COLUMNS})
        with pytest.raises(RuntimeError, match="Model not fitted"):
            trainer._predict(dummy)


class TestFeatureColumns:
    def test_count(self) -> None:
        assert len(_FEATURE_COLUMNS) == 20

    def test_has_rolling_features(self) -> None:
        rolling = [c for c in _FEATURE_COLUMNS if "_L3_" in c or "_L6_" in c]
        assert len(rolling) >= 10

    def test_has_matchup_features(self) -> None:
        matchup = [c for c in _FEATURE_COLUMNS if c.startswith("opp_")]
        assert len(matchup) >= 5

    def test_no_cpoe(self) -> None:
        """CPOE excluded due to high NaN rate in early seasons."""
        cpoe = [c for c in _FEATURE_COLUMNS if "cpoe" in c.lower()]
        assert len(cpoe) == 0

    def test_key_features_present(self) -> None:
        assert "passing_yards_L3_mean" in _FEATURE_COLUMNS
        assert "passing_yards_L6_mean" in _FEATURE_COLUMNS
        assert "opp_pass_yards_allowed_L6" in _FEATURE_COLUMNS
        assert "opp_pass_yards_allowed_rank_L6" in _FEATURE_COLUMNS

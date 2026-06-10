# tests/unit/models/test_prop_post_process.py
"""Tests for gridiron_edge.models.prop_prediction.post_process."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pytest

from gridiron_edge.models.prop_prediction.post_process import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HIGH_DISTANCE,
    DEFAULT_MODERATE_DISTANCE,
    DEFAULT_OVER_THRESHOLD,
    DEFAULT_UNDER_THRESHOLD,
    TARGET_STD_MAP,
    compute_p_over,
    compute_predicted_std,
    compute_prediction_interval,
    derive_confidence_tier,
    derive_lean,
    enrich_prop_predictions,
)


class TestConstants:
    """Verify module-level constants."""

    def test_thresholds(self) -> None:
        assert DEFAULT_OVER_THRESHOLD == 0.55
        assert DEFAULT_UNDER_THRESHOLD == 0.45

    def test_confidence_tiers(self) -> None:
        assert DEFAULT_HIGH_DISTANCE == 0.15
        assert DEFAULT_MODERATE_DISTANCE == 0.08

    def test_default_confidence(self) -> None:
        assert DEFAULT_CONFIDENCE == 0.90

    def test_target_std_map_has_all_models(self) -> None:
        assert set(TARGET_STD_MAP.keys()) == {
            "qb_pass_yards",
            "rb_rush_yards",
            "wr_rec_yards",
            "te_rec_yards",
        }

    def test_target_std_map_uses_l3(self) -> None:
        for col in TARGET_STD_MAP.values():
            assert "_L3_std" in col


class TestComputePredictedStd:
    """Verify uncertainty combination formula."""

    def test_combines_sources(self) -> None:
        """sqrt(rmse² + std²) = sqrt(72.6² + 55²) ≈ 91.1."""
        model_rmse = 72.6
        player_std = Series([55.0])
        result = compute_predicted_std(model_rmse, player_std)
        expected = np.sqrt(72.6**2 + 55.0**2)
        assert result.iloc[0] == pytest.approx(expected, rel=1e-4)

    def test_nan_defaults_to_rmse(self) -> None:
        """NaN player std → predicted_std = model RMSE."""
        model_rmse = 72.6
        player_std = Series([np.nan])
        result = compute_predicted_std(model_rmse, player_std)
        assert result.iloc[0] == pytest.approx(72.6, rel=1e-4)

    def test_zero_player_std(self) -> None:
        """Zero player std → predicted_std = model RMSE."""
        model_rmse = 50.0
        player_std = Series([0.0])
        result = compute_predicted_std(model_rmse, player_std)
        assert result.iloc[0] == pytest.approx(50.0, rel=1e-4)

    def test_vectorized(self) -> None:
        """Works on multiple rows."""
        model_rmse = 50.0
        player_std = Series([30.0, 40.0, 50.0])
        result = compute_predicted_std(model_rmse, player_std)
        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(np.sqrt(2500 + 900), rel=1e-4)


class TestComputePredictionInterval:
    """Verify prediction interval math."""

    def test_symmetric_around_mean(self) -> None:
        """Interval should be centered on predicted_mean."""
        mean = Series([200.0])
        std = Series([50.0])
        lo, hi = compute_prediction_interval(mean, std)
        midpoint = (lo.iloc[0] + hi.iloc[0]) / 2
        assert midpoint == pytest.approx(200.0, rel=1e-4)

    def test_90_interval_width(self) -> None:
        """Width = 2 x 1.645 x std."""
        mean = Series([200.0])
        std = Series([50.0])
        lo, hi = compute_prediction_interval(mean, std, confidence=0.90)
        expected_width = 2 * 1.645 * 50.0
        actual_width = hi.iloc[0] - lo.iloc[0]
        assert actual_width == pytest.approx(expected_width, rel=1e-3)

    def test_lo_clipped_at_zero(self) -> None:
        """Lower bound should never go below 0."""
        mean = Series([30.0])
        std = Series([100.0])
        lo, _hi = compute_prediction_interval(mean, std)
        assert lo.iloc[0] == 0.0

    def test_custom_confidence(self) -> None:
        """95% interval is wider than 90%."""
        mean = Series([200.0])
        std = Series([50.0])
        lo_90, hi_90 = compute_prediction_interval(mean, std, confidence=0.90)
        lo_95, hi_95 = compute_prediction_interval(mean, std, confidence=0.95)
        width_90 = hi_90.iloc[0] - lo_90.iloc[0]
        width_95 = hi_95.iloc[0] - lo_95.iloc[0]
        assert width_95 > width_90


class TestComputePOver:
    """Verify P(over) calculation."""

    def test_at_mean_is_half(self) -> None:
        """Line = predicted_mean → P(over) = 0.5."""
        mean = Series([250.0])
        std = Series([70.0])
        line = Series([250.0])
        result = compute_p_over(mean, std, line)
        assert result.iloc[0] == pytest.approx(0.5, abs=1e-6)

    def test_below_mean(self) -> None:
        """Line below mean → P(over) > 0.5."""
        mean = Series([250.0])
        std = Series([70.0])
        line = Series([200.0])
        result = compute_p_over(mean, std, line)
        assert result.iloc[0] > 0.5

    def test_above_mean(self) -> None:
        """Line above mean → P(over) < 0.5."""
        mean = Series([250.0])
        std = Series([70.0])
        line = Series([300.0])
        result = compute_p_over(mean, std, line)
        assert result.iloc[0] < 0.5

    def test_far_below_mean(self) -> None:
        """Line far below mean → P(over) ≈ 1.0."""
        mean = Series([250.0])
        std = Series([30.0])
        line = Series([100.0])
        result = compute_p_over(mean, std, line)
        assert result.iloc[0] > 0.99

    def test_far_above_mean(self) -> None:
        """Line far above mean → P(over) ≈ 0.0."""
        mean = Series([250.0])
        std = Series([30.0])
        line = Series([400.0])
        result = compute_p_over(mean, std, line)
        assert result.iloc[0] < 0.01


class TestDeriveLean:
    """Verify lean classification."""

    def test_over(self) -> None:
        result = derive_lean(Series([0.60]))
        assert result.iloc[0] == "Over"

    def test_under(self) -> None:
        result = derive_lean(Series([0.40]))
        assert result.iloc[0] == "Under"

    def test_no_edge(self) -> None:
        result = derive_lean(Series([0.50]))
        assert result.iloc[0] == "No Edge"

    def test_boundary_over(self) -> None:
        """Exactly at threshold → still No Edge (not >)."""
        result = derive_lean(Series([0.55]))
        assert result.iloc[0] == "No Edge"

    def test_boundary_under(self) -> None:
        """Exactly at threshold → still No Edge (not <)."""
        result = derive_lean(Series([0.45]))
        assert result.iloc[0] == "No Edge"

    def test_nan_produces_nan(self) -> None:
        result = derive_lean(Series([np.nan]))
        assert pd.isna(result.iloc[0])


class TestDeriveConfidenceTier:
    """Verify confidence tier classification."""

    def test_high(self) -> None:
        """p_over=0.70 → distance=0.20 → High."""
        result = derive_confidence_tier(Series([0.70]))
        assert result.iloc[0] == "High"

    def test_moderate(self) -> None:
        """p_over=0.60 → distance=0.10 → Moderate."""
        result = derive_confidence_tier(Series([0.60]))
        assert result.iloc[0] == "Moderate"

    def test_low(self) -> None:
        """p_over=0.52 → distance=0.02 → Low."""
        result = derive_confidence_tier(Series([0.52]))
        assert result.iloc[0] == "Low"

    def test_symmetric_under(self) -> None:
        """p_over=0.30 → distance=0.20 → High (works for under too)."""
        result = derive_confidence_tier(Series([0.30]))
        assert result.iloc[0] == "High"

    def test_nan_produces_nan(self) -> None:
        result = derive_confidence_tier(Series([np.nan]))
        assert pd.isna(result.iloc[0])


class TestEnrichPropPredictions:
    """Verify the orchestrator function."""

    def _make_predictions(self, n: int = 5) -> DataFrame:
        """Build a minimal predictions DataFrame."""
        rng = np.random.default_rng(42)
        return DataFrame(
            {
                "player_id": [f"P{i}" for i in range(n)],
                "predicted_mean": rng.uniform(150, 350, n),
                "passing_yards_L3_std": rng.uniform(30, 90, n),
            }
        )

    def test_raises_without_predicted_mean(self) -> None:
        df = DataFrame({"passing_yards_L3_std": [50.0]})
        with pytest.raises(ValueError, match="predicted_mean"):
            enrich_prop_predictions(df, model_rmse=72.6, target_std_col="passing_yards_L3_std")

    def test_raises_without_target_std_col(self) -> None:
        df = DataFrame({"predicted_mean": [250.0]})
        with pytest.raises(ValueError, match="passing_yards_L3_std"):
            enrich_prop_predictions(df, model_rmse=72.6, target_std_col="passing_yards_L3_std")

    def test_no_line_produces_nan_market_cols(self) -> None:
        df = self._make_predictions()
        result = enrich_prop_predictions(df, model_rmse=72.6, target_std_col="passing_yards_L3_std")
        assert result["p_over"].isna().all()
        assert result["lean"].isna().all()
        assert result["confidence_tier"].isna().all()

    def test_no_line_still_has_interval(self) -> None:
        df = self._make_predictions()
        result = enrich_prop_predictions(df, model_rmse=72.6, target_std_col="passing_yards_L3_std")
        assert "predicted_std" in result.columns
        assert "lo_90" in result.columns
        assert "hi_90" in result.columns
        assert result["predicted_std"].notna().all()

    def test_with_line_produces_all_cols(self) -> None:
        df = self._make_predictions()
        df["line"] = 250.0
        result = enrich_prop_predictions(
            df,
            model_rmse=72.6,
            target_std_col="passing_yards_L3_std",
            line_col="line",
        )
        assert result["p_over"].notna().all()
        assert result["lean"].notna().all()
        assert result["confidence_tier"].notna().all()

    def test_preserves_rows(self) -> None:
        df = self._make_predictions(n=10)
        result = enrich_prop_predictions(df, model_rmse=72.6, target_std_col="passing_yards_L3_std")
        assert len(result) == 10

    def test_does_not_modify_input(self) -> None:
        df = self._make_predictions()
        original_cols = set(df.columns)
        enrich_prop_predictions(df, model_rmse=72.6, target_std_col="passing_yards_L3_std")
        assert set(df.columns) == original_cols

    def test_lo_90_never_negative(self) -> None:
        df = self._make_predictions()
        df["predicted_mean"] = 20.0  # Low mean, high std → lo could go negative
        df["passing_yards_L3_std"] = 100.0
        result = enrich_prop_predictions(df, model_rmse=72.6, target_std_col="passing_yards_L3_std")
        assert (result["lo_90"] >= 0).all()

    def test_nan_line_rows_produce_nan_p_over(self) -> None:
        """Rows where line is NaN should get NaN p_over."""
        df = self._make_predictions(n=3)
        df["line"] = [250.0, np.nan, 275.0]
        result = enrich_prop_predictions(
            df,
            model_rmse=72.6,
            target_std_col="passing_yards_L3_std",
            line_col="line",
        )
        assert result["p_over"].notna().iloc[0]
        assert pd.isna(result["p_over"].iloc[1])
        assert result["p_over"].notna().iloc[2]

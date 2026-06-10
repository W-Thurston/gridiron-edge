# tests/unit/evaluation/test_prop_metrics.py
"""Tests for gridiron_edge.evaluation.prop_metrics."""

from __future__ import annotations

import numpy as np
from pandas import Series
import pytest

from gridiron_edge.evaluation.prop_metrics import (
    PropEvalReport,
    compute_accuracy,
    compute_bias,
    compute_calibration,
    compute_coverage,
    compute_hit_rate,
    compute_tier_analysis,
    evaluate_prop_model,
)


def _make_data(n: int = 100, seed: int = 42) -> dict[str, Series]:
    """Build synthetic evaluation data with known properties."""
    rng = np.random.default_rng(seed)
    actual = Series(rng.normal(250, 70, n))
    noise = Series(rng.normal(0, 30, n))
    predicted = actual + noise  # correlated but noisy
    std = Series(rng.uniform(50, 90, n))
    lo = predicted - 1.645 * std
    hi = predicted + 1.645 * std
    line = Series(rng.normal(250, 20, n))

    # Compute p_over from the test data
    z = (line - predicted) / std
    p_over = Series(1 - _norm_cdf(z.values))

    # Derive lean
    lean = Series("No Edge", index=predicted.index)
    lean = lean.where(~(p_over > 0.55), "Over")
    lean = lean.where(~(p_over < 0.45), "Under")

    # Derive tier
    distance = (p_over - 0.5).abs()
    tier = Series("Low", index=predicted.index)
    tier = tier.where(~(distance > 0.08), "Moderate")
    tier = tier.where(~(distance > 0.15), "High")

    return {
        "actual": actual,
        "predicted": predicted,
        "std": std,
        "lo": lo,
        "hi": hi,
        "line": line,
        "p_over": p_over,
        "lean": lean,
        "tier": tier,
    }


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF for test data generation."""
    from scipy.stats import norm  # type: ignore[import-untyped]

    return norm.cdf(x)


class TestComputeAccuracy:
    """Verify accuracy metric computation."""

    def test_perfect_predictions(self) -> None:
        actual = Series([100.0, 200.0, 300.0])
        predicted = Series([100.0, 200.0, 300.0])
        result = compute_accuracy(actual, predicted)
        assert result.mae == 0.0
        assert result.rmse == 0.0
        assert result.r2 == 1.0

    def test_mae_correct(self) -> None:
        actual = Series([100.0, 200.0, 300.0])
        predicted = Series([110.0, 190.0, 310.0])
        result = compute_accuracy(actual, predicted)
        assert result.mae == pytest.approx(10.0)

    def test_rmse_gte_mae(self) -> None:
        """RMSE >= MAE always."""
        data = _make_data()
        result = compute_accuracy(data["actual"], data["predicted"])
        assert result.rmse >= result.mae

    def test_n_correct(self) -> None:
        data = _make_data(n=50)
        result = compute_accuracy(data["actual"], data["predicted"])
        assert result.n == 50

    def test_median_ae(self) -> None:
        actual = Series([100.0, 200.0, 300.0])
        predicted = Series([110.0, 200.0, 340.0])
        result = compute_accuracy(actual, predicted)
        assert result.median_ae == pytest.approx(10.0)


class TestComputeBias:
    """Verify bias metric computation."""

    def test_no_bias(self) -> None:
        actual = Series([100.0, 200.0, 300.0])
        predicted = Series([100.0, 200.0, 300.0])
        result = compute_bias(actual, predicted)
        assert result.mean_error == pytest.approx(0.0)

    def test_over_prediction(self) -> None:
        actual = Series([100.0, 200.0, 300.0])
        predicted = Series([110.0, 210.0, 310.0])
        result = compute_bias(actual, predicted)
        assert result.mean_error > 0
        assert result.pct_over_predicted == 1.0

    def test_under_prediction(self) -> None:
        actual = Series([100.0, 200.0, 300.0])
        predicted = Series([90.0, 190.0, 290.0])
        result = compute_bias(actual, predicted)
        assert result.mean_error < 0
        assert result.pct_over_predicted == 0.0


class TestComputeCoverage:
    """Verify prediction interval coverage."""

    def test_perfect_coverage(self) -> None:
        """All actuals inside wide intervals → 100% coverage."""
        actual = Series([100.0, 200.0, 300.0])
        lo = Series([0.0, 0.0, 0.0])
        hi = Series([500.0, 500.0, 500.0])
        result = compute_coverage(actual, lo, hi)
        assert result.actual_coverage == 1.0

    def test_zero_coverage(self) -> None:
        """All actuals outside narrow intervals → 0% coverage."""
        actual = Series([100.0, 200.0, 300.0])
        lo = Series([150.0, 250.0, 350.0])
        hi = Series([160.0, 260.0, 360.0])
        result = compute_coverage(actual, lo, hi)
        assert result.actual_coverage == 0.0

    def test_interval_width(self) -> None:
        lo = Series([100.0, 150.0])
        hi = Series([200.0, 250.0])
        result = compute_coverage(Series([150.0, 200.0]), lo, hi)
        assert result.mean_interval_width == pytest.approx(100.0)

    def test_nominal_stored(self) -> None:
        actual = Series([100.0])
        result = compute_coverage(actual, Series([0.0]), Series([200.0]), nominal=0.95)
        assert result.nominal_coverage == 0.95


class TestComputeCalibration:
    """Verify calibration analysis."""

    def test_bucket_count(self) -> None:
        data = _make_data(n=200)
        result = compute_calibration(data["actual"], data["line"], data["p_over"], n_bins=5)
        assert len(result.buckets) == 5

    def test_perfect_calibration(self) -> None:
        """When p_over perfectly predicts outcomes, calibration error → 0."""
        # Construct: actual > line when p_over > 0.5, actual < line otherwise
        actual = Series([300.0, 300.0, 100.0, 100.0])
        line = Series([250.0, 250.0, 250.0, 250.0])
        p_over = Series([0.8, 0.9, 0.1, 0.2])
        result = compute_calibration(actual, line, p_over, n_bins=2)
        # Should be well-calibrated: high p_over → goes over, low → doesn't
        assert result.mean_abs_calibration_error < 0.3

    def test_nan_filtered(self) -> None:
        actual = Series([100.0, np.nan, 300.0])
        line = Series([250.0, 250.0, np.nan])
        p_over = Series([0.3, 0.5, 0.7])
        result = compute_calibration(actual, line, p_over)
        assert result.n == 1  # only first row has all three valid


class TestComputeHitRate:
    """Verify hit rate computation."""

    def test_perfect_over_hits(self) -> None:
        actual = Series([300.0, 350.0])
        line = Series([250.0, 250.0])
        lean = Series(["Over", "Over"])
        result = compute_hit_rate(actual, line, lean)
        assert result.over_hit_rate == 1.0

    def test_perfect_under_hits(self) -> None:
        actual = Series([100.0, 150.0])
        line = Series([250.0, 250.0])
        lean = Series(["Under", "Under"])
        result = compute_hit_rate(actual, line, lean)
        assert result.under_hit_rate == 1.0

    def test_no_edge_excluded(self) -> None:
        actual = Series([300.0, 100.0, 250.0])
        line = Series([250.0, 250.0, 250.0])
        lean = Series(["Over", "Under", "No Edge"])
        result = compute_hit_rate(actual, line, lean)
        assert result.overall_total == 2
        assert result.no_edge_count == 1

    def test_zero_leans(self) -> None:
        actual = Series([250.0])
        line = Series([250.0])
        lean = Series(["No Edge"])
        result = compute_hit_rate(actual, line, lean)
        assert np.isnan(result.overall_hit_rate)

    def test_mixed_results(self) -> None:
        actual = Series([300.0, 100.0, 100.0, 300.0])
        line = Series([250.0, 250.0, 250.0, 250.0])
        lean = Series(["Over", "Over", "Under", "Under"])
        result = compute_hit_rate(actual, line, lean)
        # Over: 1 hit, 1 miss. Under: 1 hit, 1 miss.
        assert result.over_hit_rate == pytest.approx(0.5)
        assert result.under_hit_rate == pytest.approx(0.5)
        assert result.overall_hit_rate == pytest.approx(0.5)


class TestComputeTierAnalysis:
    """Verify per-tier breakdown."""

    def test_three_tiers(self) -> None:
        data = _make_data(n=200)
        result = compute_tier_analysis(
            actual=data["actual"],
            predicted=data["predicted"],
            line=data["line"],
            lean=data["lean"],
            confidence_tier=data["tier"],
            p_over=data["p_over"],
        )
        assert len(result.tiers) == 3
        tier_names = {t.tier for t in result.tiers}
        assert tier_names == {"High", "Moderate", "Low"}

    def test_tier_counts_sum(self) -> None:
        data = _make_data(n=100)
        result = compute_tier_analysis(
            actual=data["actual"],
            predicted=data["predicted"],
            line=data["line"],
            lean=data["lean"],
            confidence_tier=data["tier"],
            p_over=data["p_over"],
        )
        total = sum(t.count for t in result.tiers)
        assert total == result.n

    def test_no_market_data(self) -> None:
        """Without lines, hit_rate should be NaN."""
        data = _make_data(n=50)
        result = compute_tier_analysis(
            actual=data["actual"],
            predicted=data["predicted"],
            line=None,
            lean=None,
            confidence_tier=data["tier"],
            p_over=None,
        )
        for t in result.tiers:
            if t.count > 0:
                assert not np.isnan(t.mae)
                assert np.isnan(t.hit_rate)


class TestEvaluatePropModel:
    """Verify the orchestrator entry point."""

    def test_accuracy_only(self) -> None:
        """Minimum inputs → accuracy + bias only."""
        data = _make_data()
        report = evaluate_prop_model(
            model_name="test",
            actual=data["actual"],
            predicted_mean=data["predicted"],
        )
        assert isinstance(report, PropEvalReport)
        assert report.accuracy.n > 0
        assert report.bias.n > 0
        assert report.coverage is None
        assert report.calibration is None
        assert report.hit_rate is None

    def test_with_coverage(self) -> None:
        data = _make_data()
        report = evaluate_prop_model(
            model_name="test",
            actual=data["actual"],
            predicted_mean=data["predicted"],
            lo_90=data["lo"],
            hi_90=data["hi"],
        )
        assert report.coverage is not None
        assert 0 <= report.coverage.actual_coverage <= 1

    def test_full_report(self) -> None:
        data = _make_data()
        report = evaluate_prop_model(
            model_name="test",
            actual=data["actual"],
            predicted_mean=data["predicted"],
            predicted_std=data["std"],
            lo_90=data["lo"],
            hi_90=data["hi"],
            line=data["line"],
            p_over=data["p_over"],
            lean=data["lean"],
            confidence_tier=data["tier"],
        )
        assert report.coverage is not None
        assert report.calibration is not None
        assert report.hit_rate is not None
        assert report.tier_analysis is not None

    def test_print_does_not_error(self) -> None:
        data = _make_data()
        report = evaluate_prop_model(
            model_name="test",
            actual=data["actual"],
            predicted_mean=data["predicted"],
            predicted_std=data["std"],
            lo_90=data["lo"],
            hi_90=data["hi"],
            line=data["line"],
            p_over=data["p_over"],
            lean=data["lean"],
            confidence_tier=data["tier"],
        )
        # Should not raise
        report.print_summary()

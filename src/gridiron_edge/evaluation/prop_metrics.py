# src/gridiron_edge/evaluation/prop_metrics.py
"""Evaluation metrics for prop prediction models.

Provides a structured evaluation report covering accuracy, calibration,
coverage, bias, and market-facing metrics (hit rate, by-tier analysis).

Designed to work with both backfill (no lines) and live (with lines)
scenarios.  Market-dependent metrics are NaN when lines are unavailable.

Usage::

    from gridiron_edge.evaluation.prop_metrics import evaluate_prop_model

    report = evaluate_prop_model(
        actual=actuals_series,
        predicted_mean=predictions_series,
        predicted_std=std_series,
        lo_90=lo_series,
        hi_90=hi_series,
        line=line_series,  # optional
        lean=lean_series,  # optional
        confidence_tier=tier_series,  # optional
    )
    report.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from logging import Logger
from typing import Any

import numpy as np
from numpy import dtype, float64, ndarray
from pandas import Series

from gridiron_edge.core.enums import ConfidenceTier, Lean

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccuracyMetrics:
    """Core prediction accuracy metrics."""

    mae: float
    rmse: float
    r2: float
    median_ae: float
    n: int


@dataclass(frozen=True)
class BiasMetrics:
    """Systematic prediction bias analysis."""

    mean_error: float  # mean(predicted - actual), positive = overpredict
    mean_abs_error: float
    pct_over_predicted: float  # fraction where predicted > actual
    n: int


@dataclass(frozen=True)
class CoverageMetrics:
    """Prediction interval coverage analysis."""

    nominal_coverage: float  # e.g., 0.90 for 90% intervals
    actual_coverage: float  # fraction of actuals within [lo, hi]
    mean_interval_width: float
    median_interval_width: float
    n: int


@dataclass(frozen=True)
class CalibrationBucket:
    """One bucket in a calibration analysis."""

    bin_center: float  # center of the predicted probability bin
    predicted_prob: float  # mean predicted probability in this bin
    observed_freq: float  # actual frequency of "over" in this bin
    count: int  # number of predictions in this bin


@dataclass(frozen=True)
class CalibrationMetrics:
    """P(over) calibration analysis."""

    buckets: list[CalibrationBucket]
    mean_abs_calibration_error: float  # avg |predicted - observed| across bins
    n: int  # total predictions with valid p_over + actual + line


@dataclass(frozen=True)
class HitRateMetrics:
    """Lean prediction hit rate analysis."""

    over_hits: int
    over_total: int
    over_hit_rate: float  # NaN if over_total == 0
    under_hits: int
    under_total: int
    under_hit_rate: float  # NaN if under_total == 0
    overall_hits: int
    overall_total: int
    overall_hit_rate: float  # NaN if overall_total == 0
    no_edge_count: int
    n: int


@dataclass(frozen=True)
class TierBreakdown:
    """Metrics for a single confidence tier."""

    tier: str
    count: int
    mae: float
    hit_rate: float  # NaN if no leans in this tier
    mean_p_over_distance: float  # mean |p_over - 0.5|


@dataclass(frozen=True)
class TierMetrics:
    """By-tier analysis."""

    tiers: list[TierBreakdown]
    n: int


@dataclass
class PropEvalReport:
    """Complete evaluation report for a prop model."""

    model_name: str
    accuracy: AccuracyMetrics
    bias: BiasMetrics
    coverage: CoverageMetrics | None = None
    calibration: CalibrationMetrics | None = None
    hit_rate: HitRateMetrics | None = None
    tier_analysis: TierMetrics | None = None

    def print_summary(self) -> None:
        """Print a formatted summary to the logger."""
        lines: list[str] = [
            f"\n{'=' * 60}",
            f"  Prop Model Evaluation: {self.model_name}",
            f"{'=' * 60}",
            "",
            "  ACCURACY",
            f"    MAE:       {self.accuracy.mae:.1f}",
            f"    RMSE:      {self.accuracy.rmse:.1f}",
            f"    R²:        {self.accuracy.r2:.3f}",
            f"    Median AE: {self.accuracy.median_ae:.1f}",
            f"    N:         {self.accuracy.n:,}",
            "",
            "  BIAS",
            f"    Mean Error:      {self.bias.mean_error:+.1f}"
            f"  ({'over' if self.bias.mean_error > 0 else 'under'}-predicting)",
            f"    % Over-Predicted: {self.bias.pct_over_predicted:.1%}",
        ]

        if self.coverage is not None:
            lines.extend(
                [
                    "",
                    "  COVERAGE",
                    f"    Nominal:  {self.coverage.nominal_coverage:.0%}",
                    f"    Actual:   {self.coverage.actual_coverage:.1%}",
                    f"    Mean Width: {self.coverage.mean_interval_width:.1f}",
                    f"    Median Width: {self.coverage.median_interval_width:.1f}",
                ]
            )

        if self.calibration is not None:
            lines.extend(
                [
                    "",
                    "  CALIBRATION",
                    f"    Mean Abs Cal Error: {self.calibration.mean_abs_calibration_error:.3f}",
                    "    Buckets (predicted → observed):",
                ]
            )
            for b in self.calibration.buckets:
                if b.count > 0:
                    lines.append(
                        f"      {b.predicted_prob:.2f} → {b.observed_freq:.2f}  (n={b.count})"
                    )

        if self.hit_rate is not None:
            lines.extend(
                [
                    "",
                    "  HIT RATE",
                    f"    Over:    {self.hit_rate.over_hits}/{self.hit_rate.over_total}"
                    f"  ({_fmt_pct(self.hit_rate.over_hit_rate)})",
                    f"    Under:   {self.hit_rate.under_hits}/{self.hit_rate.under_total}"
                    f"  ({_fmt_pct(self.hit_rate.under_hit_rate)})",
                    f"    Overall: {self.hit_rate.overall_hits}/{self.hit_rate.overall_total}"
                    f"  ({_fmt_pct(self.hit_rate.overall_hit_rate)})",
                    f"    No Edge: {self.hit_rate.no_edge_count}",
                ]
            )

        if self.tier_analysis is not None:
            lines.extend(
                [
                    "",
                    "  BY-TIER ANALYSIS",
                ]
            )
            for t in self.tier_analysis.tiers:
                lines.append(
                    f"    {t.tier:<10} n={t.count:<5} MAE={t.mae:.1f}"
                    f"  Hit={_fmt_pct(t.hit_rate)}"
                    f"  |p-0.5|={t.mean_p_over_distance:.3f}"
                )

        lines.append(f"\n{'=' * 60}")
        logger.info("\n".join(lines))


def _fmt_pct(val: float) -> str:
    """Format a float as percentage, handling NaN."""
    if np.isnan(val):
        return "N/A"
    return f"{val:.1%}"


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------


def compute_accuracy(actual: Series, predicted: Series) -> AccuracyMetrics:
    """Compute core accuracy metrics.

    Args:
        actual: Observed stat values.
        predicted: Model point predictions.

    Returns:
        AccuracyMetrics with MAE, RMSE, R², median AE.
    """
    errors: Series = predicted - actual
    abs_errors: Series = errors.abs()
    ss_res = (errors**2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()

    return AccuracyMetrics(
        mae=abs_errors.mean(),
        rmse=float(np.sqrt((errors**2).mean())),
        r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        median_ae=abs_errors.median(),
        n=len(actual),
    )


def compute_bias(actual: Series, predicted: Series) -> BiasMetrics:
    """Compute systematic bias metrics.

    Args:
        actual: Observed stat values.
        predicted: Model point predictions.

    Returns:
        BiasMetrics with mean error, mean absolute error,
        and fraction over-predicted.
    """
    errors: Series = predicted - actual
    return BiasMetrics(
        mean_error=errors.mean(),
        mean_abs_error=errors.abs().mean(),
        pct_over_predicted=(predicted > actual).mean(),
        n=len(actual),
    )


def compute_coverage(
    actual: Series,
    lo: Series,
    hi: Series,
    nominal: float = 0.90,
) -> CoverageMetrics:
    """Compute prediction interval coverage.

    Args:
        actual: Observed stat values.
        lo: Lower bound of prediction interval.
        hi: Upper bound of prediction interval.
        nominal: Nominal coverage level (default 0.90).

    Returns:
        CoverageMetrics with actual vs nominal coverage and interval widths.
    """
    within: Series[bool] = (actual >= lo) & (actual <= hi)
    widths: Series = hi - lo
    return CoverageMetrics(
        nominal_coverage=nominal,
        actual_coverage=within.mean(),
        mean_interval_width=widths.mean(),
        median_interval_width=widths.median(),
        n=len(actual),
    )


def compute_calibration(
    actual: Series,
    line: Series,
    p_over: Series,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute P(over) calibration (reliability diagram data).

    Bins predictions by predicted P(over), then compares the mean
    predicted probability in each bin to the actual fraction of
    outcomes that went over the line.

    Args:
        actual: Observed stat values.
        line: Betting lines.
        p_over: Predicted P(over) from the model.
        n_bins: Number of calibration bins (default 10).

    Returns:
        CalibrationMetrics with per-bin data and mean absolute
        calibration error.
    """
    # Filter to rows where all three are available
    mask: Series[bool] = actual.notna() & line.notna() & p_over.notna()
    actual_f = actual[mask]
    line_f = line[mask]
    p_over_f = p_over[mask]

    went_over = (actual_f > line_f).astype(float)

    bin_edges: ndarray[tuple[Any, ...], dtype[float64]] = np.linspace(0, 1, n_bins + 1)
    buckets: list[CalibrationBucket] = []
    abs_errors: list[float] = []

    for i in range(n_bins):
        lo_edge = bin_edges[i]
        hi_edge = bin_edges[i + 1]
        center = (lo_edge + hi_edge) / 2

        if i < n_bins - 1:
            in_bin = (p_over_f >= lo_edge) & (p_over_f < hi_edge)
        else:
            # Last bin includes the right edge
            in_bin = (p_over_f >= lo_edge) & (p_over_f <= hi_edge)

        count = int(in_bin.sum())
        if count > 0:
            pred_prob = float(p_over_f[in_bin].mean())
            obs_freq = float(went_over[in_bin].mean())
            abs_errors.append(abs(pred_prob - obs_freq))
        else:
            pred_prob = center
            obs_freq = float("nan")

        buckets.append(
            CalibrationBucket(
                bin_center=float(center),
                predicted_prob=pred_prob,
                observed_freq=obs_freq,
                count=count,
            )
        )

    mace: float = float(np.mean(abs_errors)) if abs_errors else float("nan")

    return CalibrationMetrics(
        buckets=buckets,
        mean_abs_calibration_error=mace,
        n=mask.sum(),
    )


def compute_hit_rate(
    actual: Series,
    line: Series,
    lean: Series,
) -> HitRateMetrics:
    """Compute lean prediction hit rate.

    A "hit" is when the model's lean matches the outcome:
    - "Over" lean + actual > line = hit
    - "Under" lean + actual < line = hit

    Pushes (actual == line) are excluded from hit/miss counting.

    Args:
        actual: Observed stat values.
        line: Betting lines.
        lean: Model lean predictions ("Over", "Under", "No Edge").

    Returns:
        HitRateMetrics with per-direction and overall hit rates.
    """
    mask: Series[bool] = actual.notna() & line.notna() & lean.notna()
    actual_f = actual[mask]
    line_f = line[mask]
    lean_f = lean[mask]

    went_over = actual_f > line_f
    went_under = actual_f < line_f
    # Push (actual == line) — excluded from both hit and miss

    # Over leans
    over_mask = lean_f == Lean.OVER.value
    over_total = int(over_mask.sum())
    over_hits: int = int((over_mask & went_over).sum()) if over_total > 0 else 0

    # Under leans
    under_mask = lean_f == Lean.UNDER.value
    under_total = int(under_mask.sum())
    under_hits: int = int((under_mask & went_under).sum()) if under_total > 0 else 0

    # No Edge
    no_edge_count = int((lean_f == Lean.NO_EDGE.value).sum())

    # Overall (Over + Under leans only)
    overall_total: int = over_total + under_total
    overall_hits: int = over_hits + under_hits

    return HitRateMetrics(
        over_hits=over_hits,
        over_total=over_total,
        over_hit_rate=over_hits / over_total if over_total > 0 else float("nan"),
        under_hits=under_hits,
        under_total=under_total,
        under_hit_rate=under_hits / under_total if under_total > 0 else float("nan"),
        overall_hits=overall_hits,
        overall_total=overall_total,
        overall_hit_rate=overall_hits / overall_total if overall_total > 0 else float("nan"),
        no_edge_count=no_edge_count,
        n=mask.sum(),
    )


def compute_tier_analysis(
    actual: Series,
    predicted: Series,
    line: Series | None,
    lean: Series | None,
    confidence_tier: Series,
    p_over: Series | None,
) -> TierMetrics:
    """Compute per-tier breakdown of accuracy and hit rate.

    Args:
        actual: Observed stat values.
        predicted: Model point predictions.
        line: Betting lines (optional — hit rate NaN without lines).
        lean: Model lean predictions (optional).
        confidence_tier: Confidence tier labels.
        p_over: Predicted P(over) (optional).

    Returns:
        TierMetrics with per-tier MAE, hit rate, and p_over distance.
    """
    mask: Series[bool] = actual.notna() & predicted.notna() & confidence_tier.notna()
    actual_f = actual[mask]
    predicted_f = predicted[mask]
    tier_f = confidence_tier[mask]

    has_market: bool = line is not None and lean is not None and p_over is not None

    tiers: list[TierBreakdown] = []
    for tier in (ConfidenceTier.HIGH, ConfidenceTier.MODERATE, ConfidenceTier.LOW):
        tier_name: str = tier.value
        tier_mask = tier_f == tier_name
        count = int(tier_mask.sum())

        if count == 0:
            tiers.append(
                TierBreakdown(
                    tier=tier_name,
                    count=0,
                    mae=float("nan"),
                    hit_rate=float("nan"),
                    mean_p_over_distance=float("nan"),
                )
            )
            continue

        tier_actual = actual_f[tier_mask]
        tier_predicted = predicted_f[tier_mask]
        mae = float((tier_predicted - tier_actual).abs().mean())

        # Hit rate for this tier
        hit_rate = float("nan")
        if has_market:
            assert line is not None and lean is not None
            line_f = line[mask][tier_mask]
            lean_f = lean[mask][tier_mask]
            leans_with_opinion = lean_f.isin([Lean.OVER.value, Lean.UNDER.value])
            if leans_with_opinion.any():
                over_hit = (lean_f == Lean.OVER.value) & (tier_actual > line_f)
                under_hit = (lean_f == Lean.UNDER.value) & (tier_actual < line_f)
                hits = int((over_hit | under_hit).sum())
                total = int(leans_with_opinion.sum())
                hit_rate: float = hits / total if total > 0 else float("nan")

        # Mean |p_over - 0.5|
        p_dist = float("nan")
        if p_over is not None:
            p_over_f = p_over[mask][tier_mask]
            valid_p = p_over_f.dropna()
            if len(valid_p) > 0:
                p_dist = float((valid_p - 0.5).abs().mean())

        tiers.append(
            TierBreakdown(
                tier=tier_name,
                count=count,
                mae=mae,
                hit_rate=hit_rate,
                mean_p_over_distance=p_dist,
            )
        )

    return TierMetrics(tiers=tiers, n=mask.sum())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate_prop_model(
    model_name: str,
    actual: Series,
    predicted_mean: Series,
    predicted_std: Series | None = None,
    lo_90: Series | None = None,
    hi_90: Series | None = None,
    line: Series | None = None,
    p_over: Series | None = None,
    lean: Series | None = None,
    confidence_tier: Series | None = None,
    coverage_nominal: float = 0.90,
) -> PropEvalReport:
    """Generate a complete prop model evaluation report.

    Computes all available metrics based on which inputs are provided.
    Accuracy and bias are always computed.  Coverage requires prediction
    intervals.  Calibration, hit rate, and tier analysis require market
    lines and enrichment columns.

    Args:
        model_name: Name of the model being evaluated.
        actual: Observed stat values.
        predicted_mean: Model point predictions.
        predicted_std: Per-prediction standard deviations.
        lo_90: Lower bound of 90% prediction interval.
        hi_90: Upper bound of 90% prediction interval.
        line: Betting lines.
        p_over: Predicted P(over).
        lean: Model lean predictions.
        confidence_tier: Confidence tier labels.
        coverage_nominal: Nominal coverage level (default 0.90).

    Returns:
        PropEvalReport with all available metrics.
    """
    # Always available
    accuracy: AccuracyMetrics = compute_accuracy(actual, predicted_mean)
    bias: BiasMetrics = compute_bias(actual, predicted_mean)

    # Coverage — requires prediction intervals
    coverage: CoverageMetrics | None = None
    if lo_90 is not None and hi_90 is not None:
        coverage = compute_coverage(actual, lo_90, hi_90, nominal=coverage_nominal)

    # Calibration — requires lines and p_over
    calibration: CalibrationMetrics | None = None
    if line is not None and p_over is not None:
        calibration = compute_calibration(actual, line, p_over)

    # Hit rate — requires lines and lean
    hit_rate: HitRateMetrics | None = None
    if line is not None and lean is not None:
        hit_rate = compute_hit_rate(actual, line, lean)

    # Tier analysis — requires confidence_tier
    tier_analysis: TierMetrics | None = None
    if confidence_tier is not None:
        tier_analysis = compute_tier_analysis(
            actual=actual,
            predicted=predicted_mean,
            line=line,
            lean=lean,
            confidence_tier=confidence_tier,
            p_over=p_over,
        )

    report = PropEvalReport(
        model_name=model_name,
        accuracy=accuracy,
        bias=bias,
        coverage=coverage,
        calibration=calibration,
        hit_rate=hit_rate,
        tier_analysis=tier_analysis,
    )

    report.print_summary()
    return report

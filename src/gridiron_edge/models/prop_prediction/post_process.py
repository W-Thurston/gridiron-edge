# src/gridiron_edge/models/prop_prediction/post_process.py
"""Post-processing enrichment for prop model predictions.

Transforms raw yard projections into actionable betting signals:
predicted uncertainty, prediction intervals, P(over), lean, and
confidence tier.

Design decisions (2026-06-10):
- predicted_std = sqrt(model_rmse² + player_L3_std²).  Two uncertainty
  sources: model residual variance (same for all players) and player
  rolling variance (per-player volatility).  When player std is NaN
  (early-season), defaults to model RMSE alone (conservative).
- Normal CDF for P(over) in V1.  Can upgrade to empirical distribution
  or quantile regression later.
- Lean thresholds (0.55/0.45) and confidence tiers (distance-based
  0.15/0.08) are consistent with game model post-processing.
- lo_90 clipped at 0 — negative stat projections are theoretically
  possible but not actionable for display or betting.
- Line input is optional.  When no line is provided, P(over), lean,
  and confidence_tier are NaN — MAE and calibration can still be
  evaluated without market data.

Target std column mapping:
    qb_pass_yards  → passing_yards_L3_std
    rb_rush_yards  → rushing_yards_L3_std
    wr_rec_yards   → receiving_yards_L3_std
    te_rec_yards   → receiving_yards_L3_std

Usage::

    from gridiron_edge.models.prop_prediction.post_process import (
        enrich_prop_predictions,
    )

    enriched = enrich_prop_predictions(
        df=predictions_df,
        model_rmse=72.6,
        target_std_col="passing_yards_L3_std",
        line_col="line",
    )
"""

from __future__ import annotations

from typing import Final

import numpy as np
from pandas import DataFrame, Series
from scipy.stats import norm  # type: ignore[import-untyped]

from gridiron_edge.core.enums import ConfidenceTier, Lean

# ---------------------------------------------------------------------------
# Default thresholds — consistent with game model post-processing
# ---------------------------------------------------------------------------

DEFAULT_OVER_THRESHOLD: Final[float] = 0.55
DEFAULT_UNDER_THRESHOLD: Final[float] = 0.45
DEFAULT_HIGH_DISTANCE: Final[float] = 0.15
DEFAULT_MODERATE_DISTANCE: Final[float] = 0.08
DEFAULT_CONFIDENCE: Final[float] = 0.90

# Mapping from prop model name to the rolling std column to use
# for player-level uncertainty.  L3 is more reactive to recent form.
TARGET_STD_MAP: Final[dict[str, str]] = {
    "qb_pass_yards": "passing_yards_L3_std",
    "qb_rush_yards": "rushing_yards_L3_std",
    "rb_rush_yards": "rushing_yards_L3_std",
    "wr_rec_yards": "receiving_yards_L3_std",
    "te_rec_yards": "receiving_yards_L3_std",
}


def compute_predicted_std(
    model_rmse: float,
    player_rolling_std: Series,
) -> Series:
    """Combine model and player uncertainty into per-prediction std.

    Formula: ``sqrt(model_rmse² + player_rolling_std²)``

    When ``player_rolling_std`` is NaN (early-season, insufficient
    games), defaults to ``model_rmse`` alone — a conservative estimate
    that produces wider prediction intervals.

    Args:
        model_rmse: Model residual RMSE from holdout evaluation.
        player_rolling_std: Per-player rolling standard deviation of
            the target stat (e.g., ``passing_yards_L3_std``).

    Returns:
        Series of per-prediction standard deviations.
    """
    player_var = player_rolling_std.fillna(0.0) ** 2
    model_var = model_rmse**2
    return Series(np.sqrt(model_var + player_var), index=player_rolling_std.index)


def compute_prediction_interval(
    predicted_mean: Series,
    predicted_std: Series,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[Series, Series]:
    """Compute symmetric prediction interval bounds.

    Uses the normal distribution z-score for the given confidence level.
    Lower bound is clipped at 0 — negative stat projections are not
    actionable for display or betting.

    Args:
        predicted_mean: Point predictions from the model.
        predicted_std: Per-prediction standard deviations.
        confidence: Confidence level (default 0.90 for 90% interval).

    Returns:
        Tuple of (lo, hi) Series.
    """
    z = float(norm.ppf((1 + confidence) / 2))
    lo: Series = (predicted_mean - z * predicted_std).clip(lower=0)
    hi: Series = predicted_mean + z * predicted_std
    return lo, hi


def compute_p_over(
    predicted_mean: Series,
    predicted_std: Series,
    line: Series,
) -> Series:
    """Compute P(actual > line) assuming a normal distribution.

    ``P(over) = 1 - Φ((line - predicted_mean) / predicted_std)``

    When any input is NaN, the result is NaN for that row.

    Args:
        predicted_mean: Point predictions from the model.
        predicted_std: Per-prediction standard deviations.
        line: Betting line (e.g., 274.5 for "Over 274.5 passing yards").

    Returns:
        Series of probabilities in [0, 1].
    """
    z_scores: Series = (line - predicted_mean) / predicted_std
    return Series(1 - norm.cdf(z_scores), index=predicted_mean.index)


def derive_lean(
    p_over: Series,
    over_threshold: float = DEFAULT_OVER_THRESHOLD,
    under_threshold: float = DEFAULT_UNDER_THRESHOLD,
) -> Series:
    """Classify predictions as Over, Under, or No Edge.

    Args:
        p_over: Probability of exceeding the line.
        over_threshold: P(over) above this → :data:`Lean.OVER` (default 0.55).
        under_threshold: P(over) below this → :data:`Lean.UNDER` (default 0.45).

    Returns:
        Series of :class:`Lean` string values (each member's ``.value`` is the
        underlying string label for backward compat with archived data).
    """
    lean = Series(Lean.NO_EDGE.value, index=p_over.index, dtype="object")
    lean = lean.where(~(p_over > over_threshold), Lean.OVER.value)
    lean = lean.where(~(p_over < under_threshold), Lean.UNDER.value)
    lean = lean.where(p_over.notna(), np.nan)
    return lean


def derive_confidence_tier(
    p_over: Series,
    high_distance: float = DEFAULT_HIGH_DISTANCE,
    moderate_distance: float = DEFAULT_MODERATE_DISTANCE,
) -> Series:
    """Classify prediction confidence as High, Moderate, or Low.

    Based on ``|p_over - 0.5|`` — how far the probability is from a
    coin flip. Consistent with game model confidence tiers.

    Args:
        p_over: Probability of exceeding the line.
        high_distance: Distance threshold for :data:`ConfidenceTier.HIGH`
            (default 0.15).
        moderate_distance: Distance threshold for
            :data:`ConfidenceTier.MODERATE` (default 0.08).

    Returns:
        Series of :class:`ConfidenceTier` string values (each member's
        ``.value`` is the underlying string label for backward compat
        with archived data).
    """
    distance: Series = (p_over - 0.5).abs()
    tier = Series(ConfidenceTier.LOW.value, index=p_over.index, dtype="object")
    tier = tier.where(~(distance > moderate_distance), ConfidenceTier.MODERATE.value)
    tier = tier.where(~(distance > high_distance), ConfidenceTier.HIGH.value)
    tier = tier.where(p_over.notna(), np.nan)
    return tier


def enrich_prop_predictions(
    df: DataFrame,
    model_rmse: float,
    target_std_col: str,
    line_col: str | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    over_threshold: float = DEFAULT_OVER_THRESHOLD,
    under_threshold: float = DEFAULT_UNDER_THRESHOLD,
    high_distance: float = DEFAULT_HIGH_DISTANCE,
    moderate_distance: float = DEFAULT_MODERATE_DISTANCE,
) -> DataFrame:
    """Add all post-processing enrichment columns to prop predictions.

    This is the main entry point.  Takes a DataFrame with at least
    ``predicted_mean`` and the target rolling std column, and adds:
    ``predicted_std``, ``lo_90``, ``hi_90``, and optionally ``p_over``,
    ``lean``, ``confidence_tier`` (when a line column is provided).

    Args:
        df: Predictions DataFrame.  Must contain ``predicted_mean`` and
            the column named by ``target_std_col``.
        model_rmse: Model residual RMSE from holdout evaluation.
        target_std_col: Column name for per-player rolling std
            (e.g., ``"passing_yards_L3_std"``).
        line_col: Optional column name containing the betting line.
            If ``None`` or column is absent, P(over)/lean/tier are NaN.
        confidence: Confidence level for prediction interval (default 0.90).
        over_threshold: P(over) threshold for "Over" lean.
        under_threshold: P(over) threshold for "Under" lean.
        high_distance: |p_over - 0.5| threshold for "High" confidence.
        moderate_distance: |p_over - 0.5| threshold for "Moderate" confidence.

    Returns:
        Copy of ``df`` with enrichment columns added.

    Raises:
        ValueError: If ``predicted_mean`` or ``target_std_col`` is missing.
    """
    result: DataFrame = df.copy()

    if "predicted_mean" not in result.columns:
        msg = "DataFrame must contain 'predicted_mean' column"
        raise ValueError(msg)

    if target_std_col not in result.columns:
        msg = f"DataFrame must contain '{target_std_col}' column"
        raise ValueError(msg)

    # Uncertainty
    result["predicted_std"] = compute_predicted_std(
        model_rmse=model_rmse,
        player_rolling_std=result[target_std_col],
    )

    # Prediction interval
    lo, hi = compute_prediction_interval(
        predicted_mean=result["predicted_mean"],
        predicted_std=result["predicted_std"],
        confidence=confidence,
    )
    result["lo_90"] = lo
    result["hi_90"] = hi

    # Market-dependent enrichment
    has_line = (
        line_col is not None and line_col in result.columns and result[line_col].notna().any()
    )

    if has_line:
        assert line_col is not None  # for type checker
        result["p_over"] = compute_p_over(
            predicted_mean=result["predicted_mean"],
            predicted_std=result["predicted_std"],
            line=result[line_col],
        )
        result["lean"] = derive_lean(
            result["p_over"],
            over_threshold=over_threshold,
            under_threshold=under_threshold,
        )
        result["confidence_tier"] = derive_confidence_tier(
            result["p_over"],
            high_distance=high_distance,
            moderate_distance=moderate_distance,
        )
    else:
        result["p_over"] = np.nan
        result["lean"] = np.nan
        result["confidence_tier"] = np.nan

    return result

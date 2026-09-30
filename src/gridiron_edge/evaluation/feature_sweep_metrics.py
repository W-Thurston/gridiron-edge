# src/gridiron_edge/evaluation/feature_sweep_metrics.py

"""Metrics and paired-bootstrap deltas for the consolidated feature sweep.

Pure computation over aligned holdout prediction arrays - no I/O, no
training. Scalar metrics reuse ``evaluation/metrics.py``. A delta is always
reported as an *improvement* (previous step minus new step for a loss), so a
positive value means the newer step is better. The bootstrap is paired: the
same resampled game indices score both steps, and the same seed and game
count give the same indices for every comparison.

Public API
----------
WinMetrics, TotalMetrics        dataclasses of scalar metrics
win_metrics, total_metrics      predictions -> scalar metrics
brier_losses, absolute_losses   per-game losses
BootstrapDelta                  improvement with interval and one-sided p
paired_bootstrap_improvement    two per-game loss vectors -> BootstrapDelta
season_improvements             per-season mean improvement
holm_adjust                     Holm-adjusted p-values
bh_adjust                       Benjamini-Hochberg adjusted p-values (FDR)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import Series

from gridiron_edge.evaluation.metrics import (
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
    log_loss,
    median_absolute_error,
    roc_auc,
)

DEFAULT_BOOTSTRAP_DRAWS: Final[int] = 2000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20260928
DEFAULT_INTERVAL_LEVEL: Final[float] = 0.90


@dataclass(frozen=True)
class WinMetrics:
    """Scalar Win metrics on one holdout set."""

    n: int
    brier: float
    log_loss: float
    ece: float
    auc: float
    calibration_slope: float
    calibration_intercept: float


@dataclass(frozen=True)
class TotalMetrics:
    """Scalar Total metrics on one holdout set."""

    n: int
    mae: float
    rmse: float
    bias: float
    median_ae: float


def _clean(values: Series | NDArray[np.float64]) -> Series:
    return pd.Series(np.asarray(values, dtype=float))


def win_metrics(p: Series | NDArray[np.float64], y: Series | NDArray[np.float64]) -> WinMetrics:
    """Compute Win metrics for probabilities ``p`` of ``HOME_WIN`` outcomes ``y``."""
    ps, ys = _clean(p), _clean(y)
    slope, intercept = calibration_slope_intercept(ps, ys)
    return WinMetrics(
        n=len(ps),
        brier=float(brier_score(ps, ys)),
        log_loss=float(log_loss(ps, ys)),
        ece=float(expected_calibration_error(ps, ys)),
        auc=float(roc_auc(ps, ys)),
        calibration_slope=slope,
        calibration_intercept=intercept,
    )


def total_metrics(
    pred: Series | NDArray[np.float64], y: Series | NDArray[np.float64]
) -> TotalMetrics:
    """Compute Total metrics; ``bias`` is mean(pred - actual)."""
    ps, ys = _clean(pred), _clean(y)
    err: Series = ps - ys
    return TotalMetrics(
        n=len(ps),
        mae=float(err.abs().mean()),
        rmse=float(np.sqrt((err**2).mean())),
        bias=float(err.mean()),
        median_ae=median_absolute_error(ps, ys),
    )


def brier_losses(p: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-game squared probability error."""
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def absolute_losses(pred: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-game absolute error."""
    return np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))


@dataclass(frozen=True)
class BootstrapDelta:
    """Improvement of a new step over the previous one.

    Attributes:
        improvement: mean(previous loss) - mean(new loss); positive is better.
        lower: Lower end of the two-sided interval.
        upper: Upper end of the two-sided interval.
        p_no_improvement: One-sided bootstrap p that the true improvement is
            at most zero, with a +1 correction so it is never exactly zero.
        n_games: Games scored.
        level: Two-sided interval level.
    """

    improvement: float
    lower: float
    upper: float
    p_no_improvement: float
    n_games: int
    level: float


def paired_bootstrap_improvement(
    loss_previous: NDArray[np.float64],
    loss_new: NDArray[np.float64],
    *,
    n_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    level: float = DEFAULT_INTERVAL_LEVEL,
) -> BootstrapDelta:
    """Paired game-bootstrap of the improvement in mean per-game loss.

    Args:
        loss_previous: Per-game losses of the previous step.
        loss_new: Per-game losses of the new step, same games and order.
        n_draws: Bootstrap resamples.
        seed: Generator seed; identical for identical game counts.
        level: Two-sided interval level (0.90 gives the 5th/95th percentiles).

    Raises:
        ValueError: If the vectors differ in length, are empty, or contain NaN.
    """
    a = np.asarray(loss_previous, dtype=float)
    b = np.asarray(loss_new, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        msg = "Loss vectors must be one-dimensional and the same length"
        raise ValueError(msg)
    if a.size == 0:
        msg = "Loss vectors are empty"
        raise ValueError(msg)
    if np.isnan(a).any() or np.isnan(b).any():
        msg = "Loss vectors contain NaN"
        raise ValueError(msg)

    diff: NDArray[np.float64] = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_draws, diff.size))
    draws: NDArray[np.float64] = diff[idx].mean(axis=1)
    tail: float = (1.0 - level) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    p_value: float = float((np.sum(draws <= 0.0) + 1) / (n_draws + 1))
    return BootstrapDelta(
        improvement=float(diff.mean()),
        lower=float(lower),
        upper=float(upper),
        p_no_improvement=p_value,
        n_games=int(diff.size),
        level=level,
    )


def season_improvements(
    loss_previous: NDArray[np.float64],
    loss_new: NDArray[np.float64],
    seasons: Series | NDArray[np.str_],
) -> dict[str, float]:
    """Mean improvement per season, keyed by season label in sorted order."""
    diff = np.asarray(loss_previous, dtype=float) - np.asarray(loss_new, dtype=float)
    labels = np.asarray(seasons).astype(str)
    if labels.shape != diff.shape:
        msg = "Season labels must align with the loss vectors"
        raise ValueError(msg)
    return {s: float(diff[labels == s].mean()) for s in sorted(set(labels.tolist()))}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def bh_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank in range(m - 1, -1, -1):
        key, p = ordered[rank]
        running = min(running, p * m / (rank + 1))
        adjusted[key] = min(1.0, running)
    return {key: adjusted[key] for key in p_values}

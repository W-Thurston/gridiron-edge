# src/gridiron_edge/evaluation/feature_sweep_diagnostics.py

"""Per-column diagnostics for the consolidated feature sweep.

Evidence only: none of these change an adopt/reject bar. Every statistic is
computed from training rows, or compares training to holdout distributions
without using holdout labels, so nothing here selects on the holdout's
outcomes.

Public API
----------
leakage_smell_test      new columns with implausibly strong training signal
population_stability    train-vs-holdout drift per column (PSI)
collinearity_report     condition number and VIF on training rows
suspect_correlations    pairwise correlations for named suspect columns
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame, Series

#: A single new column reaching this univariate AUC on training data is more
#: predictive than any legitimate pre-kickoff team-strength feature here.
LEAKAGE_AUC_THRESHOLD = 0.70
LEAKAGE_CORRELATION_THRESHOLD = 0.50
PSI_BINS = 10
PSI_ALERT = 0.25


@dataclass(frozen=True)
class LeakageFlag:
    """A column whose training signal is implausibly strong."""

    column: str
    auc: float
    correlation: float


def _univariate_auc(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Rank-based AUC of ``x`` for binary ``y`` (0.5 = no signal); NaN if degenerate."""
    y = np.asarray(y, dtype=float)
    pos = y == 1.0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(x).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def leakage_smell_test(
    x_train: DataFrame,
    y_train: Series,
    columns: list[str],
    *,
    binary_target: bool,
) -> list[LeakageFlag]:
    """Flag columns with implausibly high univariate signal on training rows.

    For a binary target the statistic is the larger of AUC and 1 - AUC; for a
    continuous target only correlation applies. Returns flags sorted by
    strength.
    """
    y = y_train.to_numpy(dtype=float)
    flags: list[LeakageFlag] = []
    for column in columns:
        values = x_train[column].to_numpy(dtype=float)
        if np.nanstd(values) == 0.0:
            continue
        corr = float(abs(np.corrcoef(values, y)[0, 1]))
        auc = _univariate_auc(values, y) if binary_target else float("nan")
        strong_auc = binary_target and max(auc, 1.0 - auc) >= LEAKAGE_AUC_THRESHOLD
        if strong_auc or corr >= LEAKAGE_CORRELATION_THRESHOLD:
            flags.append(LeakageFlag(column, auc, corr))
    return sorted(flags, key=lambda f: -f.correlation)


def population_stability(train: Series, holdout: Series, *, n_bins: int = PSI_BINS) -> float:
    """Population stability index of ``holdout`` against ``train`` quantile bins.

    Bins are the training deciles (deduplicated); a constant training column
    returns 0.0 when the holdout is constant at the same value, else ``inf``.
    """
    t = train.dropna().to_numpy(dtype=float)
    h = holdout.dropna().to_numpy(dtype=float)
    if t.size == 0 or h.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(t, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return 0.0 if np.all(h == t[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-4
    p_t = np.histogram(t, bins=edges)[0] / t.size + eps
    p_h = np.histogram(h, bins=edges)[0] / h.size + eps
    return float(np.sum((p_h - p_t) * np.log(p_h / p_t)))


@dataclass(frozen=True)
class CollinearityReport:
    """Collinearity of a standardized training design matrix.

    Attributes:
        condition_number: Ratio of largest to smallest singular value.
        vif: Variance inflation factor per column, largest first.
    """

    condition_number: float
    vif: dict[str, float]


def collinearity_report(x_train: DataFrame) -> CollinearityReport:
    """Condition number and VIF of the standardized training columns.

    VIF for column j is the j-th diagonal of the inverse correlation matrix
    (pseudo-inverse, so perfectly collinear columns give very large values
    instead of failing).
    """
    x = x_train.to_numpy(dtype=float)
    std = x.std(axis=0)
    keep = std > 0
    z = (x[:, keep] - x[:, keep].mean(axis=0)) / std[keep]
    names = [c for c, k in zip(x_train.columns, keep, strict=True) if k]
    singular = np.linalg.svd(z, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    corr = np.corrcoef(z, rowvar=False)
    vif = np.diag(np.linalg.pinv(corr))
    order = np.argsort(-vif)
    return CollinearityReport(condition, {names[i]: float(vif[i]) for i in order})


def suspect_correlations(
    frame: DataFrame, suspects: dict[str, list[str]]
) -> dict[str, dict[str, float]]:
    """Pearson correlation of each suspect column with its named partners."""
    return {
        column: {
            partner: float(frame[column].corr(frame[partner]))
            for partner in partners
            if partner in frame.columns
        }
        for column, partners in suspects.items()
        if column in frame.columns
    }

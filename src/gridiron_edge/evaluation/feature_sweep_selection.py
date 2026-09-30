# src/gridiron_edge/evaluation/feature_sweep_selection.py

"""Model-driven feature selection on training rows only.

Each procedure sees only the rows it is given (the caller passes training
rows) and picks columns by time-ordered cross-validation stability, so the
holdout never informs a choice. A placebo control is built in: random-noise
columns are added to the candidate pool alongside the real columns, and the
result reports how many of them the procedure keeps.

- Logistic: L1 stability selection. The penalty strength is chosen by
  time-ordered CV Brier on the pool with the one-standard-error rule; a
  column is kept when its coefficient is non-zero in at least
  ``min_fraction`` of the CV training folds.
- Trees: shadow-feature permutation importance. On each CV validation fold a
  fitted model's permutation importance is compared with the largest
  importance among separate noise "shadow" columns; a column scores a hit
  when it beats every shadow, and is kept when it hits in at least
  ``min_fraction`` of folds.

Public API
----------
SelectionResult         selected columns, stability, placebo survival
select_logistic         L1 stability selection
select_tree             shadow-feature permutation selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec, fit_estimator

MIN_FOLD_FRACTION: Final[float] = 0.6
N_SPLITS: Final[int] = 5
MIN_TRAIN_ROWS: Final[int] = 1500
N_REPEATS: Final[int] = 3
N_SHADOW: Final[int] = 30
MIN_PLACEBO: Final[int] = 10
PLACEBO_FRACTION: Final[float] = 0.10
SELECTION_SEED: Final[int] = 20260929
C_GRID: Final[tuple[float, ...]] = tuple(float(c) for c in np.logspace(-3.0, 0.5, 8))


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of one selection procedure.

    Attributes:
        method: Procedure name.
        selected: Real columns kept, in pool order.
        stability: Fraction of scored folds in which each real column was
            selected or scored a hit.
        n_pool: Real candidate columns.
        n_folds: Folds scored.
        placebo_candidates: Noise columns entered as candidates.
        placebo_selected: Noise candidates the procedure kept.
        detail: Method-specific numbers (chosen penalty, shadow levels).
    """

    method: str
    selected: tuple[str, ...]
    stability: dict[str, float]
    n_pool: int
    n_folds: int
    placebo_candidates: int
    placebo_selected: int
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def placebo_survival_rate(self) -> float:
        """Share of noise candidates kept (the false-positive rate)."""
        if self.placebo_candidates == 0:
            return 0.0
        return self.placebo_selected / self.placebo_candidates


def _folds(
    n_rows: int, n_splits: int, min_train_rows: int
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Time-ordered expanding folds, skipping those with too little training data."""
    # pyrefly: ignore [missing-import]
    from sklearn.model_selection import TimeSeriesSplit

    dummy = np.zeros(n_rows)
    folds = [
        (tr, va)
        for tr, va in TimeSeriesSplit(n_splits=n_splits).split(dummy)
        if len(tr) >= min_train_rows
    ]
    if not folds:
        msg = f"No CV fold has at least {min_train_rows} training rows ({n_rows} rows)"
        raise ValueError(msg)
    return folds


def _noise(index: pd.Index, prefix: str, count: int, seed: int) -> DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(len(index), count)),
        index=index,
        columns=[f"__{prefix}_{i}" for i in range(count)],
    )


def _n_placebo(n_columns: int, requested: int | None) -> int:
    if requested is not None:
        return requested
    return max(MIN_PLACEBO, round(PLACEBO_FRACTION * n_columns))


def select_logistic(
    x_train: DataFrame,
    y_train: Series,
    *,
    n_placebo: int | None = None,
    seed: int = SELECTION_SEED,
    min_fraction: float = MIN_FOLD_FRACTION,
    n_splits: int = N_SPLITS,
    min_train_rows: int = MIN_TRAIN_ROWS,
    c_grid: tuple[float, ...] = C_GRID,
) -> SelectionResult:
    """L1 stability selection for the Logistic family.

    Args:
        x_train: Training features (real columns only).
        y_train: Binary target aligned with ``x_train``.
        n_placebo: Noise candidates to add; ``None`` uses 10% of the pool
            (at least 10).
        seed: Noise seed.
        min_fraction: Fraction of folds a coefficient must be non-zero in.
        n_splits: CV splits.
        min_train_rows: Folds with fewer training rows are skipped.
        c_grid: Inverse-penalty grid, searched by time-ordered CV Brier.
    """
    # pyrefly: ignore [missing-import]
    from sklearn.linear_model import LogisticRegression

    # pyrefly: ignore [missing-import]
    from sklearn.preprocessing import StandardScaler

    real = list(x_train.columns)
    k = _n_placebo(len(real), n_placebo)
    pool = pd.concat([x_train, _noise(x_train.index, "placebo", k, seed)], axis=1)
    x = pool.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)
    folds = _folds(len(pool), n_splits, min_train_rows)

    def fit(train: NDArray[np.intp], c: float) -> tuple[LogisticRegression, StandardScaler]:
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(
            l1_ratio=1.0, solver="liblinear", C=c, max_iter=2000, random_state=seed
        )
        model.fit(scaler.transform(x[train]), y[train])
        return model, scaler

    fold_scores: dict[float, list[float]] = {}
    for c in c_grid:
        scores_c = []
        for tr, va in folds:
            model, scaler = fit(tr, c)
            p = model.predict_proba(scaler.transform(x[va]))[:, 1]
            scores_c.append(float(np.mean((p - y[va]) ** 2)))
        fold_scores[c] = scores_c
    means = {c: float(np.mean(v)) for c, v in fold_scores.items()}
    best = min(means, key=lambda c: means[c])
    # One-standard-error rule: the strongest penalty (smallest C) whose CV
    # Brier is within one standard error of the best. The CV-optimal C keeps
    # many noise columns; the placebo count in the result measures it.
    se = float(np.std(fold_scores[best], ddof=1) / np.sqrt(len(folds))) if len(folds) > 1 else 0.0
    best_c = min(c for c in c_grid if means[c] <= means[best] + se)
    scores = means

    nonzero = np.zeros(len(pool.columns))
    for tr, _ in folds:
        model, _scaler = fit(tr, best_c)
        nonzero += (model.coef_[0] != 0.0).astype(float)
    frac = nonzero / len(folds)
    keep = frac >= min_fraction
    return SelectionResult(
        method="l1_stability",
        selected=tuple(n for n, kept in zip(real, keep[: len(real)], strict=True) if kept),
        stability=dict(zip(real, frac[: len(real)].tolist(), strict=True)),
        n_pool=len(real),
        n_folds=len(folds),
        placebo_candidates=k,
        placebo_selected=int(keep[len(real) :].sum()),
        detail={"chosen_C": best_c, "cv_brier": scores[best_c]},
    )


def _loss(task: str, pred: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    if task == "classification":
        return float(np.mean((pred - truth) ** 2))
    return float(np.mean(np.abs(pred - truth)))


def select_tree(
    spec: FamilySpec,
    params: dict[str, Any],
    x_train: DataFrame,
    y_train: Series,
    *,
    n_placebo: int | None = None,
    n_shadow: int = N_SHADOW,
    seed: int = SELECTION_SEED,
    min_fraction: float = MIN_FOLD_FRACTION,
    n_splits: int = N_SPLITS,
    min_train_rows: int = MIN_TRAIN_ROWS,
    n_repeats: int = N_REPEATS,
    xgb_recalibrate: bool = False,
    threads: int | None = None,
) -> SelectionResult:
    """Shadow-feature permutation selection for a tree family.

    Args:
        spec: Tree family (Win or Total).
        params: Frozen estimator hyperparameters (no ``epa_window``); the
            caller supplies the columns for the matching EPA window.
        x_train: Training features (real columns only).
        y_train: Training target aligned with ``x_train``.
        n_placebo: Noise columns entered as candidates; ``None`` uses 10%.
        n_shadow: Separate noise columns that set each fold's bar.
        seed: Noise and permutation seed.
        min_fraction: Fraction of folds a column must beat every shadow in.
        n_splits: CV splits.
        min_train_rows: Folds with fewer training rows are skipped.
        n_repeats: Permutations per column and fold.
        xgb_recalibrate: Apply the isotonic wrap (XGBoost Win).
        threads: Estimator thread count.
    """
    real = list(x_train.columns)
    k = _n_placebo(len(real), n_placebo)
    shadow_names = [f"__shadow_{i}" for i in range(n_shadow)]
    pool = pd.concat(
        [
            x_train,
            _noise(x_train.index, "placebo", k, seed),
            _noise(x_train.index, "shadow", n_shadow, seed + 1),
        ],
        axis=1,
    )
    names = list(pool.columns)
    n_cand = len(real) + k
    x = pool.to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)
    folds = _folds(len(pool), n_splits, min_train_rows)
    rng = np.random.default_rng(seed + 2)

    hits = np.zeros(n_cand)
    shadow_levels: list[float] = []
    for tr, va in folds:
        predictor = fit_estimator(
            spec,
            params,
            pool.iloc[tr],
            y_train.iloc[tr],
            xgb_recalibrate=xgb_recalibrate,
            threads=threads,
        )
        base = _loss(spec.task, predictor(x[va]), y[va])
        working = x[va].copy()
        importance = np.zeros(len(names))
        for j in range(len(names)):
            original = working[:, j].copy()
            total = 0.0
            for _ in range(n_repeats):
                working[:, j] = original[rng.permutation(len(va))]
                total += _loss(spec.task, predictor(working), y[va]) - base
            working[:, j] = original
            importance[j] = total / n_repeats
        bar = float(max(importance[n_cand + i] for i in range(len(shadow_names))))
        shadow_levels.append(bar)
        hits += (importance[:n_cand] > bar).astype(float)

    frac = hits / len(folds)
    keep = frac >= min_fraction
    return SelectionResult(
        method="shadow_permutation",
        selected=tuple(n for n, kept in zip(real, keep[: len(real)], strict=True) if kept),
        stability=dict(zip(real, frac[: len(real)].tolist(), strict=True)),
        n_pool=len(real),
        n_folds=len(folds),
        placebo_candidates=k,
        placebo_selected=int(keep[len(real) :].sum()),
        detail={"mean_shadow_bar": float(np.mean(shadow_levels))},
    )

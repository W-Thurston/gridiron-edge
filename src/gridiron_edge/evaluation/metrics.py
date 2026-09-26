# src/gridiron_edge/evaluation/metrics.py

"""Evaluation metrics for game prediction models.

All functions accept a standard *evaluation DataFrame* produced by
``build_forecast_run_evaluation_df``, which evaluates one exact immutable
backfilled forecast run.  The schema is:

    game_id          str   - canonical YYYY_WW_AWAY_HOME identifier
    season           str   - e.g. "2024-2025"
    week             int   - NFL week number
    away_team        str
    home_team        str
    away_win_prob    float - model's predicted probability that away team wins
    away_team_won    float - 1.0 if away won, 0.0 if home won, 0.5 if tied
    model_name       str   - model purpose (e.g. "win_prob", "total")
    model_type       str   - model algorithm (e.g. "random_forest", "elo")

Public API
----------
build_forecast_run_evaluation_df   Evaluate one exact immutable backfill run;
                                    primary entry point.
summarise                    Grouped Brier/accuracy table.
calibration_table            Predicted vs actual win-rate by bucket.
brier_score                  Scalar Brier score.
log_loss                     Scalar log loss.
accuracy                     Fraction of games where argmax matches outcome.
roc_auc                      ROC-AUC.
expected_calibration_error   ECE (single-number calibration summary).
brier_decomposition          Murphy (1973) decomposition: reliability,
                             resolution, uncertainty.

----------------
brier_by_confidence_tier     Brier + calibration gap per predicted-prob bucket.
brier_by_season              Per-season Brier with delta vs mean; drift detection.
biggest_misses               Top-N games by |predicted_prob - outcome|.
calibration_slope_intercept  Logistic-fit calibration slope/intercept.
sharpness                    Variance of predicted probabilities.
season_stability             Stdev of per-season Brier scores.

Total (regression) metrics operate on a *Total evaluation DataFrame* from
``build_forecast_run_total_evaluation_df`` instead, with columns
``model_total``/``actual_total`` in place of ``away_win_prob``/
``away_team_won``:

median_absolute_error        Median |predicted - actual| total points.
interval_coverage            Actual vs nominal coverage of a residual-std
                              prediction interval.
environment_slice_metrics    Total error metrics sliced by an environment
                              dimension (e.g. dome/outdoor, temperature band).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final

import numpy as np
from numpy import dtype, float64, ndarray, signedinteger
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.datasets import loaders
from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_selection import select_forecast_run
from gridiron_edge.evaluation.forecast_store import load_forecast_events

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default confidence tiers for brier_by_confidence_tier.
# Each tuple is a half-open interval [lo, hi).  The last bucket closes at 1.0.
DEFAULT_CONFIDENCE_TIERS: Final[list[tuple[float, float]]] = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 1.01),  # upper bound > 1.0 to include exact 1.0 edge
]

# Threshold above which overconfidence in a high-confidence tier is flagged.
_HIGH_CONFIDENCE_WARN_THRESHOLD: Final[float] = 0.70
_CALIBRATION_GAP_WARN: Final[float] = 0.03

_EVALUATION_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "away_win_prob",
    "away_team_won",
    "model_name",
    "model_type",
)
_REQUIRED_OUTCOME_COLUMNS: Final[frozenset[str]] = frozenset(
    {"GAME_ID", "AWAY_SCORE", "HOME_SCORE"}
)

_TOTAL_EVALUATION_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "model_total",
    "actual_total",
    "model_name",
    "model_type",
)

# Default nominal coverage level for residual-std prediction intervals.
_DEFAULT_NOMINAL_COVERAGE: Final[float] = 0.90


# ---------------------------------------------------------------------------
# Scalar metric functions
# ---------------------------------------------------------------------------


def brier_score(p: Series, y: Series) -> float:
    """Compute the Brier score: mean squared error between probabilities and outcomes.

    Args:
        p: Predicted probabilities (floats in [0, 1]).
        y: Binary outcomes (0 or 1).

    Returns:
        Brier score (lower is better).
    """
    return ((p - y) ** 2).mean()


def log_loss(p: Series, y: Series, *, eps: float = 1e-7) -> float:
    """Compute binary log loss.

    Args:
        p: Predicted probabilities (floats in [0, 1]).
        y: Binary outcomes (0 or 1).
        eps: Clipping epsilon to avoid log(0).

    Returns:
        Log loss (lower is better).
    """
    p_clipped: Series = p.clip(eps, 1 - eps)
    return -(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)).mean()


def accuracy(p: Series, y: Series) -> float:
    """Fraction of games where the predicted favorite actually won.

    Args:
        p: Predicted probabilities for the away team.
        y: Binary outcomes (1 = away team won).

    Returns:
        Accuracy (higher is better).
    """
    predicted_away_wins: Series[bool] = p >= 0.5
    return (predicted_away_wins == y).mean()


def roc_auc(p: Series, y: Series) -> float:
    """Compute ROC-AUC.

    Args:
        p: Predicted probabilities (floats in [0, 1]).
        y: Binary outcomes (0 or 1).

    Returns:
        ROC-AUC score (higher is better, 0.5 = random).
    """
    # pyrefly: ignore [missing-import]
    from sklearn.metrics import roc_auc_score

    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def expected_calibration_error(p: Series, y: Series, *, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Divides predictions into equal-width bins and computes the
    weighted average gap between mean predicted probability and
    actual win rate within each bin.

    Args:
        p: Predicted probabilities (floats in [0, 1]).
        y: Binary outcomes (0 or 1).
        n_bins: Number of equal-width calibration bins.

    Returns:
        ECE (lower is better; 0.0 = perfectly calibrated).
    """
    bins: ndarray[tuple[Any, ...], dtype[float64]] = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices: ndarray[tuple[Any, ...], dtype[signedinteger]] = np.digitize(p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece_total = 0.0
    n_total: int = len(p)
    for i in range(n_bins):
        mask = bin_indices == i
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        mean_pred = float(p[mask].mean())
        mean_actual = float(y[mask].mean())
        ece_total += (n_bin / n_total) * abs(mean_pred - mean_actual)
    return ece_total


def brier_decomposition(p: Series, y: Series, *, n_bins: int = 10) -> dict[str, float]:
    """Decompose the Brier score into reliability, resolution, and uncertainty.

    Based on the Murphy (1973) decomposition:

        BS = Reliability - Resolution + Uncertainty

    where:
        Reliability  - calibration error (lower is better).  Mean squared
                       gap between the model's predicted probabilities and
                       the observed win rate within each forecast bin.
        Resolution   - sharpness (higher is better).  Mean squared deviation
                       of each bin's observed win rate from the overall base
                       rate.  A model that spreads predictions wide and is
                       right about that spread has high resolution.
        Uncertainty  - irreducible noise, ``base_rate * (1 - base_rate)``.
                       Fixed for a given dataset; not affected by the model.

    The identity ``BS ≈ Reliability - Resolution + Uncertainty`` holds
    exactly when forecasts are discrete (same predicted value within each
    bin).  For continuous probability outputs the within-bin variance
    introduces a small approximation error (typically < 0.002) that
    decreases with more bins.  This is expected and documented in the
    literature - the decomposition is a diagnostic, not an accounting identity.

    Args:
        p: Predicted probabilities (floats in [0, 1]).
        y: Binary outcomes (0 or 1).
        n_bins: Number of equal-width forecast bins (default 10).

    Returns:
        Dict with keys: ``"reliability"``, ``"resolution"``, ``"uncertainty"``,
        ``"brier_score"``.  All values are floats rounded to 6 decimal places.
    """
    n_total: int = len(p)
    base_rate: float = y.mean()
    uncertainty: float = base_rate * (1.0 - base_rate)

    bins: ndarray[tuple[Any, ...], dtype[float64]] = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices: ndarray[tuple[Any, ...], dtype[signedinteger]] = np.clip(
        np.digitize(p, bins) - 1, 0, n_bins - 1
    )

    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        # Use the bin's mean observed rate (not mean predicted) as the
        # representative forecast for the reliability term.  This is the
        # standard Murphy (1973) formulation and ensures the identity
        # BS = Reliability - Resolution + Uncertainty holds exactly.
        obs_rate = float(y[mask].mean())
        mean_pred = float(p[mask].mean())
        reliability += (n_bin / n_total) * (mean_pred - obs_rate) ** 2
        resolution += (n_bin / n_total) * (obs_rate - base_rate) ** 2

    bs: float = ((p - y) ** 2).mean()
    return {
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "brier_score": round(bs, 6),
    }


def calibration_slope_intercept(
    p: Series,
    y: Series,
    *,
    eps: float = 1e-7,
) -> tuple[float, float]:
    """Fit calibration slope and intercept via logistic regression on logit(p).

    Regresses outcome ``y`` on a single feature, ``logit(clip(p, eps, 1-eps))``,
    using an unregularized logistic regression so the fitted coefficient is
    not shrunk toward zero. A perfectly calibrated model has slope ≈ 1.0 and
    intercept ≈ 0.0: slope < 1 indicates the model is too extreme (overconfident
    at the tails), slope > 1 indicates it is too conservative, and a nonzero
    intercept indicates a systematic bias toward one side.

    Tied games (``y == 0.5``) are excluded before fitting, since the fit
    requires a strictly binary outcome; ``p`` is unaffected.

    Args:
        p: Predicted probabilities (floats in [0, 1]).
        y: Outcomes - 0 or 1, or 0.5 for a tied game (excluded from the fit).
        eps: Clipping epsilon to avoid infinite logits at 0 or 1.

    Returns:
        ``(slope, intercept)``. Both are ``nan`` if fewer than two distinct
        binary outcomes remain after excluding ties (the fit is undefined).
    """
    # pyrefly: ignore [missing-import]
    from sklearn.linear_model import LogisticRegression

    binary_mask: Series[bool] = y.isin([0.0, 1.0])
    p_binary: Series = p[binary_mask]
    y_binary: Series = y[binary_mask]
    if y_binary.nunique() < 2:
        return float("nan"), float("nan")

    p_clipped: ndarray = p_binary.clip(eps, 1 - eps).to_numpy()
    logit: ndarray = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)

    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit, y_binary.to_numpy())
    slope: float = float(model.coef_[0][0])
    intercept: float = float(model.intercept_[0])
    return slope, intercept


def sharpness(p: Series) -> float:
    """Compute the sharpness (variance) of predicted probabilities.

    Sharpness measures how spread out a model's forecasts are, independent of
    whether those forecasts are correct. A model that always predicts near
    0.5 has low sharpness (uninformative); one that confidently predicts near
    0 or 1 has high sharpness. Unlike ``brier_decomposition``'s ``resolution``
    term (which is conditioned on observed outcome within each bin), this is
    the raw variance of the predictions themselves.

    Args:
        p: Predicted probabilities (floats in [0, 1]).

    Returns:
        Variance of ``p`` (higher means more confident/spread-out forecasts).
    """
    return float(np.var(p.to_numpy(dtype=float)))


def season_stability(df: DataFrame) -> float:
    """Compute the standard deviation of per-season Brier scores.

    Built on top of ``brier_by_season``. A model whose Brier score varies
    widely from season to season is less trustworthy going forward than one
    with a stable, consistent Brier score across seasons, even if their
    average Brier scores are similar.

    Args:
        df: Evaluation DataFrame from ``build_forecast_run_evaluation_df`` or
            ``build_forecast_run_evaluation_df``.

    Returns:
        Standard deviation (ddof=1) of per-season Brier scores. ``nan`` if
        fewer than two seasons are present (matches ``pandas.Series.std``'s
        behavior for a single observation).
    """
    by_season: DataFrame = brier_by_season(df)
    if by_season.empty:
        return float("nan")
    return float(by_season["brier"].std())


# ---------------------------------------------------------------------------
# Total (regression) metric functions
# ---------------------------------------------------------------------------


def median_absolute_error(y_pred: Series, y_true: Series) -> float:
    """Compute the median absolute error between predicted and actual totals.

    Less sensitive to outlier games (e.g. a blowout) than mean absolute
    error, which is already available from persisted training metadata.

    Args:
        y_pred: Predicted total points.
        y_true: Actual total points.

    Returns:
        Median of ``|y_pred - y_true|`` (lower is better).
    """
    return float((y_pred - y_true).abs().median())


def interval_coverage(
    y_true: Series,
    y_pred: Series,
    *,
    residual_std: float,
    nominal: float = _DEFAULT_NOMINAL_COVERAGE,
) -> dict[str, float]:
    """Compute actual vs. nominal coverage of a residual-std prediction interval.

    Total game models do not persist a per-game prediction interval, so the
    interval here is constructed from the evaluated run's own holdout
    residual standard deviation: a symmetric normal-quantile band around each
    point prediction, ``y_pred ± z * residual_std``, where ``z`` is the
    two-sided normal quantile for ``nominal`` coverage. This is a descriptive
    evaluation diagnostic (how wide would an interval need to be, and does
    that width actually achieve its nominal coverage on this run), not a
    live-serving prediction interval.

    Args:
        y_true: Actual total points.
        y_pred: Predicted total points.
        residual_std: Standard deviation of holdout residuals for this run.
        nominal: Nominal (target) coverage level, e.g. 0.90 for a 90% interval.

    Returns:
        Dict with keys ``"nominal_coverage"``, ``"actual_coverage"``, and
        ``"mean_interval_width"``. ``actual_coverage`` and
        ``mean_interval_width`` are ``nan`` if ``residual_std`` is not a
        positive finite number.
    """
    if not np.isfinite(residual_std) or residual_std <= 0:
        return {
            "nominal_coverage": nominal,
            "actual_coverage": float("nan"),
            "mean_interval_width": float("nan"),
        }

    z: float = NormalDist().inv_cdf((1.0 + nominal) / 2.0)
    half_width: float = z * residual_std
    lower: Series = y_pred - half_width
    upper: Series = y_pred + half_width
    within: Series[bool] = (y_true >= lower) & (y_true <= upper)
    return {
        "nominal_coverage": nominal,
        "actual_coverage": float(within.mean()),
        "mean_interval_width": float(2.0 * half_width),
    }


def environment_slice_metrics(df: DataFrame, *, dimension: str) -> DataFrame:
    """Break down Total error metrics by an environment dimension.

    Args:
        df: Total evaluation DataFrame (``model_total``/``actual_total``
            columns) with an additional slice-label column named
            ``dimension`` already computed by the caller (e.g. a boolean
            ``IS_DOME`` column, or a precomputed temperature/wind band).
        dimension: Name of the slice-label column to group by.

    Returns:
        DataFrame with one row per non-empty slice, columns:

            <dimension>            the slice label
            n_games         int
            mae             float - mean absolute error within the slice
            median_absolute_error  float
            bias            float - mean(predicted - actual); (+) = over-predicts

    Raises:
        ValueError: If ``dimension`` is not a column in ``df``.
    """
    if dimension not in df.columns:
        raise ValueError(f"dimension column not found in evaluation frame: {dimension!r}")

    rows: list[dict] = []
    for label, group in df.groupby(dimension):
        y_pred: Series = group["model_total"]
        y_true: Series = group["actual_total"]
        errors: Series = y_pred - y_true
        rows.append(
            {
                dimension: label,
                "n_games": len(group),
                "mae": round(float(errors.abs().mean()), 4),
                "median_absolute_error": round(median_absolute_error(y_pred, y_true), 4),
                "bias": round(float(errors.mean()), 4),
            }
        )
    return DataFrame(rows)


# ---------------------------------------------------------------------------
# Archive access
# ---------------------------------------------------------------------------


def _empty_evaluation_df() -> DataFrame:
    """Return an empty frame with the canonical evaluation columns."""
    return DataFrame(columns=list(_EVALUATION_COLUMNS))


def _join_completed_outcomes(
    predictions: DataFrame,
    *,
    repo: Path,
) -> DataFrame:
    """Join canonical completed outcomes to prediction rows."""
    if predictions.empty:
        return _empty_evaluation_df()
    if "game_id" not in predictions.columns:
        raise ValueError("Prediction rows are missing required column: game_id")

    normalized = predictions.copy()
    if normalized["game_id"].isna().any():
        raise ValueError("Prediction game IDs must not be null.")
    normalized["game_id"] = normalized["game_id"].astype(str)
    if normalized["game_id"].str.strip().eq("").any():
        raise ValueError("Prediction game IDs must not be empty.")

    games = loaders.load_games(repo)
    missing = sorted(_REQUIRED_OUTCOME_COLUMNS - set(games.columns))
    if missing:
        raise ValueError("Canonical games are missing required columns: " + ", ".join(missing))
    outcomes = games.loc[:, ["GAME_ID", "AWAY_SCORE", "HOME_SCORE"]].copy()
    if outcomes["GAME_ID"].isna().any():
        raise ValueError("Canonical game IDs must not be null.")
    outcomes["GAME_ID"] = outcomes["GAME_ID"].astype(str)
    if outcomes["GAME_ID"].str.strip().eq("").any():
        raise ValueError("Canonical game IDs must not be empty.")
    if outcomes["GAME_ID"].duplicated().any():
        duplicates = sorted(
            outcomes.loc[outcomes["GAME_ID"].duplicated(keep=False), "GAME_ID"].unique().tolist()
        )
        raise ValueError("Canonical games contain duplicate game IDs: " + ", ".join(duplicates))

    outcomes["AWAY_SCORE"] = pd.to_numeric(outcomes["AWAY_SCORE"], errors="coerce")
    outcomes["HOME_SCORE"] = pd.to_numeric(outcomes["HOME_SCORE"], errors="coerce")
    outcomes = outcomes.dropna(subset=["AWAY_SCORE", "HOME_SCORE"]).copy()
    outcomes["away_team_won"] = 0.0
    outcomes.loc[outcomes["AWAY_SCORE"] > outcomes["HOME_SCORE"], "away_team_won"] = 1.0
    outcomes.loc[outcomes["AWAY_SCORE"] == outcomes["HOME_SCORE"], "away_team_won"] = 0.5

    joined = normalized.merge(
        outcomes.loc[:, ["GAME_ID", "away_team_won"]],
        how="inner",
        left_on="game_id",
        right_on="GAME_ID",
        validate="many_to_one",
    ).drop(columns=["GAME_ID"])
    joined["away_team_won"] = joined["away_team_won"].astype(float)
    available = [column for column in _EVALUATION_COLUMNS if column in joined.columns]
    return joined.loc[:, available].reset_index(drop=True)


def build_forecast_run_evaluation_df(
    *,
    run_id: str,
    model_name: str,
    model_type: str,
    repo: Path,
) -> DataFrame:
    """Evaluate one exact immutable backfilled forecast run."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty.")
    events = load_forecast_events(
        run_id=run_id,
        model_name=model_name,
        model_type=model_type,
        role=ForecastRole.BACKFILLED,
        repo=repo,
    )
    selected = select_forecast_run(events, run_id=run_id)
    if not selected.found:
        raise ValueError(f"Backfilled forecast run is unavailable: {run_id!r}.")
    selected_events = selected.events.copy()
    if set(selected_events["run_id"].astype(str)) != {run_id}:
        raise ValueError("Selected forecast events have the wrong run identity.")
    if set(selected_events["model_name"].astype(str)) != {model_name}:
        raise ValueError("Selected forecast events have the wrong model name.")
    if set(selected_events["model_type"].astype(str)) != {model_type}:
        raise ValueError("Selected forecast events have the wrong model type.")
    if set(selected_events["role"].astype(str)) != {ForecastRole.BACKFILLED.value}:
        raise ValueError("Selected forecast events must be backfilled.")
    if selected_events["game_id"].astype(str).duplicated().any():
        raise ValueError("Selected forecast run contains duplicate game IDs.")
    return _join_completed_outcomes(selected_events, repo=repo)


def _join_total_completed_outcomes(
    predictions: DataFrame,
    *,
    repo: Path,
) -> DataFrame:
    """Join canonical completed outcomes (actual total points) to Total predictions."""
    if predictions.empty:
        return DataFrame(columns=list(_TOTAL_EVALUATION_COLUMNS))
    if "game_id" not in predictions.columns:
        raise ValueError("Prediction rows are missing required column: game_id")

    normalized = predictions.copy()
    if normalized["game_id"].isna().any():
        raise ValueError("Prediction game IDs must not be null.")
    normalized["game_id"] = normalized["game_id"].astype(str)
    if normalized["game_id"].str.strip().eq("").any():
        raise ValueError("Prediction game IDs must not be empty.")

    games = loaders.load_games(repo)
    missing = sorted(_REQUIRED_OUTCOME_COLUMNS - set(games.columns))
    if missing:
        raise ValueError("Canonical games are missing required columns: " + ", ".join(missing))
    outcomes = games.loc[:, ["GAME_ID", "AWAY_SCORE", "HOME_SCORE"]].copy()
    if outcomes["GAME_ID"].isna().any():
        raise ValueError("Canonical game IDs must not be null.")
    outcomes["GAME_ID"] = outcomes["GAME_ID"].astype(str)
    if outcomes["GAME_ID"].str.strip().eq("").any():
        raise ValueError("Canonical game IDs must not be empty.")
    if outcomes["GAME_ID"].duplicated().any():
        duplicates = sorted(
            outcomes.loc[outcomes["GAME_ID"].duplicated(keep=False), "GAME_ID"].unique().tolist()
        )
        raise ValueError("Canonical games contain duplicate game IDs: " + ", ".join(duplicates))

    outcomes["AWAY_SCORE"] = pd.to_numeric(outcomes["AWAY_SCORE"], errors="coerce")
    outcomes["HOME_SCORE"] = pd.to_numeric(outcomes["HOME_SCORE"], errors="coerce")
    outcomes = outcomes.dropna(subset=["AWAY_SCORE", "HOME_SCORE"]).copy()
    outcomes["actual_total"] = outcomes["AWAY_SCORE"] + outcomes["HOME_SCORE"]

    joined = normalized.merge(
        outcomes.loc[:, ["GAME_ID", "actual_total"]],
        how="inner",
        left_on="game_id",
        right_on="GAME_ID",
        validate="many_to_one",
    ).drop(columns=["GAME_ID"])
    joined["actual_total"] = joined["actual_total"].astype(float)
    available = [column for column in _TOTAL_EVALUATION_COLUMNS if column in joined.columns]
    return joined.loc[:, available].reset_index(drop=True)


def build_forecast_run_total_evaluation_df(
    *,
    run_id: str,
    model_type: str,
    repo: Path,
) -> DataFrame:
    """Evaluate one exact immutable backfilled Total forecast run.

    Mirrors ``build_forecast_run_evaluation_df`` for the Total (regression)
    task: the model purpose is fixed to ``"total"`` and the returned frame
    carries ``model_total``/``actual_total`` in place of
    ``away_win_prob``/``away_team_won``.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty.")
    events = load_forecast_events(
        run_id=run_id,
        model_name="total",
        model_type=model_type,
        role=ForecastRole.BACKFILLED,
        repo=repo,
    )
    selected = select_forecast_run(events, run_id=run_id)
    if not selected.found:
        raise ValueError(f"Backfilled forecast run is unavailable: {run_id!r}.")
    selected_events = selected.events.copy()
    if set(selected_events["run_id"].astype(str)) != {run_id}:
        raise ValueError("Selected forecast events have the wrong run identity.")
    if set(selected_events["model_name"].astype(str)) != {"total"}:
        raise ValueError("Selected forecast events have the wrong model name.")
    if set(selected_events["model_type"].astype(str)) != {model_type}:
        raise ValueError("Selected forecast events have the wrong model type.")
    if set(selected_events["role"].astype(str)) != {ForecastRole.BACKFILLED.value}:
        raise ValueError("Selected forecast events must be backfilled.")
    if selected_events["game_id"].astype(str).duplicated().any():
        raise ValueError("Selected forecast run contains duplicate game IDs.")
    if selected_events["model_total"].isna().any():
        raise ValueError("Selected Total forecast run has missing model_total values.")
    return _join_total_completed_outcomes(selected_events, repo=repo)


# ---------------------------------------------------------------------------
# Aggregate metric tables
# ---------------------------------------------------------------------------


def summarise(df: DataFrame, *, group_by: str = "season") -> DataFrame:
    """Compute grouped Brier score and accuracy summary.

    Args:
        df: Evaluation DataFrame from ``build_forecast_run_evaluation_df``.
        group_by: Column to group by - one of ``"season"``, ``"week"``,
            ``"model_name"``, or ``"model_type"``.

    Returns:
        Summary DataFrame with columns: [group_by], n_games, brier, accuracy.

    Raises:
        ValueError: If ``group_by`` is not a recognised column.
    """
    valid: set[str] = {"season", "week", "model_name", "model_type"}
    if group_by not in valid:
        raise ValueError(f"group_by must be one of {valid!r}, got {group_by!r}")

    groups = df.groupby(group_by)
    rows: list[dict] = []
    for name, grp in groups:
        gp: Series = grp["away_win_prob"]
        gy: Series = grp["away_team_won"]
        rows.append(
            {
                group_by: name,
                "n_games": len(grp),
                "brier": round(brier_score(gp, gy), 5),
                "accuracy": round(accuracy(gp, gy), 5),
            }
        )

    return DataFrame(rows)


def calibration_table(df: DataFrame, *, n_buckets: int = 10) -> DataFrame:
    """Build a calibration table: predicted probability bucket vs actual win rate.

    Args:
        df: Evaluation DataFrame from ``build_forecast_run_evaluation_df``.
        n_buckets: Number of equal-width probability buckets (default 10).

    Returns:
        DataFrame with columns: bucket_lo, bucket_hi, bucket_mid, n_games,
        mean_predicted_prob, actual_win_rate, calibration_gap.

        Column names are chosen to match the expectations of
        ``diagnostics.py`` plot functions.
    """
    p: Series = df["away_win_prob"]
    y: Series = df["away_team_won"]

    edges: ndarray[tuple[Any, ...], dtype[float64]] = np.linspace(0.0, 1.0, n_buckets + 1)
    rows: list[dict] = []
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & (p < hi)
        if lo == edges[-2]:  # last bucket: include right edge
            mask = (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(p[mask].mean())
        actual_rate = float(y[mask].mean())
        rows.append(
            {
                "bucket_lo": round(lo, 2),
                "bucket_hi": round(hi, 2),
                "bucket_mid": round((lo + hi) / 2, 2),
                "n_games": n,
                "mean_predicted_prob": round(mean_pred, 4),
                "actual_win_rate": round(actual_rate, 4),
                "calibration_gap": round(mean_pred - actual_rate, 4),
            }
        )
    return DataFrame(rows)


# ---------------------------------------------------------------------------
# Report-quality metric functions
# ---------------------------------------------------------------------------


def brier_by_confidence_tier(
    df: DataFrame,
    *,
    tiers: list[tuple[float, float]] | None = None,
) -> DataFrame:
    """Break down Brier score and calibration gap by predicted-probability tier.

    Groups games by the model's predicted win probability and reports accuracy
    within each confidence band.  High-confidence tiers with a large calibration
    gap indicate overconfidence - the primary betting danger signal.

    The ``predicted_avg`` column shows what the model actually predicted on
    average within the tier (not the tier midpoint), making it useful for
    diagnosing exact overconfidence magnitude.

    Args:
        df: Evaluation DataFrame from ``build_forecast_run_evaluation_df``.
        tiers: List of ``(lo, hi)`` half-open intervals defining confidence
            bands.  Defaults to ``DEFAULT_CONFIDENCE_TIERS`` - four bands:
            50-60 %, 60-70 %, 70-80 %, 80-100 %.

    Returns:
        DataFrame with one row per non-empty tier, columns:

            tier            str   - e.g. "60-70%"
            n_games         int
            brier           float - Brier score within tier
            predicted_avg   float - mean predicted probability within tier
            actual_win_rate float - fraction of games the predicted team won
            calibration_gap float - predicted_avg - actual_win_rate
                                    (+) = overconfident, (-) = underconfident
    """
    resolved_tiers: list[tuple[float, float]] = tiers or DEFAULT_CONFIDENCE_TIERS
    p: Series = df["away_win_prob"]
    y: Series = df["away_team_won"]

    rows: list[dict] = []
    for lo, hi in resolved_tiers:
        # Use the model's predicted side as the "confident" side:
        # when p < 0.5 the model is confident about the home team.
        # Confidence = max(p, 1-p) so the tier label always reflects
        # the model's stated certainty regardless of direction.
        confidence: Series = p.where(p >= 0.5, 1.0 - p)  # type: ignore[operator]
        # Flip y so "1" always means the confident team won
        y_aligned: Series = y.where(p >= 0.5, 1 - y)  # type: ignore[operator]
        p_aligned: Series = confidence

        mask: Series[bool] = (confidence >= lo) & (confidence < hi)
        if lo >= 1.0 or hi > 1.0:  # last bucket closes at 1.0
            mask = confidence >= lo

        n: int = mask.sum()
        if n == 0:
            continue

        tier_lo_pct = int(lo * 100)
        tier_hi_pct: int = 100 if hi > 1.0 else int(hi * 100)
        tier_label: str = f"{tier_lo_pct}-{tier_hi_pct}%"

        gp = p_aligned[mask]
        gy = y_aligned[mask]

        pred_avg = float(gp.mean())
        actual_rate = float(gy.mean())

        rows.append(
            {
                "confidence_tier": tier_label,
                "n_games": n,
                "brier": round(brier_score(gp, gy), 5),
                "predicted_avg": round(pred_avg, 4),
                "actual_win_rate": round(actual_rate, 4),
                "calibration_gap": round(pred_avg - actual_rate, 4),
            }
        )

    return DataFrame(rows)


def brier_by_season(df: DataFrame) -> DataFrame:
    """Compute per-season Brier score with delta vs overall mean.

    Useful for detecting concept drift - a model whose Brier score is
    increasing season-over-season may be degrading relative to a shifting
    NFL environment.

    The ``delta_vs_mean`` column is positive when the season is *worse* than
    average (higher Brier) and negative when it is *better* than average.

    Args:
        df: Evaluation DataFrame from ``build_forecast_run_evaluation_df``.

    Returns:
        DataFrame sorted by season with columns:

            season          str   - e.g. "2024-2025"
            n_games         int
            brier           float
            delta_vs_mean   float - season_brier - mean_brier across all seasons
            trend           str   - "✓" (below mean), "~" (within ±0.005 of mean),
                                    "⚠" (above mean by >0.005)
    """
    trend_threshold: float = 0.005

    seasons: list[str] = sorted(df["season"].unique())
    rows: list[dict] = []

    for season in seasons:
        mask: Series[bool] = df["season"] == season
        gp: Series = df.loc[mask, "away_win_prob"]
        gy: Series = df.loc[mask, "away_team_won"]
        rows.append(
            {
                "season": season,
                "n_games": mask.sum(),
                "brier": round(brier_score(gp, gy), 5),
            }
        )

    result = DataFrame(rows)
    if result.empty:
        return result

    mean_brier: float = result["brier"].mean()
    result["delta_vs_mean"] = (result["brier"] - mean_brier).round(5)

    def _trend(delta: float) -> str:
        if delta > trend_threshold:
            return "⚠"
        if delta < -trend_threshold:
            return "✓"
        return "~"

    result["trend"] = result["delta_vs_mean"].apply(_trend)
    return result


def biggest_misses(df: DataFrame, *, n: int = 10) -> DataFrame:
    """Surface the N games where the model was most wrong.

    Ranks games by the magnitude of the model's error - ``|predicted_prob -
    outcome|`` - where outcome is 1 if the predicted team won, 0 if they lost.
    A game predicted at 85 % where the favorite lost has an error of 0.85.

    The ``predicted_team`` and ``actual_result`` columns make the output
    readable without reference to the away/home convention.

    Args:
        df: Evaluation DataFrame from ``build_forecast_run_evaluation_df``.
        n: Number of top misses to return (default 10).

    Returns:
        DataFrame with columns:

            season          str
            week            int
            away_team       str
            home_team       str
            predicted_team  str   - the team the model was most confident about
            confidence      float - model's stated win probability for that team
            actual_result   str   - "WIN" or "LOSS" for the predicted team
            error           float - |confidence - int(actual_result == "WIN")|
    """
    p: Series = df["away_win_prob"]
    y: Series = df["away_team_won"]

    if df.empty:
        return DataFrame(
            columns=[
                "season",
                "week",
                "away_team",
                "home_team",
                "predicted_team",
                "confidence",
                "actual_result",
                "error",
            ]
        )

    # Align to "confident side": always express as confidence in the team
    # the model favoured, regardless of home/away convention.
    confident_away: Series = p >= 0.5
    confidence: Series = p.where(confident_away, 1.0 - p)  # type: ignore[operator]
    predicted_team: Series = df["away_team"].where(confident_away, df["home_team"])
    confident_won: Series = y.where(confident_away, 1 - y)  # type: ignore[operator]

    error: Series = (confidence - confident_won.astype(float)).abs()

    result: DataFrame = df.loc[:, ["season", "week", "away_team", "home_team"]].copy()
    result["predicted_team"] = predicted_team.values
    result["confidence"] = confidence.round(4).values
    result["actual_result"] = confident_won.map({1: "WIN", 0: "LOSS"}).values  # type: ignore[call-overload]
    result["error"] = error.round(4).values

    # pyrefly: ignore [no-matching-overload]
    return result.sort_values("error", ascending=False).head(n).reset_index(drop=True)

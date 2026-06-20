# src/gridiron_edge/models/game_prediction/post_process.py
"""Post-processing enrichment for game prediction outputs.

Derives additional outputs from base model win probabilities without
modifying model internals. Every predictor that outputs ``home_win_prob``
can be enriched by calling ``enrich_predictions()`` on its output
DataFrame.

The spread derivation uses the probit link: the relationship between
win probability and point spread is mediated by the margin-of-victory
distribution, which is approximately normal in the NFL. The conversion
is::

    model_spread = -sigma * Phi_inv(home_win_prob)

where *sigma* is the standard deviation of the margin-of-victory
distribution (approx 13.86 league-wide) and *Phi_inv* is the inverse
normal CDF. Sigma is calibrated per ``(model_name, model_type)`` pair
for maximum accuracy.

Model identification (Workstream 2):
    Models are identified by the pair ``(model_name, model_type)`` —
    e.g. ``("win_prob", "random_forest")``. Sigma and margin_std maps
    are keyed by this tuple. Calibrators live alongside model artifacts
    at ``data/models/{model_name}/{model_type}/calibrator.joblib``.

Spread + sigma calibration:
    win_prob_to_spread       home_win_prob -> NFL point spread
    spread_to_win_prob       NFL point spread -> home_win_prob (inverse)
    calibrate_spread_sigma   Fit sigma from historical predictions + outcomes
    register_sigma           Add a calibrated sigma to the in-memory map
    get_sigma                Look up per-model calibrated sigma

Isotonic recalibration:
    fit_recalibration        Fit isotonic mapping with temporal split
    apply_recalibration      Apply fitted calibrator to probabilities
    save_calibrator          Persist calibrator alongside the model artifact
    load_calibrator          Load calibrator (None if absent)

Uncertainty bands + confidence tiers:
    compute_margin_std       Residual std from predicted vs actual margins
    get_margin_std           Look up per-model margin std
    win_prob_bands           Derive (win_prob_lo, win_prob_hi) credible interval
    classify_confidence_tier Band width -> High / Moderate / Low

Projected scores:
    projected_scores         Derive home/away scores from spread + total

Orchestrator:
    enrich_predictions       Apply all enrichments to a predictions DataFrame
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Any, Final, Literal

# pyrefly: ignore [missing-import]
import joblib
import numpy as np
from numpy import dtype, float64, ndarray
import pandas as pd
from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
from scipy.optimize import minimize_scalar

# pyrefly: ignore [missing-import]
from scipy.stats import norm

# pyrefly: ignore [missing-import]
from sklearn.isotonic import IsotonicRegression

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NFL historical margin-of-victory standard deviation. Used as the
# fallback when no per-model calibrated sigma is available. Derived
# from league-wide game results 2000-2024.
_NFL_DEFAULT_SIGMA: Final[float] = 13.86

# Probability clamp bounds -- prevents +/-inf from the probit inverse
# at p = 0 or p = 1.
_PROB_FLOOR: Final[float] = 0.001
_PROB_CEIL: Final[float] = 0.999

# Sigma calibration search bounds -- reasonable NFL range.
# Values below ~8 imply unrealistically tight games; above ~22 implies
# unrealistically volatile margins.
_SIGMA_LO: Final[float] = 8.0
_SIGMA_HI: Final[float] = 22.0

# Per-model calibrated sigma values, keyed by (model_name, model_type).
# When a pair is not found here, ``get_sigma`` falls back to
# ``_NFL_DEFAULT_SIGMA``.
#
# Calibrated 2026-06-04 via ``calibrate_spread_sigma`` against the full
# prediction archive after the TimeSeriesSplit champion retrain. The Elo
# entry carries the legacy v2 calibration; re-run ``calibrate_spread_sigma``
# after the next Elo backfill to refresh.
#
# TODO: Wire sigma calibration into the training harness so this map
# is populated automatically when a new model is trained.
_MODEL_SIGMAS: dict[tuple[str, str], float] = {
    ("win_prob", "random_forest"): 10.6252,
    ("win_prob", "xgboost"): 11.4309,
    ("win_prob", "logistic"): 11.9914,
    ("win_prob", "elo"): 13.60,
}

# Default z-score for credible intervals. 1.645 corresponds to a 90% CI.
_DEFAULT_Z: Final[float] = 1.645

# Default margin standard deviation (league-wide fallback). This is the
# standard deviation of (predicted_margin - actual_margin) residuals.
_DEFAULT_MARGIN_STD: Final[float] = 13.45

# Per-model margin std, keyed by (model_name, model_type). Derived from
# sqrt(MSE) at optimal sigma during sigma calibration (2026-06-04). The
# Elo entry carries the legacy v2 calibration; re-run after the next
# Elo backfill to refresh.
#
# TODO: Wire margin_std computation into the training harness.
_MODEL_MARGIN_STDS: dict[tuple[str, str], float] = {
    ("win_prob", "random_forest"): 13.54,
    ("win_prob", "xgboost"): 13.34,
    ("win_prob", "logistic"): 13.29,
    ("win_prob", "elo"): 13.89,
}

# Confidence tier: how far is the prediction from pick'em (0.5)?
#   distance >= HIGH  -> "High"     (prob >= 0.70 or <= 0.30)
#   distance >= MOD   -> "Moderate" (prob 0.60-0.70 or 0.30-0.40)
#   distance <  MOD   -> "Low"      (prob 0.40-0.60, near toss-up)
_TIER_HIGH_PROB: Final[float] = 0.70
_TIER_MODERATE_PROB: Final[float] = 0.60

# Calibrator artifact filename. Lives in the same directory as the model
# artifact: ``data/models/{model_name}/{model_type}/calibrator.joblib``.
_CALIBRATOR_FILENAME: Final[str] = "calibrator.joblib"

# Default number of most-recent seasons held out for calibrator validation.
_DEFAULT_HOLDOUT_SEASONS: Final[int] = 2


# ---------------------------------------------------------------------------
# Spread <-> Win Probability Conversion
# ---------------------------------------------------------------------------


def win_prob_to_spread(
    home_win_prob: float,
    *,
    sigma: float = _NFL_DEFAULT_SIGMA,
) -> float:
    """Convert a home-team win probability to an NFL point spread.

    Uses the probit (inverse-normal) link function. The returned spread
    follows NFL convention: **negative** means home team is favored,
    **positive** means away team is favored.

    Examples::

        win_prob_to_spread(0.50)  ->  0.0   (pick'em)
        win_prob_to_spread(0.75)  -> -9.35  (home favored by ~9 points)
        win_prob_to_spread(0.25)  -> +9.35  (away favored by ~9 points)

    Args:
        home_win_prob: Predicted probability that the home team wins,
            in the range [0, 1]. Values at the extremes are clamped
            to (_PROB_FLOOR, _PROB_CEIL) to avoid +/-inf.
        sigma: Standard deviation of the margin-of-victory distribution.
            Defaults to ``_NFL_DEFAULT_SIGMA`` (13.86).

    Returns:
        Point spread (float). Negative = home favored.
    """
    clamped: float = float(np.clip(home_win_prob, _PROB_FLOOR, _PROB_CEIL))
    return -sigma * float(norm.ppf(clamped))


def spread_to_win_prob(
    spread: float,
    *,
    sigma: float = _NFL_DEFAULT_SIGMA,
) -> float:
    """Convert an NFL point spread to a home-team win probability.

    Inverse of ``win_prob_to_spread``. Uses the normal CDF.

    Args:
        spread: Point spread (negative = home favored).
        sigma: Standard deviation of the margin-of-victory distribution.
            Defaults to ``_NFL_DEFAULT_SIGMA`` (13.86).

    Returns:
        Home-team win probability in (0, 1).
    """
    return float(norm.cdf(-spread / sigma))


# ---------------------------------------------------------------------------
# Per-Model Sigma Calibration
# ---------------------------------------------------------------------------


def calibrate_spread_sigma(
    home_win_probs: Series,
    actual_margins: Series,
) -> float:
    """Fit the optimal sigma for a model by minimizing spread prediction error.

    Given a model's historical home-team win probabilities and the actual
    game margins (home_score - away_score), finds the sigma that minimizes
    the mean-squared error between the probit-derived predicted margin
    and the actual margin.

    The predicted margin at a candidate sigma *s* is::

        predicted_margin = s * Phi_inv(home_win_prob)

    (Note: this is ``-spread``, since a negative spread means a positive
    expected home margin.)

    Args:
        home_win_probs: Series of home-team win probabilities from the
            model's archived predictions.
        actual_margins: Series of actual game margins (home - away),
            aligned with ``home_win_probs``.

    Returns:
        Optimal sigma (float) in the range [_SIGMA_LO, _SIGMA_HI].

    Raises:
        ValueError: If the input Series are empty or have mismatched lengths.
    """
    if len(home_win_probs) == 0:
        raise ValueError("home_win_probs must not be empty")
    if len(home_win_probs) != len(actual_margins):
        raise ValueError(
            f"Length mismatch: home_win_probs ({len(home_win_probs)}) "
            f"vs actual_margins ({len(actual_margins)})"
        )

    # Pre-compute the probit values (constant across sigma candidates).
    clamped: np.ndarray = np.clip(
        np.asarray(home_win_probs, dtype=float),
        _PROB_FLOOR,
        _PROB_CEIL,
    )
    ppf_values: np.ndarray = norm.ppf(clamped)
    actual: np.ndarray = np.asarray(actual_margins, dtype=float)

    def _mse(sigma: float) -> float:
        predicted_margin: ndarray[tuple[Any, ...], dtype[float64]] = sigma * ppf_values
        return float(np.mean((predicted_margin - actual) ** 2))

    result = minimize_scalar(
        _mse,
        bounds=(_SIGMA_LO, _SIGMA_HI),
        method="bounded",
    )

    optimal_sigma: float = round(float(result.x), 4)

    logger.info(
        "calibrate_spread_sigma: optimal sigma=%.4f  MSE=%.4f  n=%d",
        optimal_sigma,
        result.fun,
        len(home_win_probs),
    )

    return optimal_sigma


def register_sigma(model_name: str, model_type: str, sigma: float) -> None:
    """Register a calibrated sigma for a specific (model_name, model_type) pair.

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        sigma: Calibrated sigma value.
    """
    _MODEL_SIGMAS[(model_name, model_type)] = sigma
    logger.info("register_sigma: (%s, %s) -> %.4f", model_name, model_type, sigma)


def get_sigma(
    model_name: str | None = None,
    model_type: str | None = None,
) -> float:
    """Look up the calibrated sigma for a (model_name, model_type) pair.

    Falls back to ``_NFL_DEFAULT_SIGMA`` if the pair is not registered
    or if either argument is ``None``.

    Args:
        model_name: Model purpose, or ``None`` for the default.
        model_type: Model algorithm, or ``None`` for the default.

    Returns:
        Sigma value (float).
    """
    if model_name is None or model_type is None:
        return _NFL_DEFAULT_SIGMA
    return _MODEL_SIGMAS.get((model_name, model_type), _NFL_DEFAULT_SIGMA)


# ---------------------------------------------------------------------------
# Isotonic Recalibration
# ---------------------------------------------------------------------------
# The existing tree model training uses StratifiedKFold(shuffle=True) and
# CalibratedClassifierCV(cv=3), which do not respect temporal ordering.
# This second-pass recalibration uses strict temporal splits. See PLAN.md
# for details on the temporal leakage discovery.
#
# Evaluation showed random_forest is already well-calibrated on recent data
# (holdout ECE 0.036), so the calibrator was NOT saved. The
# infrastructure is retained for future model versions.


def fit_recalibration(
    predicted_probs: Series,
    actual_outcomes: Series,
    seasons: Series,
    *,
    holdout_seasons: int = _DEFAULT_HOLDOUT_SEASONS,
) -> tuple[IsotonicRegression, dict[str, Any]]:
    """Fit an isotonic recalibration mapping with a strict temporal split.

    Trains an ``IsotonicRegression`` on all seasons except the most
    recent *holdout_seasons*, then validates on the holdout to verify
    that the calibration generalises forward in time.

    This guarantees zero temporal leakage: the calibrator never sees
    future outcomes during fitting.

    Args:
        predicted_probs: Model's predicted win probabilities.
        actual_outcomes: Binary outcomes (1 = that team won, 0 = lost),
            aligned with *predicted_probs*.
        seasons: Season labels (e.g. ``"2023-2024"``), aligned with
            *predicted_probs*. Used for the temporal split.
        holdout_seasons: Number of most-recent seasons to hold out
            for validation. Defaults to 2.

    Returns:
        Tuple of (fitted IsotonicRegression, diagnostics dict).

        Diagnostics keys:
            n_train, n_holdout, train_seasons, holdout_seasons,
            train_mean_pred, train_mean_actual,
            holdout_mean_pred, holdout_mean_actual

    Raises:
        ValueError: If there are fewer unique seasons than
            ``holdout_seasons + 1`` (need at least one training season).
    """
    unique_seasons: list[str] = sorted(seasons.unique().tolist())

    if len(unique_seasons) < holdout_seasons + 1:
        raise ValueError(
            f"Need at least {holdout_seasons + 1} unique seasons for a "
            f"temporal split with holdout_seasons={holdout_seasons}, "
            f"but only found {len(unique_seasons)}: {unique_seasons}"
        )

    train_seasons: list[str] = unique_seasons[:-holdout_seasons]
    holdout_season_list: list[str] = unique_seasons[-holdout_seasons:]

    train_mask: Series[bool] = seasons.isin(train_seasons)
    holdout_mask: Series[bool] = seasons.isin(holdout_season_list)

    p_train: ndarray = np.asarray(predicted_probs[train_mask], dtype=float)
    y_train: ndarray = np.asarray(actual_outcomes[train_mask], dtype=float)
    p_holdout: ndarray = np.asarray(predicted_probs[holdout_mask], dtype=float)
    y_holdout: ndarray = np.asarray(actual_outcomes[holdout_mask], dtype=float)

    # Fit isotonic regression on training partition only.
    calibrator = IsotonicRegression(
        y_min=_PROB_FLOOR,
        y_max=_PROB_CEIL,
        out_of_bounds="clip",
    )
    calibrator.fit(p_train, y_train)

    diagnostics: dict[str, Any] = {
        "n_train": len(p_train),
        "n_holdout": len(p_holdout),
        "train_seasons": train_seasons,
        "holdout_seasons": holdout_season_list,
        "train_mean_pred": float(np.mean(p_train)),
        "train_mean_actual": float(np.mean(y_train)),
        "holdout_mean_pred": float(np.mean(p_holdout)),
        "holdout_mean_actual": float(np.mean(y_holdout)),
    }

    logger.info(
        "fit_recalibration: n_train=%d  n_holdout=%d  train_seasons=%s  holdout_seasons=%s",
        diagnostics["n_train"],
        diagnostics["n_holdout"],
        diagnostics["train_seasons"],
        diagnostics["holdout_seasons"],
    )

    return calibrator, diagnostics


def apply_recalibration(
    probs: np.ndarray | Series,
    calibrator: IsotonicRegression,
) -> np.ndarray:
    """Apply a fitted isotonic calibrator to an array of probabilities.

    Args:
        probs: Raw model probabilities (array-like).
        calibrator: A fitted ``IsotonicRegression`` instance.

    Returns:
        Calibrated probabilities as a numpy array, clamped to
        (_PROB_FLOOR, _PROB_CEIL).
    """
    raw: np.ndarray = np.asarray(probs, dtype=float)
    calibrated: np.ndarray = calibrator.predict(raw)
    return np.clip(calibrated, _PROB_FLOOR, _PROB_CEIL)


def save_calibrator(
    calibrator: IsotonicRegression,
    model_name: str,
    model_type: str,
    repo: Path | None = None,
) -> Path:
    """Persist an isotonic calibrator alongside a model's artifact.

    Saves to ``data/models/{model_name}/{model_type}/calibrator.joblib``,
    the same directory as the model artifact itself.

    Args:
        calibrator: Fitted ``IsotonicRegression`` to persist.
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        repo: Repository root. If ``None``, uses ``get_settings().repo_root``.

    Returns:
        Path to the saved calibrator file.
    """
    if repo is None:
        from gridiron_edge.core.settings import get_settings

        repo = get_settings().repo_root

    cal_dir: Path = repo / "data" / "models" / model_name / model_type
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path: Path = cal_dir / _CALIBRATOR_FILENAME

    joblib.dump(calibrator, cal_path)
    logger.info("save_calibrator: saved to %s", cal_path)

    return cal_path


def load_calibrator(
    model_name: str,
    model_type: str,
    repo: Path | None = None,
) -> IsotonicRegression | None:
    """Load an isotonic calibrator for a (model_name, model_type) pair.

    Returns ``None`` if no calibrator file exists (graceful fallback).

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        repo: Repository root. If ``None``, uses ``get_settings().repo_root``.

    Returns:
        Fitted ``IsotonicRegression``, or ``None`` if not found.
    """
    if repo is None:
        from gridiron_edge.core.settings import get_settings

        repo = get_settings().repo_root

    cal_path: Path = repo / "data" / "models" / model_name / model_type / _CALIBRATOR_FILENAME

    if not cal_path.exists():
        logger.debug("load_calibrator: no calibrator at %s", cal_path)
        return None

    calibrator: IsotonicRegression = joblib.load(cal_path)
    logger.info("load_calibrator: loaded from %s", cal_path)
    return calibrator


# ---------------------------------------------------------------------------
# Uncertainty Bands & Confidence Tiers
# ---------------------------------------------------------------------------

# Uses the per-model residual standard deviation (margin_std) to build
# credible intervals around the point-estimate win probability. The
# spread +/- z * margin_std interval is converted back to probability space
# via the probit link, producing (win_prob_lo, win_prob_hi). Band width
# determines the confidence tier.


def compute_margin_std(
    home_win_probs: Series,
    actual_margins: Series,
    sigma: float,
) -> float:
    """Compute the standard deviation of spread prediction residuals.

    The residual for each game is::

        predicted_margin - actual_margin

    where ``predicted_margin = sigma * Phi_inv(home_win_prob)``.

    Args:
        home_win_probs: Model's predicted home-team win probabilities.
        actual_margins: Actual game margins (home - away).
        sigma: The model's calibrated sigma.

    Returns:
        Standard deviation of residuals (float, ddof=1).

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if len(home_win_probs) == 0:
        raise ValueError("home_win_probs must not be empty")
    if len(home_win_probs) != len(actual_margins):
        raise ValueError(
            f"Length mismatch: home_win_probs ({len(home_win_probs)}) "
            f"vs actual_margins ({len(actual_margins)})"
        )

    clamped: np.ndarray = np.clip(
        np.asarray(home_win_probs, dtype=float),
        _PROB_FLOOR,
        _PROB_CEIL,
    )
    predicted: np.ndarray = sigma * norm.ppf(clamped)
    actual: np.ndarray = np.asarray(actual_margins, dtype=float)
    residuals: np.ndarray = predicted - actual

    return float(np.std(residuals, ddof=1))


def get_margin_std(
    model_name: str | None = None,
    model_type: str | None = None,
) -> float:
    """Look up the residual margin std for a (model_name, model_type) pair.

    Falls back to ``_DEFAULT_MARGIN_STD`` if the pair is not registered
    or if either argument is ``None``.
    """
    if model_name is None or model_type is None:
        return _DEFAULT_MARGIN_STD
    return _MODEL_MARGIN_STDS.get((model_name, model_type), _DEFAULT_MARGIN_STD)


def get_total_std(
    model_name: str | None = None,
    model_type: str | None = None,
    *,
    repo: Path | None = None,
    default: float = 13.0,
) -> float:
    """Look up the holdout RMSE for a total model from its artifact metadata.

    Used by edge calculations as the standard deviation of total-points
    residuals (``total_std``). Falls back to ``default`` when:

    - ``model_name`` or ``model_type`` is ``None``,
    - no trained artifact exists for the pair,
    - the artifact metadata does not record an ``rmse`` metric,
    - the recorded ``rmse`` is NaN.

    Args:
        model_name: Total model purpose (typically ``"total"``).
        model_type: Total model algorithm (e.g. ``"random_forest"``).
        repo: Repository root override.
        default: Value returned when artifact lookup yields no usable
            RMSE.

    Returns:
        Total holdout RMSE, or ``default``.
    """
    import math

    if model_name is None or model_type is None:
        return default

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    if not store.is_trained(model_name, model_type):
        return default

    meta = store.read_metadata(model_name, model_type)
    rmse: object = meta.metrics.get("rmse")
    if not isinstance(rmse, int | float):
        return default
    rmse_float = float(rmse)
    if math.isnan(rmse_float):
        return default
    return rmse_float


def win_prob_bands(
    home_win_prob: float,
    *,
    margin_std: float,
    sigma: float,
    z: float = _DEFAULT_Z,
) -> tuple[float, float]:
    """Derive a credible interval around a home-team win probability.

    Converts the point-estimate spread to a spread interval using
    ``spread +/- z * margin_std``, then converts each bound back to
    probability space via the probit link.

    Args:
        home_win_prob: Point-estimate probability.
        margin_std: Residual standard deviation for this model.
        sigma: Calibrated sigma for this model.
        z: Z-score for the interval. Default 1.645 (90% CI).

    Returns:
        Tuple of (win_prob_lo, win_prob_hi). ``win_prob_lo`` is always
        the lower probability (worse for home team).
    """
    spread: float = win_prob_to_spread(home_win_prob, sigma=sigma)
    lo_spread: float = spread + z * margin_std  # worse for home
    hi_spread: float = spread - z * margin_std  # better for home
    prob_lo: float = spread_to_win_prob(lo_spread, sigma=sigma)
    prob_hi: float = spread_to_win_prob(hi_spread, sigma=sigma)
    return (prob_lo, prob_hi)


def classify_confidence_tier(home_win_prob: float) -> str:
    """Classify a win probability into a confidence tier.

    Folds the probability to the favorite's side [0.5, 1.0] and
    compares directly against thresholds, avoiding floating-point
    subtraction artifacts at boundaries.

    Args:
        home_win_prob: Model's home win probability.

    Returns:
        ``"High"``, ``"Moderate"``, or ``"Low"``.
    """
    prob: float = max(home_win_prob, 1.0 - home_win_prob)
    if prob >= _TIER_HIGH_PROB:
        return "High"
    if prob >= _TIER_MODERATE_PROB:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Projected Scores
# ---------------------------------------------------------------------------


def projected_scores(
    model_spread: float,
    model_total: float,
) -> tuple[float, float]:
    """Derive projected home and away scores from spread and total.

    Args:
        model_spread: Point spread (negative = home favored).
        model_total: Predicted combined score.

    Returns:
        Tuple of (projected_home_score, projected_away_score).
        Note: ``model_spread`` is negative when home is favored, so
        ``-model_spread`` is the home team's expected margin.
    """
    projected_home: float = (model_total - model_spread) / 2.0
    projected_away: float = (model_total + model_spread) / 2.0
    return (projected_home, projected_away)


# ---------------------------------------------------------------------------
# Prediction Enrichment Orchestrator
# ---------------------------------------------------------------------------


def enrich_predictions(
    df: pd.DataFrame,
    *,
    model_name: str | None = None,
    model_type: str | None = None,
    recalibrate: bool = True,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Add post-processed columns to a predictions DataFrame.

    Takes a DataFrame containing at least ``home_win_prob`` (or
    ``HOME_WIN_PROB``) and returns a copy with additional derived
    columns.

    When *recalibrate* is ``True`` (the default), the function attempts
    to load a saved isotonic calibrator for the (model_name, model_type)
    pair. If found, ``home_win_prob`` and ``away_win_prob`` are replaced
    in-place with calibrated values before deriving the spread.

    Spread columns:
        model_spread       float -- NFL point spread (neg = home favored)

    Uncertainty columns:
        margin_std         float -- residual std for this model
        win_prob_lo        float -- lower bound of 90% credible interval
        win_prob_hi        float -- upper bound of 90% credible interval
        confidence_tier    str   -- "High" / "Moderate" / "Low"

    Args:
        df: Predictions DataFrame. Must contain a home win probability
            column (case-insensitive: ``home_win_prob`` or
            ``HOME_WIN_PROB``).
        model_name: Model purpose (e.g. ``"win_prob"``). If ``None``,
            uses the default sigma and skips recalibration.
        model_type: Model algorithm (e.g. ``"random_forest"``). If
            ``None``, uses the default sigma and skips recalibration.
        recalibrate: If ``True``, attempt to load and apply an isotonic
            calibrator for the pair. Set to ``False`` to skip.
        repo: Repository root for calibrator loading. If ``None``,
            uses ``get_settings().repo_root``.

    Returns:
        Copy of *df* with enrichment columns appended.

    Raises:
        KeyError: If no home win probability column is found.
    """
    out: DataFrame = df.copy()

    # Resolve the home win probability column name (support both cases).
    if "home_win_prob" in out.columns:
        prob_col = "home_win_prob"
        away_col = "away_win_prob"
    elif "HOME_WIN_PROB" in out.columns:
        prob_col = "HOME_WIN_PROB"
        away_col = "AWAY_WIN_PROB"
    else:
        raise KeyError("DataFrame must contain 'home_win_prob' or 'HOME_WIN_PROB'")

    # --- Isotonic recalibration ---
    if recalibrate and model_name is not None and model_type is not None:
        calibrator = load_calibrator(model_name, model_type, repo=repo)
        if calibrator is not None:
            out[prob_col] = apply_recalibration(out[prob_col], calibrator)
            if away_col in out.columns:
                out[away_col] = 1.0 - out[prob_col]
            logger.debug(
                "enrich_predictions: applied recalibration for (%s, %s)",
                model_name,
                model_type,
            )

    # --- Spread derivation ---
    sigma: float = get_sigma(model_name, model_type)

    out["model_spread"] = out[prob_col].apply(lambda p: win_prob_to_spread(p, sigma=sigma))

    # --- Uncertainty bands + confidence tier ---
    ms: float = get_margin_std(model_name, model_type)
    out["margin_std"] = ms

    def _bands(p: float) -> tuple[float, float]:
        return win_prob_bands(p, margin_std=ms, sigma=sigma)

    bands: Series = out[prob_col].apply(_bands)
    out["win_prob_lo"] = bands.apply(lambda t: t[0])
    out["win_prob_hi"] = bands.apply(lambda t: t[1])
    _prob_key: Literal["HOME_WIN_PROB", "home_win_prob"] = (
        "home_win_prob" if "home_win_prob" in out.columns else "HOME_WIN_PROB"
    )
    out["confidence_tier"] = out[_prob_key].apply(classify_confidence_tier)

    # --- Projected scores (requires model_total column) ---
    if "model_total" in out.columns:
        scores = out.apply(
            lambda row: projected_scores(row["model_spread"], row["model_total"]),
            axis=1,
        )
        out["projected_home_score"] = scores.apply(lambda t: t[0])
        out["projected_away_score"] = scores.apply(lambda t: t[1])

    logger.debug(
        "enrich_predictions: enriched %d rows (sigma=%.4f, margin_std=%.2f, model=(%s, %s))",
        len(out),
        sigma,
        ms,
        model_name or "default",
        model_type or "default",
    )

    return out

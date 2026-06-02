# src/gridiron_edge/models/game_prediction/post_process.py
"""Post-processing enrichment for game prediction outputs (W2).

Derives additional outputs from base model win probabilities without
modifying model internals.  Every predictor that outputs ``home_win_prob``
can be enriched by calling ``enrich_predictions()`` on its output
DataFrame.

The spread derivation uses the probit link: the relationship between
win probability and point spread is mediated by the margin-of-victory
distribution, which is approximately normal in the NFL.  The conversion
is::

    model_spread = -sigma * Phi_inv(home_win_prob)

where *sigma* is the standard deviation of the margin-of-victory
distribution (approx 13.86 league-wide) and *Phi_inv* is the inverse normal
CDF.  Sigma is calibrated per model version for maximum accuracy.

Phase A deliverables (spread + sigma calibration):
    win_prob_to_spread   home_win_prob -> NFL point spread
    spread_to_win_prob   NFL point spread -> home_win_prob (inverse)
    calibrate_spread_sigma  Fit sigma from historical predictions + outcomes
    get_sigma            Look up per-model calibrated sigma
    enrich_predictions   Orchestrator: adds post-processed columns to a
                         predictions DataFrame

Phase A.5 deliverables (isotonic recalibration):
    fit_recalibration    Fit isotonic mapping with temporal split
    apply_recalibration  Apply fitted calibrator to probabilities
    save_calibrator      Persist calibrator to data/models/{version}_cal/
    load_calibrator      Load calibrator (None if absent)

Future phases will extend ``enrich_predictions`` with:
    Phase B -- margin_std, win_prob_lo, win_prob_hi, confidence_tier
    Phase C -- model_total, projected_home_score, projected_away_score
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from pandas import Series
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NFL historical margin-of-victory standard deviation.  Used as the
# fallback when no per-model calibrated sigma is available.  Derived
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

# Per-model-version calibrated sigma values.  Keyed by model_version
# string (e.g. "random_forest_v3").  When a model_version is not found
# here, ``get_sigma`` falls back to ``_NFL_DEFAULT_SIGMA``.
#
# Calibrated 2026-06-01 via ``calibrate_spread_sigma`` against the full
# prediction archive (5,705-7,276 games per model).  Best spread MAE:
# random_forest_v3 (sigma=13.97, MAE=9.92).
_MODEL_SIGMAS: dict[str, float] = {
    "elo_v1": 20.3525,
    "elo_v2": 12.1753,
    "elo_v3": 12.2485,
    "logistic_v1": 12.9332,
    "logistic_v2": 12.9382,
    "logistic_v3": 12.7466,
    "logistic_v4": 12.9515,
    "random_forest_v1": 12.9482,
    "random_forest_v2": 12.6314,
    "random_forest_v3": 13.9732,
    "xgboost_v1": 14.2323,
    "xgboost_v2": 14.3169,
    "xgboost_v3": 13.951,
}
# TODO(W2-D): Wire sigma calibration into the training harness so this
# dict is populated automatically when a new model version is trained.
# See Phase D in PLAN.md.

# Calibrator artifact filename.
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

    Uses the probit (inverse-normal) link function.  The returned spread
    follows NFL convention: **negative** means home team is favored,
    **positive** means away team is favored.

    Examples::

        win_prob_to_spread(0.50)  ->  0.0   (pick'em)
        win_prob_to_spread(0.75)  -> -9.35  (home favored by ~9 points)
        win_prob_to_spread(0.25)  -> +9.35  (away favored by ~9 points)

    Args:
        home_win_prob: Predicted probability that the home team wins,
            in the range [0, 1].  Values at the extremes are clamped
            to (_PROB_FLOOR, _PROB_CEIL) to avoid +/-inf.
        sigma: Standard deviation of the margin-of-victory distribution.
            Defaults to ``_NFL_DEFAULT_SIGMA`` (13.86).

    Returns:
        Point spread (float).  Negative = home favored.
    """
    clamped: float = float(np.clip(home_win_prob, _PROB_FLOOR, _PROB_CEIL))
    return -sigma * float(norm.ppf(clamped))


def spread_to_win_prob(
    spread: float,
    *,
    sigma: float = _NFL_DEFAULT_SIGMA,
) -> float:
    """Convert an NFL point spread to a home-team win probability.

    Inverse of ``win_prob_to_spread``.  Uses the normal CDF.

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
        predicted_margin = sigma * ppf_values
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


def register_sigma(model_version: str, sigma: float) -> None:
    """Register a calibrated sigma for a specific model version.

    Args:
        model_version: Model identifier (e.g. ``"random_forest_v3"``).
        sigma: Calibrated sigma value.
    """
    _MODEL_SIGMAS[model_version] = sigma
    logger.info("register_sigma: %s -> %.4f", model_version, sigma)


def get_sigma(model_version: str | None = None) -> float:
    """Look up the calibrated sigma for a model version.

    Falls back to ``_NFL_DEFAULT_SIGMA`` if the model version is not
    registered or is ``None``.

    Args:
        model_version: Model identifier, or ``None`` for the default.

    Returns:
        Sigma value (float).
    """
    if model_version is None:
        return _NFL_DEFAULT_SIGMA
    return _MODEL_SIGMAS.get(model_version, _NFL_DEFAULT_SIGMA)


# ---------------------------------------------------------------------------
# Isotonic Recalibration (Phase A.5)
# ---------------------------------------------------------------------------
# The existing tree model training uses StratifiedKFold(shuffle=True) and
# CalibratedClassifierCV(cv=3), which do not respect temporal ordering.
# This second-pass recalibration uses strict temporal splits to fix the
# residual underconfidence without retraining the model.  See PLAN.md
# "Discovery: Temporal Leakage in Tree Model Training" for details.


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
            *predicted_probs*.  Used for the temporal split.
        holdout_seasons: Number of most-recent seasons to hold out
            for validation.  Defaults to 2.

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

    train_seasons = unique_seasons[:-holdout_seasons]
    holdout_season_list = unique_seasons[-holdout_seasons:]

    train_mask = seasons.isin(train_seasons)
    holdout_mask = seasons.isin(holdout_season_list)

    p_train = np.asarray(predicted_probs[train_mask], dtype=float)
    y_train = np.asarray(actual_outcomes[train_mask], dtype=float)
    p_holdout = np.asarray(predicted_probs[holdout_mask], dtype=float)
    y_holdout = np.asarray(actual_outcomes[holdout_mask], dtype=float)

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
    model_version: str,
    repo: Path | None = None,
) -> Path:
    """Persist an isotonic calibrator alongside a model's artifacts.

    Saves to ``data/models/{model_version}_cal/calibrator.joblib``.

    Args:
        calibrator: Fitted ``IsotonicRegression`` to persist.
        model_version: Base model identifier (e.g. ``"random_forest_v3"``).
            The calibrator is stored under ``{model_version}_cal/``.
        repo: Repository root.  If ``None``, uses ``get_settings().repo_root``.

    Returns:
        Path to the saved calibrator file.
    """
    if repo is None:
        from gridiron_edge.core.settings import get_settings

        repo = get_settings().repo_root

    cal_dir: Path = repo / "data" / "models" / f"{model_version}_cal"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path: Path = cal_dir / _CALIBRATOR_FILENAME

    joblib.dump(calibrator, cal_path)
    logger.info("save_calibrator: saved to %s", cal_path)

    return cal_path


def load_calibrator(
    model_version: str,
    repo: Path | None = None,
) -> IsotonicRegression | None:
    """Load an isotonic calibrator for a model version.

    Returns ``None`` if no calibrator file exists (graceful fallback).

    Args:
        model_version: Base model identifier (e.g. ``"random_forest_v3"``).
        repo: Repository root.  If ``None``, uses ``get_settings().repo_root``.

    Returns:
        Fitted ``IsotonicRegression``, or ``None`` if not found.
    """
    if repo is None:
        from gridiron_edge.core.settings import get_settings

        repo = get_settings().repo_root

    cal_path: Path = repo / "data" / "models" / f"{model_version}_cal" / _CALIBRATOR_FILENAME

    if not cal_path.exists():
        logger.debug("load_calibrator: no calibrator at %s", cal_path)
        return None

    calibrator: IsotonicRegression = joblib.load(cal_path)
    logger.info("load_calibrator: loaded from %s", cal_path)
    return calibrator


# ---------------------------------------------------------------------------
# Prediction Enrichment Orchestrator
# ---------------------------------------------------------------------------


def enrich_predictions(
    df: pd.DataFrame,
    *,
    model_version: str | None = None,
    recalibrate: bool = True,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Add post-processed columns to a predictions DataFrame.

    Takes a DataFrame containing at least ``home_win_prob`` (or
    ``HOME_WIN_PROB``) and returns a copy with additional derived
    columns.

    When *recalibrate* is ``True`` (the default), the function attempts
    to load a saved isotonic calibrator for *model_version*.  If found,
    ``home_win_prob`` and ``away_win_prob`` are replaced in-place with
    calibrated values before deriving the spread.

    Phase A adds:
        model_spread   float -- NFL point spread (negative = home favored)

    Future phases will add model_total, projected scores, uncertainty
    bands, confidence_tier, and margin_std.

    Args:
        df: Predictions DataFrame.  Must contain a home win probability
            column (case-insensitive: ``home_win_prob`` or
            ``HOME_WIN_PROB``).
        model_version: Model identifier for sigma lookup and calibrator
            loading.  If ``None``, uses the default sigma and skips
            recalibration.
        recalibrate: If ``True``, attempt to load and apply an isotonic
            calibrator for *model_version*.  Set to ``False`` to skip.
        repo: Repository root for calibrator loading.  If ``None``,
            uses ``get_settings().repo_root``.

    Returns:
        Copy of *df* with enrichment columns appended.

    Raises:
        KeyError: If no home win probability column is found.
    """
    out = df.copy()

    # Resolve the home win probability column name (support both cases).
    if "home_win_prob" in out.columns:
        prob_col = "home_win_prob"
        away_col = "away_win_prob"
    elif "HOME_WIN_PROB" in out.columns:
        prob_col = "HOME_WIN_PROB"
        away_col = "AWAY_WIN_PROB"
    else:
        raise KeyError("DataFrame must contain 'home_win_prob' or 'HOME_WIN_PROB'")

    # --- Phase A.5: Isotonic recalibration ---
    if recalibrate and model_version is not None:
        calibrator = load_calibrator(model_version, repo=repo)
        if calibrator is not None:
            out[prob_col] = apply_recalibration(out[prob_col], calibrator)
            if away_col in out.columns:
                out[away_col] = 1.0 - out[prob_col]
            logger.debug(
                "enrich_predictions: applied recalibration for %s",
                model_version,
            )

    # --- Phase A: Spread derivation ---
    sigma = get_sigma(model_version)

    out["model_spread"] = out[prob_col].apply(lambda p: win_prob_to_spread(p, sigma=sigma))

    logger.debug(
        "enrich_predictions: added model_spread (sigma=%.4f, model=%s, n=%d)",
        sigma,
        model_version or "default",
        len(out),
    )

    return out

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
distribution (≈ 13.86 league-wide) and *Phi_inv* is the inverse normal
CDF.  Sigma is calibrated per model version for maximum accuracy.

Phase A deliverables (spread + sigma calibration):
    win_prob_to_spread   home_win_prob → NFL point spread
    spread_to_win_prob   NFL point spread → home_win_prob (inverse)
    calibrate_spread_sigma  Fit sigma from historical predictions + outcomes
    get_sigma            Look up per-model calibrated sigma
    enrich_predictions   Orchestrator: adds post-processed columns to a
                         predictions DataFrame

Future phases will extend ``enrich_predictions`` with:
    Phase B — margin_std, win_prob_lo, win_prob_hi, confidence_tier
    Phase C — model_total, projected_home_score, projected_away_score
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import Any, Final

import numpy as np
from numpy import dtype, float64, ndarray
import pandas as pd
from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
from scipy.optimize import minimize_scalar

# pyrefly: ignore [missing-import]
from scipy.stats import norm

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NFL historical margin-of-victory standard deviation.  Used as the
# fallback when no per-model calibrated sigma is available.  Derived
# from league-wide game results 2000-2024.
_NFL_DEFAULT_SIGMA: Final[float] = 13.86

# Probability clamp bounds — prevents ±inf from the probit inverse
# at p = 0 or p = 1.
_PROB_FLOOR: Final[float] = 0.001
_PROB_CEIL: Final[float] = 0.999

# Sigma calibration search bounds — reasonable NFL range.
# Values below ~8 imply unrealistically tight games; above ~22 implies
# unrealistically volatile margins.
_SIGMA_LO: Final[float] = 8.0
_SIGMA_HI: Final[float] = 22.0

# Per-model-version calibrated sigma values.  Populated by
# ``calibrate_spread_sigma`` or set manually.  Keyed by model_version
# string (e.g. "random_forest_v3").  When a model_version is not found
# here, ``get_sigma`` falls back to ``_NFL_DEFAULT_SIGMA``.
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


# ---------------------------------------------------------------------------
# Spread ↔ Win Probability Conversion
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

        win_prob_to_spread(0.50)  →  0.0   (pick'em)
        win_prob_to_spread(0.75)  → -9.35  (home favored by ~9 points)
        win_prob_to_spread(0.25)  → +9.35  (away favored by ~9 points)

    Args:
        home_win_prob: Predicted probability that the home team wins,
            in the range [0, 1].  Values at the extremes are clamped
            to (_PROB_FLOOR, _PROB_CEIL) to avoid ±inf.
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


def register_sigma(model_version: str, sigma: float) -> None:
    """Register a calibrated sigma for a specific model version.

    Args:
        model_version: Model identifier (e.g. ``"random_forest_v3"``).
        sigma: Calibrated sigma value.
    """
    _MODEL_SIGMAS[model_version] = sigma
    logger.info("register_sigma: %s → %.4f", model_version, sigma)


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
# Prediction Enrichment Orchestrator
# ---------------------------------------------------------------------------


def enrich_predictions(
    df: pd.DataFrame,
    *,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Add post-processed columns to a predictions DataFrame.

    Takes a DataFrame containing at least ``home_win_prob`` (or
    ``HOME_WIN_PROB``) and returns a copy with additional derived
    columns.

    Phase A adds:
        model_spread   float — NFL point spread (negative = home favored)

    Future phases will add model_total, projected scores, uncertainty
    bands, confidence_tier, and margin_std.

    Args:
        df: Predictions DataFrame.  Must contain a home win probability
            column (case-insensitive: ``home_win_prob`` or
            ``HOME_WIN_PROB``).
        model_version: Model identifier for sigma lookup.  If ``None``,
            uses the default sigma.

    Returns:
        Copy of *df* with enrichment columns appended.

    Raises:
        KeyError: If no home win probability column is found.
    """
    out: DataFrame = df.copy()

    # Resolve the home win probability column name (support both cases).
    if "home_win_prob" in out.columns:
        prob_col = "home_win_prob"
    elif "HOME_WIN_PROB" in out.columns:
        prob_col = "HOME_WIN_PROB"
    else:
        raise KeyError("DataFrame must contain 'home_win_prob' or 'HOME_WIN_PROB'")

    sigma: float = get_sigma(model_version)

    out["model_spread"] = out[prob_col].apply(lambda p: win_prob_to_spread(p, sigma=sigma))

    logger.debug(
        "enrich_predictions: added model_spread (sigma=%.4f, model=%s, n=%d)",
        sigma,
        model_version or "default",
        len(out),
    )

    return out

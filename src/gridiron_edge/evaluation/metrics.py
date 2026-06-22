# src/gridiron_edge/evaluation/metrics.py

"""Evaluation metrics for game prediction models.

All functions accept a standard *evaluation DataFrame* produced by
``build_evaluation_df``.  The schema is:

    game_id          str   - canonical YYYY_WW_AWAY_HOME identifier
    season           str   - e.g. "2024-2025"
    week             int   - NFL week number
    away_team        str
    home_team        str
    away_win_prob    float - model's predicted probability that away team wins
    away_team_won    int   - 1 if away team won, 0 if home team won
    model_name       str   - model purpose (e.g. "win_prob", "total")
    model_type       str   - model algorithm (e.g. "random_forest", "elo")

Public API
----------
build_evaluation_df          Join archive to outcomes; primary entry point.
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
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy import dtype, float64, ndarray, signedinteger
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.constants import AWAY_WIN_LOCATION as _AWAY_WIN_LOCATION
from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets import loaders
from gridiron_edge.evaluation.archive import load_prediction_log

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


# ---------------------------------------------------------------------------
# Archive access
# ---------------------------------------------------------------------------


def build_evaluation_df(
    *,
    model_name: str | None = None,
    model_type: str | None = None,
    season: str | None = None,
    repo: Path | None = None,
) -> DataFrame:
    """Join the prediction archive to game outcomes.

    Loads the prediction log, joins to the canonical games table to obtain
    outcomes, and returns a clean evaluation DataFrame.  Missing outcomes
    (upcoming games) are dropped.

    Args:
        model_name: Filter to a specific model purpose (e.g. ``"win_prob"``).
            If ``None``, all purposes are returned.
        model_type: Filter to a specific model algorithm
            (e.g. ``"random_forest"``). If ``None``, all algorithms are
            returned.
        season: Filter to a specific season (e.g. ``"2024-2025"``).
            If ``None``, all seasons are returned.
        repo: Repository root.  Defaults to settings repo root.

    Returns:
        DataFrame with columns: game_id, season, week, away_team, home_team,
        away_win_prob, away_team_won, model_name, model_type.
        Empty if no data.
    """
    resolved_repo: Path = repo or get_settings().repo_root

    log: DataFrame = load_prediction_log(
        model_name=model_name,
        model_type=model_type,
        season=season,
        repo=resolved_repo,
    )

    if log.empty:
        return DataFrame()

    # Join outcomes from canonical games table
    games: DataFrame = loaders.load_games(resolved_repo)

    # Build a lookup: game_id → away_team_won (1 = away won, 0 = home won)
    # The canonical games table uses WINNER/LOSER/GAME_LOCATION convention.
    # GAME_LOCATION == _AWAY_WIN_LOCATION means the winner was the away team.
    away_won_mask: Series = games["GAME_LOCATION"] == _AWAY_WIN_LOCATION
    outcome_map: dict[str, int] = {}
    for _, row in games.iterrows():
        gid = row["GAME_ID"]
        if pd.isna(row.get("WIN_OR_TIE")):
            continue  # upcoming game
        outcome_map[gid] = 1 if away_won_mask[row.name] else 0  # type: ignore[index]

    log["away_team_won"] = log["game_id"].map(outcome_map)
    log = log.dropna(subset=["away_team_won"]).copy()
    log["away_team_won"] = log["away_team_won"].astype(int)

    cols: list[str] = [
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "away_win_prob",
        "away_team_won",
        "model_name",
        "model_type",
    ]
    available: list[str] = [c for c in cols if c in log.columns]
    return log.loc[:, available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregate metric tables
# ---------------------------------------------------------------------------


def summarise(df: DataFrame, *, group_by: str = "season") -> DataFrame:
    """Compute grouped Brier score and accuracy summary.

    Args:
        df: Evaluation DataFrame from ``build_evaluation_df``.
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
        df: Evaluation DataFrame from ``build_evaluation_df``.
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
        df: Evaluation DataFrame from ``build_evaluation_df``.
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
        df: Evaluation DataFrame from ``build_evaluation_df``.

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
        df: Evaluation DataFrame from ``build_evaluation_df``.
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

# src/gridiron_edge/evaluation/metrics.py

"""Model evaluation metrics — calibration, Brier score, log loss.

Joins the prediction archive against historical game results to produce
accuracy and calibration metrics. All metrics operate on the away team's
win probability, which is the canonical prediction unit in the archive.

Typical usage::

    from gridiron_edge.evaluation.metrics import build_evaluation_df, summarise

    df_eval = build_evaluation_df()  # join predictions to outcomes
    summary = summarise(df_eval)  # overall + by-season breakdown
    calibration = calibration_table(df_eval)  # bucket-level calibration data
"""

from __future__ import annotations

import logging
from logging import Logger
import math
from pathlib import Path

import pandas as pd

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets import loaders
from gridiron_edge.evaluation.archive import load_prediction_log

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Outcome join
# ---------------------------------------------------------------------------

# GAME_LOCATION convention from canonical games schema:
#   "NULL_VALUE" → winner is the home team  → away_team_won = 0
#   "@"          → winner is the away team  → away_team_won = 1
#   "N"          → neutral site (winner is still recorded correctly)
_AWAY_WIN_LOCATION = "@"


def _outcomes_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """Extract a slim outcome table from the canonical games DataFrame.

    Produces one row per completed game with columns:
        game_id, away_team_won (1.0 | 0.0 | 0.5)

    WIN_OR_TIE values:
        1.0 → decisive result (winner beat loser)
        0.5 → tie

    For a tie, away_team_won = 0.5 (split credit, consistent with
    probability scoring rules).

    Args:
        games: Canonical games DataFrame (``NFL_wk_by_wk_cleaned.csv``).

    Returns:
        DataFrame with columns ``game_id`` and ``away_team_won``.
    """
    df = games.copy()

    # Only keep completed games (WIN_OR_TIE is non-null)
    df = df.dropna(subset=["WIN_OR_TIE"]).copy()

    # Reconstruct away_team_won:
    # GAME_LOCATION == "@" means the WINNER was the away team.
    # For ties WIN_OR_TIE == 0.5, split credit regardless of location.
    tie_mask = df["WIN_OR_TIE"] == 0.5
    away_won_mask = df["GAME_LOCATION"] == _AWAY_WIN_LOCATION

    df["away_team_won"] = 0.0
    df.loc[away_won_mask & ~tie_mask, "away_team_won"] = 1.0
    df.loc[tie_mask, "away_team_won"] = 0.5

    return df[["GAME_ID", "away_team_won"]].rename(columns={"GAME_ID": "game_id"})


def build_evaluation_df(
    *,
    model_version: str | None = None,
    season: str | None = None,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Join prediction archive against actual game outcomes.

    Only games that appear in both the prediction archive and the completed
    games dataset are included. Unplayed games (in the archive but not yet
    in results) are silently excluded.

    Args:
        model_version: Filter to a specific model (e.g. ``"elo_v1"``).
            If ``None``, all models are included.
        season: Filter to a specific season (e.g. ``"2025-2026"``).
            If ``None``, all seasons are included.
        repo: Repository root. Defaults to ``repo_root()``.

    Returns:
        DataFrame with columns:
            predicted_at, model_version, season, week, game_id,
            away_team, home_team, away_win_prob, home_win_prob,
            away_team_won (1.0 | 0.5 | 0.0)

        Empty DataFrame if no matching predictions exist.
    """
    resolved_repo = repo or repo_root()

    predictions = load_prediction_log(
        model_version=model_version,
        season=season,
        repo=resolved_repo,
    )

    if predictions.empty:
        logger.warning(
            "No predictions found in archive%s%s.",
            f" for model '{model_version}'" if model_version else "",
            f" season {season}" if season else "",
        )
        return pd.DataFrame()

    games = loaders.load_games(resolved_repo)
    outcomes = _outcomes_from_games(games)

    eval_df = predictions.merge(outcomes, on="game_id", how="inner")

    n_dropped = len(predictions) - len(eval_df)
    if n_dropped:
        logger.debug(
            "%d archived predictions have no completed outcome yet (excluded).",
            n_dropped,
        )

    return eval_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------


def brier_score(away_win_prob: pd.Series, away_team_won: pd.Series) -> float:
    """Compute the Brier score for away-team win probability predictions.

    Lower is better. A perfect forecaster scores 0.0; random (0.5 always)
    scores 0.25.

    Brier score = mean((predicted_prob - actual_outcome)^2)

    Args:
        away_win_prob: Predicted probability that the away team wins [0, 1].
        away_team_won: Actual outcome (1.0 = away win, 0.0 = home win,
            0.5 = tie).

    Returns:
        Brier score as a float.
    """
    diff = away_win_prob - away_team_won
    return float((diff**2).mean())


def log_loss(away_win_prob: pd.Series, away_team_won: pd.Series) -> float:
    """Compute log loss for away-team win probability predictions.

    Lower is better. Penalises confident wrong predictions heavily.

    Log loss = -mean(y * log(p) + (1-y) * log(1-p))

    Probabilities are clipped to [1e-7, 1 - 1e-7] to avoid log(0).
    Ties (away_team_won == 0.5) are excluded as log loss is undefined
    for fractional outcomes.

    Args:
        away_win_prob: Predicted probability that the away team wins [0, 1].
        away_team_won: Actual outcome (1.0 = away win, 0.0 = home win).
            Tie rows (0.5) are excluded automatically.

    Returns:
        Log loss as a float, or NaN if no non-tie rows remain.
    """
    mask = away_team_won != 0.5
    p = away_win_prob[mask].clip(1e-7, 1 - 1e-7)
    y = away_team_won[mask]

    if len(p) == 0:
        return float("nan")

    losses = -(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log))
    return float(losses.mean())


def accuracy(away_win_prob: pd.Series, away_team_won: pd.Series) -> float:
    """Compute prediction accuracy (predicted winner matches actual winner).

    Ties are excluded. A game is correctly predicted when the team with
    probability > 0.5 wins.

    Args:
        away_win_prob: Predicted probability that the away team wins [0, 1].
        away_team_won: Actual outcome (1.0 = away win, 0.0 = home win).
            Tie rows (0.5) are excluded automatically.

    Returns:
        Accuracy as a float in [0, 1].
    """
    mask = away_team_won != 0.5
    p = away_win_prob[mask]
    y = away_team_won[mask]

    if len(p) == 0:
        return float("nan")

    predicted_away = p > 0.5
    actual_away = y == 1.0
    return float((predicted_away == actual_away).mean())


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def summarise(
    eval_df: pd.DataFrame,
    *,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Compute Brier score, log loss, and accuracy for evaluated predictions.

    Args:
        eval_df: Output of ``build_evaluation_df()``.
        group_by: Optional column to group by before computing metrics.
            Typical values: ``"season"``, ``"week"``, ``"model_version"``.
            If ``None``, returns a single-row overall summary.

    Returns:
        DataFrame with columns:
            [group_by,] n_games, brier_score, log_loss, accuracy
        Sorted by the group column if provided, otherwise a single row.
    """
    if eval_df.empty:
        return pd.DataFrame(
            columns=[group_by, "n_games", "brier_score", "log_loss", "accuracy"]
            if group_by
            else ["n_games", "brier_score", "log_loss", "accuracy"]
        )

    metric_cols: list[str] = ["away_win_prob", "away_team_won"]

    def _metrics(df: pd.DataFrame) -> pd.Series:
        # Explicitly select only the columns needed so the grouping key
        # is never passed in — avoids include_groups which has incomplete
        # type stubs and is deprecated in newer pandas.
        d = df[metric_cols]
        return pd.Series(
            {
                "n_games": len(d),
                "brier_score": round(brier_score(d["away_win_prob"], d["away_team_won"]), 4),
                "log_loss": round(log_loss(d["away_win_prob"], d["away_team_won"]), 4),
                "accuracy": round(accuracy(d["away_win_prob"], d["away_team_won"]), 4),
            }
        )

    if group_by is None:
        return _metrics(eval_df).to_frame().T.reset_index(drop=True)

    return (
        eval_df.groupby(group_by)[metric_cols]
        .apply(_metrics)
        .reset_index()
        .sort_values(group_by)
        .reset_index(drop=True)
    )


def calibration_table(
    eval_df: pd.DataFrame,
    *,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Bucket predictions by probability and compare to actual win rates.

    A well-calibrated model should have actual win rate ≈ predicted
    probability in each bucket. Large deviations indicate over- or
    under-confidence.

    Ties (away_team_won == 0.5) are excluded.

    Args:
        eval_df: Output of ``build_evaluation_df()``.
        n_buckets: Number of equal-width probability buckets. Default 10
            gives buckets of width 0.1 (0-10%, 10-20%, ..., 90-100%).

    Returns:
        DataFrame with columns:
            bucket_low, bucket_high, bucket_mid, n_games,
            mean_predicted_prob, actual_win_rate, error
        where error = actual_win_rate - mean_predicted_prob.
    """
    if eval_df.empty:
        return pd.DataFrame()

    df = eval_df[eval_df["away_team_won"] != 0.5].copy()

    edges = [i / n_buckets for i in range(n_buckets + 1)]
    labels = list(range(n_buckets))

    df["bucket"] = pd.cut(
        df["away_win_prob"],
        bins=edges,
        labels=labels,
        include_lowest=True,
    )

    rows = []
    for bucket_idx in labels:
        low = edges[bucket_idx]
        high = edges[bucket_idx + 1]
        mid = (low + high) / 2
        subset = df[df["bucket"] == bucket_idx]
        if subset.empty:
            continue
        mean_pred = float(subset["away_win_prob"].mean())
        actual_rate = float(subset["away_team_won"].mean())
        rows.append(
            {
                "bucket_low": round(low, 2),
                "bucket_high": round(high, 2),
                "bucket_mid": round(mid, 2),
                "n_games": len(subset),
                "mean_predicted_prob": round(mean_pred, 4),
                "actual_win_rate": round(actual_rate, 4),
                "error": round(actual_rate - mean_pred, 4),
            }
        )

    return pd.DataFrame(rows)

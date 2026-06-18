# src/gridiron_edge/models/game_prediction/pipeline.py
"""Prediction pipeline — feature prep → inference → enrichment.

Orchestrates the steps that produce a fully enriched game-level
predictions DataFrame. Each step is a composable function:

    load features → run model(s) → build game rows → enrich → return

Adding a new model type (e.g. a new sport or a new target variable)
means adding one inference call and one column — not rewriting the
pipeline.

Public API:
    predict_games            Full pipeline for historical predictions
    build_game_predictions   Map raw model output onto game-level rows

Model identification (Workstream 2):
    Win-probability and total models are identified by the pair
    ``(model_name, model_type)``. The archive schema carries both as
    separate columns rather than a composite ``model_version`` string.
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import logging
from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

logger: Logger = logging.getLogger(__name__)


def build_game_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    *,
    model_name: str,
    model_type: str,
    is_backfilled: bool = True,
    totals: pd.Series | None = None,
) -> pd.DataFrame:
    """Map raw model outputs onto game-level prediction rows.

    The modeling DataFrame has one row per team-game (two rows per game).
    This function filters to the away-team rows (``HOME_FIELD == 0``),
    deduplicates on GAME_ID, and constructs the standard prediction
    schema.

    Args:
        df: Modeling DataFrame (must include GAME_ID, TEAM_A, TEAM_B,
            YEAR, WEEK_NUM, HOME_FIELD). Aligned with *probs*.
        probs: Predicted probability that TEAM_A wins, aligned with *df*.
        model_name: Win-probability model purpose (e.g. ``"win_prob"``).
        model_type: Win-probability model algorithm (e.g. ``"random_forest"``).
        is_backfilled: Whether these are historical backfill predictions.
        totals: Optional predicted game totals, aligned with *df*.

    Returns:
        Game-level predictions DataFrame with one row per game.
    """
    work: DataFrame = df.copy()
    work["_prob"] = probs
    if totals is not None:
        work["_total"] = totals

    # One row per game — keep the away-team perspective.
    # pyrefly: ignore [no-matching-overload]
    away: DataFrame = work.loc[work["HOME_FIELD"] == 0].drop_duplicates(subset=["GAME_ID"])

    ts = dt.datetime.now(tz=dt.UTC).replace(tzinfo=None)

    result = pd.DataFrame(
        {
            "predicted_at": ts,
            "is_backfilled": is_backfilled,
            "model_name": model_name,
            "model_type": model_type,
            "season": away["YEAR"],
            "week": away["WEEK_NUM"].astype(int),
            "game_id": away["GAME_ID"],
            "game_date": "",
            "away_team": away["TEAM_A"],
            "home_team": away["TEAM_B"],
            "away_elo": float("nan"),
            "home_elo": float("nan"),
            "away_win_prob": away["_prob"],
            "home_win_prob": 1.0 - away["_prob"],
        }
    )

    if "_total" in away.columns:
        result["model_total"] = away["_total"].values

    return result.reset_index(drop=True)


def predict_games(
    *,
    model_name: str,
    model_type: str,
    feature_fn: Callable,
    repo: Path | None = None,
    total_model_name: str = "total",
    total_model_type: str = "random_forest",
    is_backfilled: bool = True,
) -> pd.DataFrame:
    """Full prediction pipeline: load → predict → enrich.

    Args:
        model_name: Win-probability model purpose (e.g. ``"win_prob"``).
        model_type: Win-probability model algorithm (e.g. ``"random_forest"``).
        feature_fn: Feature engineering function.
        repo: Repository root.
        total_model_name: Total model purpose (defaults to ``"total"``).
            Pass an empty string to skip total prediction.
        total_model_type: Total model algorithm (defaults to ``"random_forest"``).
        is_backfilled: Whether these are backfill predictions.

    Returns:
        Enriched game-level predictions DataFrame.
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_modeling_file
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.game_prediction._columns import _SCHEMA_VERSION
    from gridiron_edge.models.game_prediction.post_process import enrich_predictions

    resolved_repo: Path = repo or get_settings().repo_root
    store = ArtifactStore(resolved_repo)

    # --- Step 1: Load features ---
    if not store.is_trained(model_name, model_type):
        logger.warning("predict_games: (%s, %s) not trained.", model_name, model_type)
        return pd.DataFrame()

    df: DataFrame = load_modeling_file(resolved_repo, required_schema_version=_SCHEMA_VERSION)
    features = feature_fn(df)
    valid = features.notna().all(axis=1)
    df_valid = df.loc[valid].copy()
    x_feat = features.loc[valid]

    if x_feat.empty:
        return pd.DataFrame()

    # --- Step 2: Win probability inference ---
    pipeline = store.load(model_name, model_type)
    probs = pipeline.predict_proba(x_feat)[:, 1]

    # --- Step 3: Total points inference (optional) ---
    totals: pd.Series | None = None
    if total_model_name:
        try:
            from gridiron_edge.models.game_prediction.total import predict_total

            totals = predict_total(
                df_valid,
                model_name=total_model_name,
                model_type=total_model_type,
                repo=resolved_repo,
            )
        except FileNotFoundError:
            logger.debug(
                "predict_games: total model (%s, %s) not available",
                total_model_name,
                total_model_type,
            )

    # --- Step 4: Build game-level rows ---
    result: DataFrame = build_game_predictions(
        df_valid,
        probs,
        model_name=model_name,
        model_type=model_type,
        is_backfilled=is_backfilled,
        totals=totals,
    )

    # --- Step 5: Enrich ---
    result = enrich_predictions(
        result,
        model_name=model_name,
        model_type=model_type,
        recalibrate=True,
        repo=resolved_repo,
    )

    return result

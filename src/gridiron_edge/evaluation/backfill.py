# src/gridiron_edge/evaluation/backfill.py

"""Bulk historical prediction archiving.

Generates and archives predictions for every historical game in a single
pass. Uses the PredictorRegistry to retrieve any registered model by
composite ``(model_name, model_type)`` key, so adding a new model
requires zero changes here.

Typical usage::

    from gridiron_edge.evaluation.backfill import backfill_model

    n = backfill_model(model_name="win_prob", model_type="random_forest")
    n = backfill_model(model_name="total", model_type="xgboost")

Registry key construction:
    The function builds the flat ``PredictorRegistry`` key as
    ``f"{model_name}_{model_type}"``. This matches the composite-key
    convention introduced in Workstream 2 D2b.1.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.constants import AWAY_WIN_LOCATION as _AWAY_WIN_LOCATION
from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets import loaders
from gridiron_edge.evaluation.archive import load_prediction_log, write_archive_rows
from gridiron_edge.models.base import Predictor

logger: Logger = logging.getLogger(__name__)

# Week range considered valid for archiving (regular season + postseason).
_VALID_WEEK_RANGE: Final[range] = range(1, 23)


def _reconstruct_away_home(games: pd.DataFrame) -> pd.DataFrame:
    """Add AWAY_TEAM and HOME_TEAM columns to the canonical games DataFrame.

    Args:
        games: Canonical games DataFrame with WINNER, LOSER, GAME_LOCATION.

    Returns:
        Input DataFrame with AWAY_TEAM and HOME_TEAM columns added.
    """
    away_won: Series[bool] = games["GAME_LOCATION"] == _AWAY_WIN_LOCATION
    games = games.copy()
    games["AWAY_TEAM"] = games["WINNER"].where(away_won, games["LOSER"])
    games["HOME_TEAM"] = games["LOSER"].where(away_won, games["WINNER"])
    return games


def backfill_model(
    *,
    model_name: str,
    model_type: str,
    overwrite: bool = False,
    repo: Path | None = None,
) -> int:
    """Archive predictions for all historical games using any registered model.

    Retrieves the predictor from ``PredictorRegistry`` using the composite
    key ``f"{model_name}_{model_type}"``, runs ``predict_historical`` on
    the full games dataset, and appends the results to the prediction
    archive. Deduplicates on ``(game_id, model_name, model_type)`` so
    re-running is safe.

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        overwrite: If ``True``, re-archive all games even if predictions
            for this ``(model_name, model_type)`` pair already exist.
        repo: Repository root. Defaults to settings repo root.

    Returns:
        Number of new prediction rows written to the archive.

    Raises:
        KeyError: If no predictor is registered for the composite key.
    """
    # Import here to trigger registration of all predictor modules.
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import PredictorRegistry

    resolved_repo: Path = repo or get_settings().repo_root
    registry_key: str = f"{model_name}_{model_type}"

    predictor: Predictor = PredictorRegistry.get(registry_key)()

    games_raw: DataFrame = loaders.load_games(resolved_repo)
    games: DataFrame = games_raw.loc[games_raw["WIN_OR_TIE"].notna(), :].copy()

    df_new: DataFrame = predictor.predict_historical(games, repo=resolved_repo)

    if df_new.empty:
        logger.warning(
            "backfill_model: no predictions generated for (%s, %s).",
            model_name,
            model_type,
        )
        return 0

    if not overwrite:
        existing: DataFrame = load_prediction_log(
            model_name=model_name,
            model_type=model_type,
            repo=resolved_repo,
        )
        if not existing.empty:
            already_archived: set = set(existing["game_id"].unique())
            df_new = df_new.loc[~df_new["game_id"].isin(already_archived), :].copy()
            if df_new.empty:
                logger.info(
                    "All historical games already archived for (%s, %s).",
                    model_name,
                    model_type,
                )
                return 0

    n_new: int = len(df_new)
    write_archive_rows(df_new, repo=resolved_repo)
    logger.info(
        "backfill_model: %d predictions archived for (%s, %s).",
        n_new,
        model_name,
        model_type,
    )
    return n_new

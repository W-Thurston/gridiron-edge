# src/gridiron_edge/evaluation/backfill.py

"""Bulk historical prediction archiving.

Generates and archives predictions for every historical game in a single
pass. Uses the PredictorRegistry to retrieve any registered model by name,
so adding a new model requires zero changes here.

Typical usage::

    from gridiron_edge.evaluation.backfill import backfill_model

    n = backfill_model("elo_v1")
    n = backfill_model("logistic_v1")  # once registered
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets import loaders
from gridiron_edge.evaluation.archive import load_prediction_log, write_archive_rows
from gridiron_edge.models.base import Predictor

logger: Logger = logging.getLogger(__name__)

# Week range considered valid for archiving (regular season + postseason).
_VALID_WEEK_RANGE: Final[range] = range(1, 23)

# GAME_LOCATION value indicating the winner was the away team.
_AWAY_WIN_LOCATION: Final[str] = "@"


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
    model_version: str,
    *,
    overwrite: bool = False,
    repo: Path | None = None,
) -> int:
    """Archive predictions for all historical games using any registered model.

    Retrieves the predictor from PredictorRegistry, runs predict_historical
    on the full games dataset, and appends the results to the prediction
    archive. Deduplicates on (game_id, model_version) so re-running is safe.

    Args:
        model_version: Registered model version string (e.g. ``"elo_v1"``).
            Run ``gridiron evaluate list-models`` to see available versions.
        overwrite: If ``True``, re-archive all games even if predictions
            for this model version already exist.
        repo: Repository root. Defaults to settings repo root.

    Returns:
        Number of new prediction rows written to the archive.

    Raises:
        KeyError: If ``model_version`` is not registered.
    """
    # Import here to trigger registration of all predictor modules.
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import PredictorRegistry

    resolved_repo: Path = repo or get_settings().repo_root
    predictor: Predictor = PredictorRegistry.get(model_version)()

    games_raw: DataFrame = loaders.load_games(resolved_repo)
    games: DataFrame = games_raw.loc[games_raw["WIN_OR_TIE"].notna(), :].copy()

    df_new: DataFrame = predictor.predict_historical(games, repo=resolved_repo)

    if df_new.empty:
        logger.warning("backfill_model: no predictions generated for '%s'.", model_version)
        return 0

    if not overwrite:
        existing: DataFrame = load_prediction_log(model_version=model_version, repo=resolved_repo)
        if not existing.empty:
            already_archived: set = set(existing["game_id"].unique())
            df_new = df_new.loc[~df_new["game_id"].isin(already_archived), :].copy()
            if df_new.empty:
                logger.info("All historical games already archived for '%s'.", model_version)
                return 0

    n_new: int = len(df_new)
    write_archive_rows(df_new, repo=resolved_repo)
    logger.info("backfill_model: %d predictions archived for '%s'.", n_new, model_version)
    return n_new

# src/gridiron_edge/evaluation/backfill.py

"""Bulk historical prediction archiving.

Two modes:

- **Walk-forward** (default for trained ML models): for each historical
  season N, retrain the model on data strictly through season N-1 using
  fixed hyperparameters from the current spec, then predict season N.
  Intermediate models are discarded after their predictions are written.
  Honest with respect to model weights; mild HP-leakage is an accepted
  tradeoff against the substantially longer runtime of also walk-forward
  searching hyperparameters per season.

- **Current-model** (default for analytic models like elo): use the
  currently-trained predictor for all historical games. Honest for
  Elo because its state is built chronologically game-by-game.

Typical usage::

    from gridiron_edge.evaluation.backfill import backfill_model

    # Walk-forward for ML models (the right default):
    n = backfill_model(model_name="win_prob", model_type="random_forest")

    # Current-model for Elo (the right default for analytic models):
    n = backfill_model(model_name="win_prob", model_type="elo")

    # Explicit override if needed:
    n = backfill_model(
        model_name="win_prob",
        model_type="random_forest",
        mode="current-model",  # use existing artifact, not walk-forward
    )
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets import loaders
from gridiron_edge.datasets.loaders import load_modeling_file
from gridiron_edge.evaluation.archive import load_prediction_log, write_archive_rows
from gridiron_edge.models.game_prediction.base import GameModelType, GamesTrainer
from gridiron_edge.models.game_prediction.predictor import (
    build_game_predictions,
    build_regression_predictions,
)

logger: Logger = logging.getLogger(__name__)

#: Models that use current-model mode by default. Elo's state is built
#: chronologically by ``ratings/elo/table.py`` so ``predict_historical``
#: on the current state is honest for in-sample seasons. Adding new
#: analytic (non-trained-ML) models here keeps the default sensible.
_CURRENT_MODEL_DEFAULTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("win_prob", "elo"),
    }
)

# ---------------------------------------------------------------------------
# Walk-forward data-sufficiency contract
# ---------------------------------------------------------------------------
#
# Two knobs together determine when walk-forward will attempt a cutoff:
#
#   1. ``_MIN_WALK_FORWARD_TRAIN_SEASONS`` (below) sets a *season* floor.
#      _backfill_walk_forward will not attempt any cutoff whose training
#      pool covers fewer than this many prior seasons. Prevents obviously
#      degenerate early cutoffs from entering the loop at all.
#
#   2. ``min_cv_train_rows`` passed to ``trainer.train(...)`` sets a *row*
#      floor inside HP search. TimeSeriesSplit(n_splits=5) produces
#      training folds of roughly N/6 ... 5N/6. Any fold below this row
#      count is skipped by ``GamesTrainer._cv_score``. If every fold is
#      skipped for every combo, ``_run_hp_search`` raises.
#
# For champion training on the full-history split (~13k rows), the module
# default from ``_features.MIN_CV_TRAIN_ROWS`` (4000) leaves folds 2-5
# surviving. Walk-forward's training pools are much smaller, so
# ``_walk_forward_one_season`` explicitly overrides with a lower value.
# The two knobs must stay in rough agreement: at
# ``_MIN_WALK_FORWARD_TRAIN_SEASONS`` seasons, at least one CV fold must
# clear ``min_cv_train_rows``, otherwise the earliest cutoff will raise
# and the whole retrain fails.
#
# ~272 games/season x 2 rows (home/away perspective) ≈ 544 rows/season.
# At 3 seasons, training pool ≈ 1,632 rows; largest fold (5/6) ≈ 1,360.
# The walk-forward override of ``min_cv_train_rows=200`` clears that with
# ample margin.
#
# ---------------------------------------------------------------------------

#: Minimum number of prior seasons required before walk-forward will
#: attempt a cutoff. See "Walk-forward data-sufficiency contract" above.
_MIN_WALK_FORWARD_TRAIN_SEASONS: int = 3

#: Minimum training rows per CV fold for the walk-forward path.
#: Overrides the champion-training default of ``MIN_CV_TRAIN_ROWS``.
#: See "Walk-forward data-sufficiency contract" above.
_WALK_FORWARD_MIN_CV_TRAIN_ROWS: int = 200

BackfillMode = Literal["walk-forward", "current-model"]


def _resolve_mode(
    model_name: str,
    model_type: str,
    mode: BackfillMode | None,
) -> BackfillMode:
    """Resolve the backfill mode for a given model.

    If the caller passed an explicit mode, use it. Otherwise, fall back
    to ``_CURRENT_MODEL_DEFAULTS`` for analytic models or
    ``"walk-forward"`` for everything else.
    """
    if mode is not None:
        return mode
    if (model_name, model_type) in _CURRENT_MODEL_DEFAULTS:
        return "current-model"
    return "walk-forward"


def _backfill_current_model(
    *,
    model_name: str,
    model_type: str,
    repo: Path,
) -> DataFrame:
    """Generate predictions for all historical games using the current artifact.

    Used for analytic models (elo) where state is built chronologically
    and the current artifact produces honest historical predictions.
    """
    from typing import cast

    from gridiron_edge.models.base import GameModel
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import PredictorRegistry

    registry_key: str = f"{model_name}_{model_type}"

    predictor = cast(GameModel, PredictorRegistry.get(registry_key)())

    games_raw: DataFrame = loaders.load_games(repo)
    games: DataFrame = games_raw.loc[games_raw["WIN_OR_TIE"].notna(), :].copy()

    return predictor.predict_historical(games, repo=repo)


def _resolve_walk_forward_trainer(
    model_name: str,
) -> tuple[GamesTrainer, type[GameModelType]]:
    """Resolve the GamesTrainer subclass for a walk-forward backfill.

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``, ``"total"``).

    Returns:
        Tuple of (trainer instance, GameModelType enum class).

    Raises:
        ValueError: If walk-forward is not yet supported for this model_name.
    """
    from gridiron_edge.models.game_prediction.total import TotalTrainer
    from gridiron_edge.models.game_prediction.win_prob import WinProbTrainer

    if model_name == "win_prob":
        return WinProbTrainer(), GameModelType
    if model_name == "total":
        return TotalTrainer(), GameModelType

    msg = (
        f"Walk-forward backfill not supported for model_name={model_name!r}. "
        f"Use mode='current-model' or extend _resolve_walk_forward_trainer."
    )
    raise ValueError(msg)


def _resolve_season_range(
    df: pd.DataFrame,
    start_season: str | None,
    end_season: str | None,
) -> tuple[list[str], list[str]]:
    """Resolve the season range to walk-forward through.

    Returns:
        Tuple of (all_seasons_sorted, seasons_to_predict).
    """
    seasons = sorted(df["YEAR"].astype(str).unique().tolist())

    if start_season is None:
        # Need enough prior seasons to survive TimeSeriesSplit + the
        # MIN_CV_TRAIN_ROWS guard. Otherwise all HP combos return inf.
        floor_idx: int = _MIN_WALK_FORWARD_TRAIN_SEASONS
        start_season = seasons[floor_idx] if len(seasons) > floor_idx else seasons[-1]

    if end_season is None:
        end_season = seasons[-1]

    seasons_to_predict = [s for s in seasons if start_season <= s <= end_season]
    return seasons, seasons_to_predict


def _walk_forward_one_season(
    *,
    trainer: GamesTrainer,
    gm_type: GameModelType,
    df: pd.DataFrame,
    target_season: str,
    train_through_season: str,
    model_name: str,
    model_type: str,
    repo: Path,
) -> pd.DataFrame:
    """Single walk-forward iteration: retrain then predict one season.

    Returns:
        DataFrame of predictions for ``target_season``, or empty if
        no valid rows.
    """
    logger.info(
        "Walk-forward iteration: train through %s, predict %s",
        train_through_season,
        target_season,
    )

    # Retrain (persist=False keeps this as an intermediate)
    meta = trainer.train(
        df,
        model_type=gm_type,
        repo=repo,
        train_through_season=train_through_season,
        persist=False,
        min_cv_train_rows=_WALK_FORWARD_MIN_CV_TRAIN_ROWS,
    )

    # Filter df to target season
    target_df = df.loc[df["YEAR"] == target_season, :].copy()
    if target_df.empty:
        logger.warning("No rows for target season %s; skipping", target_season)
        return pd.DataFrame()

    # Compute features
    feature_fn = trainer.spec.feature_set[gm_type].feature_fn
    features = feature_fn(target_df)
    valid = features.notna().all(axis=1)
    target_df_valid = target_df.loc[valid].copy()
    x_feat = features.loc[valid]

    if x_feat.empty:
        logger.warning("No valid rows for target season %s; skipping", target_season)
        return pd.DataFrame()

    # Predict
    x_feat_arr = trainer._scaler.transform(x_feat) if trainer._scaler is not None else x_feat.values

    if trainer.spec.task == "classification":
        probs = trainer._model.predict_proba(x_feat_arr)[:, 1]
        season_preds: DataFrame = build_game_predictions(
            target_df_valid,
            probs,
        )
    else:
        preds = trainer._model.predict(x_feat_arr)
        season_preds = build_regression_predictions(
            target_df_valid,
            preds,
        )

    logger.info(
        "Walk-forward predictions for %s: %d rows (cv_score=%s)",
        target_season,
        len(season_preds),
        meta.parameters.get("cv_brier") or meta.parameters.get("cv_mae"),
    )
    return season_preds


def _backfill_walk_forward(
    *,
    model_name: str,
    model_type: str,
    repo: Path,
    start_season: str | None = None,
    end_season: str | None = None,
) -> DataFrame:
    """Walk-forward retraining: for each season N, train on data through N-1, then predict season N.

    Each iteration retrains the model with ``train_through_season=N-1``
    using fixed hyperparameters from the current spec. The intermediate
    model is discarded after writing its season-N predictions.

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        repo: Repository root.
        start_season: First season to predict (e.g. ``"2000-2001"``).
            If ``None``, defaults to the second season in the modeling data.
        end_season: Last season to predict. If ``None``, defaults to the
            most recent season in the modeling data.

    Returns:
        Concatenated DataFrame of predictions across all walk-forward
        iterations.
    """
    from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION

    trainer, gm_type_cls = _resolve_walk_forward_trainer(model_name)
    gm_type = gm_type_cls(model_type)

    df: DataFrame = load_modeling_file(repo, required_schema_version=CURRENT_SCHEMA_VERSION)
    seasons, seasons_to_predict = _resolve_season_range(df, start_season, end_season)

    if not seasons_to_predict:
        logger.warning(
            "_backfill_walk_forward: no seasons in range for (%s, %s)",
            model_name,
            model_type,
        )
        return pd.DataFrame()

    logger.info(
        "Walk-forward backfill for (%s, %s): %d seasons from %s to %s",
        model_name,
        model_type,
        len(seasons_to_predict),
        seasons_to_predict[0],
        seasons_to_predict[-1],
    )

    all_predictions: list[DataFrame] = []
    for target_season in seasons_to_predict:
        target_start_year = int(target_season.split("-")[0])
        train_through_start = target_start_year - 1
        train_through_season = f"{train_through_start}-{train_through_start + 1}"

        if train_through_season not in seasons:
            logger.warning(
                "Skipping %s: no training data through %s",
                target_season,
                train_through_season,
            )
            continue

        season_preds = _walk_forward_one_season(
            trainer=trainer,
            gm_type=gm_type,
            df=df,
            target_season=target_season,
            train_through_season=train_through_season,
            model_name=model_name,
            model_type=model_type,
            repo=repo,
        )
        if not season_preds.empty:
            all_predictions.append(season_preds)

    if not all_predictions:
        return pd.DataFrame()

    # At the end of _backfill_walk_forward, after the concat:
    predicted_seasons: list[str] = sorted(
        [str(df["season"].iloc[0]) for df in all_predictions if not df.empty]
    )
    skipped_seasons: list[str] = sorted(set(seasons_to_predict) - set(predicted_seasons))

    logger.warning(
        "Walk-forward summary for (%s, %s): %d predicted, %d skipped",
        model_name,
        model_type,
        len(predicted_seasons),
        len(skipped_seasons),
    )
    if skipped_seasons:
        logger.warning("Skipped seasons: %s", skipped_seasons)

    return pd.concat(all_predictions, ignore_index=True)


def backfill_model(
    *,
    model_name: str,
    model_type: str,
    mode: BackfillMode | None = None,
    overwrite: bool = False,
    start_season: str | None = None,
    end_season: str | None = None,
    repo: Path | None = None,
) -> int:
    """Archive predictions for all historical games.

    Dispatches to walk-forward retraining (the right default for trained
    ML models) or current-model prediction (the right default for
    analytic models like Elo). Override with ``mode`` if needed.

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        mode: Explicit mode override. ``None`` selects the default for
            this model (see ``_CURRENT_MODEL_DEFAULTS``).
        overwrite: If ``True``, re-archive all games even if predictions
            for this ``(model_name, model_type)`` pair already exist.
        start_season: First season to predict (walk-forward only).
            Defaults to the second-earliest available season.
        end_season: Last season to predict (walk-forward only).
            Defaults to the most recent season.
        repo: Repository root. Defaults to settings repo root.

    Returns:
        Number of new prediction rows written to the archive.

    Raises:
        KeyError: If no predictor is registered for the composite key
            (current-model mode only).
        ValueError: If walk-forward is requested for a model_name not
            yet supported (currently only ``"win_prob"`` and ``"total"``).
    """
    resolved_repo: Path = repo or get_settings().repo_root
    resolved_mode = _resolve_mode(model_name, model_type, mode)

    logger.info(
        "backfill_model: (%s, %s) mode=%s",
        model_name,
        model_type,
        resolved_mode,
    )

    if resolved_mode == "current-model":
        df_new = _backfill_current_model(
            model_name=model_name,
            model_type=model_type,
            repo=resolved_repo,
        )
    else:
        df_new = _backfill_walk_forward(
            model_name=model_name,
            model_type=model_type,
            repo=resolved_repo,
            start_season=start_season,
            end_season=end_season,
        )

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
        "backfill_model: %d predictions archived for (%s, %s) via %s mode.",
        n_new,
        model_name,
        model_type,
        resolved_mode,
    )
    return n_new

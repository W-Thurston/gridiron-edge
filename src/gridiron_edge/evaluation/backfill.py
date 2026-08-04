# src/gridiron_edge/evaluation/backfill.py

"""Bulk historical prediction archiving.

Two modes:

- **Walk-forward** (default for trained ML models): for each historical
  season N, retrain the model on data strictly through season N-1 using
  fixed hyperparameters from the current spec, then predict season N.
  Intermediate models are discarded after their predictions are written.
  Each cutoff uses only modeling rows available through the preceding
  season.

- **Current-model** (default for analytic models like elo): use the
  currently-trained model for all historical games. Honest for
  Elo because its state is built chronologically game-by-game.

Typical usage::

    from gridiron_edge.evaluation.backfill import backfill_model

    # Walk-forward for ML models (the right default):
    result = backfill_model(model_name="win_prob", model_type="random_forest")

    # Current-model for Elo (the right default for analytic models):
    result = backfill_model(model_name="win_prob", model_type="elo")

    # Explicit override if needed:
    result = backfill_model(
        model_name="win_prob",
        model_type="random_forest",
        mode="current-model",  # use existing artifact, not walk-forward
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import logging
from logging import Logger
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets import loaders
from gridiron_edge.datasets.loaders import load_modeling_file
from gridiron_edge.evaluation.forecast_contracts import (
    ForecastRole,
    new_forecast_run_id,
)
from gridiron_edge.evaluation.forecast_events import build_forecast_events
from gridiron_edge.evaluation.forecast_store import write_forecast_events
from gridiron_edge.models.game_prediction.base import GameModelType, GamesTrainer
from gridiron_edge.models.game_prediction.model import (
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
# ~272 games/season x 1 canonical row per game ≈ 272 rows/season.
# At 3 seasons, the training pool is approximately 816 rows and the
# largest TimeSeriesSplit training fold is approximately 680 rows.
# The walk-forward override of ``min_cv_train_rows=200`` allows multiple
# folds to survive at the earliest supported cutoff.
#
# ---------------------------------------------------------------------------

#: Minimum number of prior seasons required before walk-forward will
#: attempt a cutoff. See "Walk-forward data-sufficiency contract" above.
_MIN_WALK_FORWARD_TRAIN_SEASONS: int = 3

#: Minimum training rows per CV fold for the walk-forward path.
#: Overrides the champion-training default of ``MIN_CV_TRAIN_ROWS``.
#: See "Walk-forward data-sufficiency contract" above.
_WALK_FORWARD_MIN_CV_TRAIN_ROWS: int = 200


class BackfillMode(StrEnum):
    """Supported historical reconstruction strategies."""

    WALK_FORWARD = "walk-forward"
    CURRENT_MODEL = "current-model"


class BackfillSeasonStatus(StrEnum):
    """Terminal outcome for one requested walk-forward season."""

    PREDICTED = "predicted"
    SKIPPED_NO_PRIOR_SEASON = "skipped_no_prior_season"
    SKIPPED_NO_TARGET_ROWS = "skipped_no_target_rows"
    SKIPPED_NO_VALID_ROWS = "skipped_no_valid_rows"


@dataclass(frozen=True, slots=True)
class BackfillSeasonResult:
    """Generated count or explicit skip reason for one season."""

    season: str
    status: BackfillSeasonStatus
    generated_count: int
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate status-specific count and reason invariants."""
        if not self.season.strip():
            raise ValueError("season must not be empty.")
        if self.status is BackfillSeasonStatus.PREDICTED:
            if self.generated_count <= 0:
                raise ValueError("Predicted seasons require generated_count > 0.")
            if self.reason is not None:
                raise ValueError("Predicted seasons must not include a reason.")
            return
        if self.generated_count != 0:
            raise ValueError("Skipped seasons require generated_count == 0.")
        if self.reason is None or not self.reason.strip():
            raise ValueError("Skipped seasons require a nonempty reason.")


@dataclass(frozen=True, slots=True)
class BackfillGeneration:
    """Prediction rows and per-season outcomes before event persistence."""

    predictions: DataFrame
    seasons: tuple[BackfillSeasonResult, ...]


@dataclass(frozen=True, slots=True)
class WalkForwardSeasonOutput:
    """Prediction rows and terminal status for one walk-forward season."""

    predictions: DataFrame
    result: BackfillSeasonResult


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """Accounting and provenance for one immutable historical run."""

    model_name: str
    model_type: str
    mode: BackfillMode
    run_id: str | None
    generated_count: int
    inserted_count: int
    existing_count: int
    seasons: tuple[BackfillSeasonResult, ...]

    def __post_init__(self) -> None:
        """Validate result identity, counts, run state, and season totals."""
        for field_name, value in (
            ("model_name", self.model_name),
            ("model_type", self.model_type),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
        for field_name, value in (
            ("generated_count", self.generated_count),
            ("inserted_count", self.inserted_count),
            ("existing_count", self.existing_count),
        ):
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative.")
        if self.inserted_count + self.existing_count != self.generated_count:
            raise ValueError("inserted_count plus existing_count must equal generated_count.")
        if self.generated_count == 0 and self.run_id is not None:
            raise ValueError("A zero-generation result must not have a run_id.")
        if self.generated_count > 0 and not self.run_id:
            raise ValueError("A generated backfill result requires a run_id.")
        if sum(item.generated_count for item in self.seasons) != self.generated_count:
            raise ValueError("Season counts must sum to generated_count.")

    @property
    def predicted_seasons(self) -> tuple[str, ...]:
        """Return seasons that generated historical forecast rows."""
        return tuple(
            item.season for item in self.seasons if item.status is BackfillSeasonStatus.PREDICTED
        )

    @property
    def skipped_seasons(self) -> tuple[str, ...]:
        """Return requested seasons that did not generate forecast rows."""
        return tuple(
            item.season
            for item in self.seasons
            if item.status is not BackfillSeasonStatus.PREDICTED
        )


def _resolve_mode(
    model_name: str,
    model_type: str,
    mode: BackfillMode | str | None,
) -> BackfillMode:
    """Resolve the backfill mode for a given model.

    If the caller passed an explicit mode, use it. Otherwise, fall back
    to ``_CURRENT_MODEL_DEFAULTS`` for analytic models or
    ``"walk-forward"`` for everything else.
    """
    if mode is not None:
        return BackfillMode(mode)
    if (model_name, model_type) in _CURRENT_MODEL_DEFAULTS:
        return BackfillMode.CURRENT_MODEL
    return BackfillMode.WALK_FORWARD


def _validate_season_label(value: str, *, field_name: str) -> str:
    """Validate and return one canonical ``YYYY-YYYY+1`` season label."""
    parts = value.split("-")
    if len(parts) != 2 or any(len(part) != 4 or not part.isdigit() for part in parts):
        raise ValueError(f"{field_name} must use YYYY-YYYY+1 format, got {value!r}.")
    start, end = (int(part) for part in parts)
    if end != start + 1:
        raise ValueError(f"{field_name} must use consecutive years, got {value!r}.")
    return value


def _validate_backfill_request(
    *,
    mode: BackfillMode,
    start_season: str | None,
    end_season: str | None,
) -> None:
    """Validate mode-specific season bounds before loading model data."""
    if start_season is not None:
        _validate_season_label(start_season, field_name="start_season")
    if end_season is not None:
        _validate_season_label(end_season, field_name="end_season")
    if start_season is not None and end_season is not None and start_season > end_season:
        raise ValueError("start_season must not be later than end_season.")
    if mode is BackfillMode.CURRENT_MODEL and (start_season is not None or end_season is not None):
        raise ValueError("Season bounds are supported only in walk-forward mode.")


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
    import gridiron_edge.models.elo.model
    import gridiron_edge.models.game_prediction.model  # noqa: F401
    from gridiron_edge.models.registry import ModelRegistry

    registry_key: str = f"{model_name}_{model_type}"

    model = cast(GameModel, ModelRegistry.get(registry_key)())

    games_raw: DataFrame = loaders.load_games(repo)
    completed_mask = games_raw["AWAY_SCORE"].notna() & games_raw["HOME_SCORE"].notna()
    games: DataFrame = games_raw.loc[
        completed_mask,
        :,
    ].copy()

    return model.predict_historical(games, repo=repo)


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
) -> WalkForwardSeasonOutput:
    """Retrain through the prior season and predict one target season."""
    logger.info(
        "Walk-forward iteration: train through %s, predict %s",
        train_through_season,
        target_season,
    )
    meta = trainer.train(
        df,
        model_type=gm_type,
        repo=repo,
        train_through_season=train_through_season,
        persist=False,
        min_cv_train_rows=_WALK_FORWARD_MIN_CV_TRAIN_ROWS,
    )
    target_df = df.loc[df["YEAR"] == target_season, :].copy()
    if target_df.empty:
        reason = "no target rows"
        logger.warning("No rows for target season %s; skipping", target_season)
        return WalkForwardSeasonOutput(
            predictions=DataFrame(),
            result=BackfillSeasonResult(
                season=target_season,
                status=BackfillSeasonStatus.SKIPPED_NO_TARGET_ROWS,
                generated_count=0,
                reason=reason,
            ),
        )

    feature_fn = trainer.spec.feature_set[gm_type].feature_fn
    features = feature_fn(target_df)
    valid = features.notna().all(axis=1)
    target_df_valid = target_df.loc[valid].copy()
    x_feat = features.loc[valid]
    if x_feat.empty:
        reason = "no target rows with complete model features"
        logger.warning("No valid rows for target season %s; skipping", target_season)
        return WalkForwardSeasonOutput(
            predictions=DataFrame(),
            result=BackfillSeasonResult(
                season=target_season,
                status=BackfillSeasonStatus.SKIPPED_NO_VALID_ROWS,
                generated_count=0,
                reason=reason,
            ),
        )

    x_feat_arr = trainer._scaler.transform(x_feat) if trainer._scaler is not None else x_feat.values
    if trainer.spec.task == "classification":
        probs = trainer._model.predict_proba(x_feat_arr)[:, 1]
        predictions = build_game_predictions(target_df_valid, probs)
    else:
        values = trainer._model.predict(x_feat_arr)
        predictions = build_regression_predictions(target_df_valid, values)

    logger.info(
        "Walk-forward predictions for %s: %d rows (cv_score=%s)",
        target_season,
        len(predictions),
        meta.parameters.get("cv_brier") or meta.parameters.get("cv_mae"),
    )
    return WalkForwardSeasonOutput(
        predictions=predictions,
        result=BackfillSeasonResult(
            season=target_season,
            status=BackfillSeasonStatus.PREDICTED,
            generated_count=len(predictions),
        ),
    )


def _backfill_walk_forward(
    *,
    model_name: str,
    model_type: str,
    repo: Path,
    start_season: str | None = None,
    end_season: str | None = None,
) -> BackfillGeneration:
    """Walk forward by season and preserve every terminal season outcome."""
    from gridiron_edge.features.manifest import CURRENT_SCHEMA_VERSION

    trainer, gm_type_cls = _resolve_walk_forward_trainer(model_name)
    gm_type = gm_type_cls(model_type)
    df = load_modeling_file(repo, required_schema_version=CURRENT_SCHEMA_VERSION)
    seasons, targets = _resolve_season_range(df, start_season, end_season)
    if not targets:
        logger.warning(
            "_backfill_walk_forward: no seasons in range for (%s, %s)",
            model_name,
            model_type,
        )
        return BackfillGeneration(predictions=DataFrame(), seasons=())

    all_predictions: list[DataFrame] = []
    season_results: list[BackfillSeasonResult] = []
    for target_season in targets:
        target_start_year = int(target_season.split("-")[0])
        training_start = target_start_year - 1
        train_through = f"{training_start}-{training_start + 1}"
        if train_through not in seasons:
            reason = f"no training data through {train_through}"
            logger.warning("Skipping %s: %s", target_season, reason)
            season_results.append(
                BackfillSeasonResult(
                    season=target_season,
                    status=BackfillSeasonStatus.SKIPPED_NO_PRIOR_SEASON,
                    generated_count=0,
                    reason=reason,
                )
            )
            continue

        output = _walk_forward_one_season(
            trainer=trainer,
            gm_type=gm_type,
            df=df,
            target_season=target_season,
            train_through_season=train_through,
            model_name=model_name,
            model_type=model_type,
            repo=repo,
        )
        season_results.append(output.result)
        if not output.predictions.empty:
            all_predictions.append(output.predictions)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else DataFrame()
    return BackfillGeneration(
        predictions=predictions,
        seasons=tuple(season_results),
    )


def backfill_model(
    *,
    model_name: str,
    model_type: str,
    mode: BackfillMode | str | None = None,
    start_season: str | None = None,
    end_season: str | None = None,
    repo: Path | None = None,
) -> BackfillResult:
    """Generate and store one immutable historical forecast run.

    Dispatches to walk-forward retraining (the right default for trained
    ML models) or current-model prediction (the right default for
    analytic models like Elo). Override with ``mode`` if needed.

    Each successful invocation creates a distinct backfilled forecast run.
    Existing live and backfilled events are never replaced or skipped.

    Args:
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``).
        mode: Explicit mode override. ``None`` selects the default for
            this model (see ``_CURRENT_MODEL_DEFAULTS``).
        start_season: First season to predict (walk-forward only).
            Defaults to the second-earliest available season.
        end_season: Last season to predict (walk-forward only).
            Defaults to the most recent season.
        repo: Repository root. Defaults to settings repo root.

    Returns:
        Structured generation, insertion, run, mode, and season accounting.

    Raises:
        KeyError: If no model is registered for the composite key
            (current-model mode only).
        ValueError: If walk-forward is requested for a model_name not
            yet supported (currently only ``"win_prob"`` and ``"total"``).
    """
    resolved_mode = _resolve_mode(model_name, model_type, mode)
    _validate_backfill_request(
        mode=resolved_mode,
        start_season=start_season,
        end_season=end_season,
    )
    resolved_repo: Path = repo or get_settings().repo_root

    logger.info(
        "backfill_model: (%s, %s) mode=%s",
        model_name,
        model_type,
        resolved_mode,
    )

    if resolved_mode is BackfillMode.CURRENT_MODEL:
        df_new = _backfill_current_model(
            model_name=model_name,
            model_type=model_type,
            repo=resolved_repo,
        )
        season_counts = df_new.groupby("season", sort=True).size().items()
        generation = BackfillGeneration(
            predictions=df_new,
            seasons=tuple(
                BackfillSeasonResult(
                    season=str(season),
                    status=BackfillSeasonStatus.PREDICTED,
                    generated_count=int(count),
                )
                for season, count in season_counts
            ),
        )
    else:
        generation = _backfill_walk_forward(
            model_name=model_name,
            model_type=model_type,
            repo=resolved_repo,
            start_season=start_season,
            end_season=end_season,
        )
    df_new = generation.predictions

    if df_new.empty:
        logger.warning(
            "backfill_model: no predictions generated for (%s, %s).",
            model_name,
            model_type,
        )
        return BackfillResult(
            model_name=model_name,
            model_type=model_type,
            mode=resolved_mode,
            run_id=None,
            generated_count=0,
            inserted_count=0,
            existing_count=0,
            seasons=generation.seasons,
        )

    n_new: int = len(df_new)
    run_id: str = new_forecast_run_id()
    generated_at: datetime = datetime.now(UTC)

    events: DataFrame = build_forecast_events(
        df_new,
        model_name=model_name,
        model_type=model_type,
        run_id=run_id,
        role=ForecastRole.BACKFILLED,
        generated_at=generated_at,
    )

    write_result = write_forecast_events(
        events,
        repo=resolved_repo,
    )

    logger.info(
        "backfill_model: %d forecast events written for (%s, %s) via %s mode in run %s.",
        n_new,
        model_name,
        model_type,
        resolved_mode,
        run_id,
    )
    return BackfillResult(
        model_name=model_name,
        model_type=model_type,
        mode=resolved_mode,
        run_id=run_id,
        generated_count=n_new,
        inserted_count=write_result.inserted_count,
        existing_count=write_result.existing_count,
        seasons=generation.seasons,
    )

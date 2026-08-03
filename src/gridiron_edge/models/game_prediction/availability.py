# src/gridiron_edge/models/game_prediction/availability.py
"""Read-only weekly game-model availability inspection."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
from pandas import DataFrame

from gridiron_edge.datasets import loaders
from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.pipeline import CANONICAL_FEATURES
from gridiron_edge.features.registry import run_features
from gridiron_edge.models.artifact import ArtifactStore
from gridiron_edge.models.game_prediction.model import GamesModel
from gridiron_edge.models.game_prediction.prediction_policy import PredictionAvailability
from gridiron_edge.models.registry import ModelRegistry
from gridiron_edge.ratings.elo.predict import _build_elo_schedule, _validate_elo_identity

_MODEL_REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
    ("win_prob", "logistic", "classification"),
    ("win_prob", "random_forest", "classification"),
    ("win_prob", "xgboost", "classification"),
    ("total", "random_forest", "regression"),
    ("total", "xgboost", "regression"),
)


def _scope_schedule(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Validate and scope the rich schedule to one requested week."""
    if not season.strip():
        raise ValueError("season must not be empty.")
    if week < 1:
        raise ValueError("week must be at least 1.")

    required = {
        "season",
        "week",
        "game_id",
        "game_day_of_week",
        "game_date",
        "game_time",
        "away_team",
        "home_team",
        "neutral_site",
    }
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(
            "Rich upcoming schedule is missing required columns: " + ", ".join(missing)
        )

    scoped = schedule.loc[
        (schedule["season"].astype(str) == season)
        & (pd.to_numeric(schedule["week"], errors="coerce") == week),
        :,
    ].copy()
    if scoped.empty:
        raise ValueError(f"Rich upcoming schedule has no games for {season} week {week}.")

    duplicate_ids = scoped["game_id"].astype(str).duplicated(keep=False)
    if duplicate_ids.any():
        values = sorted(scoped.loc[duplicate_ids, "game_id"].astype(str).unique())
        raise ValueError(
            "Rich upcoming schedule has duplicate game_id values: " + ", ".join(values)
        )
    return scoped


def _inspect_elo(canonical_schedule: DataFrame, *, repo: Path) -> bool:
    """Return whether every scoped game has exact-week Away and Home Elo."""
    try:
        elo = loaders.load_elo_state(repo)
    except FileNotFoundError:
        return False

    required = {"NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"}
    missing = sorted(required - set(elo.columns))
    if missing:
        raise ValueError("Elo state is missing required columns: " + ", ".join(missing))
    _validate_elo_identity(elo)

    identities = elo.loc[:, ["NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"]]
    away = canonical_schedule.merge(
        identities,
        how="left",
        left_on=["AWAY_TEAM", "YEAR", "WEEK_NUM"],
        right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
    )["ELO"]
    home = canonical_schedule.merge(
        identities,
        how="left",
        left_on=["HOME_TEAM", "YEAR", "WEEK_NUM"],
        right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
    )["ELO"]
    return bool(away.notna().all() and home.notna().all())


def _inspect_trained_model(
    enriched: DataFrame,
    *,
    model_name: str,
    model_type: str,
    expected_task: str,
    repo: Path,
) -> bool:
    """Inspect one exact persisted game model without loading its estimator."""
    store = ArtifactStore(repo)
    if not store.is_trained(model_name, model_type):
        return False

    metadata = store.read_metadata(model_name, model_type)
    if metadata.model_name != model_name or metadata.model_type != model_type:
        raise ValueError("Artifact metadata identity does not match the requested model.")
    if metadata.kind != "game":
        raise ValueError("Prediction availability requires game artifact metadata.")
    if metadata.task != expected_task:
        raise ValueError("Artifact metadata task does not match the requested model family.")

    model_path = store.artifact_dir(model_name, model_type) / "model.joblib"
    if not model_path.is_file():
        return False

    registry_key = f"{model_name}_{model_type}"
    model = cast(GamesModel, ModelRegistry.get(registry_key)())
    feature_set = model.prediction_feature_set()
    if metadata.feature_columns != feature_set.feature_names:
        return False

    features = feature_set.feature_fn(enriched)
    if features.columns.tolist() != feature_set.feature_names:
        raise ValueError(
            f"{registry_key} produced a feature schema that differs from its contract."
        )
    if not features.index.equals(enriched.index):
        raise ValueError(f"{registry_key} feature rows are not aligned to the weekly schedule.")
    return bool(len(features) == len(enriched) and features.notna().all(axis=1).all())


def inspect_prediction_availability(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
    repo: Path,
) -> PredictionAvailability:
    """Inspect exact-model input availability for one complete weekly schedule."""
    scoped = _scope_schedule(schedule, season=season, week=week)
    canonical = _build_elo_schedule(scoped.copy())
    datasets = DatasetAccessor(repo=repo)
    enriched = run_features(
        df=canonical.copy(),
        feature_names=CANONICAL_FEATURES,
        datasets=datasets,
    )

    availability: dict[tuple[str, str], bool] = {}
    for model_name, model_type, task in _MODEL_REQUIREMENTS:
        availability[(model_name, model_type)] = _inspect_trained_model(
            enriched,
            model_name=model_name,
            model_type=model_type,
            expected_task=task,
            repo=repo,
        )

    return PredictionAvailability(
        season=season,
        week=week,
        elo_available=_inspect_elo(canonical, repo=repo),
        win_logistic_features_available=availability[("win_prob", "logistic")],
        win_random_forest_features_available=availability[("win_prob", "random_forest")],
        win_xgboost_features_available=availability[("win_prob", "xgboost")],
        total_random_forest_features_available=availability[("total", "random_forest")],
        total_xgboost_features_available=availability[("total", "xgboost")],
    )

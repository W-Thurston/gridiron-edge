# src/gridiron_edge/models/game_prediction/weekly_execution.py
"""Execute one availability-aware weekly game prediction policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
from pandas import DataFrame

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_events import build_forecast_events
from gridiron_edge.models.base import GameModel
from gridiron_edge.models.game_prediction.availability import (
    inspect_prediction_availability,
)
from gridiron_edge.models.game_prediction.prediction_policy import (
    PredictionModelDecision,
    PredictionModelStatus,
    PredictionPolicy,
    load_prediction_policy,
)
from gridiron_edge.models.registry import ModelRegistry
from gridiron_edge.ratings.elo.predict import (
    _build_elo_schedule,
    format_elo_prediction_percentages,
)


@dataclass(frozen=True)
class WeeklyPredictionExecution:
    """Policy, immutable event rows, and selected Win display output."""

    policy: PredictionPolicy
    events: DataFrame
    win_display: DataFrame | None


def _scope_schedule(schedule: DataFrame, *, season: str, week: int) -> DataFrame:
    """Return one nonempty, uniquely identified weekly rich schedule."""
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
    duplicated = scoped["game_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        values = sorted(scoped.loc[duplicated, "game_id"].astype(str).unique())
        raise ValueError("Weekly schedule has duplicate game IDs: " + ", ".join(values))
    return scoped.reset_index(drop=True)


def _execute_decision(
    decision: PredictionModelDecision,
    canonical_schedule: DataFrame,
    *,
    repo: Path,
) -> DataFrame | None:
    """Execute one selected family through its exact registered model."""
    if decision.status is PredictionModelStatus.UNAVAILABLE:
        return None
    if decision.model_type is None:
        raise ValueError("Selected prediction decision has no model_type.")

    registry_key = f"{decision.model_name}_{decision.model_type}"
    model = cast(GameModel, ModelRegistry.get(registry_key)())
    return model.predict_upcoming(canonical_schedule.copy(), repo=repo)


def _validate_coverage(
    predictions: DataFrame,
    expected_ids: list[str],
    *,
    family: str,
) -> None:
    """Require exactly one returned prediction for every scheduled game."""
    if "GAME_ID" not in predictions.columns:
        raise ValueError(f"{family} predictions are missing GAME_ID.")
    actual = predictions["GAME_ID"]
    if actual.isna().any():
        raise ValueError(f"{family} predictions contain null GAME_ID values.")
    actual_ids = actual.astype(str)
    duplicated = actual_ids.duplicated(keep=False)
    if duplicated.any():
        values = sorted(actual_ids.loc[duplicated].unique())
        raise ValueError(f"{family} predictions contain duplicate Game IDs: " + ", ".join(values))

    expected_set = set(expected_ids)
    actual_set = set(actual_ids.tolist())
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if len(predictions) != len(expected_ids) or missing or unexpected:
        raise ValueError(
            f"{family} prediction coverage does not match the weekly schedule; "
            f"missing={missing}, unexpected={unexpected}."
        )


def _canonical_win_rows(
    predictions: DataFrame,
    scoped: DataFrame,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Map selected Win output to canonical forecast rows."""
    identity = scoped.loc[:, ["game_id", "game_date", "away_team", "home_team"]]
    source = identity.merge(
        predictions,
        how="left",
        left_on="game_id",
        right_on="GAME_ID",
        validate="one_to_one",
    )
    output = DataFrame(
        {
            "season": [season] * len(source),
            "week": [week] * len(source),
            "game_id": source["game_id"],
            "game_date": source["game_date"],
            "away_team": source["away_team"],
            "home_team": source["home_team"],
            "away_elo": source.get("AWAY_TEAM_ELO"),
            "home_elo": source.get("HOME_TEAM_ELO"),
            "away_win_prob": source["AWAY_WIN_PROB"],
            "home_win_prob": source["HOME_WIN_PROB"],
        }
    )
    for column in (
        "model_spread",
        "projected_home_score",
        "projected_away_score",
        "margin_std",
        "win_prob_lo",
        "win_prob_hi",
        "confidence_tier",
    ):
        if column in source.columns:
            output[column] = source[column]
    return output


def _canonical_total_rows(
    predictions: DataFrame,
    scoped: DataFrame,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Map selected Total output to canonical forecast rows."""
    identity = scoped.loc[:, ["game_id", "game_date", "away_team", "home_team"]]
    source = identity.merge(
        predictions,
        how="left",
        left_on="game_id",
        right_on="GAME_ID",
        validate="one_to_one",
    )
    return DataFrame(
        {
            "season": [season] * len(source),
            "week": [week] * len(source),
            "game_id": source["game_id"],
            "game_date": source["game_date"],
            "away_team": source["away_team"],
            "home_team": source["home_team"],
            "model_total": source["model_total"],
        }
    )


def _win_display_frame(predictions: DataFrame, scoped: DataFrame) -> DataFrame:
    """Attach rich schedule metadata required by existing renderers."""
    metadata = scoped.loc[
        :,
        ["game_id", "game_date", "game_time", "game_day_of_week"],
    ].rename(
        columns={
            "game_id": "GAME_ID",
            "game_date": "GAME_DATE",
            "game_time": "GAMETIME",
            "game_day_of_week": "GAME_DAY_OF_WEEK",
        }
    )
    display = metadata.merge(predictions, how="left", on="GAME_ID", validate="one_to_one")
    return format_elo_prediction_percentages(display)


def execute_weekly_prediction_policy(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
    repo: Path,
    run_id: str,
    generated_at: datetime,
    win_override: str | None = None,
    total_override: str | None = None,
) -> WeeklyPredictionExecution:
    """Resolve and execute the exact selected weekly Win and Total models."""
    import gridiron_edge.models.elo.model
    import gridiron_edge.models.game_prediction.model  # noqa: F401

    scoped = _scope_schedule(schedule, season=season, week=week)
    availability = inspect_prediction_availability(
        schedule,
        season=season,
        week=week,
        repo=repo,
    )
    policy = load_prediction_policy(
        availability,
        repo=repo,
        win_override=win_override,
        total_override=total_override,
    )
    if (
        policy.win.status is PredictionModelStatus.UNAVAILABLE
        and policy.total.status is PredictionModelStatus.UNAVAILABLE
    ):
        raise ValueError("Prediction policy selected no available Win or Total model.")

    canonical_schedule = _build_elo_schedule(scoped.copy())
    expected_ids = scoped["game_id"].astype(str).tolist()
    win_predictions = _execute_decision(policy.win, canonical_schedule, repo=repo)
    total_predictions = _execute_decision(policy.total, canonical_schedule, repo=repo)

    if win_predictions is not None:
        _validate_coverage(win_predictions, expected_ids, family="Win")
    if total_predictions is not None:
        _validate_coverage(total_predictions, expected_ids, family="Total")

    event_frames: list[DataFrame] = []
    win_display: DataFrame | None = None
    if win_predictions is not None:
        if policy.win.model_type is None:
            raise ValueError("Selected Win policy has no model_type.")
        win_rows = _canonical_win_rows(
            win_predictions,
            scoped,
            season=season,
            week=week,
        )
        event_frames.append(
            build_forecast_events(
                win_rows,
                model_name="win_prob",
                model_type=policy.win.model_type,
                run_id=run_id,
                role=ForecastRole.LIVE,
                generated_at=generated_at,
            )
        )
        win_display = _win_display_frame(win_predictions, scoped)

    if total_predictions is not None:
        if policy.total.model_type is None:
            raise ValueError("Selected Total policy has no model_type.")
        total_rows = _canonical_total_rows(
            total_predictions,
            scoped,
            season=season,
            week=week,
        )
        event_frames.append(
            build_forecast_events(
                total_rows,
                model_name="total",
                model_type=policy.total.model_type,
                run_id=run_id,
                role=ForecastRole.LIVE,
                generated_at=generated_at,
            )
        )

    events = pd.concat(event_frames, ignore_index=True)
    return WeeklyPredictionExecution(
        policy=policy,
        events=events,
        win_display=win_display,
    )

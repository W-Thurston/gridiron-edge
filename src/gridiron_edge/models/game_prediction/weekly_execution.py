# src/gridiron_edge/models/game_prediction/weekly_execution.py
"""Execute one availability-aware weekly game prediction policy."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_events import build_forecast_events
from gridiron_edge.evaluation.prediction_input_evidence import (
    EloPredictionEventEvidence,
    JsonScalar,
    PredictionInputEvidence,
    SourceArtifactReference,
    SourceRevision,
    StatisticalPredictionEventEvidence,
    authenticate_prediction_input_evidence,
    create_elo_prediction_input_evidence,
    create_statistical_prediction_input_evidence,
)
from gridiron_edge.models.elo.model import WinProbEloModel
from gridiron_edge.models.game_prediction.availability import (
    inspect_prediction_availability,
)
from gridiron_edge.models.game_prediction.prediction_execution import (
    EloPredictionExecution,
    StatisticalPredictionExecution,
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
    """Policy, immutable events, input evidence, and Win display output."""

    policy: PredictionPolicy
    events: DataFrame
    input_evidence: tuple[PredictionInputEvidence, ...]
    win_display: DataFrame | None


@dataclass(frozen=True)
class _DecisionExecution:
    """Predictions plus exact execution-kind-specific evidence."""

    predictions: DataFrame
    elo_execution: EloPredictionExecution | None
    statistical_execution: StatisticalPredictionExecution | None


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
    source_revision: SourceRevision | None,
    source_artifacts: tuple[SourceArtifactReference, ...] | None,
) -> _DecisionExecution | None:
    """Execute one selected family through its evidence-aware model boundary."""
    if decision.status is PredictionModelStatus.UNAVAILABLE:
        return None
    if decision.model_type is None:
        raise ValueError("Selected prediction decision has no model_type.")
    if source_revision is None:
        raise ValueError("Selected weekly execution requires source_revision.")
    if source_artifacts is None:
        raise ValueError("Selected weekly execution requires source_artifacts.")

    registry_key = f"{decision.model_name}_{decision.model_type}"
    model = ModelRegistry.get(registry_key)()
    if decision.model_name == "win_prob" and decision.model_type == "elo":
        if not isinstance(model, WinProbEloModel):
            raise TypeError("Registered win_prob/elo model is not WinProbEloModel.")
        execution = model.predict_upcoming_with_evidence(
            canonical_schedule.copy(),
            source_revision=source_revision,
            source_artifacts=source_artifacts,
            repo=repo,
        )
        return _DecisionExecution(execution.predictions, execution, None)

    prediction_method = getattr(model, "predict_upcoming_with_evidence", None)
    if not callable(prediction_method):
        raise TypeError(f"Registered {registry_key} model has no evidence-aware method.")
    execution = prediction_method(
        canonical_schedule.copy(),
        source_revision=source_revision,
        source_artifacts=source_artifacts,
        repo=repo,
    )
    if not isinstance(execution, StatisticalPredictionExecution):
        raise TypeError(f"Registered {registry_key} model returned invalid execution evidence.")
    return _DecisionExecution(execution.predictions, None, execution)


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
    actual_ids = [str(value) for value in actual.tolist()]
    duplicate_ids = sorted(game_id for game_id, count in Counter(actual_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(
            f"{family} predictions contain duplicate Game IDs: " + ", ".join(duplicate_ids)
        )
    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
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


def _elo_input_evidence(
    execution: EloPredictionExecution,
    events: DataFrame,
    *,
    run_id: str,
    generated_at: datetime,
    season: str,
    week: int,
) -> PredictionInputEvidence:
    """Bind generated forecast UUIDs to exact Elo computations by game ID."""
    event_rows = events.set_index("game_id", drop=False)
    computations = {value.game_id: value for value in execution.computations}
    if set(event_rows.index.astype(str)) != set(computations):
        raise ValueError("Elo computations do not match generated forecast-event games.")

    evidence_events: list[EloPredictionEventEvidence] = []
    for game_id in sorted(computations):
        computation = computations[game_id]
        row = event_rows.loc[game_id]
        evidence_events.append(
            EloPredictionEventEvidence(
                event_id=str(row["event_id"]),
                game_id=game_id,
                season=season,
                week=week,
                away_team=computation.away_team,
                home_team=computation.home_team,
                away_elo=computation.away_elo,
                home_elo=computation.home_elo,
                formula_id=computation.formula_id,
                divisor=computation.divisor,
                away_win_probability=computation.away_win_probability,
                home_win_probability=computation.home_win_probability,
                final_outputs=(
                    ("away_elo", computation.away_elo),
                    (
                        "away_win_prob",
                        computation.away_win_probability,
                    ),
                    ("home_elo", computation.home_elo),
                    (
                        "home_win_prob",
                        computation.home_win_probability,
                    ),
                ),
            )
        )

    evidence = create_elo_prediction_input_evidence(
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
        source_revision=execution.source_revision,
        source_artifacts=execution.source_artifacts,
        binary_artifacts=execution.binary_artifacts,
        events=tuple(evidence_events),
    )
    authenticate_prediction_input_evidence(evidence, forecast_events=events)
    return evidence


def _event_final_outputs(
    row: Series,
    *,
    task: str,
) -> tuple[tuple[str, JsonScalar], ...]:
    """Return exact persisted forecast outputs for one statistical event."""
    if task == "classification":
        names = (
            "away_elo",
            "away_win_prob",
            "confidence_tier",
            "home_elo",
            "home_win_prob",
            "margin_std",
            "model_spread",
            "projected_away_score",
            "projected_home_score",
            "win_prob_hi",
            "win_prob_lo",
        )
    elif task == "regression":
        names = ("model_total",)
    else:
        raise ValueError(f"Unsupported statistical prediction task: {task!r}.")

    outputs: list[tuple[str, JsonScalar]] = []
    for name in names:
        value = row[name]

        if pd.isna(value):
            normalized: JsonScalar = None
        else:
            scalar = value.item() if hasattr(value, "item") else value
            normalized = cast(JsonScalar, scalar)

        outputs.append((name, normalized))

    return tuple(outputs)


def _statistical_input_evidence(
    execution: StatisticalPredictionExecution,
    events: DataFrame,
    *,
    run_id: str,
    generated_at: datetime,
    season: str,
    week: int,
) -> PredictionInputEvidence:
    """Bind generated forecast UUIDs to exact statistical computations."""
    event_rows = events.set_index("game_id", drop=False)
    computations = {value.game_id: value for value in execution.computations}
    if set(event_rows.index.astype(str)) != set(computations):
        raise ValueError("Statistical computations do not match generated forecast-event games.")

    evidence_events: list[StatisticalPredictionEventEvidence] = []

    for game_id in sorted(computations):
        computation = computations[game_id]
        row = cast(Series, event_rows.loc[game_id])

        evidence_events.append(
            StatisticalPredictionEventEvidence(
                event_id=str(row["event_id"]),
                game_id=game_id,
                raw_feature_values=computation.raw_feature_values,
                transformed_feature_values=(computation.transformed_feature_values),
                raw_estimator_output=(computation.raw_estimator_output),
                post_estimator_output=(computation.post_estimator_output),
                final_outputs=_event_final_outputs(
                    row,
                    task=execution.feature_schema.task,
                ),
            )
        )
    evidence = create_statistical_prediction_input_evidence(
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
        model_name=execution.feature_schema.model_name,
        model_type=execution.feature_schema.model_type,
        source_revision=execution.source_revision,
        source_artifacts=execution.source_artifacts,
        binary_artifacts=execution.binary_artifacts,
        feature_schema=execution.feature_schema,
        post_processing=execution.post_processing,
        events=tuple(evidence_events),
    )
    authenticate_prediction_input_evidence(evidence, forecast_events=events)
    return evidence


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
    source_revision: SourceRevision | None = None,
    source_artifacts: tuple[SourceArtifactReference, ...] | None = None,
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
    expected_ids = [str(value) for value in scoped["game_id"].tolist()]
    win_execution = _execute_decision(
        policy.win,
        canonical_schedule,
        repo=repo,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
    )
    total_execution = _execute_decision(
        policy.total,
        canonical_schedule,
        repo=repo,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
    )

    if win_execution is not None:
        _validate_coverage(win_execution.predictions, expected_ids, family="Win")
    if total_execution is not None:
        _validate_coverage(total_execution.predictions, expected_ids, family="Total")

    event_frames: list[DataFrame] = []
    input_evidence: list[PredictionInputEvidence] = []
    win_display: DataFrame | None = None
    if win_execution is not None:
        if policy.win.model_type is None:
            raise ValueError("Selected Win policy has no model_type.")
        win_rows = _canonical_win_rows(
            win_execution.predictions,
            scoped,
            season=season,
            week=week,
        )
        win_events = build_forecast_events(
            win_rows,
            model_name="win_prob",
            model_type=policy.win.model_type,
            run_id=run_id,
            role=ForecastRole.LIVE,
            generated_at=generated_at,
        )
        event_frames.append(win_events)
        if win_execution.elo_execution is not None:
            input_evidence.append(
                _elo_input_evidence(
                    win_execution.elo_execution,
                    win_events,
                    run_id=run_id,
                    generated_at=generated_at,
                    season=season,
                    week=week,
                )
            )
        elif win_execution.statistical_execution is not None:
            input_evidence.append(
                _statistical_input_evidence(
                    win_execution.statistical_execution,
                    win_events,
                    run_id=run_id,
                    generated_at=generated_at,
                    season=season,
                    week=week,
                )
            )
        else:
            raise ValueError("Selected Win execution returned no input evidence.")
        win_display = _win_display_frame(win_execution.predictions, scoped)

    if total_execution is not None:
        if policy.total.model_type is None:
            raise ValueError("Selected Total policy has no model_type.")
        total_rows = _canonical_total_rows(
            total_execution.predictions,
            scoped,
            season=season,
            week=week,
        )
        total_events = build_forecast_events(
            total_rows,
            model_name="total",
            model_type=policy.total.model_type,
            run_id=run_id,
            role=ForecastRole.LIVE,
            generated_at=generated_at,
        )
        event_frames.append(total_events)
        if total_execution.statistical_execution is None:
            raise ValueError("Selected Total execution returned no statistical input evidence.")
        input_evidence.append(
            _statistical_input_evidence(
                total_execution.statistical_execution,
                total_events,
                run_id=run_id,
                generated_at=generated_at,
                season=season,
                week=week,
            )
        )

    events = pd.concat(event_frames, ignore_index=True)
    return WeeklyPredictionExecution(
        policy=policy,
        events=events,
        input_evidence=tuple(input_evidence),
        win_display=win_display,
    )

# src/gridiron_edge/models/game_prediction/weekly_win_product.py

"""Schedule-complete weekly win prediction product composition."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.evaluation.forecast_selection import (
    ForecastCandidateResolution,
    ForecastCandidateStatus,
)
from gridiron_edge.evaluation.forecast_store import validate_forecast_events
from gridiron_edge.models.game_prediction.prediction_policy import (
    PredictionModelStatus,
    PredictionPolicy,
)


class WeeklyWinStatus(StrEnum):
    """Availability state for one scheduled game's win prediction."""

    AVAILABLE = "available"
    POLICY_UNAVAILABLE = "policy_unavailable"
    FORECAST_MISSING = "forecast_missing"
    FORECAST_AMBIGUOUS = "forecast_ambiguous"


_REQUIRED_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
)

_PRODUCT_COLUMNS: Final[tuple[str, ...]] = (
    "win_status",
    "win_selection_status",
    "away_win_prob",
    "home_win_prob",
    "win_model_name",
    "win_model_type",
    "win_event_id",
    "win_run_id",
    "win_generated_at",
    "win_role",
)


def _require_schedule_columns(schedule: DataFrame) -> None:
    """Require canonical rich-schedule identity columns."""
    missing = sorted(set(_REQUIRED_SCHEDULE_COLUMNS) - set(schedule.columns))
    if missing:
        raise ValueError("Schedule is missing required columns: " + ", ".join(missing))


def _scope_schedule(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Return the requested schedule rows in source order."""
    _require_schedule_columns(schedule)

    scoped = schedule.loc[
        (schedule["season"].astype(str) == season) & (schedule["week"] == week),
        :,
    ].copy()

    duplicated = scoped["game_id"].duplicated(keep=False)
    if duplicated.any():
        game_ids = sorted(scoped.loc[duplicated, "game_id"].astype(str).unique().tolist())
        raise ValueError("Schedule contains duplicate game IDs: " + ", ".join(game_ids))

    return scoped.reset_index(drop=True)


def _resolution_map(
    resolutions: Sequence[ForecastCandidateResolution],
    *,
    model_type: str,
) -> dict[str, ForecastCandidateResolution]:
    """Validate and index explicit win-forecast resolutions."""
    indexed: dict[str, ForecastCandidateResolution] = {}

    for resolution in resolutions:
        identity = resolution.identity
        if identity.model_name != "win_prob":
            raise ValueError("Win product resolution must use model_name 'win_prob'.")
        if identity.model_type != model_type:
            raise ValueError(
                "Win product resolution model_type does not match policy: "
                f"{identity.model_type!r} != {model_type!r}."
            )
        if identity.game_id in indexed:
            raise ValueError(
                f"Win product resolutions contain duplicate game ID: {identity.game_id}."
            )
        indexed[identity.game_id] = resolution

    return indexed


def _event_map(events: DataFrame) -> dict[str, Series]:
    """Validate forecast events and index rows by immutable event ID."""
    normalized = validate_forecast_events(events)
    return {str(row["event_id"]): row for _, row in normalized.iterrows()}


def _empty_product_values(
    status: WeeklyWinStatus,
    *,
    selection_status: str,
) -> dict[str, object]:
    """Return nullable prediction fields for one unavailable row."""
    return {
        "win_status": status.value,
        "win_selection_status": selection_status,
        "away_win_prob": pd.NA,
        "home_win_prob": pd.NA,
        "win_model_name": pd.NA,
        "win_model_type": pd.NA,
        "win_event_id": pd.NA,
        "win_run_id": pd.NA,
        "win_generated_at": pd.NaT,
        "win_role": pd.NA,
    }


def _validate_selected_event(
    event: Series,
    *,
    schedule_row: Series,
    selected_event_id: str,
    model_type: str,
    season: str,
    week: int,
) -> tuple[float, float]:
    """Validate selected event identity, orientation, and probabilities."""
    if str(event["event_id"]) != selected_event_id:
        raise ValueError("Selected forecast event ID does not match event row.")
    if str(event["season"]) != season or int(event["week"]) != week:
        raise ValueError("Selected forecast event is outside product scope.")
    if str(event["game_id"]) != str(schedule_row["game_id"]):
        raise ValueError("Selected forecast game_id does not match schedule.")
    if str(event["model_name"]) != "win_prob":
        raise ValueError("Selected forecast must use model_name 'win_prob'.")
    if str(event["model_type"]) != model_type:
        raise ValueError("Selected forecast model_type does not match policy.")
    if str(event["away_team"]) != str(schedule_row["away_team"]):
        raise ValueError("Selected forecast away_team does not match schedule.")
    if str(event["home_team"]) != str(schedule_row["home_team"]):
        raise ValueError("Selected forecast home_team does not match schedule.")

    away_probability = event["away_win_prob"]
    home_probability = event["home_win_prob"]
    if pd.isna(away_probability) or pd.isna(home_probability):
        raise ValueError("Selected win forecast must contain both probabilities.")

    away = float(away_probability)
    home = float(home_probability)
    if not 0.0 <= away <= 1.0 or not 0.0 <= home <= 1.0:
        raise ValueError("Selected win probabilities must be between 0 and 1.")
    if abs((away + home) - 1.0) > 1e-9:
        raise ValueError("Selected win probabilities must sum to 1.")

    return away, home


def build_weekly_win_product(
    schedule: DataFrame,
    events: DataFrame,
    resolutions: Sequence[ForecastCandidateResolution],
    *,
    policy: PredictionPolicy,
    season: str,
    week: int,
) -> DataFrame:
    """Attach explicitly selected win forecasts to every scheduled game.

    The function performs no model execution, champion resolution, forecast
    generation, storage reads, timestamp-based selection, or filesystem I/O.
    """
    if policy.availability.season != season or policy.availability.week != week:
        raise ValueError("Prediction policy scope does not match product scope.")

    scoped = _scope_schedule(schedule, season=season, week=week)
    if scoped.empty:
        return scoped.assign(**{column: pd.Series(dtype="object") for column in _PRODUCT_COLUMNS})

    if policy.win.status is PredictionModelStatus.UNAVAILABLE:
        if resolutions:
            raise ValueError("Unavailable win policy must not contain forecast resolutions.")
        values = [
            _empty_product_values(
                WeeklyWinStatus.POLICY_UNAVAILABLE,
                selection_status="not_requested",
            )
            for _ in range(len(scoped))
        ]
        return pd.concat(
            [scoped, DataFrame(values, index=scoped.index)],
            axis=1,
        )

    model_type = policy.win.model_type
    if model_type is None:
        raise ValueError("Selected win policy requires model_type.")

    indexed_resolutions = _resolution_map(
        resolutions,
        model_type=model_type,
    )
    indexed_events = _event_map(events)
    product_values: list[dict[str, object]] = []

    for _, schedule_row in scoped.iterrows():
        game_id = str(schedule_row["game_id"])
        resolution = indexed_resolutions.get(game_id)

        if resolution is None or resolution.status is ForecastCandidateStatus.MISSING:
            product_values.append(
                _empty_product_values(
                    WeeklyWinStatus.FORECAST_MISSING,
                    selection_status=ForecastCandidateStatus.MISSING.value,
                )
            )
            continue

        if resolution.status is ForecastCandidateStatus.AMBIGUOUS:
            product_values.append(
                _empty_product_values(
                    WeeklyWinStatus.FORECAST_AMBIGUOUS,
                    selection_status=ForecastCandidateStatus.AMBIGUOUS.value,
                )
            )
            continue

        selected = resolution.selected
        if selected is None:
            raise ValueError("Selected resolution requires a selected forecast.")
        if selected.event_id not in indexed_events:
            product_values.append(
                _empty_product_values(
                    WeeklyWinStatus.FORECAST_MISSING,
                    selection_status=ForecastCandidateStatus.SELECTED.value,
                )
            )
            continue

        event = indexed_events[selected.event_id]
        away_probability, home_probability = _validate_selected_event(
            event,
            schedule_row=schedule_row,
            selected_event_id=selected.event_id,
            model_type=model_type,
            season=season,
            week=week,
        )
        product_values.append(
            {
                "win_status": WeeklyWinStatus.AVAILABLE.value,
                "win_selection_status": ForecastCandidateStatus.SELECTED.value,
                "away_win_prob": away_probability,
                "home_win_prob": home_probability,
                "win_model_name": str(event["model_name"]),
                "win_model_type": str(event["model_type"]),
                "win_event_id": str(event["event_id"]),
                "win_run_id": str(event["run_id"]),
                "win_generated_at": event["generated_at"],
                "win_role": str(event["role"]),
            }
        )

    return pd.concat(
        [scoped, DataFrame(product_values, index=scoped.index)],
        axis=1,
    )

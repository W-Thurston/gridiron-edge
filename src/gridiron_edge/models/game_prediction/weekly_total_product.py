# src/gridiron_edge/models/game_prediction/weekly_total_product.py

"""Independent total attachment for the schedule-complete weekly product."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.evaluation.forecast_selection import (
    ForecastCandidateResolution,
    ForecastCandidateStatus,
)
from gridiron_edge.evaluation.forecast_store import validate_forecast_events
from gridiron_edge.models.artifact import ArtifactStore
from gridiron_edge.models.game_prediction.prediction_policy import (
    PredictionModelStatus,
    PredictionPolicy,
)


class WeeklyTotalStatus(StrEnum):
    """Availability state for one scheduled game's total prediction."""

    AVAILABLE = "available"
    POLICY_UNAVAILABLE = "policy_unavailable"
    FORECAST_MISSING = "forecast_missing"
    FORECAST_AMBIGUOUS = "forecast_ambiguous"
    UNCERTAINTY_UNAVAILABLE = "uncertainty_unavailable"


@dataclass(frozen=True)
class TotalUncertainty:
    """Strict uncertainty metadata for one total model artifact."""

    model_name: str
    model_type: str
    total_std: float
    trained_at: str

    def __post_init__(self) -> None:
        """Validate uncertainty identity and value."""
        for field_name, value in (
            ("model_name", self.model_name),
            ("model_type", self.model_type),
            ("trained_at", self.trained_at),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
        if not math.isfinite(self.total_std) or self.total_std <= 0:
            raise ValueError("total_std must be finite and greater than 0.")


_TOTAL_COLUMNS: Final[tuple[str, ...]] = (
    "total_status",
    "model_total",
    "total_uncertainty",
    "total_model_name",
    "total_model_type",
    "total_event_id",
    "total_run_id",
    "total_generated_at",
    "total_role",
    "total_selection_status",
    "total_uncertainty_trained_at",
)

_REQUIRED_PRODUCT_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
)


def load_total_uncertainty(
    model_name: str,
    model_type: str,
    *,
    repo: Path,
) -> TotalUncertainty | None:
    """Load exact artifact RMSE without substituting a default value."""
    store = ArtifactStore(repo)
    if not store.is_trained(model_name, model_type):
        return None

    metadata = store.read_metadata(model_name, model_type)
    if metadata.model_name != model_name or metadata.model_type != model_type:
        raise ValueError("Total artifact metadata identity does not match lookup.")
    if metadata.task != "regression":
        raise ValueError("Total uncertainty requires regression metadata.")

    rmse = metadata.metrics.get("rmse")
    if not isinstance(rmse, int | float):
        return None
    total_std = float(rmse)
    if not math.isfinite(total_std) or total_std <= 0:
        return None
    if not metadata.trained_at.strip():
        return None

    return TotalUncertainty(
        model_name=model_name,
        model_type=model_type,
        total_std=total_std,
        trained_at=metadata.trained_at,
    )


def _require_product_columns(product: DataFrame) -> None:
    """Require schedule identity columns from the existing weekly product."""
    missing = sorted(set(_REQUIRED_PRODUCT_COLUMNS) - set(product.columns))
    if missing:
        raise ValueError("Weekly product is missing required columns: " + ", ".join(missing))


def _resolution_map(
    resolutions: Sequence[ForecastCandidateResolution],
    *,
    model_type: str,
) -> dict[str, ForecastCandidateResolution]:
    """Validate and index explicit total forecast resolutions."""
    indexed: dict[str, ForecastCandidateResolution] = {}
    for resolution in resolutions:
        identity = resolution.identity
        if identity.model_name != "total":
            raise ValueError("Total resolution must use model_name 'total'.")
        if identity.model_type != model_type:
            raise ValueError(
                "Total resolution model_type does not match policy: "
                f"{identity.model_type!r} != {model_type!r}."
            )
        if identity.game_id in indexed:
            raise ValueError(f"Total resolutions contain duplicate game ID: {identity.game_id}.")
        indexed[identity.game_id] = resolution
    return indexed


def _event_map(events: DataFrame) -> dict[str, Series]:
    """Validate immutable event rows and index them by exact event ID."""
    normalized = validate_forecast_events(events)
    return {str(row["event_id"]): row for _, row in normalized.iterrows()}


def _blocked_values(
    status: WeeklyTotalStatus,
    *,
    selection_status: str,
) -> dict[str, object]:
    """Return nullable total fields for one blocked row."""
    return {
        "total_status": status.value,
        "model_total": pd.NA,
        "total_uncertainty": pd.NA,
        "total_model_name": pd.NA,
        "total_model_type": pd.NA,
        "total_event_id": pd.NA,
        "total_run_id": pd.NA,
        "total_generated_at": pd.NaT,
        "total_role": pd.NA,
        "total_selection_status": selection_status,
        "total_uncertainty_trained_at": pd.NA,
    }


def _validate_total_event(
    event: Series,
    product_row: Series,
    *,
    event_id: str,
    model_type: str,
    season: str,
    week: int,
) -> float:
    """Validate selected event identity, orientation, and point estimate."""
    if str(event["event_id"]) != event_id:
        raise ValueError("Selected total event ID does not match event row.")
    if str(event["season"]) != season or int(event["week"]) != week:
        raise ValueError("Selected total event is outside product scope.")
    if str(event["game_id"]) != str(product_row["game_id"]):
        raise ValueError("Selected total game_id does not match product row.")
    if str(event["away_team"]) != str(product_row["away_team"]):
        raise ValueError("Selected total away_team does not match product row.")
    if str(event["home_team"]) != str(product_row["home_team"]):
        raise ValueError("Selected total home_team does not match product row.")
    if str(event["model_name"]) != "total":
        raise ValueError("Selected total event must use model_name 'total'.")
    if str(event["model_type"]) != model_type:
        raise ValueError("Selected total model_type does not match policy.")

    value = event["model_total"]
    if pd.isna(value):
        raise ValueError("Selected total event must contain model_total.")
    total = float(value)
    if not math.isfinite(total):
        raise ValueError("Selected model_total must be finite.")
    return total


def _available_values(
    event: Series,
    *,
    model_total: float,
    uncertainty: TotalUncertainty | None,
) -> dict[str, object]:
    """Return available total fields with optional exact uncertainty."""
    model_name = str(event["model_name"])
    model_type = str(event["model_type"])
    if uncertainty is not None:
        if uncertainty.model_name != model_name:
            raise ValueError("Total uncertainty model_name does not match forecast.")
        if uncertainty.model_type != model_type:
            raise ValueError("Total uncertainty model_type does not match forecast.")

    return {
        "total_status": (
            WeeklyTotalStatus.AVAILABLE.value
            if uncertainty is not None
            else WeeklyTotalStatus.UNCERTAINTY_UNAVAILABLE.value
        ),
        "model_total": model_total,
        "total_uncertainty": (uncertainty.total_std if uncertainty is not None else pd.NA),
        "total_model_name": model_name,
        "total_model_type": model_type,
        "total_event_id": str(event["event_id"]),
        "total_run_id": str(event["run_id"]),
        "total_generated_at": event["generated_at"],
        "total_role": str(event["role"]),
        "total_selection_status": ForecastCandidateStatus.SELECTED.value,
        "total_uncertainty_trained_at": (
            uncertainty.trained_at if uncertainty is not None else pd.NA
        ),
    }


def attach_selected_totals(
    weekly_product: DataFrame,
    events: DataFrame,
    resolutions: Sequence[ForecastCandidateResolution],
    uncertainties: dict[tuple[str, str], TotalUncertainty],
    *,
    policy: PredictionPolicy,
    season: str,
    week: int,
) -> DataFrame:
    """Attach independent selected total forecasts to every weekly product row.

    Total identity comes only from policy.total and explicit total forecast
    resolutions. Win and spread model identities are not consulted.
    """
    _require_product_columns(weekly_product)
    if policy.availability.season != season or policy.availability.week != week:
        raise ValueError("Prediction policy scope does not match total product scope.")

    source = (
        weekly_product.loc[
            (weekly_product["season"].astype(str) == season) & (weekly_product["week"] == week),
            :,
        ]
        .copy()
        .reset_index(drop=True)
    )

    if source["game_id"].duplicated().any():
        raise ValueError("Weekly product contains duplicate game IDs.")

    if policy.total.status is PredictionModelStatus.UNAVAILABLE:
        if resolutions:
            raise ValueError("Unavailable total policy must not contain resolutions.")
        values = [
            _blocked_values(
                WeeklyTotalStatus.POLICY_UNAVAILABLE,
                selection_status="not_requested",
            )
            for _ in range(len(source))
        ]
        return pd.concat([source, DataFrame(values, index=source.index)], axis=1)

    model_type = policy.total.model_type
    if model_type is None:
        raise ValueError("Selected total policy requires model_type.")

    indexed_resolutions = _resolution_map(resolutions, model_type=model_type)
    indexed_events = _event_map(events)
    values: list[dict[str, object]] = []

    for _, row in source.iterrows():
        game_id = str(row["game_id"])
        resolution = indexed_resolutions.get(game_id)
        if resolution is None or resolution.status is ForecastCandidateStatus.MISSING:
            values.append(
                _blocked_values(
                    WeeklyTotalStatus.FORECAST_MISSING,
                    selection_status=ForecastCandidateStatus.MISSING.value,
                )
            )
            continue
        if resolution.status is ForecastCandidateStatus.AMBIGUOUS:
            values.append(
                _blocked_values(
                    WeeklyTotalStatus.FORECAST_AMBIGUOUS,
                    selection_status=ForecastCandidateStatus.AMBIGUOUS.value,
                )
            )
            continue

        selected = resolution.selected
        if selected is None:
            raise ValueError("Selected total resolution requires a forecast reference.")
        event = indexed_events.get(selected.event_id)
        if event is None:
            values.append(
                _blocked_values(
                    WeeklyTotalStatus.FORECAST_MISSING,
                    selection_status=ForecastCandidateStatus.SELECTED.value,
                )
            )
            continue

        model_total = _validate_total_event(
            event,
            row,
            event_id=selected.event_id,
            model_type=model_type,
            season=season,
            week=week,
        )
        uncertainty = uncertainties.get(("total", model_type))
        values.append(
            _available_values(
                event,
                model_total=model_total,
                uncertainty=uncertainty,
            )
        )

    return pd.concat([source, DataFrame(values, index=source.index)], axis=1)


def load_and_attach_selected_totals(
    weekly_product: DataFrame,
    events: DataFrame,
    resolutions: Sequence[ForecastCandidateResolution],
    *,
    policy: PredictionPolicy,
    season: str,
    week: int,
    repo: Path,
) -> DataFrame:
    """Load exact selected-total uncertainty and compose the total component."""
    uncertainties: dict[tuple[str, str], TotalUncertainty] = {}
    if (
        policy.total.status is PredictionModelStatus.SELECTED
        and policy.total.model_type is not None
    ):
        uncertainty = load_total_uncertainty(
            "total",
            policy.total.model_type,
            repo=repo,
        )
        if uncertainty is not None:
            uncertainties[("total", policy.total.model_type)] = uncertainty

    return attach_selected_totals(
        weekly_product,
        events,
        resolutions,
        uncertainties,
        policy=policy,
        season=season,
        week=week,
    )

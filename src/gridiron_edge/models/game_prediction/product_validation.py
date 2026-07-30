# src/gridiron_edge/models/game_prediction/product_validation.py

"""Validation for composed schedule-complete weekly game products."""

from __future__ import annotations

import math
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.models.game_prediction.weekly_game_product import (
    ProjectedScoreStatus,
)
from gridiron_edge.models.game_prediction.weekly_spread_product import WeeklySpreadStatus
from gridiron_edge.models.game_prediction.weekly_total_product import WeeklyTotalStatus
from gridiron_edge.models.game_prediction.weekly_win_product import WeeklyWinStatus

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "win_status",
    "away_win_prob",
    "home_win_prob",
    "win_model_name",
    "win_model_type",
    "win_event_id",
    "spread_status",
    "model_spread",
    "spread_uncertainty",
    "spread_source_event_id",
    "spread_model_name",
    "spread_model_type",
    "spread_calibration_key",
    "spread_calibration_updated_at",
    "total_status",
    "model_total",
    "total_uncertainty",
    "total_model_name",
    "total_model_type",
    "total_event_id",
    "total_uncertainty_trained_at",
    "projected_score_status",
    "projected_home_score",
    "projected_away_score",
)

_TOLERANCE: Final[float] = 1e-9


def _require_columns(product: DataFrame) -> None:
    missing = sorted(set(_REQUIRED_COLUMNS) - set(product.columns))
    if missing:
        raise ValueError("Weekly game product is missing required columns: " + ", ".join(missing))


def _require_text(row: Series, columns: tuple[str, ...], *, context: str) -> None:
    for column in columns:
        value = row[column]
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"{context} requires nonempty {column}.")


def _finite(row: Series, column: str, *, context: str) -> float:
    value = row[column]
    if pd.isna(value):
        raise ValueError(f"{context} requires {column}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} requires finite {column}.")
    return number


def _null(row: Series, columns: tuple[str, ...], *, context: str) -> None:
    populated = [column for column in columns if not pd.isna(row[column])]
    if populated:
        raise ValueError(f"{context} requires null fields: " + ", ".join(populated))


def _validate_win(row: Series) -> None:
    if str(row["win_status"]) != WeeklyWinStatus.AVAILABLE.value:
        return

    away = _finite(row, "away_win_prob", context="Available win")
    home = _finite(row, "home_win_prob", context="Available win")
    if not 0.0 <= away <= 1.0 or not 0.0 <= home <= 1.0:
        raise ValueError("Available win probabilities must be between 0 and 1.")
    if abs((away + home) - 1.0) > _TOLERANCE:
        raise ValueError("Available win probabilities must sum to 1.")
    _require_text(
        row,
        ("win_model_name", "win_model_type", "win_event_id"),
        context="Available win",
    )


def _validate_spread(row: Series) -> None:
    if str(row["spread_status"]) != WeeklySpreadStatus.AVAILABLE.value:
        _null(
            row,
            ("model_spread", "spread_uncertainty"),
            context="Unavailable spread",
        )
        return

    _finite(row, "model_spread", context="Available spread")
    uncertainty = _finite(row, "spread_uncertainty", context="Available spread")
    if uncertainty <= 0:
        raise ValueError("Available spread uncertainty must be greater than 0.")
    _require_text(
        row,
        (
            "spread_source_event_id",
            "spread_model_name",
            "spread_model_type",
            "spread_calibration_key",
            "spread_calibration_updated_at",
        ),
        context="Available spread",
    )
    if str(row["spread_source_event_id"]) != str(row["win_event_id"]):
        raise ValueError("Spread source event must match selected win event.")
    if str(row["spread_model_name"]) != str(row["win_model_name"]):
        raise ValueError("Spread model_name must match selected win model.")
    if str(row["spread_model_type"]) != str(row["win_model_type"]):
        raise ValueError("Spread model_type must match selected win model.")


def _validate_total(row: Series) -> None:
    status = str(row["total_status"])
    if status not in {
        WeeklyTotalStatus.AVAILABLE.value,
        WeeklyTotalStatus.UNCERTAINTY_UNAVAILABLE.value,
    }:
        _null(row, ("model_total", "total_uncertainty"), context="Unavailable total")
        return

    _finite(row, "model_total", context="Available total")
    _require_text(
        row,
        ("total_model_name", "total_model_type", "total_event_id"),
        context="Available total",
    )
    if str(row["total_model_name"]) != "total":
        raise ValueError("Available total must use model_name 'total'.")

    if status == WeeklyTotalStatus.AVAILABLE.value:
        uncertainty = _finite(row, "total_uncertainty", context="Available total")
        if uncertainty <= 0:
            raise ValueError("Available total uncertainty must be greater than 0.")
        _require_text(
            row,
            ("total_uncertainty_trained_at",),
            context="Available total",
        )
        return

    _null(
        row,
        ("total_uncertainty", "total_uncertainty_trained_at"),
        context="Total with unavailable uncertainty",
    )


def _validate_projected_scores(row: Series) -> None:
    status = str(row["projected_score_status"])
    if status != ProjectedScoreStatus.AVAILABLE.value:
        _null(
            row,
            ("projected_home_score", "projected_away_score"),
            context="Unavailable projected scores",
        )
        return

    home = _finite(row, "projected_home_score", context="Available projected scores")
    away = _finite(row, "projected_away_score", context="Available projected scores")
    spread = _finite(row, "model_spread", context="Available projected scores")
    total = _finite(row, "model_total", context="Available projected scores")

    if abs((home + away) - total) > _TOLERANCE:
        raise ValueError("Projected scores must reconcile to model_total.")
    if abs((away - home) - spread) > _TOLERANCE:
        raise ValueError("Projected score difference must reconcile to model_spread.")


def validate_weekly_game_product(product: DataFrame) -> DataFrame:
    """Validate final weekly product invariants and return a defensive copy."""
    _require_columns(product)
    normalized = product.copy()

    if normalized["game_id"].isna().any():
        raise ValueError("Weekly game product game_id must not be null.")
    if normalized["game_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Weekly game product game_id must not be empty.")
    if normalized["game_id"].duplicated().any():
        raise ValueError("Weekly game product contains duplicate game IDs.")

    for _, row in normalized.iterrows():
        _validate_win(row)
        _validate_spread(row)
        _validate_total(row)
        _validate_projected_scores(row)

    return normalized

# src/gridiron_edge/evaluation/forecast_events.py

"""Composition of canonical game predictions into forecast events.

This module maps storage-independent game prediction rows into the strict
forecast-event schema. Persistence remains the responsibility of
``forecast_store``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final
from uuid import uuid4

from pandas import DataFrame, Series

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_store import (
    FORECAST_EVENT_COLUMNS,
    validate_forecast_events,
)

_REQUIRED_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
)

_OPTIONAL_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "game_date",
    "away_elo",
    "home_elo",
    "away_win_prob",
    "home_win_prob",
    "model_spread",
    "model_total",
    "projected_home_score",
    "projected_away_score",
    "margin_std",
    "win_prob_lo",
    "win_prob_hi",
    "confidence_tier",
)


def _optional_column(
    predictions: DataFrame,
    column: str,
) -> Series:
    """Return an optional prediction column aligned to the source index."""
    if column in predictions.columns:
        return predictions[column].copy()

    return Series(
        [None] * len(predictions),
        index=predictions.index,
        dtype=object,
    )


def build_forecast_events(
    predictions: DataFrame,
    *,
    model_name: str,
    model_type: str,
    run_id: str,
    role: ForecastRole,
    generated_at: datetime,
) -> DataFrame:
    """Compose canonical prediction rows into immutable forecast events.

    Args:
        predictions: Canonical lowercase game-prediction rows. Each row
            represents one game forecast.
        model_name: Prediction family, such as ``"win_prob"`` or ``"total"``.
        model_type: Model implementation, such as ``"elo"`` or
            ``"random_forest"``.
        run_id: Shared identity for this prediction invocation.
        role: Whether the invocation is live or historically backfilled.
        generated_at: Shared timezone-aware UTC generation timestamp.

    Returns:
        A validated forecast-event DataFrame in canonical column order.

    Raises:
        ValueError: If required prediction columns or event values violate
            the forecast-event contract.
    """
    missing = sorted(set(_REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError("Prediction rows are missing required columns: " + ", ".join(missing))

    if predictions.empty:
        return DataFrame(columns=FORECAST_EVENT_COLUMNS)

    source = predictions.reset_index(drop=True)

    events = DataFrame(
        {
            "event_id": [str(uuid4()) for _ in range(len(source))],
            "run_id": [run_id] * len(source),
            "role": [role.value] * len(source),
            "generated_at": [generated_at] * len(source),
            "season": source["season"],
            "week": source["week"],
            "game_id": source["game_id"],
            "model_name": [model_name] * len(source),
            "model_type": [model_type] * len(source),
            **{column: _optional_column(source, column) for column in _OPTIONAL_PREDICTION_COLUMNS},
            "away_team": source["away_team"],
            "home_team": source["home_team"],
        }
    )

    return validate_forecast_events(events)

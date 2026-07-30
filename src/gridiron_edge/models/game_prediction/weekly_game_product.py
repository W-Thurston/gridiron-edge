# src/gridiron_edge/models/game_prediction/weekly_game_product.py

"""Projected-score composition for the complete weekly game product."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.models.game_prediction.post_process import projected_scores
from gridiron_edge.models.game_prediction.weekly_spread_product import WeeklySpreadStatus
from gridiron_edge.models.game_prediction.weekly_total_product import WeeklyTotalStatus


class ProjectedScoreStatus(StrEnum):
    """Availability state for one game's projected scores."""

    AVAILABLE = "available"
    SPREAD_UNAVAILABLE = "spread_unavailable"
    TOTAL_UNAVAILABLE = "total_unavailable"
    SPREAD_AND_TOTAL_UNAVAILABLE = "spread_and_total_unavailable"


_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "spread_status",
    "model_spread",
    "total_status",
    "model_total",
)

_SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "projected_score_status",
    "projected_home_score",
    "projected_away_score",
)

_TOTAL_POINT_ESTIMATE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        WeeklyTotalStatus.AVAILABLE.value,
        WeeklyTotalStatus.UNCERTAINTY_UNAVAILABLE.value,
    }
)


def _require_columns(product: DataFrame) -> None:
    """Require spread and total component columns."""
    missing = sorted(set(_REQUIRED_COLUMNS) - set(product.columns))
    if missing:
        raise ValueError("Weekly product is missing required columns: " + ", ".join(missing))


def _score_status(row: Series) -> ProjectedScoreStatus:
    """Return projected-score availability from component statuses."""
    spread_available = str(row["spread_status"]) == WeeklySpreadStatus.AVAILABLE.value
    total_available = str(row["total_status"]) in _TOTAL_POINT_ESTIMATE_STATUSES

    if spread_available and total_available:
        return ProjectedScoreStatus.AVAILABLE
    if not spread_available and not total_available:
        return ProjectedScoreStatus.SPREAD_AND_TOTAL_UNAVAILABLE
    if not spread_available:
        return ProjectedScoreStatus.SPREAD_UNAVAILABLE
    return ProjectedScoreStatus.TOTAL_UNAVAILABLE


def _project_row(row: Series) -> dict[str, object]:
    """Derive projected scores when both point estimates are available."""
    status = _score_status(row)
    if status is not ProjectedScoreStatus.AVAILABLE:
        return {
            "projected_score_status": status.value,
            "projected_home_score": pd.NA,
            "projected_away_score": pd.NA,
        }

    spread_value = row["model_spread"]
    total_value = row["model_total"]
    if pd.isna(spread_value) or pd.isna(total_value):
        raise ValueError("Available projected-score inputs require model_spread and model_total.")

    home_score, away_score = projected_scores(
        float(spread_value),
        float(total_value),
    )
    return {
        "projected_score_status": status.value,
        "projected_home_score": home_score,
        "projected_away_score": away_score,
    }


def attach_projected_scores(weekly_product: DataFrame) -> DataFrame:
    """Attach projected scores or a granular blocker to every product row."""
    _require_columns(weekly_product)
    source = weekly_product.copy()
    values = [_project_row(row) for _, row in source.iterrows()]
    score_frame = DataFrame(values, index=source.index)
    return pd.concat([source, score_frame], axis=1)


def build_weekly_game_product(weekly_product: DataFrame) -> DataFrame:
    """Attach projected scores and return a validated complete game product."""
    from gridiron_edge.models.game_prediction.product_validation import (
        validate_weekly_game_product,
    )

    return validate_weekly_game_product(attach_projected_scores(weekly_product))

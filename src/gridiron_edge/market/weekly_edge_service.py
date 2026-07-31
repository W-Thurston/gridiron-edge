# src/gridiron_edge/market/weekly_edge_service.py

"""Domain service for persisted weekly products and current market snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.loaders import load_current_weekly_product
from gridiron_edge.ingest.odds.store import load_current_odds
from gridiron_edge.market.recommendations import EdgeResult, build_edge_result
from gridiron_edge.models.game_prediction.weekly_spread_product import (
    WeeklySpreadStatus,
)
from gridiron_edge.models.game_prediction.weekly_total_product import (
    WeeklyTotalStatus,
)
from gridiron_edge.models.game_prediction.weekly_win_product import WeeklyWinStatus

_EMPTY_PREDICTION_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
)
_EMPTY_MARKET_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "market",
    "side",
    "odds",
    "line",
)


def _empty_predictions() -> DataFrame:
    """Return the minimum empty prediction contract for diagnostics."""
    return DataFrame(columns=list(_EMPTY_PREDICTION_COLUMNS))


def _empty_markets() -> DataFrame:
    """Return the minimum empty market contract for diagnostics."""
    return DataFrame(columns=list(_EMPTY_MARKET_COLUMNS))


def _load_selected_product(
    *,
    repo: Path,
    season: str,
    week: int,
) -> DataFrame:
    """Load the explicit current product or return canonical absence."""
    try:
        return load_current_weekly_product(
            repo,
            season=season,
            week=week,
        )
    except FileNotFoundError as exc:
        if "No current weekly product selected" not in str(exc):
            raise
        return _empty_predictions()


def _finite_unique_uncertainty(
    product: DataFrame,
    *,
    status_column: str,
    available_status: str,
    uncertainty_column: str,
    label: str,
) -> float | None:
    """Return one persisted uncertainty or reject mixed available values."""
    if product.empty:
        return None
    available = product.loc[
        product[status_column].astype(str) == available_status,
        uncertainty_column,
    ]
    values = {
        float(value)
        # pyrefly: ignore [missing-attribute]
        for value in pd.to_numeric(available, errors="coerce").dropna()
        if math.isfinite(float(value)) and float(value) > 0.0
    }
    if len(values) > 1:
        rendered = ", ".join(str(value) for value in sorted(values))
        raise ValueError(f"Current weekly product contains mixed {label} values: {rendered}.")
    return next(iter(values), None)


def _adapt_weekly_product(product: DataFrame) -> DataFrame:
    """Adapt persisted component statuses to recommendation inputs."""
    if product.empty:
        return _empty_predictions()

    adapted = product.copy()
    win_available = adapted["win_status"].astype(str) == WeeklyWinStatus.AVAILABLE.value
    spread_available = adapted["spread_status"].astype(str) == WeeklySpreadStatus.AVAILABLE.value
    total_available = adapted["total_status"].astype(str) == WeeklyTotalStatus.AVAILABLE.value

    adapted.loc[~win_available, ["away_win_prob", "home_win_prob"]] = pd.NA
    adapted.loc[~spread_available, "model_spread"] = pd.NA
    adapted.loc[~total_available, "model_total"] = pd.NA
    adapted["model_name"] = adapted["win_model_name"]
    adapted["model_type"] = adapted["win_model_type"]
    return adapted


def build_weekly_edge_result(
    *,
    season: str,
    week: int,
    bankroll: float | None = None,
    kelly_multiplier: float = 0.25,
    min_ev: float = 0.0,
    repo: Path | None = None,
    as_of: datetime | None = None,
    max_market_age: timedelta | None = None,
) -> EdgeResult:
    """Build one edge result from the selected product and market snapshot."""
    root = repo or get_settings().repo_root
    product = _load_selected_product(
        repo=root,
        season=season,
        week=week,
    )
    markets = load_current_odds(repo=root)
    market_input = _empty_markets() if markets is None else markets

    margin_std = _finite_unique_uncertainty(
        product,
        status_column="spread_status",
        available_status=WeeklySpreadStatus.AVAILABLE.value,
        uncertainty_column="spread_uncertainty",
        label="spread uncertainty",
    )
    total_std = _finite_unique_uncertainty(
        product,
        status_column="total_status",
        available_status=WeeklyTotalStatus.AVAILABLE.value,
        uncertainty_column="total_uncertainty",
        label="total uncertainty",
    )

    return build_edge_result(
        _adapt_weekly_product(product),
        market_input,
        season=season,
        week=week,
        margin_std=margin_std,
        total_std=total_std,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        min_ev=min_ev,
        as_of=as_of,
        max_market_age=max_market_age,
    )

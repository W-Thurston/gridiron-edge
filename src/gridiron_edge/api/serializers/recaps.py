# src/gridiron_edge/api/serializers/recaps.py

"""Serialize persisted weekly recaps (``DECISIONS.md`` D86) into the /recaps responses.

Nothing is scored or recomputed here: the week metrics come from the manifest and every game
value from the stored game frame. Missing frame values (NaN, NaT, NA) become null beside the
row's own status.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import asdict
from datetime import datetime
from typing import Any, Final, cast

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.meta import ResponseMeta
from gridiron_edge.api.schemas.recaps import (
    RecapClosingMarket,
    RecapConventionsResponse,
    RecapCountsResponse,
    RecapFamilyAvailability,
    RecapGame,
    RecapGameForecast,
    RecapGameResult,
    RecapMarketAtForecast,
    RecapPointForecast,
    RecapProjectedScoreForecast,
    RecapWeekMetrics,
    RecapWinForecast,
    WeeklyRecapList,
    WeeklyRecapResponse,
    WeeklyRecapSummary,
)
from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.weekly_recap import (
    Closer,
    ClosingLineStatus,
    ForecastMarketStatus,
    ForecastStatus,
    ProductRowStatus,
    WeeklyRecap,
)
from gridiron_edge.evaluation.weekly_recap_store import WeeklyRecapSelection

# Sections the recap does not carry yet; null and marked pending (D14).
PENDING_RECAP_SECTIONS: Final[tuple[str, ...]] = (
    "starting_qbs",
    "edges",
    "wagers",
    "season_to_date",
)


def serialize_weekly_recap_list(
    recaps: list[tuple[WeeklyRecapSelection, WeeklyRecap, DataFrame]],
) -> WeeklyRecapList:
    """Return one summary per selected recap, in the order given."""
    items = [_summary(selection, recap, games) for selection, recap, games in recaps]
    return WeeklyRecapList(items=items)


def serialize_weekly_recap(
    selection: WeeklyRecapSelection,
    recap: WeeklyRecap,
    games: DataFrame,
) -> WeeklyRecapResponse:
    """Return the selected recap of one week with every game row."""
    meta = ResponseMeta()
    for section in PENDING_RECAP_SECTIONS:
        meta = meta.with_pending(section)
    return WeeklyRecapResponse(
        season=recap.season,
        week=recap.week,
        recap_id=recap.recap_id,
        built_at=recap.built_at,
        selected_at=selection.selected_at,
        product_id=recap.product_id,
        product_run_id=recap.product_run_id,
        carried_run_ids=list(recap.carried_run_ids),
        product_schema_version=recap.product_schema_version,
        forecast_role=_role(recap.forecast_role),
        conventions=RecapConventionsResponse(**asdict(recap.conventions)),
        counts=RecapCountsResponse(**asdict(recap.counts)),
        metrics=RecapWeekMetrics.model_validate(asdict(recap.metrics)),
        games=[_game(row) for row in games.to_dict("records")],
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )


def _summary(
    selection: WeeklyRecapSelection,
    recap: WeeklyRecap,
    games: DataFrame,
) -> WeeklyRecapSummary:
    available = ForecastStatus.AVAILABLE.value
    roles = sorted({str(role) for role in games["forecast_role"].dropna()})
    return WeeklyRecapSummary(
        season=recap.season,
        week=recap.week,
        recap_id=recap.recap_id,
        built_at=recap.built_at,
        selected_at=selection.selected_at,
        product_id=recap.product_id,
        forecast_role=_role(recap.forecast_role),
        row_forecast_roles=[ForecastRole(role) for role in roles],
        scheduled_games=recap.counts.scheduled_games,
        in_product=recap.counts.in_product,
        available=RecapFamilyAvailability(
            win=int(games["win_status"].eq(available).sum()),
            spread=int(games["spread_status"].eq(available).sum()),
            total=int(games["total_status"].eq(available).sum()),
            projected_score=int(games["projected_score_status"].eq(available).sum()),
        ),
    )


def _game(row: Mapping[Hashable, Any]) -> RecapGame:
    return RecapGame(
        game_id=str(row["game_id"]),
        away_team=str(row["away_team"]),
        home_team=str(row["home_team"]),
        kickoff=cast(datetime, _timestamp(row["kickoff"])),
        product_row_status=ProductRowStatus(row["product_row_status"]),
        forecast=RecapGameForecast(
            role=_role(_text(row["forecast_role"])),
            run_id=_text(row["forecast_run_id"]),
            generated_at=_timestamp(row["forecast_generated_at"]),
            carried=_flag(row["carried"]),
        ),
        result=RecapGameResult(
            away_score=float(row["away_score"]),
            home_score=float(row["home_score"]),
            home_margin=float(row["actual_margin"]),
            total=float(row["actual_total"]),
            home_win=_flag(row["actual_home_win"]),
        ),
        market_at_forecast=RecapMarketAtForecast(
            status=ForecastMarketStatus(row["forecast_market_status"]),
            home_margin=_number(row["forecast_home_margin_line"]),
            total=_number(row["forecast_total_line"]),
            ingested_at=_timestamp(row["forecast_market_ingested_at"]),
        ),
        closing_market=RecapClosingMarket(
            spread_status=ClosingLineStatus(row["closing_spread_status"]),
            home_margin=_number(row["closing_home_margin"]),
            home_win_prob=_number(row["closing_home_win_prob"]),
            total_status=ClosingLineStatus(row["closing_total_status"]),
            total=_number(row["closing_total"]),
            implied_away_score=_number(row["market_implied_away_score"]),
            implied_home_score=_number(row["market_implied_home_score"]),
        ),
        win=RecapWinForecast(
            status=ForecastStatus(row["win_status"]),
            home_win_prob=_number(row["home_win_prob"]),
            model_brier=_number(row["win_model_brier"]),
            market_brier=_number(row["win_market_brier"]),
            closer=_closer(row["win_closer"]),
        ),
        spread=RecapPointForecast(
            status=ForecastStatus(row["spread_status"]),
            value=_number(row["projected_home_margin"]),
            model_error=_number(row["spread_model_error"]),
            market_error=_number(row["spread_market_error"]),
            closer=_closer(row["spread_closer"]),
        ),
        total=RecapPointForecast(
            status=ForecastStatus(row["total_status"]),
            value=_number(row["model_total"]),
            model_error=_number(row["total_model_error"]),
            market_error=_number(row["total_market_error"]),
            closer=_closer(row["total_closer"]),
        ),
        projected_score=RecapProjectedScoreForecast(
            status=ForecastStatus(row["projected_score_status"]),
            away_score=_number(row["projected_away_score"]),
            home_score=_number(row["projected_home_score"]),
            model_abs_error=_number(row["projected_score_model_abs_error"]),
            market_abs_error=_number(row["projected_score_market_abs_error"]),
            closer=_closer(row["projected_score_closer"]),
        ),
    )


def _missing(value: object) -> bool:
    return value is None or bool(pd.isna(cast(Any, value)))


def _number(value: object) -> float | None:
    return None if _missing(value) else float(cast(float, value))


def _text(value: object) -> str | None:
    return None if _missing(value) else str(value)


def _flag(value: object) -> bool | None:
    return None if _missing(value) else bool(value)


def _timestamp(value: object) -> datetime | None:
    return None if _missing(value) else pd.Timestamp(cast(Any, value)).to_pydatetime()


def _role(value: str | None) -> ForecastRole | None:
    return None if value is None else ForecastRole(value)


def _closer(value: object) -> Closer | None:
    text = _text(value)
    return None if text is None else Closer(text)

"""Serialize classified current quotes for the Line Shopping API."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from datetime import date, datetime
import math
from numbers import Real
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.meta import ResponseMeta
from gridiron_edge.api.schemas.lines import (
    GuidanceStatus,
    LineOffer,
    LineOutcomeGuidance,
    LineShoppingGame,
    LineShoppingList,
    MarketName,
    MarketSide,
)


def _is_missing_scalar(value: object) -> bool:
    """Return whether one normalized scalar represents a missing value."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, Real) and math.isnan(float(value))


def _datetime_or_none(value: object) -> datetime | None:
    """Return one optional datetime-compatible scalar as a datetime."""
    if _is_missing_scalar(value):
        return None
    supported = str | int | float | date | datetime | np.datetime64
    if not isinstance(value, supported):
        raise ValueError("Line-shopping timestamp must be datetime-compatible")
    return pd.Timestamp(value).to_pydatetime()


def _required_datetime(value: object) -> datetime:
    timestamp = _datetime_or_none(value)
    if timestamp is None:
        raise ValueError("Line-shopping fetched_at cannot be null")
    return timestamp


def _text_or_none(value: object) -> str | None:
    if _is_missing_scalar(value):
        return None
    return str(value)


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Line-shopping {label} must be numeric")
    return float(value)


def _optional_number(value: object) -> float | None:
    return None if _is_missing_scalar(value) else _number(value, label="model value")


def _optional_bool(value: object) -> bool | None:
    if _is_missing_scalar(value):
        return None
    if not isinstance(value, bool | np.bool_):
        raise ValueError("Line-shopping model approval must be boolean")
    return bool(value)


def _serialize_offer(row: Mapping[Hashable, object]) -> LineOffer:
    return LineOffer(
        provider=str(row["provider"]),
        provider_event_id=_text_or_none(row.get("provider_event_id")),
        sportsbook=str(row["sportsbook"]),
        sportsbook_updated_at=_datetime_or_none(row.get("sportsbook_updated_at")),
        market_fetched_at=_required_datetime(row["fetched_at"]),
        commence_time=_datetime_or_none(row.get("commence_time")),
        is_live=bool(row["is_live"]),
        market=cast(MarketName, str(row["market"])),
        side=cast(MarketSide, str(row["side"])),
        line=(None if _is_missing_scalar(row.get("line")) else _number(row["line"], label="line")),
        american_odds=int(_number(row["odds"], label="odds")),
        is_best_line=bool(row["is_best_line"]),
        is_best_price=bool(row["is_best_price"]),
        model_status=cast(
            "GuidanceStatus",
            str(row.get("model_status", "model_unavailable")),
        ),
        model_value=_optional_number(row.get("model_value")),
        model_probability=_optional_number(row.get("model_probability")),
        expected_value=_optional_number(row.get("expected_value")),
        is_model_approved=_optional_bool(row.get("is_model_approved")),
        is_best_model_approved_offer=bool(row.get("is_best_model_approved_offer", False)),
        product_id=_text_or_none(row.get("product_id")),
        product_run_id=_text_or_none(row.get("product_run_id")),
    )


def _serialize_guidance(
    guidance: DataFrame | None,
    *,
    game_id: str,
) -> list[LineOutcomeGuidance]:
    if guidance is None or guidance.empty:
        return []
    scoped = guidance.loc[guidance["game_id"].astype(str) == game_id, :]
    return [
        LineOutcomeGuidance(
            side=cast(MarketSide, str(row["side"])),
            model_status=cast(GuidanceStatus, str(row["model_status"])),
            model_value=_optional_number(row.get("model_value")),
            playable_line=_optional_number(row.get("playable_line")),
            reference_odds=(
                None
                if _is_missing_scalar(row.get("reference_odds"))
                else int(_number(row["reference_odds"], label="reference odds"))
            ),
            fair_american_odds=(
                None
                if _is_missing_scalar(row.get("fair_american_odds"))
                else int(_number(row["fair_american_odds"], label="fair odds"))
            ),
            product_id=_text_or_none(row.get("product_id")),
            product_run_id=_text_or_none(row.get("product_run_id")),
        )
        for row in scoped.to_dict(orient="records")
    ]


def _line_shopping_game_sort_key(
    game: LineShoppingGame,
) -> tuple[int, datetime, str, str, str, str]:
    fallback = datetime.max
    return (
        1 if game.commence_time is None else 0,
        game.commence_time or fallback,
        game.game_date,
        game.away_team.casefold(),
        game.home_team.casefold(),
        game.game_id,
    )


def serialize_line_shopping_list(
    rows: DataFrame,
    *,
    season: str,
    week: int,
    market: MarketName | None,
    sportsbooks: tuple[str, ...] | None = None,
    guidance: DataFrame | None = None,
    response_meta: ResponseMeta | None = None,
) -> LineShoppingList:
    """Group classified offers by game without fabricating missing books."""
    games: list[LineShoppingGame] = []
    for game_id, group in rows.groupby("game_id", sort=True):
        first = group.iloc[0]
        commence_values = group["commence_time"].dropna()
        commence_time = (
            _datetime_or_none(commence_values.iloc[0]) if not commence_values.empty else None
        )
        offers = [_serialize_offer(record) for record in group.to_dict(orient="records")]
        game_guidance = _serialize_guidance(
            guidance,
            game_id=str(game_id),
        )
        games.append(
            LineShoppingGame(
                game_id=str(game_id),
                season=str(first["season"]),
                week=int(first["week"]),
                game_date=str(first["game_date"]),
                away_team=str(first["away_team"]),
                home_team=str(first["home_team"]),
                commence_time=commence_time,
                offers=offers,
                guidance=game_guidance,
            )
        )

    games.sort(key=_line_shopping_game_sort_key)

    response_sportsbooks = (
        sportsbooks
        if sportsbooks is not None
        else tuple(sorted(rows["sportsbook"].astype(str).unique()))
        if not rows.empty
        else ()
    )
    timestamps = (
        tuple(
            pd.Timestamp(value).to_pydatetime()
            for value in sorted(rows["fetched_at"].dropna().unique())
        )
        if not rows.empty
        else ()
    )
    return LineShoppingList(
        season=season,
        week=week,
        market=market,
        items=games,
        total=len(games),
        sportsbooks=response_sportsbooks,
        market_fetched_at=timestamps,
        # pyrefly: ignore [unexpected-keyword]
        response_meta=response_meta,
    )

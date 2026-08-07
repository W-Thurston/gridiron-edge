"""Routes for current multi-book line shopping."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_games_for_week,
    resolve_current_season_week,
)
from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.lines import LineShoppingList, MarketName
from gridiron_edge.api.serializers.lines import serialize_line_shopping_list
from gridiron_edge.ingest.odds.store import empty_quote_frame, load_current_odds
from gridiron_edge.market.line_shopping import (
    classify_line_shopping_offers,
    evaluate_line_shopping_guidance,
)

router = APIRouter(prefix="/lines", tags=["lines"])


@router.get(
    "",
    response_model=LineShoppingList,
    summary="Compare current sportsbook offers across the selected slate.",
)
def list_lines(
    settings: SettingsDep,
    season: Annotated[str | None, Query()] = None,
    week: Annotated[int | None, Query(ge=1)] = None,
    market: Annotated[MarketName | None, Query()] = None,
) -> LineShoppingList:
    """Return exact current quotes with best-line and exact-line price flags."""
    if season is None or week is None:
        current_season, current_week = resolve_current_season_week(settings)
    else:
        current_season, current_week = "", 0
    resolved_season = season or current_season
    resolved_week = week or current_week

    snapshot = load_current_odds(repo=settings.repo_root)
    if snapshot is None:
        meta = ResponseMeta().with_blocked(
            "items",
            *Unavailable.NO_ODDS_AVAILABLE,
        )
        return serialize_line_shopping_list(
            empty_quote_frame(),
            season=resolved_season,
            week=resolved_week,
            market=market,
            response_meta=meta,
        )

    scoped = snapshot.loc[
        (snapshot["season"] == resolved_season) & (snapshot["week"] == resolved_week),
        :,
    ].copy()
    sportsbooks = tuple(sorted(scoped["sportsbook"].astype(str).unique()))
    if market is not None:
        scoped = scoped.loc[scoped["market"] == market, :].copy()

    try:
        product = load_games_for_week(
            settings,
            season=resolved_season,
            week=resolved_week,
        )
    except FileNotFoundError:
        product = None

    if product is None or scoped.empty:
        classified = classify_line_shopping_offers(scoped)
        guidance = None
    else:
        evaluated = evaluate_line_shopping_guidance(product, scoped)
        classified = evaluated.offers
        guidance = evaluated.guidance

    return serialize_line_shopping_list(
        classified,
        season=resolved_season,
        week=resolved_week,
        market=market,
        sportsbooks=sportsbooks,
        guidance=guidance,
    )

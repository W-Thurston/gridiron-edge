# src/gridiron_edge/api/routes/edges.py

"""Edges endpoint backed by the selected persisted weekly product."""

from __future__ import annotations

from fastapi import APIRouter, Query

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_edges_for_week,
    resolve_current_season_week,
)
from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.edges import EdgeList
from gridiron_edge.api.serializers.edges import serialize_edges_list
from gridiron_edge.market.edge_diagnostics import EdgeDiagnosticBlocker

router = APIRouter(prefix="/edges", tags=["edges"])


def _resolve_scope(
    settings: SettingsDep,
    season: str | None,
    week: int | None,
) -> tuple[str, int]:
    """Return explicit scope or resolve only the missing values."""
    if season is not None and week is not None:
        return season, week

    resolved_season, resolved_week = resolve_current_season_week(settings)
    return (
        season if season is not None else resolved_season,
        week if week is not None else resolved_week,
    )


@router.get("", response_model=EdgeList)
def list_edges(
    settings: SettingsDep,
    *,
    season: str | None = Query(
        default=None,
        description="Season, e.g. '2026-2027'. Defaults to current.",
    ),
    week: int | None = Query(
        default=None,
        description="Week number. Defaults to current.",
    ),
    min_ev: float = Query(
        default=0.0,
        ge=0.0,
        description="Minimum EV threshold. Rows with ev <= min_ev excluded.",
    ),
    bankroll: float | None = Query(
        default=None,
        ge=0.0,
        description=(
            "Bankroll for Kelly stake sizing. When omitted, "
            "kelly_stake is unavailable while kelly_frac remains populated."
        ),
    ),
    kelly_multiplier: float = Query(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=("Fraction of full Kelly, constrained to [0, 1] (e.g. 0.25 = quarter-Kelly)."),
    ),
) -> EdgeList:
    """Return ranked edges from the selected persisted weekly product."""
    resolved_season, resolved_week = _resolve_scope(settings, season, week)
    result = load_edges_for_week(
        settings,
        season=resolved_season,
        week=resolved_week,
        min_ev=min_ev,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
    )

    unavailable = {
        EdgeDiagnosticBlocker.NO_PREDICTIONS: Unavailable.NO_WEEKLY_PRODUCT,
        EdgeDiagnosticBlocker.NO_MARKET_DATA: Unavailable.NO_ODDS_AVAILABLE,
        EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE: (Unavailable.MARKET_SCOPE_MISMATCH),
        EdgeDiagnosticBlocker.MARKET_STALE: Unavailable.STALE_MARKET_DATA,
        EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES: (Unavailable.ZERO_EDGE_GAME_MATCHES),
        EdgeDiagnosticBlocker.INCOMPLETE_MARKETS: (Unavailable.INCOMPLETE_MARKET_DATA),
    }
    meta: ResponseMeta | None = None
    if result.diagnostics.blockers:
        first_blocker = result.diagnostics.blockers[0]
        meta = ResponseMeta().with_blocked(
            "items",
            *unavailable[first_blocker],
        )

    return serialize_edges_list(
        result.rows,
        season=resolved_season,
        week=resolved_week,
        min_ev=min_ev,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        response_meta=meta,
    )

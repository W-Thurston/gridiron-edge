# src/gridiron_edge/api/routes/edges.py

"""Edges endpoint — ranked market edges for the current champion."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pandas import DataFrame

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.exceptions import OddsUnavailableError
from gridiron_edge.api.loaders import (
    load_edges_for_week,
    resolve_current_season_week,
)
from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.edges import EdgeList
from gridiron_edge.api.serializers.edges import serialize_edges_list
from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError

router = APIRouter(prefix="/edges", tags=["edges"])


def _resolve_scope(
    settings: SettingsDep,
    season: str | None,
    week: int | None,
) -> tuple[str, int]:
    """Return (season, week), defaulting to current when not provided."""
    resolved_season, resolved_week = resolve_current_season_week(settings)
    return (season or resolved_season, week or resolved_week)


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
    """Return ranked edges for (season, week) using the champion model.

    - Missing champion manifest: 200 with empty list,
      ``_meta.field_status["items"]`` marked NO_CHAMPION_MANIFEST.
    - Missing odds snapshot: 200 with empty list,
      ``_meta.field_status["items"]`` marked NO_ODDS_AVAILABLE.
    - No predictions or no positive-EV edges: 200 with empty list, no
      field_status (legitimate empty state).
    """
    resolved_season, resolved_week = _resolve_scope(settings, season, week)

    try:
        rows: DataFrame = load_edges_for_week(
            settings,
            season=resolved_season,
            week=resolved_week,
            min_ev=min_ev,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
        )
    except ChampionNotFoundError:
        meta: ResponseMeta = ResponseMeta().with_blocked(
            "items",
            *Unavailable.NO_CHAMPION_MANIFEST,
        )
        return EdgeList(
            season=resolved_season,
            week=resolved_week,
            min_ev=min_ev,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore [unexpected-keyword]
        )
    except OddsUnavailableError:
        meta = ResponseMeta().with_blocked(
            "items",
            *Unavailable.NO_ODDS_AVAILABLE,
        )
        return EdgeList(
            season=resolved_season,
            week=resolved_week,
            min_ev=min_ev,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore [unexpected-keyword]
        )

    return serialize_edges_list(
        rows,
        season=resolved_season,
        week=resolved_week,
        min_ev=min_ev,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
    )

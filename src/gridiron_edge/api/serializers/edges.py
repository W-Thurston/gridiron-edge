"""Mechanical serializers for unified weekly edge results."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from gridiron_edge.api.meta import ResponseMeta
from gridiron_edge.api.schemas.edges import (
    EdgeDiagnosticsResponse,
    EdgeList,
    EdgeProvenanceResponse,
    EdgeRow,
)
from gridiron_edge.market.edge import EdgeStrength
from gridiron_edge.market.recommendations import EdgeResult


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; otherwise preserve the value."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _row_to_edge(row: dict) -> EdgeRow:
    """Convert one service recommendation row to its API schema."""
    return EdgeRow(
        game_id=str(row["game_id"]),
        game_date=_none_if_nan(row.get("game_date")),
        season=_none_if_nan(row.get("season")),
        week=_none_if_nan(row.get("week")),
        away_team=str(row["away_team"]),
        home_team=str(row["home_team"]),
        model_key=str(row["model_key"]),
        confidence_tier=_none_if_nan(row.get("confidence_tier")),
        market_type=str(row["market_type"]),
        side=str(row["side"]),
        model_value=_none_if_nan(row.get("model_value")),
        market_value=_none_if_nan(row.get("market_value")),
        american_odds=int(row["american_odds"]),
        point_edge=_none_if_nan(row.get("point_edge")),
        cover_prob=_none_if_nan(row.get("cover_prob")),
        ev=float(row["ev"]),
        edge_strength=cast(EdgeStrength, str(row["edge_strength"])),
        kelly_frac=_none_if_nan(row.get("kelly_frac")),
        kelly_stake=_none_if_nan(row.get("kelly_stake")),
    )


def _serialize_diagnostics(result: EdgeResult) -> EdgeDiagnosticsResponse:
    """Serialize service diagnostics without deriving or collapsing values."""
    diagnostics = result.diagnostics
    provenance = diagnostics.provenance
    return EdgeDiagnosticsResponse(
        season=diagnostics.season,
        week=diagnostics.week,
        prediction_game_count=diagnostics.prediction_game_count,
        market_game_count=diagnostics.market_game_count,
        matched_game_count=diagnostics.matched_game_count,
        complete_moneyline_count=diagnostics.complete_moneyline_count,
        complete_spread_count=diagnostics.complete_spread_count,
        complete_total_count=diagnostics.complete_total_count,
        eligible_market_count=diagnostics.eligible_market_count,
        calculated_edge_count=diagnostics.calculated_edge_count,
        positive_edge_count=diagnostics.positive_edge_count,
        filtered_edge_count=diagnostics.filtered_edge_count,
        state=diagnostics.state,
        blockers=diagnostics.blockers,
        provenance=EdgeProvenanceResponse(
            win_event_ids=provenance.win_event_ids,
            win_run_ids=provenance.win_run_ids,
            win_model_names=provenance.win_model_names,
            win_model_types=provenance.win_model_types,
            total_event_ids=provenance.total_event_ids,
            total_run_ids=provenance.total_run_ids,
            total_model_names=provenance.total_model_names,
            total_model_types=provenance.total_model_types,
            product_ids=provenance.product_ids,
            product_run_ids=provenance.product_run_ids,
            market_providers=provenance.market_providers,
            market_sportsbooks=provenance.market_sportsbooks,
            market_fetched_at=provenance.market_fetched_at,
        ),
    )


def serialize_edges_list(
    result: EdgeResult,
    *,
    min_ev: float | None,
    bankroll: float | None,
    kelly_multiplier: float | None,
    response_meta: ResponseMeta | None = None,
) -> EdgeList:
    """Serialize one complete unified weekly edge result."""
    items = [_row_to_edge(row.to_dict()) for _, row in result.rows.iterrows()]
    diagnostics = _serialize_diagnostics(result)
    return EdgeList(
        items=items,
        total=len(items),
        season=diagnostics.season,
        week=diagnostics.week,
        min_ev=min_ev,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        diagnostics=diagnostics,
        response_meta=response_meta,  # pyrefly: ignore [unexpected-keyword]
    )

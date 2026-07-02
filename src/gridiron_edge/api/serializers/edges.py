# src/gridiron_edge/api/serializers/edges.py

"""Serializers for /edges.

Per D17, hand-written. Per D18, owns _meta.field_status construction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.schemas.edges import EdgeList, EdgeRow


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; else the value."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _row_to_edge(row: dict) -> EdgeRow:
    """Convert one edge-report row (dict) to EdgeRow.

    Normalizes NaN → None for optional numeric fields. Required fields
    (``game_id``, ``away_team``, ``home_team``, ``model_key``,
    ``market_type``, ``side``, ``ev``, ``edge_strength``) are passed
    through as-is.
    """
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
        point_edge=_none_if_nan(row.get("point_edge")),
        cover_prob=_none_if_nan(row.get("cover_prob")),
        ev=float(row["ev"]),
        edge_strength=str(row["edge_strength"]),
        kelly_frac=_none_if_nan(row.get("kelly_frac")),
        kelly_stake=_none_if_nan(row.get("kelly_stake")),
    )


def serialize_edges_list(
    rows: DataFrame,
    *,
    season: str | None,
    week: int | None,
    min_ev: float | None,
) -> EdgeList:
    """Build the /edges list response from a loader DataFrame.

    Empty rows return an empty items list. The route layer handles
    ChampionNotFoundError / OddsUnavailableError translation to
    field_status; this serializer covers only the happy-path shape.
    """
    items = [_row_to_edge(r.to_dict()) for _, r in rows.iterrows()]
    return EdgeList(
        items=items,
        total=len(items),
        season=season,
        week=week,
        min_ev=min_ev,
    )

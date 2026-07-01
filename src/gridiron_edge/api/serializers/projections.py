# src/gridiron_edge/api/serializers/projections.py

"""Serializer for /projections.

Per D17, hand-written. Per D18, owns _meta construction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.projections import (
    ProjectionsList,
    TeamProjectionRow,
)


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; else the value."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def serialize_projections(
    df: DataFrame,
    long_to_short: dict[str, str],
    season: str,
    computed_at: str | None,
) -> ProjectionsList:
    """Build the /projections response from the projections summary CSV.

    Maps CSV columns to schema fields:
        TEAM → abbr (already short)
        AVG_WINS → avg_wins
        P_MAKE_PLAYOFFS → make_playoffs
        P_REACH_DIV → reach_div
        P_REACH_CONF → reach_conf
        P_REACH_SB → reach_sb
        P_WIN_SB → win_sb
    """
    # Invert long_to_short for name resolution: {abbr → long_name}
    short_to_long = {v: k for k, v in long_to_short.items()}

    meta = ResponseMeta()

    if df.empty:
        # No projections CSV or empty file. Mark items as unavailable.
        meta = meta.with_blocked("items", *Unavailable.NO_PROJECTIONS_DATA)
        return ProjectionsList(
            season=season,
            computed_at=computed_at,
            n_simulations=None,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    # Sort by SB win probability descending.
    df_sorted = df.sort_values("P_WIN_SB", ascending=False).reset_index(drop=True)

    rows = [
        TeamProjectionRow(
            abbr=str(row["TEAM"]),
            name=short_to_long.get(str(row["TEAM"]), str(row["TEAM"])),
            avg_wins=_none_if_nan(row.get("AVG_WINS")),
            make_playoffs=_none_if_nan(row.get("P_MAKE_PLAYOFFS")),
            reach_div=_none_if_nan(row.get("P_REACH_DIV")),
            reach_conf=_none_if_nan(row.get("P_REACH_CONF")),
            reach_sb=_none_if_nan(row.get("P_REACH_SB")),
            win_sb=_none_if_nan(row.get("P_WIN_SB")),
        )
        for _, row in df_sorted.iterrows()
    ]

    # Mark pending / blocked fields on every row (n_simulations is response-level).
    meta = meta.with_pending("n_simulations")
    meta = meta.with_blocked(
        "items.week_over_week_delta",
        *Unavailable.NO_PRIOR_SNAPSHOT,
    )
    meta = meta.with_pending("items.clinched")
    meta = meta.with_pending("items.eliminated")

    return ProjectionsList(
        season=season,
        computed_at=computed_at,
        n_simulations=None,
        items=rows,
        total=len(rows),
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )

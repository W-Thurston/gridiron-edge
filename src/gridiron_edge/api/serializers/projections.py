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
    n_simulations: int | None,
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
        elo_delta → elo_delta  (Elo rating change from prior same-season week)
    """
    # Invert long_to_short for name resolution: {abbr → long_name}
    short_to_long: dict[str, str] = {v: k for k, v in long_to_short.items()}

    meta = ResponseMeta()

    if df.empty:
        # No projections CSV or empty file. Mark items as unavailable.
        meta: ResponseMeta = meta.with_blocked("items", *Unavailable.NO_PROJECTIONS_DATA)
        return ProjectionsList(
            season=season,
            computed_at=computed_at,
            n_simulations=n_simulations,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    # Sort by SB win probability descending.
    df_sorted: DataFrame = df.sort_values("P_WIN_SB", ascending=False).reset_index(drop=True)

    rows: list[TeamProjectionRow] = [
        TeamProjectionRow(
            abbr=str(row["TEAM"]),
            name=short_to_long.get(str(row["TEAM"]), str(row["TEAM"])),
            avg_wins=_none_if_nan(row.get("AVG_WINS")),
            make_playoffs=_none_if_nan(row.get("P_MAKE_PLAYOFFS")),
            reach_div=_none_if_nan(row.get("P_REACH_DIV")),
            reach_conf=_none_if_nan(row.get("P_REACH_CONF")),
            reach_sb=_none_if_nan(row.get("P_REACH_SB")),
            win_sb=_none_if_nan(row.get("P_WIN_SB")),
            elo_delta=_none_if_nan(row.get("elo_delta")),
        )
        for _, row in df_sorted.iterrows()
    ]

    # Clinched and eliminated remain pending for every projection row.
    meta = meta.with_pending("items.clinched")
    meta = meta.with_pending("items.eliminated")

    # Elo movement requires a prior week in the same season. Week 1 and
    # equivalent no-history states therefore have no prior snapshot.
    elo_delta = df_sorted.get("elo_delta")
    if elo_delta is None or not elo_delta.notna().any():
        meta = meta.with_blocked(
            "items.elo_delta",
            *Unavailable.NO_PRIOR_SNAPSHOT,
        )

    return ProjectionsList(
        season=season,
        computed_at=computed_at,
        n_simulations=n_simulations,
        items=rows,
        total=len(rows),
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )

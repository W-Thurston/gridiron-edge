# src/gridiron_edge/api/serializers/props.py

"""Serializers for /props and /props/{prop_id}.

Per D17, hand-written. Per D18, owns _meta.field_status construction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.meta import Blocker, ResponseMeta
from gridiron_edge.api.schemas.props import (
    LineBlock,
    ProjectionBlock,
    PropDetail,
    PropList,
    PropSummary,
)


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for NaN or None; else the value."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _season_int_to_str(season_int: str | float | None) -> str | None:
    """Convert an int season (e.g. 2026) to the API string ('2026-2027').

    Returns None for missing or unparseable values.
    """
    val = _none_if_nan(season_int)
    if val is None:
        return None
    try:
        i = int(val)
        return f"{i}-{i + 1}"
    except (ValueError, TypeError):
        return None


def _build_prop_id(row: dict) -> str:
    """Build the composite prop_id from the archive row."""
    return f"{row['game_id']}__{row['player_id']}__{row['stat_type']}"


def _build_model_key(row: dict) -> str:
    """Build the composite model_key from the archive row."""
    return f"{row['model_name']}_{row['model_type']}"


def _build_projection_block(row: dict) -> ProjectionBlock:
    """Extract projection fields from an archive row.

    Always returns a block (never None). Fields are individually
    nullable if the archive row has NaN.
    """
    return ProjectionBlock(
        predicted_mean=_none_if_nan(row.get("predicted_mean")),
        predicted_std=_none_if_nan(row.get("predicted_std")),
        lo_90=_none_if_nan(row.get("lo_90")),
        hi_90=_none_if_nan(row.get("hi_90")),
    )


def _build_line_block(row: dict) -> LineBlock:
    """Extract odds-derived fields from an archive row.

    Always returns a block. In T2 all fields are null; the block is
    still emitted for consistent shape. Populates when odds-join lands.
    """
    return LineBlock(
        line=_none_if_nan(row.get("line")),
        p_over=_none_if_nan(row.get("p_over")),
        lean=_none_if_nan(row.get("lean")),
        confidence_tier=_none_if_nan(row.get("confidence_tier")),
    )


def serialize_prop_summary(row: dict) -> PropSummary:
    """Convert one archive row (Series-as-dict) to PropSummary."""
    return PropSummary(
        prop_id=_build_prop_id(row),
        game_id=str(row["game_id"]),
        season=_season_int_to_str(row.get("season")),
        week=_none_if_nan(row.get("week")),
        player_id=str(row["player_id"]),
        player_name=str(row["player_name"]),
        position=str(row["position"]),
        team=str(row["team"]),
        stat_type=str(row["stat_type"]),
        model_key=_build_model_key(row),
        projection=_build_projection_block(row),
        line_context=_build_line_block(row),
    )


def serialize_props_list(
    rows: DataFrame,
    *,
    season: str | None,
    week: int | None,
    stat_type: str | None,
    position: str | None,
) -> PropList:
    """Build the /props list response.

    field_status marks the LineBlock's four fields as pending. Detail-
    only scaffolded fields (historical_vs_opponent, etc.) are not in
    the list response.
    """
    items = [serialize_prop_summary(r.to_dict()) for _, r in rows.iterrows()]

    meta = ResponseMeta()
    meta = meta.with_pending("items.line_context.line")
    meta = meta.with_pending("items.line_context.p_over")
    meta = meta.with_pending("items.line_context.lean")
    meta = meta.with_pending("items.line_context.confidence_tier")

    return PropList(
        items=items,
        total=len(items),
        season=season,
        week=week,
        stat_type=stat_type,
        position=position,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )


def serialize_prop_detail(
    row: dict,
    situational_splits: dict | None = None,
) -> PropDetail:
    """Build the /props/{prop_id} response.

    Populated fields come from the archive row. Pending/blocked fields
    are marked in _meta.field_status per D14.
    """
    projection: ProjectionBlock = _build_projection_block(row)
    line_context: LineBlock = _build_line_block(row)

    meta: ResponseMeta = ResponseMeta()
    # LineBlock fields — pending on odds-join at prediction time.
    meta = meta.with_pending("line_context.line")
    meta = meta.with_pending("line_context.p_over")
    meta = meta.with_pending("line_context.lean")
    meta = meta.with_pending("line_context.confidence_tier")
    # PropDetail-only scaffolded fields.
    meta = meta.with_pending("historical_vs_opponent")
    if situational_splits is None:
        # Artifact not yet computed for this stat_type. Mark pending.
        meta = meta.with_pending("situational_splits")
    # else: field is populated (possibly empty dict); no marker needed.
    meta = meta.with_pending("recent_form")
    meta = meta.with_blocked("prop_reasoning", *Blocker.FEATURE_ATTRIBUTION)
    meta = meta.with_blocked("injury_status", *Blocker.INJURY_DATA)
    meta = meta.with_blocked("multi_book_shopping", *Blocker.MULTI_BOOK)

    return PropDetail(
        prop_id=_build_prop_id(row),
        game_id=str(row["game_id"]),
        season=_season_int_to_str(row.get("season")),
        week=_none_if_nan(row.get("week")),
        player_id=str(row["player_id"]),
        player_name=str(row["player_name"]),
        position=str(row["position"]),
        team=str(row["team"]),
        stat_type=str(row["stat_type"]),
        model_key=_build_model_key(row),
        projection=projection,
        line_context=line_context,
        situational_splits=situational_splits,
        response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
    )

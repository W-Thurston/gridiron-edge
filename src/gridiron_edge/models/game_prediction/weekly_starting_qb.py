# src/gridiron_edge/models/game_prediction/weekly_starting_qb.py

"""The assumed starting quarterbacks recorded in each weekly product (``DECISIONS.md`` D77)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.features.team.qb_profile import (
    QB_SOURCE_CARRIED_FORWARD,
    QB_SOURCE_NFLVERSE_SCHEDULE,
    resolve_starting_qb_provenance,
)

PROJECTED_QB_SOURCE: Final[str] = "depth_chart_report"
QB_SOURCES: Final[frozenset[str]] = frozenset(
    {PROJECTED_QB_SOURCE, QB_SOURCE_NFLVERSE_SCHEDULE, QB_SOURCE_CARRIED_FORWARD}
)
_SIDES: Final[tuple[str, ...]] = ("away", "home")
_SIDE_FIELDS: Final[tuple[str, ...]] = (
    "qb_id",
    "qb_name",
    "qb_source",
    "qb_projection_as_of",
    "qb_nflverse_id",
    "qb_carried_forward_id",
    "qb_differs_from_carry_forward",
)
STATUS_COLUMN: Final[str] = "starting_qb_status"
STARTING_QB_COLUMNS: Final[tuple[str, ...]] = (
    STATUS_COLUMN,
    *(f"{side}_{field}" for side in _SIDES for field in _SIDE_FIELDS),
)


class StartingQbStatus(StrEnum):
    """Whether a game's assumed starters were recorded."""

    AVAILABLE = "available"
    SIDECAR_UNAVAILABLE = "sidecar_unavailable"
    STARTER_UNRESOLVED = "starter_unresolved"


def has_starting_qb_columns(product: DataFrame) -> bool:
    """Return whether a product carries the complete starting-QB record."""
    present = [column in product.columns for column in STARTING_QB_COLUMNS]
    if any(present) and not all(present):
        raise ValueError("Weekly product has a partial starting-QB record.")
    return all(present)


def _empty_record(index: pd.Index, status: StartingQbStatus) -> DataFrame:
    frame = DataFrame(index=index)
    frame[STATUS_COLUMN] = status.value
    for side in _SIDES:
        for field in _SIDE_FIELDS:
            dtype = "boolean" if field == "qb_differs_from_carry_forward" else "string"
            frame[f"{side}_{field}"] = Series(pd.NA, index=index, dtype=dtype)
    return frame


def attach_starting_qb_assumptions(
    product: DataFrame,
    sidecar: DataFrame | None,
    names: Mapping[str, str] | None = None,
) -> DataFrame:
    """Attach each side's assumed starter, its source and provenance to every game.

    The record comes from the same resolver that feeds the QB features, so it
    cannot differ from what the forecast used. A missing sidecar, a game absent
    from it, or a side with no resolvable starter is recorded as a status, never
    left blank.

    Args:
        product: Weekly product rows with ``game_id``.
        sidecar: The QB/coach sidecar the forecast run was bound to, or ``None``.
        names: Player id to display name; an id without a name records a null name.
    """
    if "game_id" not in product.columns:
        raise ValueError("Weekly product is missing required columns: game_id")
    if sidecar is None:
        return pd.concat(
            [product, _empty_record(product.index, StartingQbStatus.SIDECAR_UNAVAILABLE)], axis=1
        )

    provenance = resolve_starting_qb_provenance(sidecar)
    by_key: dict[tuple[str, str], dict[str, Any]] = {
        (str(item["GAME_ID"]), str(item["SIDE"])): {str(k): v for k, v in item.items()}
        for item in provenance.to_dict("records")
    }
    lookup = names or {}
    record = _empty_record(product.index, StartingQbStatus.AVAILABLE)
    for position, game_id in zip(product.index, product["game_id"].astype(str), strict=True):
        resolved_sides = 0
        for side in _SIDES:
            key = (game_id, side.upper())
            row = by_key.get(key)
            if row is None:
                continue
            qb_id = row["RESOLVED_QB_ID"]
            if pd.isna(qb_id):
                continue
            resolved_sides += 1
            carried = row["CARRIED_FORWARD_QB_ID"]
            record.at[position, f"{side}_qb_id"] = str(qb_id)
            record.at[position, f"{side}_qb_name"] = lookup.get(str(qb_id), pd.NA)
            record.at[position, f"{side}_qb_source"] = row["QB_SOURCE"]
            record.at[position, f"{side}_qb_projection_as_of"] = row["PROJECTION_AS_OF"]
            record.at[position, f"{side}_qb_nflverse_id"] = row["NFLVERSE_QB_ID"]
            record.at[position, f"{side}_qb_carried_forward_id"] = carried
            if not pd.isna(carried):
                record.at[position, f"{side}_qb_differs_from_carry_forward"] = bool(
                    str(qb_id) != str(carried)
                )
        if resolved_sides < len(_SIDES):
            record.at[position, STATUS_COLUMN] = StartingQbStatus.STARTER_UNRESOLVED.value
    return pd.concat([product, record], axis=1)


def validate_starting_qb_record(product: DataFrame) -> None:
    """Validate the starting-QB record when the product carries one."""
    if not has_starting_qb_columns(product):
        return
    for _, row in product.iterrows():
        game = f"Starting QB record for {row['game_id']}"
        try:
            status = StartingQbStatus(str(row[STATUS_COLUMN]))
        except ValueError as exc:
            raise ValueError(f"{game} has an unsupported status: {row[STATUS_COLUMN]!r}.") from exc
        for side in _SIDES:
            qb_id = row[f"{side}_qb_id"]
            source = row[f"{side}_qb_source"]
            as_of = row[f"{side}_qb_projection_as_of"]
            if pd.isna(qb_id):
                populated = [f for f in _SIDE_FIELDS if not pd.isna(row[f"{side}_{f}"])]
                if populated:
                    raise ValueError(f"{game} has a {side} record without a starter id.")
                if status is StartingQbStatus.AVAILABLE:
                    raise ValueError(f"{game} is available but has no {side} starter id.")
                continue
            if not str(qb_id).strip():
                raise ValueError(f"{game} has an empty {side} starter id.")
            if str(source) not in QB_SOURCES:
                raise ValueError(f"{game} has an unsupported {side} source: {source!r}.")
            if (str(source) == PROJECTED_QB_SOURCE) != (not pd.isna(as_of)):
                raise ValueError(f"{game} must record a projection time exactly when projected.")

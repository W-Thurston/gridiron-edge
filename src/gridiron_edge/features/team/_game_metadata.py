# src/gridiron_edge/features/team/_game_metadata.py

"""Shared resolution of historical and upcoming game metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas import DataFrame

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor


def load_optional_upcoming_metadata(
    datasets: DatasetAccessor,
) -> DataFrame:
    """Load rich upcoming metadata when the artifact is available."""
    try:
        return datasets.schedule_upcoming_rich()
    except (AttributeError, FileNotFoundError):
        return DataFrame()


def _normalize_metadata_source(
    frame: DataFrame,
    *,
    mapping: Mapping[str, str],
    label: str,
) -> DataFrame:
    """Project and rename one metadata source."""
    output_columns = list(mapping.values())
    if frame.empty:
        return DataFrame(columns=output_columns)

    missing = sorted(set(mapping) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))

    normalized = frame.loc[:, list(mapping)].rename(columns=dict(mapping))
    normalized["GAME_ID"] = normalized["GAME_ID"].astype(str).str.strip()
    return normalized.loc[
        normalized["GAME_ID"].ne(""),
        :,
    ].reset_index(drop=True)


def _nonempty_values(values: pd.Series) -> list:
    """Return distinct non-null, nonempty metadata values."""
    cleaned: list[object] = []

    for value in values.tolist():
        if pd.isna(value):
            continue

        normalized: str | Any = value.strip() if isinstance(value, str) else value
        if normalized == "":
            continue

        if normalized not in cleaned:
            cleaned.append(normalized)

    return cleaned


def build_game_metadata_lookup(
    *,
    historical: DataFrame,
    upcoming: DataFrame,
    historical_mapping: Mapping[str, str],
    upcoming_mapping: Mapping[str, str],
) -> DataFrame:
    """Combine game metadata and reject conflicting non-null values."""
    historical_rows = _normalize_metadata_source(
        historical,
        mapping=historical_mapping,
        label="Historical game metadata",
    )
    upcoming_rows = _normalize_metadata_source(
        upcoming,
        mapping=upcoming_mapping,
        label="Upcoming game metadata",
    )

    metadata_columns = list(
        dict.fromkeys(
            [
                *historical_mapping.values(),
                *upcoming_mapping.values(),
            ]
        )
    )
    value_columns = [column for column in metadata_columns if column != "GAME_ID"]

    rows = pd.concat(
        [historical_rows, upcoming_rows],
        ignore_index=True,
    )
    if rows.empty:
        return DataFrame(columns=["GAME_ID", *value_columns])

    resolved: list[dict[str, object]] = []
    for game_id, group in rows.groupby("GAME_ID", sort=False):
        row: dict[str, object] = {"GAME_ID": game_id}
        for column in value_columns:
            values = _nonempty_values(group[column])
            if len(values) > 1:
                raise ValueError(
                    f"Game metadata contains conflicting values for {game_id}/{column}."
                )
            row[column] = values[0] if values else pd.NA
        resolved.append(row)

    return DataFrame(
        resolved,
        columns=["GAME_ID", *value_columns],
    )

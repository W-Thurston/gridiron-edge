# src/gridiron_edge/evaluation/forecast_store.py

"""Immutable storage for game forecast events.

Each row represents one forecast event identified by ``event_id``. Forecast
events are append-only and may coexist for the same game and model pair.

Writing an existing event ID is idempotent when the stored and incoming rows
are identical. Reusing an event ID for different content is rejected.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Final, cast

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.forecast_contracts import ForecastRole

FORECAST_EVENT_COLUMNS: Final[list[str]] = [
    # Event identity
    "event_id",
    "run_id",
    "role",
    "generated_at",
    # Scope and model identity
    "season",
    "week",
    "game_id",
    "model_name",
    "model_type",
    # Game context
    "game_date",
    "away_team",
    "home_team",
    # Model outputs
    "away_elo",
    "home_elo",
    "away_win_prob",
    "home_win_prob",
    "model_spread",
    "model_total",
    "projected_home_score",
    "projected_away_score",
    "margin_std",
    "win_prob_lo",
    "win_prob_hi",
    "confidence_tier",
]

_REQUIRED_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "run_id",
    "role",
    "season",
    "game_id",
    "model_name",
    "model_type",
    "away_team",
    "home_team",
)

_NULLABLE_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "away_elo",
    "home_elo",
    "away_win_prob",
    "home_win_prob",
    "model_spread",
    "model_total",
    "projected_home_score",
    "projected_away_score",
    "margin_std",
    "win_prob_lo",
    "win_prob_hi",
)


def forecast_event_path(
    repo: Path | None = None,
) -> Path:
    """Return the immutable forecast-event store path."""
    root = repo or get_settings().repo_root
    directory = root / "data" / "output" / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "forecast_events.parquet"


def empty_forecast_events() -> DataFrame:
    """Return an empty DataFrame with the forecast-event schema."""
    return DataFrame(columns=FORECAST_EVENT_COLUMNS)


def _normalize_forecast_event_schema(
    events: DataFrame,
) -> DataFrame:
    """Validate the forecast-event schema and return a canonical copy."""
    actual = set(events.columns)
    expected = set(FORECAST_EVENT_COLUMNS)

    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)

    if missing:
        raise ValueError("Forecast events are missing required columns: " + ", ".join(missing))

    if unknown:
        raise ValueError("Forecast events contain unknown columns: " + ", ".join(unknown))

    return events.loc[:, FORECAST_EVENT_COLUMNS].copy()


def _validate_forecast_event_identity(
    events: DataFrame,
) -> None:
    """Validate event IDs and required textual identity fields."""
    if events["event_id"].duplicated().any():
        duplicate_ids = sorted(
            events.loc[
                events["event_id"].duplicated(keep=False),
                "event_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Forecast event batch contains duplicate event IDs: " + ", ".join(duplicate_ids)
        )

    for column in _REQUIRED_TEXT_COLUMNS:
        values = events[column]

        if values.isna().any():
            raise ValueError(f"Forecast event column {column!r} must not contain nulls.")

        empty = values.astype(str).str.strip().eq("")
        if empty.any():
            raise ValueError(f"Forecast event column {column!r} must not contain empty values.")


def _normalize_forecast_event_week(
    events: DataFrame,
) -> None:
    """Validate and normalize the forecast week in place."""
    if events["week"].isna().any():
        raise ValueError("Forecast event column 'week' must not contain nulls.")

    events["week"] = events["week"].astype(int)

    if (events["week"] < 1).any():
        raise ValueError("Forecast event week must be at least 1.")


def _normalize_forecast_generated_at(
    events: DataFrame,
) -> None:
    """Validate and normalize forecast generation timestamps in place."""
    generated_at = pd.to_datetime(
        events["generated_at"],
        utc=True,
        errors="coerce",
    )

    # pyrefly: ignore [missing-attribute]
    if generated_at.isna().any():
        raise ValueError("Forecast event generated_at values must be valid datetimes.")

    # ``utc=True`` converts aware non-UTC values. Reject those before
    # conversion so the domain contract remains explicitly UTC.
    for value in events["generated_at"]:
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            raise ValueError("Forecast event generated_at values must be timezone-aware UTC.")

        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("Forecast event generated_at values must use UTC.")

    events["generated_at"] = generated_at


def _normalize_nullable_object(
    values: Series,
) -> Series:
    """Replace pandas null values with Python None in an object Series."""
    return Series(
        (None if pd.isna(value) else value for value in values),
        index=values.index,
        dtype=object,
    )


def validate_forecast_events(events: DataFrame) -> DataFrame:
    """Validate and normalize forecast events before persistence.

    Returns a normalized copy with canonical column order.

    Raises:
        ValueError: If the schema or event values violate the contract.
    """
    normalized = _normalize_forecast_event_schema(events)

    _validate_forecast_event_identity(normalized)

    valid_roles = {role.value for role in ForecastRole}
    invalid_roles = sorted(set(normalized["role"].astype(str)) - valid_roles)
    if invalid_roles:
        # pyrefly: ignore [no-matching-overload]
        raise ValueError("Forecast events contain invalid roles: " + ", ".join(invalid_roles))

    _normalize_forecast_event_week(normalized)
    _normalize_forecast_generated_at(normalized)

    for column in _NULLABLE_FLOAT_COLUMNS:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized["game_date"] = _normalize_nullable_object(
        normalized["game_date"],
    )
    normalized["confidence_tier"] = _normalize_nullable_object(
        normalized["confidence_tier"],
    )

    return normalized


def write_forecast_events(
    events: DataFrame,
    *,
    repo: Path | None = None,
) -> Path:
    """Append immutable forecast events to the event store.

    An existing event ID with identical content is an idempotent no-op.
    An existing event ID with different content violates event immutability.
    """
    normalized = validate_forecast_events(events)
    path = forecast_event_path(repo)

    if not path.exists():
        normalized.sort_values(
            ["generated_at", "run_id", "event_id"],
            kind="stable",
        ).to_parquet(path, index=False)
        return path

    existing = validate_forecast_events(
        pd.read_parquet(path),
    )

    existing_by_id = existing.set_index(
        "event_id",
        drop=False,
    )
    incoming_by_id = normalized.set_index(
        "event_id",
        drop=False,
    )

    overlapping_ids = set(existing_by_id.index).intersection(
        incoming_by_id.index,
    )

    new_ids: list[str] = []

    new_ids: list[str] = []

    for raw_event_id, incoming_row in incoming_by_id.iterrows():
        event_id = cast(str, raw_event_id)

        if event_id not in overlapping_ids:
            new_ids.append(event_id)
            continue

        existing_row = existing_by_id.loc[event_id]

        if isinstance(existing_row, DataFrame):
            raise ValueError(
                f"Stored forecast event ID {event_id!r} is not unique.",
            )

        if not existing_row.equals(incoming_row):
            raise ValueError(
                f"Forecast event ID cannot be reused with different content: {event_id}",
            )

    if not new_ids:
        return path

    new_rows = normalized.loc[
        normalized["event_id"].isin(new_ids),
        :,
    ]

    combined = pd.concat(
        [existing, new_rows],
        ignore_index=True,
    ).sort_values(
        ["generated_at", "run_id", "event_id"],
        kind="stable",
    )

    combined.to_parquet(path, index=False)
    return path


def load_forecast_events(
    *,
    season: str | None = None,
    week: int | None = None,
    game_id: str | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    role: ForecastRole | None = None,
    run_id: str | None = None,
    event_id: str | None = None,
    repo: Path | None = None,
) -> DataFrame:
    """Load immutable forecast events with optional filters."""
    path = forecast_event_path(repo)

    if not path.exists():
        return empty_forecast_events()

    events = validate_forecast_events(
        pd.read_parquet(path),
    )

    filters: tuple[tuple[str, object | None], ...] = (
        ("season", season),
        ("week", week),
        ("game_id", game_id),
        ("model_name", model_name),
        ("model_type", model_type),
        (
            "role",
            role.value if role is not None else None,
        ),
        ("run_id", run_id),
        ("event_id", event_id),
    )

    for column, value in filters:
        if value is not None:
            events = events.loc[
                events[column] == value,
                :,
            ]

    return events.reset_index(drop=True)

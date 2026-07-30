# src/gridiron_edge/evaluation/forecast_selection.py

"""Explicit selection of immutable forecast events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame

from gridiron_edge.evaluation.forecast_contracts import SelectedForecast
from gridiron_edge.evaluation.forecast_store import (
    FORECAST_EVENT_COLUMNS,
    validate_forecast_events,
)


@dataclass(frozen=True)
class ForecastSelectionResult:
    """Result of resolving explicit forecast references.

    Attributes:
        events: Canonical forecast-event rows selected by exact event ID.
        missing: Requested references whose event IDs were not present.
    """

    events: DataFrame
    missing: tuple[SelectedForecast, ...]

    @property
    def complete(self) -> bool:
        """Return whether every requested reference was resolved."""
        return not self.missing


def _empty_selected_events() -> DataFrame:
    """Return an empty canonical forecast-event frame."""
    return DataFrame(columns=FORECAST_EVENT_COLUMNS)


def _validate_selected_identity(
    event: DataFrame,
    selection: SelectedForecast,
) -> None:
    """Verify that an event matches its explicit selection reference."""
    row = event.iloc[0]

    expected = {
        "game_id": selection.game_id,
        "model_name": selection.model_name,
        "model_type": selection.model_type,
    }

    conflicts = [
        field_name
        for field_name, expected_value in expected.items()
        if str(row[field_name]) != expected_value
    ]

    if conflicts:
        raise ValueError(
            "Selected forecast reference does not match event identity "
            f"for event_id {selection.event_id!r}: " + ", ".join(conflicts)
        )


def select_forecast_events(
    events: DataFrame,
    selections: Sequence[SelectedForecast],
) -> ForecastSelectionResult:
    """Resolve exact immutable forecast events from explicit references.

    Selection is based only on the requested event IDs. Generation time,
    storage order, role, and model priority do not affect the result.

    Args:
        events: Forecast events conforming to the canonical event schema.
        selections: Explicit references to immutable forecast events.

    Returns:
        Selected events in reference order together with unresolved
        references.

    Raises:
        ValueError: If the event frame violates the storage contract, a
            requested event ID appears more than once, the same event ID is
            requested more than once, or a reference conflicts with the
            corresponding event's game or model identity.
    """
    normalized: DataFrame = validate_forecast_events(events)

    reference_counts: Counter[str] = Counter(selection.event_id for selection in selections)
    duplicate_reference_ids: list[str] = sorted(
        event_id for event_id, count in reference_counts.items() if count > 1
    )
    if duplicate_reference_ids:
        raise ValueError(
            "Forecast selections contain duplicate event IDs: " + ", ".join(duplicate_reference_ids)
        )

    selected_rows: list[DataFrame] = []
    missing: list[SelectedForecast] = []

    for selection in selections:
        matches = normalized.loc[
            normalized["event_id"] == selection.event_id,
            :,
        ]

        if matches.empty:
            missing.append(selection)
            continue

        if len(matches) != 1:
            raise ValueError(f"Forecast event ID is not unique: {selection.event_id}")

        _validate_selected_identity(
            matches,
            selection,
        )
        selected_rows.append(matches)

    if selected_rows:
        selected = pd.concat(
            selected_rows,
            ignore_index=True,
        ).loc[:, FORECAST_EVENT_COLUMNS]
    else:
        selected: DataFrame = _empty_selected_events()

    return ForecastSelectionResult(
        events=selected,
        missing=tuple(missing),
    )

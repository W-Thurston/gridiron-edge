# src/gridiron_edge/evaluation/forecast_selection.py

"""Explicit selection of immutable forecast events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from pandas import DataFrame

from gridiron_edge.evaluation.forecast_contracts import (
    ForecastRole,
    SelectedForecast,
)
from gridiron_edge.evaluation.forecast_store import (
    FORECAST_EVENT_COLUMNS,
    validate_forecast_events,
)

_RUN_FORECAST_KEY: tuple[str, ...] = (
    "game_id",
    "model_name",
    "model_type",
)


@dataclass(frozen=True)
class ForecastCandidateIdentity:
    """Game and model identity requiring forecast selection."""

    game_id: str
    model_name: str
    model_type: str

    def __post_init__(self) -> None:
        """Validate candidate identity fields."""
        for field_name, value in (
            ("game_id", self.game_id),
            ("model_name", self.model_name),
            ("model_type", self.model_type),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")


class ForecastCandidateStatus(StrEnum):
    """Resolution state for one forecast candidate identity."""

    SELECTED = "selected"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ForecastCandidateResolution:
    """Resolution result for one game and model identity."""

    identity: ForecastCandidateIdentity
    status: ForecastCandidateStatus
    selected: SelectedForecast | None
    eligible_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate resolution-state invariants."""
        if self.status is ForecastCandidateStatus.SELECTED:
            if self.selected is None:
                raise ValueError("Selected resolution requires a selected forecast.")
            if len(self.eligible_event_ids) != 1:
                raise ValueError("Selected resolution requires exactly one eligible event ID.")
            return

        if self.selected is not None:
            raise ValueError("Only a selected resolution may contain a selected forecast.")

        if self.status is ForecastCandidateStatus.MISSING and self.eligible_event_ids:
            raise ValueError("Missing resolution must not contain eligible event IDs.")

        if self.status is ForecastCandidateStatus.AMBIGUOUS and len(self.eligible_event_ids) < 2:
            raise ValueError("Ambiguous resolution requires at least two eligible event IDs.")


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


@dataclass(frozen=True)
class ForecastRunSelectionResult:
    """Result of selecting one explicit forecast run.

    Attributes:
        run_id: Requested forecast-run identity.
        events: Canonical events belonging to the requested run.
        found: Whether the requested run was present.
    """

    run_id: str
    events: DataFrame
    found: bool


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


def select_forecast_run(
    events: DataFrame,
    *,
    run_id: str,
) -> ForecastRunSelectionResult:
    """Select all forecast events from one explicit invocation.

    Selection is restricted to the requested run ID. Events from other
    runs are never considered, regardless of role, generation time, or
    storage order.

    Within the selected run, each game and model identity must be unique.
    Win and total prediction families remain independent because
    ``model_name`` and ``model_type`` are part of the uniqueness key.

    Args:
        events: Forecast events conforming to the canonical event schema.
        run_id: Exact run identity to select.

    Returns:
        A structured result containing the requested run's canonical
        events. A missing run returns an empty canonical frame with
        ``found=False``.

    Raises:
        ValueError: If ``run_id`` is empty, the event frame violates the
            storage contract, or the requested run contains duplicate game
            and model identities.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty.")

    normalized = validate_forecast_events(events)

    selected = normalized.loc[
        normalized["run_id"] == run_id,
        :,
    ].copy()

    if selected.empty:
        return ForecastRunSelectionResult(
            run_id=run_id,
            events=_empty_selected_events(),
            found=False,
        )

    duplicated = selected.duplicated(
        subset=list(_RUN_FORECAST_KEY),
        keep=False,
    )
    if duplicated.any():
        duplicate_rows = (
            selected.loc[
                duplicated,
                list(_RUN_FORECAST_KEY),
            ]
            .drop_duplicates()
            .sort_values(
                list(_RUN_FORECAST_KEY),
                kind="stable",
            )
        )

        duplicate_identities = [
            "/".join(str(row[column]) for column in _RUN_FORECAST_KEY)
            for _, row in duplicate_rows.iterrows()
        ]

        raise ValueError(
            f"Forecast run {run_id!r} contains duplicate "
            "game and model identities: " + ", ".join(duplicate_identities)
        )

    selected = selected.sort_values(
        [
            "season",
            "week",
            "game_id",
            "model_name",
            "model_type",
            "event_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return ForecastRunSelectionResult(
        run_id=run_id,
        events=selected.loc[:, FORECAST_EVENT_COLUMNS],
        found=True,
    )


def resolve_forecast_candidates(
    events: DataFrame,
    identities: Sequence[ForecastCandidateIdentity],
) -> tuple[ForecastCandidateResolution, ...]:
    """Resolve eligible events for explicit game and model identities.

    Live events are preferred over backfilled events. When at least one
    matching live event exists, matching backfilled events are excluded.

    Selection succeeds only when exactly one eligible event remains.
    Multiple eligible live events or multiple eligible backfilled events
    remain ambiguous. Generation time, storage order, and event ID do not
    determine which event is selected.

    Args:
        events: Forecast events conforming to the canonical event schema.
        identities: Game and model identities requiring resolution.

    Returns:
        One resolution per requested identity, preserving request order.

    Raises:
        ValueError: If the event frame violates the storage contract or a
            candidate identity is requested more than once.
    """
    normalized = validate_forecast_events(events)

    identity_counts = Counter(identities)
    duplicate_identities = sorted(
        (identity for identity, count in identity_counts.items() if count > 1),
        key=lambda identity: (
            identity.game_id,
            identity.model_name,
            identity.model_type,
        ),
    )
    if duplicate_identities:
        formatted = [
            (f"{identity.game_id}/{identity.model_name}/{identity.model_type}")
            for identity in duplicate_identities
        ]
        raise ValueError(
            "Forecast candidate identities contain duplicates: " + ", ".join(formatted)
        )

    resolutions: list[ForecastCandidateResolution] = []

    for identity in identities:
        matches = normalized.loc[
            (normalized["game_id"] == identity.game_id)
            & (normalized["model_name"] == identity.model_name)
            & (normalized["model_type"] == identity.model_type),
            :,
        ]

        if matches.empty:
            resolutions.append(
                ForecastCandidateResolution(
                    identity=identity,
                    status=ForecastCandidateStatus.MISSING,
                    selected=None,
                    eligible_event_ids=(),
                )
            )
            continue

        live_matches = matches.loc[
            matches["role"] == ForecastRole.LIVE.value,
            :,
        ]
        eligible = (
            live_matches
            if not live_matches.empty
            else matches.loc[
                matches["role"] == ForecastRole.BACKFILLED.value,
                :,
            ]
        )

        eligible_event_ids = tuple(sorted(eligible["event_id"].astype(str).tolist()))

        if len(eligible) > 1:
            resolutions.append(
                ForecastCandidateResolution(
                    identity=identity,
                    status=ForecastCandidateStatus.AMBIGUOUS,
                    selected=None,
                    eligible_event_ids=eligible_event_ids,
                )
            )
            continue

        row = eligible.iloc[0]
        selected = SelectedForecast(
            event_id=str(row["event_id"]),
            game_id=str(row["game_id"]),
            model_name=str(row["model_name"]),
            model_type=str(row["model_type"]),
        )

        resolutions.append(
            ForecastCandidateResolution(
                identity=identity,
                status=ForecastCandidateStatus.SELECTED,
                selected=selected,
                eligible_event_ids=eligible_event_ids,
            )
        )

    return tuple(resolutions)

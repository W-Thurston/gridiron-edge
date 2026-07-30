# src/gridiron_edge/evaluation/forecast_contracts.py

"""Domain contracts for immutable forecast events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4


class ForecastRole(StrEnum):
    """Operational role of a forecast event."""

    LIVE = "live"
    BACKFILLED = "backfilled"


@dataclass(frozen=True)
class ForecastEventIdentity:
    """Storage-independent identity for one immutable forecast event."""

    event_id: str
    run_id: str
    role: ForecastRole
    generated_at: datetime
    game_id: str
    model_name: str
    model_type: str

    def __post_init__(self) -> None:
        """Validate identity invariants."""
        for field_name, value in (
            ("event_id", self.event_id),
            ("run_id", self.run_id),
            ("game_id", self.game_id),
            ("model_name", self.model_name),
            ("model_type", self.model_type),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")

        if not isinstance(self.role, ForecastRole):
            raise TypeError("role must be a ForecastRole.")

        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware UTC.")

        if self.generated_at.utcoffset() != timedelta(0):
            raise ValueError("generated_at must use UTC.")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        role: ForecastRole,
        game_id: str,
        model_name: str,
        model_type: str,
        generated_at: datetime | None = None,
    ) -> ForecastEventIdentity:
        """Create an identity for a newly generated forecast event."""
        return cls(
            event_id=str(uuid4()),
            run_id=run_id,
            role=role,
            generated_at=generated_at or datetime.now(UTC),
            game_id=game_id,
            model_name=model_name,
            model_type=model_type,
        )


@dataclass(frozen=True)
class SelectedForecast:
    """Reference to the forecast event selected for one prediction family.

    Selection is explicit and references an immutable forecast event. It does
    not infer a forecast from write order, generation time, or model priority.
    """

    event_id: str
    game_id: str
    model_name: str
    model_type: str

    def __post_init__(self) -> None:
        """Validate selected-forecast identity fields."""
        for field_name, value in (
            ("event_id", self.event_id),
            ("game_id", self.game_id),
            ("model_name", self.model_name),
            ("model_type", self.model_type),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")

    @classmethod
    def from_event(
        cls,
        event: ForecastEventIdentity,
    ) -> SelectedForecast:
        """Create a selected-forecast reference from an event identity."""
        return cls(
            event_id=event.event_id,
            game_id=event.game_id,
            model_name=event.model_name,
            model_type=event.model_type,
        )


@dataclass(frozen=True)
class WeeklyProductIdentity:
    """Identity and scope of one immutable weekly prediction product."""

    product_id: str
    run_id: str
    season: str
    week: int
    generated_at: datetime

    def __post_init__(self) -> None:
        """Validate weekly product identity invariants."""
        for field_name, value in (
            ("product_id", self.product_id),
            ("run_id", self.run_id),
            ("season", self.season),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")

        if self.week < 1:
            raise ValueError("week must be at least 1.")

        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware UTC.")

        if self.generated_at.utcoffset() != timedelta(0):
            raise ValueError("generated_at must use UTC.")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        season: str,
        week: int,
        generated_at: datetime | None = None,
    ) -> WeeklyProductIdentity:
        """Create an identity for a newly composed weekly product."""
        return cls(
            product_id=str(uuid4()),
            run_id=run_id,
            season=season,
            week=week,
            generated_at=generated_at or datetime.now(UTC),
        )


def new_forecast_run_id() -> str:
    """Create an identifier shared by forecasts generated in one run."""
    return str(uuid4())

"""Domain contracts for weekly game-prediction readiness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final


class WeeklyReadinessBlocker(StrEnum):
    """Machine-readable reason a weekly product is not fully ready."""

    MISSING_SCHEDULE = "missing_schedule"
    MISSING_WIN_PREDICTIONS = "missing_win_predictions"
    PARTIAL_WIN_PREDICTION_COVERAGE = "partial_win_prediction_coverage"
    MISSING_SPREAD_VALUES = "missing_spread_values"
    PARTIAL_SPREAD_COVERAGE = "partial_spread_coverage"
    MISSING_TOTAL_PREDICTIONS = "missing_total_predictions"
    PARTIAL_TOTAL_PREDICTION_COVERAGE = "partial_total_prediction_coverage"
    MISSING_PROJECTED_SCORES = "missing_projected_scores"
    PARTIAL_PROJECTED_SCORE_COVERAGE = "partial_projected_score_coverage"
    MISSING_MODEL_PROVENANCE = "missing_model_provenance"
    PARTIAL_MODEL_PROVENANCE = "partial_model_provenance"
    MISSING_MARKET_DATA = "missing_market_data"
    PARTIAL_MARKET_COVERAGE = "partial_market_coverage"
    ZERO_PREDICTION_MARKET_MATCHES = "zero_prediction_market_matches"
    INCOMPLETE_MARKETS = "incomplete_markets"


_GAME_COVERAGE_FIELDS: Final[tuple[str, ...]] = (
    "selected_win_prediction_count",
    "spread_value_count",
    "total_prediction_count",
    "projected_score_count",
    "complete_provenance_count",
    "market_game_count",
    "prediction_market_match_count",
)


@dataclass(frozen=True)
class WeeklyReadiness:
    """Quantitative readiness state for one season and week."""

    season: str
    week: int

    scheduled_game_count: int
    selected_win_prediction_count: int
    spread_value_count: int
    total_prediction_count: int
    projected_score_count: int
    complete_provenance_count: int

    market_game_count: int
    prediction_market_match_count: int
    eligible_market_count: int
    positive_edge_count: int

    prediction_generated_at: datetime | None = None
    market_fetched_at: datetime | None = None
    market_source: str | None = None

    blockers: tuple[WeeklyReadinessBlocker, ...] = ()

    def __post_init__(self) -> None:
        """Validate weekly readiness invariants."""
        if not self.season.strip():
            raise ValueError("season must not be empty.")

        if self.week < 1:
            raise ValueError("week must be at least 1.")

        count_fields = (
            "scheduled_game_count",
            *_GAME_COVERAGE_FIELDS,
            "eligible_market_count",
            "positive_edge_count",
        )

        for field_name in count_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must not be negative.")

        for field_name in _GAME_COVERAGE_FIELDS:
            value = getattr(self, field_name)
            if value > self.scheduled_game_count:
                raise ValueError(f"{field_name} must not exceed scheduled_game_count.")

        if self.positive_edge_count > self.eligible_market_count:
            raise ValueError("positive_edge_count must not exceed eligible_market_count.")

        self._validate_utc_timestamp(
            "prediction_generated_at",
            self.prediction_generated_at,
        )
        self._validate_utc_timestamp(
            "market_fetched_at",
            self.market_fetched_at,
        )

        if self.market_source is not None and not self.market_source.strip():
            raise ValueError("market_source must not be empty when provided.")

        if len(set(self.blockers)) != len(self.blockers):
            raise ValueError("blockers must not contain duplicate values.")

    @staticmethod
    def _validate_utc_timestamp(
        field_name: str,
        value: datetime | None,
    ) -> None:
        """Validate an optional timezone-aware UTC timestamp."""
        if value is None:
            return

        if value.tzinfo is None:
            raise ValueError(f"{field_name} must be timezone-aware UTC.")

        if value.utcoffset() != timedelta(0):
            raise ValueError(f"{field_name} must use UTC.")

    @property
    def ready(self) -> bool:
        """Return whether no operational blockers are present."""
        return not self.blockers

    @property
    def has_positive_edges(self) -> bool:
        """Return whether analysis produced any positive edges."""
        return self.positive_edge_count > 0

# src/gridiron_edge/market/edge_diagnostics.py

"""Structured diagnostics for weekly edge calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class EdgeDiagnosticBlocker(StrEnum):
    """Machine-readable reasons an edge result is unavailable or incomplete."""

    NO_PREDICTIONS = "no_predictions"
    NO_MARKET_DATA = "no_market_data"
    MARKET_WRONG_SCOPE = "market_wrong_scope"
    MARKET_STALE = "market_stale"
    ZERO_MATCHED_GAMES = "zero_matched_games"
    INCOMPLETE_MARKETS = "incomplete_markets"


class EdgeResultState(StrEnum):
    """Terminal analytical state of one edge evaluation."""

    BLOCKED = "blocked"
    NO_CALCULABLE_EDGES = "no_calculable_edges"
    NO_POSITIVE_EDGES = "no_positive_edges"
    POSITIVE_EDGES = "positive_edges"


@dataclass(frozen=True, slots=True)
class EdgeProvenance:
    """Distinct prediction and market provenance present in supplied inputs."""

    win_event_ids: tuple[str, ...] = ()
    win_run_ids: tuple[str, ...] = ()
    win_model_names: tuple[str, ...] = ()
    win_model_types: tuple[str, ...] = ()
    total_event_ids: tuple[str, ...] = ()
    total_run_ids: tuple[str, ...] = ()
    total_model_names: tuple[str, ...] = ()
    total_model_types: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()
    product_run_ids: tuple[str, ...] = ()
    market_sources: tuple[str, ...] = ()
    market_fetched_at: tuple[datetime, ...] = ()

    def __post_init__(self) -> None:
        """Validate deterministic, nonempty provenance values."""
        text_fields = (
            "win_event_ids",
            "win_run_ids",
            "win_model_names",
            "win_model_types",
            "total_event_ids",
            "total_run_ids",
            "total_model_names",
            "total_model_types",
            "product_ids",
            "product_run_ids",
            "market_sources",
        )

        for field_name in text_fields:
            values: tuple[str, ...] = getattr(self, field_name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{field_name} must contain sorted unique values.")
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must not contain empty values.")

        timestamps = self.market_fetched_at
        if tuple(sorted(set(timestamps))) != timestamps:
            raise ValueError("market_fetched_at must contain sorted unique values.")

        for timestamp in timestamps:
            if timestamp.tzinfo is None:
                raise ValueError("market_fetched_at values must be timezone-aware UTC.")
            offset = timestamp.utcoffset()
            if offset is None or offset.total_seconds() != 0:
                raise ValueError("market_fetched_at values must use UTC.")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible provenance."""
        return {
            "win_event_ids": list(self.win_event_ids),
            "win_run_ids": list(self.win_run_ids),
            "win_model_names": list(self.win_model_names),
            "win_model_types": list(self.win_model_types),
            "total_event_ids": list(self.total_event_ids),
            "total_run_ids": list(self.total_run_ids),
            "total_model_names": list(self.total_model_names),
            "total_model_types": list(self.total_model_types),
            "product_ids": list(self.product_ids),
            "product_run_ids": list(self.product_run_ids),
            "market_sources": list(self.market_sources),
            "market_fetched_at": [timestamp.isoformat() for timestamp in self.market_fetched_at],
        }


@dataclass(frozen=True, slots=True)
class EdgeDiagnostics:
    """Coverage, eligibility, and result diagnostics for one weekly scope."""

    season: str
    week: int
    prediction_game_count: int
    market_game_count: int
    matched_game_count: int
    complete_moneyline_count: int
    complete_spread_count: int
    complete_total_count: int
    eligible_market_count: int
    calculated_edge_count: int
    positive_edge_count: int
    filtered_edge_count: int
    state: EdgeResultState
    blockers: tuple[EdgeDiagnosticBlocker, ...] = ()
    provenance: EdgeProvenance = field(default_factory=EdgeProvenance)

    def __post_init__(self) -> None:
        """Validate scope, count relationships, and terminal state."""
        self._validate_scope()
        self._validate_counts()
        self._validate_state()

    def _validate_scope(self) -> None:
        """Validate weekly scope identity."""
        if not self.season.strip():
            raise ValueError("season must not be empty.")
        if self.week < 1:
            raise ValueError("week must be at least 1.")

    def _validate_counts(self) -> None:
        """Validate nonnegative counts and their relationships."""
        count_fields = (
            "prediction_game_count",
            "market_game_count",
            "matched_game_count",
            "complete_moneyline_count",
            "complete_spread_count",
            "complete_total_count",
            "eligible_market_count",
            "calculated_edge_count",
            "positive_edge_count",
            "filtered_edge_count",
        )
        for field_name in count_fields:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative.")

        if self.matched_game_count > self.prediction_game_count:
            raise ValueError("matched_game_count must not exceed prediction_game_count.")
        if self.matched_game_count > self.market_game_count:
            raise ValueError("matched_game_count must not exceed market_game_count.")

        complete_count = (
            self.complete_moneyline_count + self.complete_spread_count + self.complete_total_count
        )
        if self.eligible_market_count != complete_count:
            raise ValueError(
                "eligible_market_count must equal the sum of complete "
                "Moneyline, Spread, and Total markets."
            )
        if self.positive_edge_count > self.calculated_edge_count:
            raise ValueError("positive_edge_count must not exceed calculated_edge_count.")
        if self.filtered_edge_count > self.positive_edge_count:
            raise ValueError("filtered_edge_count must not exceed positive_edge_count.")

    def _validate_state(self) -> None:
        """Validate blockers and terminal analytical state."""
        if len(set(self.blockers)) != len(self.blockers):
            raise ValueError("blockers must not contain duplicates.")
        if self.state is EdgeResultState.BLOCKED:
            if not self.blockers:
                raise ValueError("blocked diagnostics require at least one blocker.")
            return
        if self.blockers:
            raise ValueError("non-blocked diagnostics must not contain blockers.")
        if self.state is EdgeResultState.NO_CALCULABLE_EDGES:
            if self.calculated_edge_count != 0:
                raise ValueError("no_calculable_edges requires calculated_edge_count == 0.")
        elif self.state is EdgeResultState.NO_POSITIVE_EDGES:
            if self.calculated_edge_count == 0:
                raise ValueError("no_positive_edges requires calculated edge rows.")
            if self.positive_edge_count != 0:
                raise ValueError("no_positive_edges requires positive_edge_count == 0.")
        elif self.state is EdgeResultState.POSITIVE_EDGES and self.positive_edge_count == 0:
            raise ValueError("positive_edges requires positive edge rows.")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible diagnostics."""
        return {
            "season": self.season,
            "week": self.week,
            "prediction_game_count": self.prediction_game_count,
            "market_game_count": self.market_game_count,
            "matched_game_count": self.matched_game_count,
            "complete_moneyline_count": self.complete_moneyline_count,
            "complete_spread_count": self.complete_spread_count,
            "complete_total_count": self.complete_total_count,
            "eligible_market_count": self.eligible_market_count,
            "calculated_edge_count": self.calculated_edge_count,
            "positive_edge_count": self.positive_edge_count,
            "filtered_edge_count": self.filtered_edge_count,
            "state": self.state.value,
            "blockers": [blocker.value for blocker in self.blockers],
            "provenance": self.provenance.to_dict(),
        }

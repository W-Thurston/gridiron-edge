# src/gridiron_edge/market/edge_diagnostics.py

"""Structured diagnostics for weekly edge calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import pandas as pd
from pandas import DataFrame, Series


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


# ---------------------------------------------------------------------------
# Pure DataFrame evaluation
# ---------------------------------------------------------------------------

_PREDICTION_SCOPE_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
)
_MARKET_SCOPE_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "market",
    "side",
    "odds",
    "line",
)
_EDGE_SCOPE_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "ev",
)


def _require_nonempty_columns(
    frame: DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require columns only when a supplied frame contains rows."""
    if frame.empty:
        return
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _scope_frame(
    frame: DataFrame,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Return a detached copy restricted to one weekly scope."""
    if frame.empty:
        return frame.copy()
    return frame.loc[
        (frame["season"].astype(str) == season) & (frame["week"] == week),
        :,
    ].copy()


def _game_ids(frame: DataFrame) -> set[str]:
    """Return distinct nonempty game IDs from one scoped frame."""
    if frame.empty:
        return set()
    return {str(value).strip() for value in frame["game_id"].dropna() if str(value).strip()}


def _prediction_for_game(
    predictions: DataFrame,
    game_id: str,
) -> Series:
    """Return the first prediction row for one already-scoped game."""
    return predictions.loc[
        predictions["game_id"].astype(str) == game_id,
        :,
    ].iloc[0]


def _value_present(row: Series, column: str) -> bool:
    """Return whether a scalar prediction value is present."""
    return column in row.index and row[column] is not None and pd.notna(row[column])


def _market_side_complete(
    game_markets: DataFrame,
    *,
    market: str,
    side: str,
    require_line: bool,
) -> bool:
    """Return whether any supplied row has the required side values."""
    rows = game_markets.loc[
        (game_markets["market"] == market) & (game_markets["side"] == side),
        :,
    ]
    if rows.empty:
        return False
    # pyrefly: ignore [missing-attribute]
    odds_present = pd.to_numeric(rows["odds"], errors="coerce").notna()
    if not require_line:
        return bool(odds_present.any())
    # pyrefly: ignore [missing-attribute]
    line_present = pd.to_numeric(rows["line"], errors="coerce").notna()
    return bool((odds_present & line_present).any())


def _home_win_probability_present(prediction: Series) -> bool:
    """Return whether either canonical win-probability side is available."""
    return _value_present(prediction, "home_win_prob") or _value_present(
        prediction,
        "away_win_prob",
    )


def _complete_market_counts(
    predictions: DataFrame,
    markets: DataFrame,
    matched_game_ids: set[str],
) -> tuple[int, int, int]:
    """Count complete calculable Moneyline, Spread, and Total pairs."""
    moneyline_count = 0
    spread_count = 0
    total_count = 0

    for game_id in sorted(matched_game_ids):
        prediction = _prediction_for_game(predictions, game_id)
        game_markets = markets.loc[
            markets["game_id"].astype(str) == game_id,
            :,
        ]

        moneyline_complete = (
            _home_win_probability_present(prediction)
            and _market_side_complete(
                game_markets,
                market="moneyline",
                side="home",
                require_line=False,
            )
            and _market_side_complete(
                game_markets,
                market="moneyline",
                side="away",
                require_line=False,
            )
        )
        spread_complete = (
            _value_present(prediction, "model_spread")
            and _market_side_complete(
                game_markets,
                market="spread",
                side="home",
                require_line=True,
            )
            and _market_side_complete(
                game_markets,
                market="spread",
                side="away",
                require_line=False,
            )
        )
        total_complete = (
            _value_present(prediction, "model_total")
            and _market_side_complete(
                game_markets,
                market="total",
                side="over",
                require_line=True,
            )
            and _market_side_complete(
                game_markets,
                market="total",
                side="under",
                require_line=False,
            )
        )

        moneyline_count += int(moneyline_complete)
        spread_count += int(spread_complete)
        total_count += int(total_complete)

    return moneyline_count, spread_count, total_count


def _distinct_text_values(
    frame: DataFrame,
    columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Collect sorted unique nonempty values across known aliases."""
    values: set[str] = set()
    for column in columns:
        if column not in frame.columns:
            continue
        values.update(str(value).strip() for value in frame[column].dropna() if str(value).strip())
    return tuple(sorted(values))


def _market_timestamps(markets: DataFrame) -> tuple[datetime, ...]:
    """Collect sorted unique valid UTC market timestamps."""
    if markets.empty or "fetched_at" not in markets.columns:
        return ()
    # pyrefly: ignore [missing-attribute]
    timestamps = pd.to_datetime(
        markets["fetched_at"],
        utc=True,
        errors="coerce",
    ).dropna()
    return tuple(sorted({timestamp.to_pydatetime() for timestamp in timestamps}))


def _extract_provenance(
    predictions: DataFrame,
    markets: DataFrame,
) -> EdgeProvenance:
    """Extract all recognized scoped prediction and market provenance."""
    return EdgeProvenance(
        win_event_ids=_distinct_text_values(
            predictions,
            ("win_event_id", "win_forecast_event_id"),
        ),
        win_run_ids=_distinct_text_values(
            predictions,
            ("win_run_id", "win_forecast_run_id"),
        ),
        win_model_names=_distinct_text_values(
            predictions,
            ("win_model_name", "model_name"),
        ),
        win_model_types=_distinct_text_values(
            predictions,
            ("win_model_type", "model_type", "model_version"),
        ),
        total_event_ids=_distinct_text_values(
            predictions,
            ("total_event_id", "total_forecast_event_id"),
        ),
        total_run_ids=_distinct_text_values(
            predictions,
            ("total_run_id", "total_forecast_run_id"),
        ),
        total_model_names=_distinct_text_values(
            predictions,
            ("total_model_name",),
        ),
        total_model_types=_distinct_text_values(
            predictions,
            ("total_model_type",),
        ),
        product_ids=_distinct_text_values(
            predictions,
            ("product_id",),
        ),
        product_run_ids=_distinct_text_values(
            predictions,
            ("product_run_id",),
        ),
        market_sources=_distinct_text_values(
            markets,
            ("sportsbook",),
        ),
        market_fetched_at=_market_timestamps(markets),
    )


def _validate_freshness_inputs(
    *,
    as_of: datetime | None,
    max_market_age: timedelta | None,
) -> None:
    """Require a complete deterministic freshness policy when supplied."""
    if (as_of is None) != (max_market_age is None):
        raise ValueError("as_of and max_market_age must be provided together.")
    if as_of is None or max_market_age is None:
        return
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware UTC.")
    offset = as_of.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("as_of must use UTC.")
    if max_market_age < timedelta(0):
        raise ValueError("max_market_age must not be negative.")


def _market_is_stale(
    provenance: EdgeProvenance,
    *,
    as_of: datetime | None,
    max_market_age: timedelta | None,
) -> bool:
    """Return whether any scoped market timestamp violates the policy."""
    if as_of is None or max_market_age is None:
        return False
    cutoff = as_of - max_market_age
    return any(timestamp < cutoff for timestamp in provenance.market_fetched_at)


def _edge_counts(
    calculated_edges: DataFrame,
    filtered_edges: DataFrame,
) -> tuple[int, int, int]:
    """Return calculated, positive, and positively filtered row counts."""
    calculated_count = len(calculated_edges)
    if calculated_edges.empty:
        positive_count = 0
    else:
        positive_count = int(
            # pyrefly: ignore [missing-attribute]
            (
                pd.to_numeric(
                    calculated_edges["ev"],
                    errors="coerce",
                )
                > 0.0
            ).sum()
        )
    if filtered_edges.empty:
        filtered_count = 0
    else:
        filtered_count = int(
            # pyrefly: ignore [missing-attribute]
            (
                pd.to_numeric(
                    filtered_edges["ev"],
                    errors="coerce",
                )
                > 0.0
            ).sum()
        )
    return calculated_count, positive_count, filtered_count


def evaluate_edge_diagnostics(
    predictions: DataFrame,
    markets: DataFrame,
    calculated_edges: DataFrame,
    filtered_edges: DataFrame,
    *,
    season: str,
    week: int,
    as_of: datetime | None = None,
    max_market_age: timedelta | None = None,
) -> EdgeDiagnostics:
    """Derive weekly edge coverage, blockers, outcomes, and provenance.

    The evaluator is pure and does not invoke edge math, ranking, artifact
    selection, ingestion, the system clock, or file I/O.
    """
    if not season.strip():
        raise ValueError("season must not be empty.")
    if week < 1:
        raise ValueError("week must be at least 1.")
    _validate_freshness_inputs(
        as_of=as_of,
        max_market_age=max_market_age,
    )
    _require_nonempty_columns(
        predictions,
        _PREDICTION_SCOPE_COLUMNS,
        label="Predictions",
    )
    _require_nonempty_columns(
        markets,
        _MARKET_SCOPE_COLUMNS,
        label="Markets",
    )
    _require_nonempty_columns(
        calculated_edges,
        _EDGE_SCOPE_COLUMNS,
        label="Calculated edges",
    )
    _require_nonempty_columns(
        filtered_edges,
        _EDGE_SCOPE_COLUMNS,
        label="Filtered edges",
    )

    scoped_predictions = _scope_frame(
        predictions,
        season=season,
        week=week,
    )
    scoped_markets = _scope_frame(
        markets,
        season=season,
        week=week,
    )
    scoped_calculated = _scope_frame(
        calculated_edges,
        season=season,
        week=week,
    )
    scoped_filtered = _scope_frame(
        filtered_edges,
        season=season,
        week=week,
    )

    prediction_game_ids = _game_ids(scoped_predictions)
    market_game_ids = _game_ids(scoped_markets)
    matched_game_ids = prediction_game_ids.intersection(market_game_ids)

    moneyline_count, spread_count, total_count = _complete_market_counts(
        scoped_predictions,
        scoped_markets,
        matched_game_ids,
    )
    eligible_count = moneyline_count + spread_count + total_count
    calculated_count, positive_count, filtered_count = _edge_counts(
        scoped_calculated,
        scoped_filtered,
    )
    provenance = _extract_provenance(
        scoped_predictions,
        scoped_markets,
    )

    blockers: list[EdgeDiagnosticBlocker] = []
    if not prediction_game_ids:
        blockers.append(EdgeDiagnosticBlocker.NO_PREDICTIONS)
    if markets.empty:
        blockers.append(EdgeDiagnosticBlocker.NO_MARKET_DATA)
    elif not market_game_ids:
        blockers.append(EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE)
    if _market_is_stale(
        provenance,
        as_of=as_of,
        max_market_age=max_market_age,
    ):
        blockers.append(EdgeDiagnosticBlocker.MARKET_STALE)
    if prediction_game_ids and market_game_ids and not matched_game_ids:
        blockers.append(EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES)
    if matched_game_ids and eligible_count < len(matched_game_ids) * 3:
        blockers.append(EdgeDiagnosticBlocker.INCOMPLETE_MARKETS)

    if blockers:
        state = EdgeResultState.BLOCKED
    elif calculated_count == 0:
        state = EdgeResultState.NO_CALCULABLE_EDGES
    elif positive_count == 0:
        state = EdgeResultState.NO_POSITIVE_EDGES
    else:
        state = EdgeResultState.POSITIVE_EDGES

    return EdgeDiagnostics(
        season=season,
        week=week,
        prediction_game_count=len(prediction_game_ids),
        market_game_count=len(market_game_ids),
        matched_game_count=len(matched_game_ids),
        complete_moneyline_count=moneyline_count,
        complete_spread_count=spread_count,
        complete_total_count=total_count,
        eligible_market_count=eligible_count,
        calculated_edge_count=calculated_count,
        positive_edge_count=positive_count,
        filtered_edge_count=filtered_count,
        state=state,
        blockers=tuple(blockers),
        provenance=provenance,
    )

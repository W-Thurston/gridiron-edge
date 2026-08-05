# tests/unit/market/test_edge_diagnostics.py

"""Tests for immutable edge diagnostic contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
import json

import pytest

from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeDiagnostics,
    EdgeProvenance,
    EdgeResultState,
)


def _diagnostics(**overrides: object) -> EdgeDiagnostics:
    values: dict[str, object] = {
        "season": "2026-2027",
        "week": 1,
        "prediction_game_count": 2,
        "market_game_count": 2,
        "matched_game_count": 2,
        "complete_moneyline_count": 2,
        "complete_spread_count": 2,
        "complete_total_count": 2,
        "eligible_market_count": 6,
        "calculated_edge_count": 6,
        "positive_edge_count": 2,
        "filtered_edge_count": 2,
        "state": EdgeResultState.POSITIVE_EDGES,
    }
    values.update(overrides)
    return EdgeDiagnostics(**values)  # type: ignore[arg-type]


def test_enum_values_are_stable() -> None:
    assert EdgeDiagnosticBlocker.NO_PREDICTIONS.value == "no_predictions"
    assert EdgeDiagnosticBlocker.NO_MARKET_DATA.value == "no_market_data"
    assert EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE.value == "market_wrong_scope"
    assert EdgeDiagnosticBlocker.MARKET_STALE.value == "market_stale"
    assert EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES.value == "zero_matched_games"
    assert EdgeDiagnosticBlocker.INCOMPLETE_MARKETS.value == "incomplete_markets"
    assert EdgeResultState.BLOCKED.value == "blocked"
    assert EdgeResultState.NO_CALCULABLE_EDGES.value == "no_calculable_edges"
    assert EdgeResultState.NO_POSITIVE_EDGES.value == "no_positive_edges"
    assert EdgeResultState.POSITIVE_EDGES.value == "positive_edges"


def test_contracts_are_frozen() -> None:
    diagnostics = _diagnostics()
    provenance = EdgeProvenance()
    with pytest.raises(FrozenInstanceError):
        diagnostics.week = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        provenance.market_providers = ("changed",)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("season", "", "season must not be empty"),
        ("week", 0, "week must be at least 1"),
        ("prediction_game_count", -1, "must not be negative"),
        ("market_game_count", -1, "must not be negative"),
        ("matched_game_count", -1, "must not be negative"),
        ("eligible_market_count", -1, "must not be negative"),
        ("calculated_edge_count", -1, "must not be negative"),
        ("positive_edge_count", -1, "must not be negative"),
        ("filtered_edge_count", -1, "must not be negative"),
    ],
)
def test_rejects_invalid_scope_or_negative_counts(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _diagnostics(**{field: value})


def test_matched_games_cannot_exceed_predictions() -> None:
    with pytest.raises(ValueError, match="prediction_game_count"):
        _diagnostics(matched_game_count=3)


def test_matched_games_cannot_exceed_markets() -> None:
    with pytest.raises(ValueError, match="market_game_count"):
        _diagnostics(market_game_count=1)


def test_eligible_count_equals_complete_market_sum() -> None:
    with pytest.raises(ValueError, match="eligible_market_count"):
        _diagnostics(eligible_market_count=5)


def test_positive_count_cannot_exceed_calculated_count() -> None:
    with pytest.raises(ValueError, match="positive_edge_count"):
        _diagnostics(calculated_edge_count=1, positive_edge_count=2)


def test_filtered_count_cannot_exceed_positive_count() -> None:
    with pytest.raises(ValueError, match="filtered_edge_count"):
        _diagnostics(filtered_edge_count=3)


def test_blocked_state_requires_blocker() -> None:
    with pytest.raises(ValueError, match="require at least one blocker"):
        _diagnostics(state=EdgeResultState.BLOCKED)


def test_non_blocked_state_rejects_blockers() -> None:
    with pytest.raises(ValueError, match="must not contain blockers"):
        _diagnostics(blockers=(EdgeDiagnosticBlocker.NO_MARKET_DATA,))


def test_duplicate_blockers_are_rejected() -> None:
    blocker = EdgeDiagnosticBlocker.NO_MARKET_DATA
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _diagnostics(
            state=EdgeResultState.BLOCKED,
            blockers=(blocker, blocker),
        )


def test_no_calculable_state_requires_zero_calculated_rows() -> None:
    with pytest.raises(ValueError, match="calculated_edge_count == 0"):
        _diagnostics(state=EdgeResultState.NO_CALCULABLE_EDGES)


def test_no_positive_state_requires_calculated_rows() -> None:
    with pytest.raises(ValueError, match="requires calculated edge rows"):
        _diagnostics(
            state=EdgeResultState.NO_POSITIVE_EDGES,
            calculated_edge_count=0,
            positive_edge_count=0,
            filtered_edge_count=0,
        )


def test_no_positive_state_requires_zero_positive_rows() -> None:
    with pytest.raises(ValueError, match="positive_edge_count == 0"):
        _diagnostics(
            state=EdgeResultState.NO_POSITIVE_EDGES,
            positive_edge_count=1,
            filtered_edge_count=0,
        )


def test_positive_state_requires_positive_rows() -> None:
    with pytest.raises(ValueError, match="requires positive edge rows"):
        _diagnostics(
            positive_edge_count=0,
            filtered_edge_count=0,
        )


def test_valid_terminal_states() -> None:
    blocked = _diagnostics(
        state=EdgeResultState.BLOCKED,
        blockers=(EdgeDiagnosticBlocker.NO_MARKET_DATA,),
        calculated_edge_count=0,
        positive_edge_count=0,
        filtered_edge_count=0,
    )
    no_calculable = _diagnostics(
        state=EdgeResultState.NO_CALCULABLE_EDGES,
        calculated_edge_count=0,
        positive_edge_count=0,
        filtered_edge_count=0,
    )
    no_positive = _diagnostics(
        state=EdgeResultState.NO_POSITIVE_EDGES,
        positive_edge_count=0,
        filtered_edge_count=0,
    )
    assert blocked.state is EdgeResultState.BLOCKED
    assert no_calculable.state is EdgeResultState.NO_CALCULABLE_EDGES
    assert no_positive.state is EdgeResultState.NO_POSITIVE_EDGES


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("win_event_ids", ("b", "a")),
        ("win_run_ids", ("run", "run")),
        ("market_providers", ("z", "a")),
    ],
)
def test_provenance_text_values_require_sorted_unique_tuples(
    field: str,
    values: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="sorted unique"):
        EdgeProvenance(**{field: values})  # type: ignore[arg-type]


def test_provenance_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="must not contain empty values"):
        EdgeProvenance(market_providers=("",))


def test_market_timestamps_require_timezone_aware_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        EdgeProvenance(market_fetched_at=(datetime(2026, 9, 5, 12),))
    with pytest.raises(ValueError, match="must use UTC"):
        EdgeProvenance(
            market_fetched_at=(
                datetime(
                    2026,
                    9,
                    5,
                    12,
                    tzinfo=timezone(timedelta(hours=-4)),
                ),
            )
        )


def test_market_timestamps_require_sorted_unique_values() -> None:
    first = datetime(2026, 9, 5, 12, tzinfo=UTC)
    second = datetime(2026, 9, 5, 13, tzinfo=UTC)
    with pytest.raises(ValueError, match="sorted unique"):
        EdgeProvenance(market_fetched_at=(second, first))


def test_provenance_and_diagnostics_serialize_to_json() -> None:
    timestamp = datetime(2026, 9, 5, 12, tzinfo=UTC)
    provenance = EdgeProvenance(
        win_event_ids=("win-event",),
        win_run_ids=("win-run",),
        win_model_names=("win_prob",),
        win_model_types=("random_forest",),
        total_event_ids=("total-event",),
        total_run_ids=("total-run",),
        total_model_names=("total",),
        total_model_types=("random_forest",),
        product_ids=("product",),
        product_run_ids=("product-run",),
        market_providers=("nflverse",),
        market_sportsbooks=(),
        market_fetched_at=(timestamp,),
    )
    diagnostics = _diagnostics(provenance=provenance)
    encoded = json.dumps(diagnostics.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["state"] == "positive_edges"
    assert decoded["blockers"] == []
    assert decoded["provenance"]["market_fetched_at"] == ["2026-09-05T12:00:00+00:00"]
    assert decoded["provenance"]["market_providers"] == ["nflverse"]


def test_equal_values_produce_equal_contracts() -> None:
    assert _diagnostics() == _diagnostics()
    assert EdgeProvenance() == EdgeProvenance()

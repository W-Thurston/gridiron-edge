"""Tests for weekly readiness domain contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone

import pytest

from gridiron_edge.evaluation.weekly_readiness import (
    WeeklyReadiness,
    WeeklyReadinessBlocker,
)


def _readiness(
    **overrides: object,
) -> WeeklyReadiness:
    """Create one complete weekly readiness result."""
    values: dict[str, object] = {
        "season": "2026-2027",
        "week": 1,
        "scheduled_game_count": 16,
        "selected_win_prediction_count": 16,
        "spread_value_count": 16,
        "total_prediction_count": 16,
        "projected_score_count": 16,
        "complete_provenance_count": 16,
        "market_game_count": 16,
        "prediction_market_match_count": 16,
        "eligible_market_count": 48,
        "positive_edge_count": 4,
        "prediction_generated_at": datetime(
            2026,
            9,
            1,
            12,
            tzinfo=UTC,
        ),
        "market_fetched_at": datetime(
            2026,
            9,
            1,
            13,
            tzinfo=UTC,
        ),
        "market_source": "draftkings",
        "blockers": (),
    }
    values.update(overrides)

    return WeeklyReadiness(**values)  # type: ignore[arg-type]


def test_complete_result_is_ready() -> None:
    result = _readiness()

    assert result.ready
    assert result.has_positive_edges


def test_zero_positive_edges_is_valid_ready_result() -> None:
    result = _readiness(
        positive_edge_count=0,
    )

    assert result.ready
    assert not result.has_positive_edges
    assert result.blockers == ()


def test_blocker_makes_result_not_ready() -> None:
    result = _readiness(
        market_game_count=0,
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        market_fetched_at=None,
        market_source=None,
        blockers=(WeeklyReadinessBlocker.MISSING_MARKET_DATA,),
    )

    assert not result.ready
    assert result.blockers == (WeeklyReadinessBlocker.MISSING_MARKET_DATA,)


def test_partial_prediction_coverage_is_representable() -> None:
    result = _readiness(
        selected_win_prediction_count=15,
        blockers=(WeeklyReadinessBlocker.PARTIAL_WIN_PREDICTION_COVERAGE,),
    )

    assert result.scheduled_game_count == 16
    assert result.selected_win_prediction_count == 15
    assert not result.ready


def test_no_predictions_is_distinct_from_no_market_data() -> None:
    no_predictions = _readiness(
        selected_win_prediction_count=0,
        spread_value_count=0,
        total_prediction_count=0,
        projected_score_count=0,
        complete_provenance_count=0,
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        prediction_generated_at=None,
        blockers=(WeeklyReadinessBlocker.MISSING_WIN_PREDICTIONS,),
    )
    no_market = _readiness(
        market_game_count=0,
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        market_fetched_at=None,
        market_source=None,
        blockers=(WeeklyReadinessBlocker.MISSING_MARKET_DATA,),
    )

    assert no_predictions.blockers != no_market.blockers


def test_zero_matches_is_distinct_from_incomplete_markets() -> None:
    zero_matches = _readiness(
        prediction_market_match_count=0,
        eligible_market_count=0,
        positive_edge_count=0,
        blockers=(WeeklyReadinessBlocker.ZERO_PREDICTION_MARKET_MATCHES,),
    )
    incomplete = _readiness(
        eligible_market_count=0,
        positive_edge_count=0,
        blockers=(WeeklyReadinessBlocker.INCOMPLETE_MARKETS,),
    )

    assert zero_matches.blockers != incomplete.blockers


@pytest.mark.parametrize(
    "field_name",
    [
        "scheduled_game_count",
        "selected_win_prediction_count",
        "spread_value_count",
        "total_prediction_count",
        "projected_score_count",
        "complete_provenance_count",
        "market_game_count",
        "prediction_market_match_count",
        "eligible_market_count",
        "positive_edge_count",
    ],
)
def test_rejects_negative_counts(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must not be negative",
    ):
        _readiness(
            **{field_name: -1},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "selected_win_prediction_count",
        "spread_value_count",
        "total_prediction_count",
        "projected_score_count",
        "complete_provenance_count",
        "market_game_count",
        "prediction_market_match_count",
    ],
)
def test_game_coverage_cannot_exceed_schedule(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=(f"{field_name} must not exceed scheduled_game_count"),
    ):
        _readiness(
            **{field_name: 17},
        )


def test_eligible_markets_may_exceed_game_count() -> None:
    result = _readiness(
        eligible_market_count=96,
        positive_edge_count=10,
    )

    assert result.eligible_market_count == 96


def test_positive_edges_cannot_exceed_eligible_markets() -> None:
    with pytest.raises(
        ValueError,
        match=("positive_edge_count must not exceed eligible_market_count"),
    ):
        _readiness(
            eligible_market_count=2,
            positive_edge_count=3,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "prediction_generated_at",
        "market_fetched_at",
    ],
)
def test_artifact_timestamp_requires_timezone(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be timezone-aware UTC",
    ):
        _readiness(
            **{
                field_name: datetime(
                    2026,
                    9,
                    1,
                    12,
                )
            },
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "prediction_generated_at",
        "market_fetched_at",
    ],
)
def test_artifact_timestamp_requires_utc(
    field_name: str,
) -> None:
    mountain_time = timezone(timedelta(hours=-6))

    with pytest.raises(
        ValueError,
        match=f"{field_name} must use UTC",
    ):
        _readiness(
            **{
                field_name: datetime(
                    2026,
                    9,
                    1,
                    12,
                    tzinfo=mountain_time,
                )
            },
        )


def test_nullable_provenance_is_allowed() -> None:
    result = _readiness(
        prediction_generated_at=None,
        market_fetched_at=None,
        market_source=None,
    )

    assert result.prediction_generated_at is None
    assert result.market_fetched_at is None
    assert result.market_source is None


def test_rejects_empty_market_source() -> None:
    with pytest.raises(
        ValueError,
        match="market_source must not be empty",
    ):
        _readiness(
            market_source=" ",
        )


def test_rejects_duplicate_blockers() -> None:
    blocker = WeeklyReadinessBlocker.MISSING_MARKET_DATA

    with pytest.raises(
        ValueError,
        match="blockers must not contain duplicate values",
    ):
        _readiness(
            blockers=(
                blocker,
                blocker,
            ),
        )


def test_rejects_invalid_scope() -> None:
    with pytest.raises(
        ValueError,
        match="season must not be empty",
    ):
        _readiness(
            season=" ",
        )

    with pytest.raises(
        ValueError,
        match="week must be at least 1",
    ):
        _readiness(
            week=0,
        )


def test_result_is_immutable() -> None:
    result = _readiness()

    with pytest.raises(FrozenInstanceError):
        result.week = 2  # type: ignore[misc]

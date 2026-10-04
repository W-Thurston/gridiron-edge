# tests/unit/market/test_market_freshness.py

"""Tests for the market age limit and the clock every edge boundary uses."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gridiron_edge.market.market_freshness import (
    MAX_MARKET_AGE,
    format_age,
    freshness_inputs,
    market_age_label,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def test_the_limit_is_twenty_four_hours() -> None:
    assert timedelta(hours=24) == MAX_MARKET_AGE


def test_inputs_pair_the_clock_with_the_limit() -> None:
    inputs = freshness_inputs(now=NOW)

    assert inputs.as_of == NOW
    assert inputs.max_market_age == MAX_MARKET_AGE


def test_inputs_default_to_the_current_utc_time() -> None:
    before = datetime.now(UTC)
    inputs = freshness_inputs()

    assert before <= inputs.as_of <= datetime.now(UTC)


def test_a_naive_clock_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        freshness_inputs(now=datetime(2026, 10, 4, 12, 0))


@pytest.mark.parametrize(
    ("age", "text"),
    [
        (timedelta(minutes=5), "5 min"),
        (timedelta(hours=3, minutes=30), "3.5 h"),
        (timedelta(hours=47), "47.0 h"),
        (timedelta(hours=76, minutes=48), "3.2 days"),
        (timedelta(seconds=-10), "0 min"),
    ],
)
def test_ages_are_rendered_for_people(age: timedelta, text: str) -> None:
    assert format_age(age) == text


def test_the_label_says_whether_the_snapshot_is_over_the_limit() -> None:
    assert market_age_label(NOW - timedelta(hours=3), now=NOW) == "3.0 h old, within the 24 h limit"
    assert market_age_label(NOW - timedelta(hours=24), now=NOW).endswith("within the 24 h limit")
    assert market_age_label(NOW - timedelta(hours=24, minutes=1), now=NOW).endswith(
        "over the 24 h limit"
    )
    assert market_age_label(None, now=NOW) == "unknown"

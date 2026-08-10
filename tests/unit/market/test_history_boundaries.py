"""Tests for leakage-safe historical quote boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS, empty_quote_frame
from gridiron_edge.market.history_boundaries import (
    QuoteBoundaryStatus,
    QuoteHistoryBoundary,
    SelectedQuoteObservation,
    select_quote_history_boundaries,
)

KICKOFF = "2026-09-10T00:20:00Z"


def _row(
    *,
    fetched_at: str = "2026-09-01T12:00:00Z",
    provider: str = "the_odds_api",
    provider_event_id: str | None = "event-1",
    sportsbook: str | None = "fanduel",
    game_id: str = "2026_01_KC_LAC",
    market: str = "spread",
    side: str = "away",
    odds: float | None = -110.0,
    line: float | None = 3.5,
    sportsbook_updated_at: str | None = "2026-09-01T11:59:00Z",
    commence_time: str | None = KICKOFF,
    is_live: bool = False,
) -> dict[str, object]:
    """Build one canonical quote observation."""
    return {
        "fetched_at": pd.Timestamp(fetched_at),
        "provider": provider,
        "provider_event_id": provider_event_id,
        "sportsbook": sportsbook,
        "sportsbook_updated_at": pd.Timestamp(sportsbook_updated_at)
        if sportsbook_updated_at is not None
        else pd.NaT,
        "commence_time": pd.Timestamp(commence_time) if commence_time is not None else pd.NaT,
        "is_live": is_live,
        "season": "2026-2027",
        "week": 1,
        "game_id": game_id,
        "game_date": "2026-09-10",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "market": market,
        "side": side,
        "odds": odds,
        "line": line,
    }


def _frame(*rows: dict[str, object]) -> DataFrame:
    """Build canonical historical observations."""
    return DataFrame(rows, columns=list(QUOTE_COLUMNS))


def test_empty_history_returns_no_boundaries() -> None:
    """Empty observations have no historical identities."""
    assert select_quote_history_boundaries(empty_quote_frame()) == ()


def test_one_fetch_selects_both_boundaries_without_repeated_evidence() -> None:
    """One eligible observation is visible but temporally shallow."""
    result = select_quote_history_boundaries(_frame(_row()))
    assert len(result) == 1
    boundary = result[0]
    assert boundary.status is QuoteBoundaryStatus.AVAILABLE
    assert boundary.observation_count == 1
    assert boundary.distinct_fetch_count == 1
    assert not boundary.repeated_observation_evidence_available
    assert boundary.latest_eligible_pregame == boundary.earliest_observed


def test_latest_eligible_excludes_live_and_post_kickoff_rows() -> None:
    """Only non-live observations strictly before kickoff are eligible."""
    result = select_quote_history_boundaries(
        _frame(
            _row(fetched_at="2026-09-01T12:00:00Z", line=3.0),
            _row(fetched_at="2026-09-09T23:00:00Z", line=3.5),
            _row(fetched_at="2026-09-09T23:30:00Z", line=4.0, is_live=True),
            _row(fetched_at=KICKOFF, line=4.5),
            _row(fetched_at="2026-09-10T00:30:00Z", line=5.0),
        )
    )
    boundary = result[0]
    assert boundary.status is QuoteBoundaryStatus.AVAILABLE
    assert boundary.latest_eligible_pregame is not None
    assert boundary.latest_eligible_pregame.line == 3.5
    assert boundary.latest_eligible_pregame.fetched_at == datetime(
        2026,
        9,
        9,
        23,
        tzinfo=UTC,
    )


def test_missing_kickoff_is_explicit() -> None:
    """Date-only game data cannot replace a missing kickoff timestamp."""
    boundary = select_quote_history_boundaries(
        _frame(
            _row(
                provider="nflverse",
                provider_event_id=None,
                sportsbook=None,
                sportsbook_updated_at=None,
                commence_time=None,
            )
        )
    )[0]
    assert boundary.status is QuoteBoundaryStatus.KICKOFF_UNAVAILABLE
    assert boundary.latest_eligible_pregame is None
    assert boundary.earliest_observed.commence_time is None


def test_conflicting_kickoffs_are_explicit() -> None:
    """One identity cannot silently choose between conflicting kickoffs."""
    boundary = select_quote_history_boundaries(
        _frame(
            _row(),
            _row(
                fetched_at="2026-09-01T13:00:00Z",
                commence_time="2026-09-10T01:20:00Z",
            ),
        )
    )[0]
    assert boundary.status is QuoteBoundaryStatus.KICKOFF_CONFLICT
    assert boundary.latest_eligible_pregame is None


def test_no_eligible_pregame_observation_preserves_earliest() -> None:
    """Unavailable pregame selection does not erase observed history."""
    boundary = select_quote_history_boundaries(
        _frame(
            _row(fetched_at=KICKOFF),
            _row(fetched_at="2026-09-10T00:30:00Z"),
        )
    )[0]
    assert boundary.status is QuoteBoundaryStatus.NO_ELIGIBLE_PREGAME_OBSERVATION
    assert boundary.latest_eligible_pregame is None
    assert boundary.earliest_observed.fetched_at == datetime(
        2026,
        9,
        10,
        0,
        20,
        tzinfo=UTC,
    )


def test_identity_and_selected_fields_are_preserved() -> None:
    """Boundary results retain source identity and exact quote evidence."""
    boundary = select_quote_history_boundaries(
        _frame(
            _row(
                odds=-105.0,
                line=4.0,
                sportsbook_updated_at="2026-09-01T11:58:00Z",
            )
        )
    )[0]
    assert boundary.provider == "the_odds_api"
    assert boundary.provider_event_id == "event-1"
    assert boundary.sportsbook == "fanduel"
    assert boundary.game_id == "2026_01_KC_LAC"
    assert boundary.market == "spread"
    assert boundary.side == "away"
    selected = boundary.earliest_observed
    assert selected.odds == -105.0
    assert selected.line == 4.0
    assert selected.sportsbook_updated_at == datetime(
        2026,
        9,
        1,
        11,
        58,
        tzinfo=UTC,
    )
    assert selected.commence_time == datetime(
        2026,
        9,
        10,
        0,
        20,
        tzinfo=UTC,
    )
    assert not selected.is_live


def test_consensus_and_sportsbook_histories_remain_separate() -> None:
    """Nullable consensus identity never merges into a sportsbook history."""
    rows = _frame(
        _row(),
        _row(
            provider="nflverse",
            provider_event_id=None,
            sportsbook=None,
            sportsbook_updated_at=None,
            commence_time=None,
        ),
    )
    result = select_quote_history_boundaries(rows)
    assert len(result) == 2
    assert {(item.provider, item.sportsbook) for item in result} == {
        ("nflverse", None),
        ("the_odds_api", "fanduel"),
    }


def test_repeated_unchanged_observations_preserve_fetch_depth() -> None:
    """Repeated temporal evidence does not claim market movement."""
    boundary = select_quote_history_boundaries(
        _frame(
            _row(),
            _row(fetched_at="2026-09-01T13:00:00Z"),
        )
    )[0]
    assert boundary.observation_count == 2
    assert boundary.distinct_fetch_count == 2
    assert boundary.repeated_observation_evidence_available


def test_result_order_is_input_independent() -> None:
    """Historical identities return in deterministic source-aware order."""
    rows = _frame(
        _row(sportsbook="fanduel"),
        _row(sportsbook="draftkings"),
        _row(
            provider="nflverse",
            provider_event_id=None,
            sportsbook=None,
            sportsbook_updated_at=None,
            commence_time=None,
        ),
    )
    first = select_quote_history_boundaries(rows)
    reversed_rows = DataFrame(rows.iloc[::-1].reset_index(drop=True))

    second = select_quote_history_boundaries(reversed_rows)
    assert first == second


def test_contracts_are_frozen() -> None:
    """Boundary and selected-observation contracts are immutable."""
    boundary = select_quote_history_boundaries(_frame(_row()))[0]
    with pytest.raises(FrozenInstanceError):
        # pyrefly: ignore [read-only]
        boundary.status = QuoteBoundaryStatus.KICKOFF_UNAVAILABLE
    with pytest.raises(FrozenInstanceError):
        # pyrefly: ignore [read-only]
        boundary.earliest_observed.line = 7.0
    assert isinstance(boundary, QuoteHistoryBoundary)
    assert isinstance(boundary.earliest_observed, SelectedQuoteObservation)

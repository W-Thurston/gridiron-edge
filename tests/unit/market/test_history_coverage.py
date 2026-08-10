"""Tests for historical quote coverage diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS, empty_quote_frame
from gridiron_edge.market.history_coverage import (
    QuoteHistoryCoverage,
    evaluate_quote_history_coverage,
)


def _row(
    *,
    fetched_at: str = "2026-09-01T12:00:00Z",
    provider: str = "the_odds_api",
    provider_event_id: str | None = "event-1",
    sportsbook: str | None = "fanduel",
    is_live: bool = False,
    commence_time: str | None = "2026-09-10T00:20:00Z",
) -> dict[str, object]:
    """Build one canonical historical quote observation."""
    return {
        "fetched_at": pd.Timestamp(fetched_at),
        "provider": provider,
        "provider_event_id": provider_event_id,
        "sportsbook": sportsbook,
        "sportsbook_updated_at": pd.Timestamp("2026-09-01T11:59:00Z")
        if sportsbook is not None
        else pd.NaT,
        "commence_time": pd.Timestamp(commence_time) if commence_time is not None else pd.NaT,
        "is_live": is_live,
        "season": "2026-2027",
        "week": 1,
        "game_id": "2026_01_KC_LAC",
        "game_date": "2026-09-10",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "market": "moneyline",
        "side": "away",
        "odds": 125.0,
        "line": None,
    }


def _frame(*rows: dict[str, object]) -> DataFrame:
    """Build canonical quote rows."""
    return DataFrame(rows, columns=list(QUOTE_COLUMNS))


def test_empty_history_has_zero_coverage() -> None:
    """Empty evidence reports no temporal depth."""
    result = evaluate_quote_history_coverage(empty_quote_frame())
    assert result.row_count == 0
    assert result.earliest_fetched_at is None
    assert result.latest_fetched_at is None
    assert not result.repeated_observation_evidence_available


def test_single_observation_has_no_repeated_evidence() -> None:
    """One observation is not temporal history for its identity."""
    result = evaluate_quote_history_coverage(_frame(_row()))
    assert result.row_count == 1
    assert result.market_identity_count == 1
    assert result.identities_with_multiple_fetches == 0
    assert result.maximum_fetches_per_identity == 1
    assert not result.repeated_observation_evidence_available


def test_repeated_unchanged_observation_is_temporal_evidence() -> None:
    """A later fetch proves temporal depth without claiming movement."""
    result = evaluate_quote_history_coverage(
        _frame(
            _row(),
            _row(fetched_at="2026-09-01T13:00:00Z"),
        )
    )
    assert result.identities_with_multiple_observations == 1
    assert result.identities_with_multiple_fetches == 1
    assert result.maximum_observations_per_identity == 2
    assert result.maximum_fetches_per_identity == 2
    assert result.repeated_observation_evidence_available


def test_multi_source_live_and_missing_kickoff_coverage() -> None:
    """Source-neutral and live evidence remain explicit and separate."""
    result = evaluate_quote_history_coverage(
        _frame(
            _row(),
            _row(
                provider="nflverse",
                provider_event_id=None,
                sportsbook=None,
                is_live=True,
                commence_time=None,
            ),
        )
    )
    assert result.provider_count == 2
    assert result.sportsbook_count == 1
    assert result.market_identity_count == 2
    assert result.pregame_observation_count == 1
    assert result.live_observation_count == 1
    assert result.missing_commence_time_count == 1


def test_result_contract_is_frozen() -> None:
    """Coverage results are immutable."""
    result = evaluate_quote_history_coverage(_frame(_row()))
    with pytest.raises(FrozenInstanceError):
        # pyrefly: ignore [read-only]
        result.row_count = 2


def test_result_type_is_public() -> None:
    """The evaluator returns the documented public contract."""
    result = evaluate_quote_history_coverage(_frame(_row()))
    assert isinstance(result, QuoteHistoryCoverage)

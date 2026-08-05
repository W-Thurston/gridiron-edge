"""Tests for The Odds API v4 pure payload parser."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS
from gridiron_edge.ingest.odds.the_odds_api import (
    OddsPayloadError,
    parse_the_odds_api_payload,
)

SEASON = "2026-2027"
WEEK = 1
FETCHED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)


def _schedule() -> DataFrame:
    return DataFrame(
        [
            {
                "season": SEASON,
                "week": WEEK,
                "game_id": "2026_01_KC_LAC",
                "game_date": "2026-09-10",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
            }
        ]
    )


def _market(key: str, outcomes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "key": key,
        "last_update": "2026-09-01T11:59:30Z",
        "outcomes": outcomes,
    }


def _payload() -> list[dict[str, object]]:
    markets = [
        _market(
            "h2h",
            [
                {"name": "Kansas City Chiefs", "price": 125},
                {"name": "Los Angeles Chargers", "price": -145},
            ],
        ),
        _market(
            "spreads",
            [
                {"name": "Kansas City Chiefs", "price": -110, "point": 2.5},
                {"name": "Los Angeles Chargers", "price": -110, "point": -2.5},
            ],
        ),
        _market(
            "totals",
            [
                {"name": "Over", "price": -105, "point": 45.5},
                {"name": "Under", "price": -115, "point": 45.5},
            ],
        ),
    ]
    return [
        {
            "id": "provider-event-1",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2026-09-10T00:20:00Z",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "last_update": "2026-09-01T11:59:00Z",
                    "markets": markets,
                },
                {
                    "key": "fanduel",
                    "last_update": "2026-09-01T11:58:00Z",
                    "markets": markets,
                },
            ],
        }
    ]


def _parse(payload: object | None = None) -> DataFrame:
    return parse_the_odds_api_payload(
        _payload() if payload is None else payload,
        _schedule(),
        season=SEASON,
        week=WEEK,
        fetched_at=FETCHED_AT,
    )


def test_parses_all_supported_markets_for_every_book() -> None:
    result = _parse()
    assert tuple(result.columns) == QUOTE_COLUMNS
    assert len(result) == 12
    assert result["game_id"].unique().tolist() == ["2026_01_KC_LAC"]
    assert set(result["provider"]) == {"the_odds_api"}
    assert set(result["provider_event_id"]) == {"provider-event-1"}
    assert set(result["sportsbook"]) == {"draftkings", "fanduel"}
    assert not result["is_live"].any()


def test_market_and_side_mapping_preserves_provider_values() -> None:
    result = _parse()
    draftkings = result.loc[result["sportsbook"] == "draftkings"]
    values = {
        (row.market, row.side): (row.odds, row.line) for row in draftkings.itertuples(index=False)
    }
    assert values[("moneyline", "away")] == (125.0, pytest.approx(float("nan"), nan_ok=True))
    assert values[("moneyline", "home")][0] == -145.0
    assert values[("spread", "away")] == (-110.0, 2.5)
    assert values[("spread", "home")] == (-110.0, -2.5)
    assert values[("total", "over")] == (-105.0, 45.5)
    assert values[("total", "under")] == (-115.0, 45.5)


def test_uses_market_update_and_bookmaker_fallback() -> None:
    payload = _payload()
    bookmaker = payload[0]["bookmakers"][0]
    assert isinstance(bookmaker, dict)
    markets = bookmaker["markets"]
    assert isinstance(markets, list)
    markets[0].pop("last_update")
    result = _parse(payload)
    moneyline = result.loc[
        (result["sportsbook"] == "draftkings") & (result["market"] == "moneyline")
    ]
    spread = result.loc[(result["sportsbook"] == "draftkings") & (result["market"] == "spread")]
    assert moneyline["sportsbook_updated_at"].unique().tolist() == [
        pd.Timestamp("2026-09-01T11:59:00Z")
    ]
    assert spread["sportsbook_updated_at"].unique().tolist() == [
        pd.Timestamp("2026-09-01T11:59:30Z")
    ]


def test_excludes_unmatched_wrong_sport_and_started_events() -> None:
    unmatched = _payload()[0]
    unmatched["id"] = "unmatched"
    unmatched["away_team"] = "Buffalo Bills"
    unmatched["home_team"] = "Miami Dolphins"

    wrong_sport = _payload()[0]
    wrong_sport["id"] = "wrong-sport"
    wrong_sport["sport_key"] = "baseball_mlb"

    started = _payload()[0]
    started["id"] = "started"
    started["commence_time"] = "2026-09-01T11:00:00Z"

    assert _parse([unmatched, wrong_sport, started]).empty


def test_normalizes_team_whitespace_and_case_for_matching() -> None:
    payload = _payload()
    payload[0]["away_team"] = "  KANSAS   CITY chiefs "
    payload[0]["home_team"] = "los angeles CHARGERS"
    result = _parse(payload)
    assert len(result) == 12
    assert set(result["away_team"]) == {"Kansas City Chiefs"}
    assert set(result["home_team"]) == {"Los Angeles Chargers"}


def test_ignores_unsupported_market_without_collapsing_supported_rows() -> None:
    payload = _payload()
    bookmaker = payload[0]["bookmakers"][0]
    assert isinstance(bookmaker, dict)
    markets = bookmaker["markets"]
    assert isinstance(markets, list)
    markets.append(_market("outrights", [{"name": "KC", "price": 500}]))
    result = _parse(payload)
    assert len(result) == 12


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload[0].update(id=""), "event id"),
        (lambda payload: payload[0].update(commence_time="invalid"), "commence_time"),
        (
            lambda payload: payload[0]["bookmakers"][0].update(key=""),
            "bookmaker key",
        ),
        (
            lambda payload: payload[0]["bookmakers"][0]["markets"][0]["outcomes"][0].update(
                price=0
            ),
            "must not be zero",
        ),
    ],
)
def test_rejects_invalid_supported_payload_values(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(OddsPayloadError, match=message):
        _parse(payload)


def test_rejects_non_array_payload() -> None:
    with pytest.raises(OddsPayloadError, match="payload must be an array"):
        _parse({"events": []})


def test_rejects_missing_schedule_columns() -> None:
    with pytest.raises(ValueError, match="game_id"):
        parse_the_odds_api_payload(
            _payload(),
            _schedule().drop(columns="game_id"),
            season=SEASON,
            week=WEEK,
            fetched_at=FETCHED_AT,
        )


def test_inputs_are_not_mutated() -> None:
    payload = _payload()
    schedule = _schedule()
    expected_payload = repr(payload)
    expected_schedule = schedule.copy(deep=True)
    parse_the_odds_api_payload(
        payload,
        schedule,
        season=SEASON,
        week=WEEK,
        fetched_at=FETCHED_AT,
    )
    assert repr(payload) == expected_payload
    pd.testing.assert_frame_equal(schedule, expected_schedule)

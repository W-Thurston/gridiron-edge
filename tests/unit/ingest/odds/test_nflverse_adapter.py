# tests/unit/ingest/odds/test_nflverse_adapter.py

"""Tests for nflverse rich-schedule market adaptation."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.nflverse_schedule import (
    MARKET_COLUMNS,
    NFLVERSE_SCHEDULE_SOURCE,
    adapt_nflverse_schedule_markets,
)


def _schedule() -> DataFrame:
    timestamp = datetime(2026, 7, 30, 18, tzinfo=UTC)
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027", "2026-2027"],
            "week": [1, 1, 2],
            "game_id": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
                "2026_02_GB_CHI",
            ],
            "game_date": ["2026-09-10", "2026-09-13", "2026-09-20"],
            "away_team": [
                "Kansas City Chiefs",
                "Baltimore Ravens",
                "Green Bay Packers",
            ],
            "home_team": [
                "Los Angeles Chargers",
                "Buffalo Bills",
                "Chicago Bears",
            ],
            "away_moneyline": [-120.0, None, -105.0],
            "home_moneyline": [105.0, None, -115.0],
            "spread_line": [-2.5, None, -1.0],
            "away_spread_odds": [-110.0, None, -108.0],
            "home_spread_odds": [-110.0, None, -112.0],
            "total_line": [45.5, None, 42.0],
            "over_odds": [-110.0, None, -105.0],
            "under_odds": [-110.0, None, -115.0],
            "source": ["nflverse", "nflverse", "nflverse"],
            "ingested_at": [timestamp, timestamp, timestamp],
        }
    )


def test_labels_source_without_draftkings() -> None:
    result = adapt_nflverse_schedule_markets(
        _schedule(),
        season="2026-2027",
        week=1,
    )

    assert set(result["sportsbook"]) == {NFLVERSE_SCHEDULE_SOURCE}
    assert "draftkings" not in set(result["sportsbook"])


def test_preserves_ingestion_timestamp_as_fetched_at() -> None:
    result = adapt_nflverse_schedule_markets(
        _schedule(),
        season="2026-2027",
        week=1,
    )

    expected = pd.Timestamp("2026-07-30T18:00:00Z")
    assert result["fetched_at"].nunique() == 1
    assert result["fetched_at"].iloc[0] == expected


def test_normalizes_all_moneyline_spread_and_total_sides() -> None:
    result = adapt_nflverse_schedule_markets(
        _schedule(),
        season="2026-2027",
        week=1,
    )
    game = result.loc[result["game_id"] == "2026_01_KC_LAC"].reset_index(drop=True)

    assert list(zip(game["market"], game["side"], strict=True)) == [
        ("moneyline", "away"),
        ("moneyline", "home"),
        ("spread", "away"),
        ("spread", "home"),
        ("total", "over"),
        ("total", "under"),
    ]
    assert game["odds"].tolist() == [-120.0, 105.0, -110.0, -110.0, -110.0, -110.0]
    assert pd.isna(game.loc[0, "line"])
    assert pd.isna(game.loc[1, "line"])
    assert game.loc[2, "line"] == pytest.approx(2.5)
    assert game.loc[3, "line"] == pytest.approx(-2.5)
    assert game.loc[4, "line"] == pytest.approx(45.5)
    assert game.loc[5, "line"] == pytest.approx(45.5)


def test_incomplete_markets_remain_as_six_explicit_rows() -> None:
    result = adapt_nflverse_schedule_markets(
        _schedule(),
        season="2026-2027",
        week=1,
    )
    missing = result.loc[result["game_id"] == "2026_01_BAL_BUF"]

    assert len(missing) == 6
    assert missing["odds"].isna().all()
    assert missing["line"].isna().all()


def test_preserves_canonical_game_ids_and_team_orientation() -> None:
    result = adapt_nflverse_schedule_markets(
        _schedule(),
        season="2026-2027",
        week=1,
    )

    assert set(result["game_id"]) == {
        "2026_01_KC_LAC",
        "2026_01_BAL_BUF",
    }
    kc = result.loc[result["game_id"] == "2026_01_KC_LAC"].iloc[0]
    assert kc["away_team"] == "Kansas City Chiefs"
    assert kc["home_team"] == "Los Angeles Chargers"


def test_enforces_requested_season_and_week_scope() -> None:
    result = adapt_nflverse_schedule_markets(
        _schedule(),
        season="2026-2027",
        week=1,
    )

    assert set(result["week"]) == {1}
    assert "2026_02_GB_CHI" not in set(result["game_id"])
    assert len(result) == 12


def test_rejects_duplicate_game_ids_in_requested_scope() -> None:
    schedule = pd.concat([_schedule(), _schedule().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate game IDs"):
        adapt_nflverse_schedule_markets(
            schedule,
            season="2026-2027",
            week=1,
        )


def test_rejects_non_utc_ingestion_timestamp() -> None:
    schedule = _schedule()
    schedule["ingested_at"] = pd.Timestamp("2026-07-30T18:00:00-04:00")

    with pytest.raises(ValueError, match="must use UTC"):
        adapt_nflverse_schedule_markets(
            schedule,
            season="2026-2027",
            week=1,
        )


def test_rejects_mixed_ingestion_timestamps() -> None:
    schedule = _schedule()
    schedule.loc[1, "ingested_at"] = pd.Timestamp("2026-07-30T19:00:00Z")

    with pytest.raises(ValueError, match="one ingestion timestamp"):
        adapt_nflverse_schedule_markets(
            schedule,
            season="2026-2027",
            week=1,
        )


def test_rejects_non_nflverse_source() -> None:
    schedule = _schedule()
    schedule.loc[0, "source"] = "draftkings"

    with pytest.raises(ValueError, match="requires source 'nflverse'"):
        adapt_nflverse_schedule_markets(
            schedule,
            season="2026-2027",
            week=1,
        )


def test_empty_requested_scope_returns_canonical_schema() -> None:
    result = adapt_nflverse_schedule_markets(
        _schedule(),
        season="2026-2027",
        week=9,
    )

    assert result.empty
    assert tuple(result.columns) == MARKET_COLUMNS

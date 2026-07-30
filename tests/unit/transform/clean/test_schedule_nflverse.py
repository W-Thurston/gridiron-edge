# tests/unit/transform/clean/test_schedule_nflverse.py

"""Tests for rich and focused nflverse upcoming-schedule cleaning."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.loaders import (
    load_schedule_upcoming,
    load_schedule_upcoming_rich,
)
from gridiron_edge.datasets.writers import write_parquet
from gridiron_edge.transform.clean.schedule_nflverse import (
    ELO_UPCOMING_COLUMNS,
    RICH_UPCOMING_COLUMNS,
    build_elo_upcoming_schedule,
    build_rich_upcoming_schedule,
    clean_nflverse_upcoming,
)


def _raw_upcoming() -> DataFrame:
    """Create source rows with complete and missing optional fields."""
    return DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 1, 2],
            "game_type": ["REG", "REG", "REG"],
            "game_id": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
                "2026_02_GB_CHI",
            ],
            "weekday": [
                "Sunday",
                "Sunday",
                "Thursday",
            ],
            "gameday": [
                "2026-09-06",
                "2026-09-06",
                "2026-09-10",
            ],
            "gametime": [
                "20:20",
                "13:00",
                "20:15",
            ],
            "away_team": ["KC", "BAL", "GB"],
            "home_team": ["LAC", "BUF", "CHI"],
            "result": [None, None, None],
            "location": [
                "Home",
                "Neutral",
                None,
            ],
            "stadium": [
                "SoFi Stadium",
                None,
                "Soldier Field",
            ],
            "roof": [
                "dome",
                None,
                "outdoors",
            ],
            "surface": [
                "turf",
                None,
                "grass",
            ],
            "div_game": [1, 0, None],
            "away_rest": [7, 7, None],
            "home_rest": [7, 7, None],
            "away_moneyline": [-120, None, None],
            "home_moneyline": [105, None, None],
            "spread_line": [-2.5, None, None],
            "away_spread_odds": [-110, None, None],
            "home_spread_odds": [-110, None, None],
            "total_line": [45.5, None, None],
            "over_odds": [-110, None, None],
            "under_odds": [-110, None, None],
        }
    )


def test_every_upcoming_source_row_survives() -> None:
    raw = _raw_upcoming()

    rich = build_rich_upcoming_schedule(
        raw,
        ingested_at=datetime(
            2026,
            7,
            30,
            18,
            tzinfo=UTC,
        ),
    )

    assert len(rich) == len(raw)
    assert set(rich["game_id"]) == set(raw["game_id"])


def test_missing_markets_do_not_drop_games() -> None:
    rich = build_rich_upcoming_schedule(
        _raw_upcoming(),
        ingested_at=datetime(
            2026,
            7,
            30,
            18,
            tzinfo=UTC,
        ),
    )

    missing_market_game = rich.loc[rich["game_id"] == "2026_01_BAL_BUF"].iloc[0]

    assert pd.isna(missing_market_game["away_moneyline"])
    assert pd.isna(missing_market_game["spread_line"])
    assert pd.isna(missing_market_game["total_line"])


def test_rich_schema_and_provenance_are_stable() -> None:
    timestamp = datetime(
        2026,
        7,
        30,
        18,
        tzinfo=UTC,
    )

    rich = build_rich_upcoming_schedule(
        _raw_upcoming(),
        ingested_at=timestamp,
    )

    assert list(rich.columns) == list(RICH_UPCOMING_COLUMNS)
    assert set(rich["source"]) == {"nflverse"}
    assert rich["ingested_at"].nunique() == 1
    assert rich["ingested_at"].iloc[0] == pd.Timestamp(timestamp)


def test_team_and_game_identifiers_are_canonical() -> None:
    rich = build_rich_upcoming_schedule(
        _raw_upcoming(),
        ingested_at=datetime(
            2026,
            7,
            30,
            18,
            tzinfo=UTC,
        ),
    )

    row = rich.loc[rich["game_id"] == "2026_01_KC_LAC"].iloc[0]

    assert row["away_team"] == "Kansas City Chiefs"
    assert row["home_team"] == "Los Angeles Chargers"
    assert row["game_id"] == "2026_01_KC_LAC"
    assert row["season"] == "2026-2027"
    assert row["week"] == 1


def test_location_context_is_preserved() -> None:
    rich = build_rich_upcoming_schedule(
        _raw_upcoming(),
        ingested_at=datetime(
            2026,
            7,
            30,
            18,
            tzinfo=UTC,
        ),
    )

    neutral = rich.loc[rich["game_id"] == "2026_01_BAL_BUF"].iloc[0]

    assert neutral["location"] == "Neutral"
    assert bool(neutral["neutral_site"])


def test_focused_schedule_preserves_legacy_schema() -> None:
    rich = build_rich_upcoming_schedule(
        _raw_upcoming(),
        ingested_at=datetime(
            2026,
            7,
            30,
            18,
            tzinfo=UTC,
        ),
    )

    focused = build_elo_upcoming_schedule(rich)

    assert list(focused.columns) == list(ELO_UPCOMING_COLUMNS)
    assert set(focused["GAME_ID"]) == set(rich["game_id"])
    assert len(focused) == len(rich)


def test_empty_source_produces_both_empty_schemas() -> None:
    raw = _raw_upcoming().iloc[0:0].copy()

    rich = build_rich_upcoming_schedule(
        raw,
        ingested_at=datetime(
            2026,
            7,
            30,
            18,
            tzinfo=UTC,
        ),
    )
    focused = build_elo_upcoming_schedule(rich)

    assert rich.empty
    assert focused.empty
    assert list(rich.columns) == list(RICH_UPCOMING_COLUMNS)
    assert list(focused.columns) == list(ELO_UPCOMING_COLUMNS)


def test_ingestion_timestamp_requires_timezone_aware_utc() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware UTC",
    ):
        build_rich_upcoming_schedule(
            _raw_upcoming(),
            ingested_at=datetime(
                2026,
                7,
                30,
                18,
            ),
        )

    mountain_time = timezone(timedelta(hours=-6))

    with pytest.raises(
        ValueError,
        match="must use UTC",
    ):
        build_rich_upcoming_schedule(
            _raw_upcoming(),
            ingested_at=datetime(
                2026,
                7,
                30,
                18,
                tzinfo=mountain_time,
            ),
        )


def test_cleaner_writes_rich_and_focused_registered_outputs(
    tmp_path: Path,
) -> None:
    raw = _raw_upcoming()

    write_parquet(
        tmp_path,
        "schedule_upcoming_raw_nflverse",
        raw,
    )

    focused_path = clean_nflverse_upcoming(
        repo=tmp_path,
        ingested_at=datetime(
            2026,
            7,
            30,
            18,
            tzinfo=UTC,
        ),
    )

    rich = load_schedule_upcoming_rich(tmp_path)
    focused = load_schedule_upcoming(tmp_path)

    assert len(rich) == len(raw)
    assert len(focused) == len(raw)
    assert set(rich["game_id"]) == set(focused["GAME_ID"])
    assert focused_path.exists()

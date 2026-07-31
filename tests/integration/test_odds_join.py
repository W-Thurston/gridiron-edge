# tests/integration/test_odds_join.py

"""Integration: adapt and persist nflverse schedule markets by canonical game ID."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from gridiron_edge.ingest.odds.nflverse_schedule import (
    NFLVERSE_SCHEDULE_SOURCE,
    adapt_nflverse_schedule_markets,
)
from gridiron_edge.ingest.odds.store import (
    load_current_odds,
    write_current_odds_snapshot,
)


def _rich_schedule() -> DataFrame:
    """Return schedule truth with complete, incomplete, and unmatched games."""
    timestamp = datetime(2026, 7, 30, 18, tzinfo=UTC)
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027", "2026-2027"],
            "week": [1, 1, 1],
            "game_id": [
                "2026_01_KC_LAC",
                "2026_01_BAL_BUF",
                "2026_01_GB_CHI",
            ],
            "game_date": ["2026-09-10", "2026-09-13", "2026-09-13"],
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
            "away_moneyline": [-120.0, None, None],
            "home_moneyline": [105.0, None, None],
            "spread_line": [-2.5, None, None],
            "away_spread_odds": [-110.0, None, None],
            "home_spread_odds": [-110.0, None, None],
            "total_line": [45.5, None, None],
            "over_odds": [-110.0, None, None],
            "under_odds": [-110.0, None, None],
            "source": ["nflverse", "nflverse", "nflverse"],
            "ingested_at": [timestamp, timestamp, timestamp],
        }
    )


def test_nflverse_markets_roundtrip_and_join_by_schedule_game_id(
    tmp_path: Path,
) -> None:
    """Persist adapted markets and retain complete and incomplete joins."""
    schedule = _rich_schedule()
    adapted = adapt_nflverse_schedule_markets(
        schedule.iloc[:2].copy(),
        season="2026-2027",
        week=1,
    )

    snapshot_path = write_current_odds_snapshot(
        adapted,
        repo=tmp_path,
    )
    loaded = load_current_odds(repo=tmp_path)

    assert loaded is not None
    assert snapshot_path.name == "odds_current.parquet"
    assert set(loaded["sportsbook"]) == {NFLVERSE_SCHEDULE_SOURCE}
    assert "draftkings" not in set(loaded["sportsbook"])
    assert loaded["fetched_at"].nunique() == 1
    assert loaded["fetched_at"].iloc[0] == pd.Timestamp("2026-07-30T18:00:00Z")

    market_coverage = (
        loaded.groupby("game_id", sort=False)
        .agg(
            market_rows=("market", "size"),
            populated_odds=("odds", "count"),
        )
        .reset_index()
    )
    joined = schedule.merge(
        market_coverage,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    assert len(joined) == len(schedule)
    assert joined["game_id"].tolist() == schedule["game_id"].tolist()

    complete = joined.loc[joined["game_id"] == "2026_01_KC_LAC"].iloc[0]
    incomplete = joined.loc[joined["game_id"] == "2026_01_BAL_BUF"].iloc[0]
    unmatched = joined.loc[joined["game_id"] == "2026_01_GB_CHI"].iloc[0]

    assert complete["market_rows"] == 6
    assert complete["populated_odds"] == 6
    assert incomplete["market_rows"] == 6
    assert incomplete["populated_odds"] == 0
    assert pd.isna(unmatched["market_rows"])
    assert pd.isna(unmatched["populated_odds"])

    assert not (tmp_path / "data" / "odds" / "dk_odds_current.parquet").exists()


def test_loaded_spread_and_total_sides_preserve_normalized_values(
    tmp_path: Path,
) -> None:
    """Round-trip normalized spread orientation and total-side values."""
    adapted = adapt_nflverse_schedule_markets(
        _rich_schedule().iloc[:1].copy(),
        season="2026-2027",
        week=1,
    )
    write_current_odds_snapshot(adapted, repo=tmp_path)
    loaded = load_current_odds(repo=tmp_path)

    assert loaded is not None
    spread = loaded.loc[loaded["market"] == "spread"].set_index("side")
    total = loaded.loc[loaded["market"] == "total"].set_index("side")

    assert spread.loc["away", "line"] == 2.5
    assert spread.loc["home", "line"] == -2.5
    assert spread.loc["away", "odds"] == -110.0
    assert spread.loc["home", "odds"] == -110.0
    assert total.loc["over", "line"] == 45.5
    assert total.loc["under", "line"] == 45.5
    assert total.loc["over", "odds"] == -110.0
    assert total.loc["under", "odds"] == -110.0

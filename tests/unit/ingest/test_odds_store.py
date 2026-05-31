# tests/unit/ingest/test_odds_store.py
"""Tests for gridiron_edge.ingest.odds.store — wide_to_long, ledger, snapshot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame, Timestamp

from gridiron_edge.ingest.odds.store import (
    append_to_odds_ledger,
    load_current_odds,
    load_odds_ledger,
    wide_to_long,
    write_current_odds_snapshot,
)


def _make_long_odds(n: int = 2) -> pd.DataFrame:
    """Build a minimal long-format odds DataFrame matching ledger schema."""
    rows: list[dict[str, Timestamp | float | int | str]] = []
    for i in range(n):
        rows.append(
            {
                "fetched_at": pd.Timestamp("2026-09-10 12:00:00", tz="UTC"),
                "sportsbook": "draftkings",
                "season": "2026-2027",
                "week": 1,
                "game_id": f"2026_01_KC_LAC_{i}",
                "game_date": "2026-09-10",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market": "moneyline",
                "side": "away",
                "odds": -150.0,
                "line": float("nan"),
            }
        )
    return pd.DataFrame(rows)


class TestWideToLong:
    def _make_wide(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "team": ["Kansas City Chiefs", "Los Angeles Chargers"],
                "opponent": ["Los Angeles Chargers", "Kansas City Chiefs"],
                "location": [1, 0],  # 1 = home, 0 = away
                "start_time": ["2026-09-10T20:20:00Z", "2026-09-10T20:20:00Z"],
                "event_id": ["evt_001", "evt_001"],
                "moneyline": [-150, 130],
                "spread_value": [-3.0, 3.0],
                "spread_odds": [-110, -110],
                "total_OU_value": [47.5, 47.5],
                "over_total_odds": [-110, -110],
                "under_total_odds": [-110, -110],
            }
        )

    def test_returns_dataframe(self) -> None:
        result: DataFrame = wide_to_long(
            self._make_wide(),
            sportsbook="draftkings",
            season="2026-2027",
            week=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_long_format_has_required_columns(self) -> None:
        result: DataFrame = wide_to_long(
            self._make_wide(),
            sportsbook="draftkings",
            season="2026-2027",
            week=1,
        )
        required: set[str] = {"sportsbook", "season", "week", "game_id", "market", "side"}
        assert required <= set(result.columns)


class TestAppendToOddsLedger:
    def test_creates_parquet_file(self, tmp_path: Path) -> None:
        df: DataFrame = _make_long_odds()
        result_path: Path = append_to_odds_ledger(df, repo=tmp_path)
        assert result_path.is_file()
        assert result_path.suffix == ".parquet"

    def test_roundtrip_preserves_rows(self, tmp_path: Path) -> None:
        df: DataFrame = _make_long_odds(n=3)
        path: Path = append_to_odds_ledger(df, repo=tmp_path)
        loaded: DataFrame = pd.read_parquet(path)
        assert len(loaded) == 3


class TestWriteCurrentOddsSnapshot:
    def test_creates_parquet_file(self, tmp_path: Path) -> None:
        df: DataFrame = _make_long_odds()
        result_path: Path = write_current_odds_snapshot(df, repo=tmp_path)
        assert result_path.is_file()
        assert result_path.suffix == ".parquet"

    def test_snapshot_overwrites(self, tmp_path: Path) -> None:
        df1: DataFrame = _make_long_odds(n=5)
        df2: DataFrame = _make_long_odds(n=2)
        write_current_odds_snapshot(df1, repo=tmp_path)
        path: Path = write_current_odds_snapshot(df2, repo=tmp_path)
        loaded: DataFrame = pd.read_parquet(path)
        assert len(loaded) == 2  # overwritten, not appended


class TestLoadOdds:
    def test_load_current_returns_none_when_missing(self, tmp_path: Path) -> None:
        result: DataFrame | None = load_current_odds(repo=tmp_path)
        assert result is None

    def test_load_current_roundtrip(self, tmp_path: Path) -> None:
        df: DataFrame = _make_long_odds(n=3)
        write_current_odds_snapshot(df, repo=tmp_path)
        loaded: DataFrame | None = load_current_odds(repo=tmp_path)
        assert loaded is not None
        assert len(loaded) == 3

    def test_load_ledger_returns_empty_when_missing(self, tmp_path: Path) -> None:
        result: DataFrame = load_odds_ledger(repo=tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

# tests/unit/ingest/test_odds_store.py
"""Tests for gridiron_edge.ingest.odds.store - wide_to_long, ledger, snapshot."""

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
        """Wide-format fixture for KC @ LAC.

        Row 0: Los Angeles Chargers at home (location=1).
        Row 1: Kansas City Chiefs as visitors (location=0).
        Canonical game_id is YYYY_WW_AWAY_HOME = "2026_01_KC_LAC".
        """
        return pd.DataFrame(
            {
                "team": ["Los Angeles Chargers", "Kansas City Chiefs"],
                "opponent": ["Kansas City Chiefs", "Los Angeles Chargers"],
                "location": [1, 0],
                "start_time": ["2026-09-10T20:20:00Z", "2026-09-10T20:20:00Z"],
                "event_id": ["evt_001", "evt_001"],
                "moneyline": [130, -150],  # home favored less, away favored more
                "spread_value": [3.0, -3.0],  # home is underdog
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

    def test_game_id_uses_canonical_format(self) -> None:
        """game_id must be YYYY_WW_AWAY_HOME, not event_id-based.

        This is the store/H2 fix from audit_2026_06_18.md.
        """
        result: DataFrame = wide_to_long(
            self._make_wide(),
            sportsbook="draftkings",
            season="2026-2027",
            week=1,
        )
        # All rows for one game should share the same canonical game_id.
        unique_gids = result["game_id"].unique()
        assert len(unique_gids) == 1
        assert unique_gids[0] == "2026_01_KC_LAC"

    def test_skips_unresolvable_team(self) -> None:
        """Games with unknown teams should be skipped, not crash."""
        wide = pd.DataFrame(
            {
                "team": ["Fake Team", "Made Up FC"],
                "opponent": ["Made Up FC", "Fake Team"],
                "location": [1, 0],
                "start_time": ["2026-09-10T20:20:00Z", "2026-09-10T20:20:00Z"],
                "event_id": ["evt_999", "evt_999"],
                "moneyline": [-110, -110],
                "spread_value": [-3.0, 3.0],
                "spread_odds": [-110, -110],
                "total_OU_value": [47.5, 47.5],
                "over_total_odds": [-110, -110],
                "under_total_odds": [-110, -110],
            }
        )
        result: DataFrame = wide_to_long(
            wide,
            sportsbook="draftkings",
            season="2026-2027",
            week=1,
        )
        assert len(result) == 0


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

    def test_dedup_removes_duplicate_pulls(self, tmp_path: Path) -> None:
        """Calling append twice with the same df should NOT double rows.

        This is the store/C1 fix from audit_2026_06_18.md. The original
        df_long.loc[key_cols, :] indexing was silently broken, so dedup
        was a no-op and re-runs would accumulate duplicate rows.
        """
        df: DataFrame = _make_long_odds(n=2)
        append_to_odds_ledger(df, repo=tmp_path)
        path: Path = append_to_odds_ledger(df, repo=tmp_path)

        loaded: DataFrame = pd.read_parquet(path)
        # Both calls used the same fetched_at, so the second call should
        # have replaced the first call's rows, not appended to them.
        assert len(loaded) == 2

    def test_dedup_keeps_distinct_pulls(self, tmp_path: Path) -> None:
        """Two pulls at different fetched_at timestamps should both be retained."""
        df1: DataFrame = _make_long_odds(n=2)
        # Change fetched_at on the second batch
        df2: DataFrame = _make_long_odds(n=2)
        df2["fetched_at"] = pd.Timestamp("2026-09-10 18:00:00", tz="UTC")
        append_to_odds_ledger(df1, repo=tmp_path)
        path: Path = append_to_odds_ledger(df2, repo=tmp_path)

        loaded: DataFrame = pd.read_parquet(path)
        # Different fetched_at means both batches should persist
        assert len(loaded) == 4

    def test_empty_df_short_circuits(self, tmp_path: Path) -> None:
        """Appending an empty DataFrame should not touch the file."""
        df: DataFrame = _make_long_odds(n=2)
        path1: Path = append_to_odds_ledger(df, repo=tmp_path)

        # Append empty - should be a no-op
        empty = pd.DataFrame(columns=df.columns)
        path2: Path = append_to_odds_ledger(empty, repo=tmp_path)

        loaded: DataFrame = pd.read_parquet(path2)
        assert len(loaded) == 2  # original rows still there
        assert path1 == path2


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

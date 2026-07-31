# tests/unit/ingest/test_odds_store_source_neutral.py

"""Tests for source-neutral odds storage and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.store import (
    append_to_odds_ledger,
    load_current_odds,
    validate_odds_rows,
    write_current_odds_snapshot,
)


def _rows() -> DataFrame:
    return DataFrame(
        {
            "fetched_at": [pd.Timestamp("2026-07-30T18:00:00Z")],
            "sportsbook": ["nflverse_schedule"],
            "season": ["2026-2027"],
            "week": [1],
            "game_id": ["2026_01_KC_LAC"],
            "game_date": ["2026-09-10"],
            "away_team": ["Kansas City Chiefs"],
            "home_team": ["Los Angeles Chargers"],
            "market": ["moneyline"],
            "side": ["away"],
            "odds": [pd.NA],
            "line": [pd.NA],
        }
    )


def test_source_neutral_paths_replace_draftkings_names(tmp_path: Path) -> None:
    rows = _rows()
    ledger = append_to_odds_ledger(rows, repo=tmp_path)
    snapshot = write_current_odds_snapshot(rows, repo=tmp_path)

    assert ledger.name == "odds_log.parquet"
    assert snapshot.name == "odds_current.parquet"
    assert not (tmp_path / "data" / "odds" / "dk_odds_log.parquet").exists()
    assert not (tmp_path / "data" / "odds" / "dk_odds_current.parquet").exists()


def test_incomplete_rows_survive_snapshot_roundtrip(tmp_path: Path) -> None:
    write_current_odds_snapshot(_rows(), repo=tmp_path)
    loaded = load_current_odds(repo=tmp_path)

    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.loc[0, "sportsbook"] == "nflverse_schedule"
    assert pd.isna(loaded.loc[0, "odds"])
    assert pd.isna(loaded.loc[0, "line"])


def test_rejects_unknown_columns() -> None:
    rows = _rows().assign(unexpected="value")
    with pytest.raises(ValueError, match="unknown columns"):
        validate_odds_rows(rows)


def test_rejects_invalid_market_side_pair() -> None:
    rows = _rows()
    rows.loc[0, "market"] = "total"
    rows.loc[0, "side"] = "home"
    with pytest.raises(ValueError, match="invalid market/side"):
        validate_odds_rows(rows)


def test_rejects_non_utc_fetched_at() -> None:
    rows = _rows()
    rows["fetched_at"] = pd.Timestamp("2026-07-30T18:00:00-04:00")
    with pytest.raises(ValueError, match="must use UTC"):
        validate_odds_rows(rows)

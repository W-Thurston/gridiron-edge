"""Tests for provider-aware quote validation and atomic storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.store import (
    OBSERVATION_IDENTITY_COLUMNS,
    QUOTE_COLUMNS,
    append_to_odds_ledger,
    load_current_odds,
    load_odds_ledger,
    validate_quote_rows,
    write_current_odds_snapshot,
)


def _quotes(*, fetched_at: str = "2026-09-10T12:00:00Z") -> DataFrame:
    base: dict[str, object] = {
        "fetched_at": pd.Timestamp(fetched_at),
        "provider": "the_odds_api",
        "provider_event_id": "event-1",
        "sportsbook_updated_at": pd.Timestamp("2026-09-10T11:59:00Z"),
        "commence_time": pd.Timestamp("2026-09-11T00:20:00Z"),
        "is_live": False,
        "season": "2026-2027",
        "week": 1,
        "game_id": "2026_01_KC_LAC",
        "game_date": "2026-09-10",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
        "market": "moneyline",
        "side": "away",
        "odds": -150.0,
        "line": None,
    }
    return DataFrame(
        [
            {**base, "sportsbook": "draftkings"},
            {**base, "sportsbook": "fanduel", "odds": -145.0},
        ],
        columns=list(QUOTE_COLUMNS),
    )


def test_exact_schema_and_order_are_canonical() -> None:
    result = validate_quote_rows(_quotes())
    assert tuple(result.columns) == QUOTE_COLUMNS
    assert OBSERVATION_IDENTITY_COLUMNS


def test_rejects_missing_and_unknown_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_quote_rows(_quotes().drop(columns="provider"))
    with pytest.raises(ValueError, match="unknown columns"):
        validate_quote_rows(_quotes().assign(unexpected="value"))


def test_nullable_consensus_provenance_is_valid() -> None:
    rows = _quotes().iloc[[0]].copy()
    rows["provider"] = "nflverse"
    rows["provider_event_id"] = None
    rows["sportsbook"] = None
    rows["sportsbook_updated_at"] = pd.NaT
    rows["commence_time"] = pd.NaT
    result = validate_quote_rows(rows)
    assert result.loc[0, "provider"] == "nflverse"
    assert pd.isna(result.loc[0, "sportsbook"])


@pytest.mark.parametrize("column", ["fetched_at", "sportsbook_updated_at", "commence_time"])
def test_rejects_non_utc_timestamps(column: str) -> None:
    rows = _quotes()
    rows[column] = pd.Timestamp("2026-09-10T12:00:00-04:00")
    with pytest.raises(ValueError, match="must use UTC"):
        validate_quote_rows(rows)


def test_rejects_invalid_live_market_and_numeric_values() -> None:
    rows = _quotes()
    rows["is_live"] = "false"
    with pytest.raises(ValueError, match="must be boolean"):
        validate_quote_rows(rows)

    rows = _quotes()
    rows["odds"] = 0
    with pytest.raises(ValueError, match="must not be zero"):
        validate_quote_rows(rows)


def test_exact_rerun_is_idempotent_and_multiple_books_survive(tmp_path: Path) -> None:
    quotes = _quotes()
    append_to_odds_ledger(quotes, repo=tmp_path)
    path = append_to_odds_ledger(quotes, repo=tmp_path)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 2
    assert set(loaded["sportsbook"]) == {"draftkings", "fanduel"}


def test_later_local_observation_is_retained(tmp_path: Path) -> None:
    append_to_odds_ledger(_quotes(), repo=tmp_path)
    path = append_to_odds_ledger(
        _quotes(fetched_at="2026-09-10T13:00:00Z"),
        repo=tmp_path,
    )
    assert len(pd.read_parquet(path)) == 4


def test_snapshot_atomically_overwrites_and_roundtrips(tmp_path: Path) -> None:
    write_current_odds_snapshot(_quotes(), repo=tmp_path)
    replacement = _quotes().iloc[[0]].copy()
    path = write_current_odds_snapshot(replacement, repo=tmp_path)
    loaded = load_current_odds(repo=tmp_path)
    assert path.is_file()
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.loc[0, "sportsbook"] == "draftkings"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_invalid_snapshot_does_not_replace_existing_file(tmp_path: Path) -> None:
    path = write_current_odds_snapshot(_quotes(), repo=tmp_path)
    before = path.read_bytes()
    invalid = _quotes().drop(columns="provider")
    with pytest.raises(ValueError, match="missing columns"):
        write_current_odds_snapshot(invalid, repo=tmp_path)
    assert path.read_bytes() == before


def test_loaders_return_canonical_absence_and_support_filters(tmp_path: Path) -> None:
    assert load_current_odds(repo=tmp_path) is None
    assert tuple(load_odds_ledger(repo=tmp_path).columns) == QUOTE_COLUMNS
    append_to_odds_ledger(_quotes(), repo=tmp_path)
    loaded = load_odds_ledger(
        provider="the_odds_api",
        sportsbook="fanduel",
        market="moneyline",
        repo=tmp_path,
    )
    assert len(loaded) == 1
    assert loaded.loc[0, "sportsbook"] == "fanduel"

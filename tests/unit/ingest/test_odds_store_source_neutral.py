"""Focused validation tests for source-neutral quote storage."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS, validate_quote_rows


def _consensus_row() -> DataFrame:
    return DataFrame(
        [
            {
                "fetched_at": pd.Timestamp("2026-07-30T18:00:00Z"),
                "provider": "nflverse",
                "provider_event_id": None,
                "sportsbook": None,
                "sportsbook_updated_at": pd.NaT,
                "commence_time": pd.NaT,
                "is_live": False,
                "season": "2026-2027",
                "week": 1,
                "game_id": "2026_01_KC_LAC",
                "game_date": "2026-09-10",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market": "moneyline",
                "side": "away",
                "odds": pd.NA,
                "line": pd.NA,
            }
        ],
        columns=list(QUOTE_COLUMNS),
    )


def test_incomplete_consensus_quote_is_truthful() -> None:
    result = validate_quote_rows(_consensus_row())
    assert result.loc[0, "provider"] == "nflverse"
    assert pd.isna(result.loc[0, "sportsbook"])
    assert pd.isna(result.loc[0, "odds"])
    assert pd.isna(result.loc[0, "line"])


def test_optional_text_is_null_or_nonempty() -> None:
    rows = _consensus_row()
    rows["sportsbook"] = " "
    with pytest.raises(ValueError, match="null or nonempty"):
        validate_quote_rows(rows)


def test_rejects_invalid_market_side_pair() -> None:
    rows = _consensus_row()
    rows.loc[0, "market"] = "total"
    rows.loc[0, "side"] = "home"
    with pytest.raises(ValueError, match="invalid market/side"):
        validate_quote_rows(rows)

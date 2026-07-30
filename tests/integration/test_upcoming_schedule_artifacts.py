# tests/integration/test_upcoming_schedule_artifacts.py

"""Integration tests for dual upcoming-schedule artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.loaders import (
    load_schedule_upcoming,
    load_schedule_upcoming_rich,
)
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import write_parquet
from gridiron_edge.transform.clean.schedule_nflverse import (
    ELO_UPCOMING_COLUMNS,
    RICH_UPCOMING_COLUMNS,
    clean_nflverse_upcoming,
)


def _raw_schedule() -> DataFrame:
    """Create schedule rows with mixed optional-data coverage."""
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
            "away_team": [
                "KC",
                "BAL",
                "GB",
            ],
            "home_team": [
                "LAC",
                "BUF",
                "CHI",
            ],
            "result": [
                None,
                None,
                None,
            ],
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
            "div_game": [
                1,
                0,
                None,
            ],
            "away_rest": [
                7,
                7,
                None,
            ],
            "home_rest": [
                7,
                7,
                None,
            ],
            "away_moneyline": [
                -120,
                None,
                None,
            ],
            "home_moneyline": [
                105,
                None,
                None,
            ],
            "spread_line": [
                -2.5,
                None,
                None,
            ],
            "away_spread_odds": [
                -110,
                None,
                None,
            ],
            "home_spread_odds": [
                -110,
                None,
                None,
            ],
            "total_line": [
                45.5,
                None,
                None,
            ],
            "over_odds": [
                -110,
                None,
                None,
            ],
            "under_odds": [
                -110,
                None,
                None,
            ],
        }
    )


def test_raw_schedule_builds_complete_rich_and_focused_artifacts(
    tmp_path: Path,
) -> None:
    raw = _raw_schedule()
    timestamp = datetime(
        2026,
        7,
        30,
        18,
        tzinfo=UTC,
    )

    raw_path = write_parquet(
        tmp_path,
        "schedule_upcoming_raw_nflverse",
        raw,
    )

    focused_path = clean_nflverse_upcoming(
        repo=tmp_path,
        ingested_at=timestamp,
    )

    rich_path = dataset_path(
        tmp_path,
        "schedule_upcoming_rich",
    )

    assert raw_path.exists()
    assert rich_path.exists()
    assert focused_path.exists()

    rich = load_schedule_upcoming_rich(tmp_path)
    focused = load_schedule_upcoming(tmp_path)

    assert list(rich.columns) == list(RICH_UPCOMING_COLUMNS)
    assert list(focused.columns) == list(ELO_UPCOMING_COLUMNS)

    assert len(rich) == len(raw)
    assert len(focused) == len(raw)

    assert set(rich["game_id"]) == set(raw["game_id"])
    assert set(focused["GAME_ID"]) == set(raw["game_id"])
    assert set(rich["game_id"]) == set(focused["GAME_ID"])

    no_market = rich.loc[rich["game_id"] == "2026_01_BAL_BUF"].iloc[0]

    for column in (
        "away_moneyline",
        "home_moneyline",
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "total_line",
        "over_odds",
        "under_odds",
    ):
        assert pd.isna(no_market[column])

    assert rich["ingested_at"].nunique() == 1
    assert rich["ingested_at"].iloc[0] == pd.Timestamp(timestamp)
    assert set(rich["source"]) == {"nflverse"}


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_completed_source_rows_do_not_enter_upcoming_artifacts(
    tmp_path: Path,
) -> None:
    raw = _raw_schedule()
    raw["result"] = raw["result"].astype("Float64")

    completed = raw.iloc[0].copy()
    completed["game_id"] = "2026_01_COMPLETED_GAME"
    completed["result"] = 7.0

    mixed = raw.copy()
    mixed.loc[len(mixed)] = completed

    write_parquet(
        tmp_path,
        "schedule_upcoming_raw_nflverse",
        mixed,
    )

    clean_nflverse_upcoming(
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
    assert "2026_01_COMPLETED_GAME" not in set(rich["game_id"])
    assert "2026_01_COMPLETED_GAME" not in set(focused["GAME_ID"])

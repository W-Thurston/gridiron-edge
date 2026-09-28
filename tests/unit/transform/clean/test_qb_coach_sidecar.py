# tests/unit/transform/clean/test_qb_coach_sidecar.py

"""Tests for the QB-and-coach identity sidecar transform."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.loaders import load_qb_coach_sidecar
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import write_parquet
from gridiron_edge.transform.clean.qb_coach_sidecar import (
    SIDECAR_COLUMNS,
    build_qb_coach_sidecar,
)


def _raw_historical() -> DataFrame:
    return DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "game_id": ["2025_01_KC_LAC", "2025_02_BUF_MIA"],
            "away_team": ["KC", "BUF"],
            "home_team": ["LAC", "MIA"],
            "result": [7.0, -3.0],
            "away_qb_id": ["00-AAA", "00-BBB"],
            "away_qb_name": ["A. Away", "B. Away"],
            "home_qb_id": ["00-CCC", "00-DDD"],
            "home_qb_name": ["C. Home", "D. Home"],
            "away_coach": ["Coach Away1", "Coach Away2"],
            "home_coach": ["Coach Home1", "Coach Home2"],
        }
    )


def _raw_upcoming() -> DataFrame:
    return DataFrame(
        {
            "season": [2025],
            "week": [3],
            "game_type": ["REG"],
            "game_id": ["2025_03_KC_DEN"],
            "away_team": ["KC"],
            "home_team": ["DEN"],
            "result": [None],
            "away_qb_id": [None],
            "away_qb_name": [None],
            "home_qb_id": [None],
            "home_qb_name": [None],
            "away_coach": ["Coach Away1"],
            "home_coach": ["Coach Home3"],
        }
    )


def test_combines_historical_and_upcoming_rows(tmp_path: Path) -> None:
    write_parquet(tmp_path, "games_raw_nflverse", _raw_historical())
    write_parquet(tmp_path, "schedule_upcoming_raw_nflverse", _raw_upcoming())

    result_path = build_qb_coach_sidecar(repo=tmp_path)
    sidecar = load_qb_coach_sidecar(tmp_path)

    assert result_path == dataset_path(tmp_path, "qb_coach_sidecar")
    assert list(sidecar.columns) == list(SIDECAR_COLUMNS)
    assert len(sidecar) == 3
    assert set(sidecar["GAME_ID"]) == {
        "2025_01_KC_LAC",
        "2025_02_BUF_MIA",
        "2025_03_KC_DEN",
    }


def test_maps_short_team_codes_to_long_names(tmp_path: Path) -> None:
    write_parquet(tmp_path, "games_raw_nflverse", _raw_historical())
    write_parquet(tmp_path, "schedule_upcoming_raw_nflverse", _raw_upcoming())

    build_qb_coach_sidecar(repo=tmp_path)
    sidecar = load_qb_coach_sidecar(tmp_path)

    row = sidecar.loc[sidecar["GAME_ID"] == "2025_01_KC_LAC"].iloc[0]
    assert row["AWAY_TEAM"] == "Kansas City Chiefs"
    assert row["HOME_TEAM"] == "Los Angeles Chargers"


def test_upcoming_rows_have_coach_but_not_qb_identity(tmp_path: Path) -> None:
    write_parquet(tmp_path, "games_raw_nflverse", _raw_historical())
    write_parquet(tmp_path, "schedule_upcoming_raw_nflverse", _raw_upcoming())

    build_qb_coach_sidecar(repo=tmp_path)
    sidecar = load_qb_coach_sidecar(tmp_path)

    row = sidecar.loc[sidecar["GAME_ID"] == "2025_03_KC_DEN"].iloc[0]
    assert pd.isna(row["AWAY_QB_ID"])
    assert row["AWAY_COACH"] == "Coach Away1"
    assert row["HOME_COACH"] == "Coach Home3"


def test_excludes_preseason_from_historical(tmp_path: Path) -> None:
    historical = _raw_historical()
    historical["game_type"] = ["PRE", "REG"]
    write_parquet(tmp_path, "games_raw_nflverse", historical)

    build_qb_coach_sidecar(repo=tmp_path)
    sidecar = load_qb_coach_sidecar(tmp_path)

    assert "2025_01_KC_LAC" not in set(sidecar["GAME_ID"])
    assert "2025_02_BUF_MIA" in set(sidecar["GAME_ID"])


def test_missing_upcoming_cache_still_builds_historical_only(tmp_path: Path) -> None:
    write_parquet(tmp_path, "games_raw_nflverse", _raw_historical())

    build_qb_coach_sidecar(repo=tmp_path)
    sidecar = load_qb_coach_sidecar(tmp_path)

    assert len(sidecar) == 2


def test_missing_historical_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Raw nflverse games file not found"):
        build_qb_coach_sidecar(repo=tmp_path)


def test_duplicate_game_ids_raise(tmp_path: Path) -> None:
    historical = _raw_historical()
    upcoming = _raw_upcoming()
    upcoming["game_id"] = ["2025_01_KC_LAC"]
    write_parquet(tmp_path, "games_raw_nflverse", historical)
    write_parquet(tmp_path, "schedule_upcoming_raw_nflverse", upcoming)

    with pytest.raises(ValueError, match="duplicate game IDs"):
        build_qb_coach_sidecar(repo=tmp_path)

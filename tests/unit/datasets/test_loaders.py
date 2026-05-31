# tests/unit/datasets/test_loaders.py
"""Tests for gridiron_edge.datasets.loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.dataframes import make_epa_by_game, make_games, make_stadiums

from gridiron_edge.datasets.loaders import (
    load_csv,
    load_epa_by_game,
    load_games,
    load_parquet_if_exists,
    load_stadiums,
)
from gridiron_edge.datasets.writers import write_csv


class TestLoadCsv:
    def test_roundtrip_with_write_csv(self, tmp_path: Path) -> None:
        original = make_games(n=3)
        write_csv(tmp_path, "games", original)
        loaded: DataFrame = load_csv(tmp_path, "games")
        pd.testing.assert_frame_equal(loaded, original)

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path, "games")

    def test_extra_kwargs_forwarded(self, tmp_path: Path) -> None:
        df = make_games(n=5)
        write_csv(tmp_path, "games", df)
        loaded: DataFrame = load_csv(tmp_path, "games", nrows=2)
        assert len(loaded) == 2


class TestConvenienceLoaders:
    def test_load_games(self, tmp_path: Path) -> None:
        original = make_games(n=2)
        write_csv(tmp_path, "games", original)
        loaded: DataFrame = load_games(tmp_path)
        assert len(loaded) == 2
        assert "GAME_ID" in loaded.columns

    def test_load_stadiums(self, tmp_path: Path) -> None:
        original = make_stadiums()
        write_csv(tmp_path, "stadiums", original)
        loaded: DataFrame = load_stadiums(tmp_path)
        assert len(loaded) == len(original)
        assert "STADIUM" in loaded.columns


class TestLoadParquetIfExists:
    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        result: DataFrame | None = load_parquet_if_exists(tmp_path / "nonexistent.parquet")
        assert result is None

    def test_returns_dataframe_when_present(self, tmp_path: Path) -> None:
        df = make_games(n=3)
        path: Path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)
        result: DataFrame | None = load_parquet_if_exists(path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestLoadEpaByGame:
    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        result: DataFrame = load_epa_by_game(tmp_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_returns_data_when_file_exists(self, tmp_path: Path) -> None:
        epa = make_epa_by_game(teams=["KC", "SF"], seasons=[2024], weeks_per_season=2)

        # Write to the expected location
        epa_path: Path = tmp_path / "data" / "cleaned" / "epa_by_game.parquet"
        epa_path.parent.mkdir(parents=True, exist_ok=True)
        epa.to_parquet(epa_path, index=False)
        result: DataFrame = load_epa_by_game(tmp_path)
        assert len(result) > 0

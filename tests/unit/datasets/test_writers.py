# tests/unit/datasets/test_writers.py
"""Tests for gridiron_edge.datasets.writers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
from tests.fixtures.dataframes import make_games, make_stadiums

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import write_csv, write_parquet


class TestWriteCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        df = make_games(n=2)
        write_csv(tmp_path, "games", df)
        assert dataset_path(tmp_path, "games").is_file()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        df = make_games(n=1)
        # tmp_path won't have data/cleaned/ yet
        write_csv(tmp_path, "games", df)
        assert (tmp_path / "data" / "cleaned").is_dir()

    def test_returns_path(self, tmp_path: Path) -> None:
        df = make_games(n=1)
        result: Path = write_csv(tmp_path, "games", df)
        assert isinstance(result, Path)
        assert result == dataset_path(tmp_path, "games")

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        original = make_stadiums()
        write_csv(tmp_path, "stadiums", original)
        loaded: DataFrame = pd.read_csv(write_csv(tmp_path, "stadiums", original))
        pd.testing.assert_frame_equal(loaded, original)

    def test_index_false_by_default(self, tmp_path: Path) -> None:
        df = make_games(n=1)
        write_csv(tmp_path, "games", df)
        loaded: DataFrame = pd.read_csv(dataset_path(tmp_path, "games"))
        assert "Unnamed: 0" not in loaded.columns


class TestWriteParquet:
    def test_creates_file(self, tmp_path: Path) -> None:
        df = make_games(n=2)
        write_parquet(tmp_path, "modeling_base", df)
        assert dataset_path(tmp_path, "modeling_base").is_file()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        df = make_games(n=1)
        write_parquet(tmp_path, "modeling_base", df)
        assert (tmp_path / "data" / "modeling").is_dir()

    def test_returns_path(self, tmp_path: Path) -> None:
        df = make_games(n=1)
        result: Path = write_parquet(tmp_path, "modeling_base", df)
        assert isinstance(result, Path)
        assert result == dataset_path(tmp_path, "modeling_base")

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        original = make_games(n=3)
        write_parquet(tmp_path, "modeling_base", original)
        loaded: DataFrame = pd.read_parquet(dataset_path(tmp_path, "modeling_base"))
        pd.testing.assert_frame_equal(loaded, original)

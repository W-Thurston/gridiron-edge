# tests/unit/datasets/test_writers.py
"""Tests for gridiron_edge.datasets.writers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from pandas import DataFrame
from tests.fixtures.dataframes import make_games, make_stadiums

from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_csv,
    write_parquet,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity


class TestWriteCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        df = make_games(n=2)
        write_csv(tmp_path, "games", df)
        assert dataset_path(tmp_path, "games").is_file()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        df = make_games(n=1)
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


class TestWeeklyProductWriters:
    """Verify dataset writers delegate to the weekly-product store."""

    @patch("gridiron_edge.models.game_prediction.weekly_product_store.write_weekly_product")
    def test_writes_immutable_weekly_product(
        self,
        mock_write: MagicMock,
        tmp_path: Path,
    ) -> None:
        product = pd.DataFrame({"game_id": ["game-1"]})
        identity = WeeklyProductIdentity(
            product_id="product-1",
            run_id="run-1",
            season="2026-2027",
            week=8,
            generated_at=datetime(2026, 10, 20, 12, tzinfo=UTC),
        )
        expected = tmp_path / "product-1.parquet"
        mock_write.return_value = expected

        result = write_weekly_product(
            tmp_path,
            product,
            identity=identity,
        )

        assert result == expected
        mock_write.assert_called_once_with(
            product,
            identity=identity,
            repo=tmp_path,
        )

    @patch(
        "gridiron_edge.models.game_prediction.weekly_product_store.select_current_weekly_product"
    )
    def test_selects_current_weekly_product_explicitly(
        self,
        mock_select: MagicMock,
        tmp_path: Path,
    ) -> None:
        selected_at = datetime(2026, 10, 20, 12, 30, tzinfo=UTC)

        result = select_current_weekly_product(
            tmp_path,
            "product-1",
            season="2026-2027",
            week=8,
            selected_at=selected_at,
        )

        assert result is None
        mock_select.assert_called_once_with(
            "product-1",
            season="2026-2027",
            week=8,
            selected_at=selected_at,
            repo=tmp_path,
        )

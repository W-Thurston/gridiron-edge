# tests/integration/test_dataset_roundtrip.py
"""Integration: write → read → schema preserved across dataset types."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.dataframes import (
    make_elo_state,
    make_epa_by_game,
    make_games,
    make_stadiums,
    make_weather_enriched,
)

from gridiron_edge.datasets.loaders import load_csv, load_games, load_stadiums
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import write_csv, write_parquet


class TestCsvRoundtrip:
    """Write and read back CSV datasets — verify schema and data integrity."""

    @pytest.mark.parametrize(
        ("key", "factory"),
        [
            ("games", make_games),
            ("stadiums", make_stadiums),
            ("elo_state", make_elo_state),
            ("weather_enriched", make_weather_enriched),
        ],
    )
    def test_csv_roundtrip_preserves_columns(self, tmp_path: Path, key: str, factory) -> None:
        original = factory()
        write_csv(tmp_path, key, original)
        loaded: DataFrame = load_csv(tmp_path, key)
        assert list(loaded.columns) == list(original.columns)

    @pytest.mark.parametrize(
        ("key", "factory"),
        [
            ("games", make_games),
            ("stadiums", make_stadiums),
            ("elo_state", make_elo_state),
        ],
    )
    def test_csv_roundtrip_preserves_row_count(self, tmp_path: Path, key: str, factory) -> None:
        original = factory()
        write_csv(tmp_path, key, original)
        loaded: DataFrame = load_csv(tmp_path, key)
        assert len(loaded) == len(original)


class TestParquetRoundtrip:
    """Write and read back Parquet datasets."""

    def test_modeling_base_roundtrip(self, tmp_path: Path) -> None:
        original = make_games(n=4)
        write_parquet(tmp_path, "modeling_base", original)
        loaded: DataFrame = pd.read_parquet(dataset_path(tmp_path, "modeling_base"))
        pd.testing.assert_frame_equal(loaded, original)

    def test_epa_by_game_roundtrip(self, tmp_path: Path) -> None:
        original = make_epa_by_game(teams=["KC", "SF"], seasons=[2024], weeks_per_season=3)
        path: Path = tmp_path / "data" / "cleaned" / "epa_by_game.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        original.to_parquet(path, index=False)
        loaded: DataFrame = pd.read_parquet(path)
        pd.testing.assert_frame_equal(loaded, original)


class TestMiniRepoBuilderRoundtrip:
    """Verify MiniRepoBuilder datasets are loadable by production loaders."""

    def test_games_loadable(self, mini_repo: Path) -> None:
        games: DataFrame = load_games(mini_repo)
        assert len(games) > 0
        assert "GAME_ID" in games.columns

    def test_stadiums_loadable(self, mini_repo: Path) -> None:
        stadiums: DataFrame = load_stadiums(mini_repo)
        assert len(stadiums) > 0
        assert "STADIUM" in stadiums.columns

    def test_full_repo_all_datasets_loadable(self, mini_repo_full: Path) -> None:
        assert len(load_games(mini_repo_full)) > 0
        assert len(load_stadiums(mini_repo_full)) > 0
        # EPA is Parquet, verify it exists
        epa_path: Path = mini_repo_full / "data" / "cleaned" / "epa_by_game.parquet"
        assert epa_path.is_file()

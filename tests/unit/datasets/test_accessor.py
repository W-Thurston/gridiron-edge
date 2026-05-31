# tests/unit/datasets/test_accessor.py
"""Tests for gridiron_edge.datasets.accessor."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.dataframes import make_games, make_stadiums

from gridiron_edge.datasets.accessor import DatasetAccessor


class TestDatasetAccessorFrozen:
    def test_is_frozen_dataclass(self, tmp_path: Path) -> None:
        acc = DatasetAccessor(repo=tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            acc.repo = Path("/other")  # type: ignore[misc]


class TestDatasetAccessorDelegation:
    """Each method should delegate to the corresponding loader function."""

    def test_games_delegates(self, tmp_path: Path) -> None:
        expected = make_games(n=2)
        with patch("gridiron_edge.datasets.loaders.load_games", return_value=expected) as mock:
            acc = DatasetAccessor(repo=tmp_path)
            result: DataFrame = acc.games()
            mock.assert_called_once_with(tmp_path)
            pd.testing.assert_frame_equal(result, expected)

    def test_elo_state_delegates(self, tmp_path: Path) -> None:
        expected = pd.DataFrame({"NFL_TEAM": ["KC"], "ELO": [1500.0]})
        with patch("gridiron_edge.datasets.loaders.load_elo_state", return_value=expected) as mock:
            acc = DatasetAccessor(repo=tmp_path)
            result: DataFrame = acc.elo_state()
            mock.assert_called_once_with(tmp_path)
            pd.testing.assert_frame_equal(result, expected)

    def test_stadiums_delegates(self, tmp_path: Path) -> None:
        expected = make_stadiums()
        with patch("gridiron_edge.datasets.loaders.load_stadiums", return_value=expected) as mock:
            acc = DatasetAccessor(repo=tmp_path)
            result: DataFrame = acc.stadiums()
            mock.assert_called_once_with(tmp_path)
            pd.testing.assert_frame_equal(result, expected)

    def test_epa_by_game_delegates(self, tmp_path: Path) -> None:
        expected = pd.DataFrame({"game_id": ["g1"], "off_epa_per_play": [0.1]})
        with patch(
            "gridiron_edge.datasets.loaders.load_epa_by_game", return_value=expected
        ) as mock:
            acc = DatasetAccessor(repo=tmp_path)
            result: DataFrame = acc.epa_by_game()
            mock.assert_called_once_with(tmp_path)
            pd.testing.assert_frame_equal(result, expected)

    def test_weather_enriched_delegates(self, tmp_path: Path) -> None:
        expected = pd.DataFrame({"GAME_ID": ["g1"], "TEMP_F": [72.0]})
        with patch("gridiron_edge.datasets.loaders.load_csv", return_value=expected) as mock:
            acc = DatasetAccessor(repo=tmp_path)
            result: DataFrame = acc.weather_enriched()
            mock.assert_called_once_with(tmp_path, "weather_enriched")
            pd.testing.assert_frame_equal(result, expected)

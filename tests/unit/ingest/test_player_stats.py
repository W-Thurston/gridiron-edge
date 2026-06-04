# tests/unit/ingest/test_player_stats.py
"""Tests for gridiron_edge.ingest.nflverse.player_stats."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.nflverse.player_stats import (
    _KEEP_COLUMNS,
    _STATS_RELIABLE_FROM,
    fetch_player_stats,
    load_player_stats,
)


def _make_raw_player_stats(n: int = 10, season: int = 2024) -> DataFrame:
    """Build a minimal DataFrame mimicking nflreadpy.load_player_stats() output."""
    rows = []
    for i in range(n):
        row: dict = dict.fromkeys(_KEEP_COLUMNS, 0)
        row.update(
            {
                "player_id": f"00-000{i:04d}",
                "player_name": f"Player{i}",
                "player_display_name": f"Player {i}",
                "position": "QB" if i % 4 == 0 else "WR",
                "position_group": "QB" if i % 4 == 0 else "WR",
                "team": "KC",
                "opponent_team": "LV",
                "game_id": f"{season}_01_KC_LV",
                "season": season,
                "season_type": "REG",
                "week": 1,
                "passing_yards": 200 + i * 10,
                # Extra columns that should be filtered out
                "fantasy_points": 15.0,
                "headshot_url": "http://example.com/img.png",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class TestKeepColumns:
    def test_keep_columns_count(self) -> None:
        assert len(_KEEP_COLUMNS) == 43

    def test_key_columns_present(self) -> None:
        required = {
            "player_id",
            "player_name",
            "position",
            "team",
            "opponent_team",
            "season",
            "week",
            "game_id",
            "passing_yards",
            "rushing_yards",
            "receiving_yards",
            "passing_cpoe",
            "target_share",
        }
        assert required.issubset(set(_KEEP_COLUMNS))

    def test_excluded_columns(self) -> None:
        """Fantasy and headshot columns must not be in _KEEP_COLUMNS."""
        excluded = {"fantasy_points", "fantasy_points_ppr", "headshot_url"}
        assert excluded.isdisjoint(set(_KEEP_COLUMNS))

    def test_reliable_from_is_1999(self) -> None:
        assert _STATS_RELIABLE_FROM == 1999


class TestFetchPlayerStats:
    @patch("gridiron_edge.ingest.nflverse.player_stats.nfl")
    def test_writes_parquet(self, mock_nfl: MagicMock, tmp_path: Path) -> None:
        raw = _make_raw_player_stats(season=2024)
        mock_nfl.load_player_stats.return_value = MagicMock(to_pandas=MagicMock(return_value=raw))

        paths = fetch_player_stats(seasons=[2024], repo=tmp_path)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].name == "player_stats_2024.parquet"

    @patch("gridiron_edge.ingest.nflverse.player_stats.nfl")
    def test_filters_to_keep_columns(self, mock_nfl: MagicMock, tmp_path: Path) -> None:
        """Only _KEEP_COLUMNS should survive in the written Parquet."""
        raw = _make_raw_player_stats(season=2024)
        mock_nfl.load_player_stats.return_value = MagicMock(to_pandas=MagicMock(return_value=raw))

        fetch_player_stats(seasons=[2024], repo=tmp_path)

        written = pd.read_parquet(
            tmp_path / "data" / "raw" / "player_stats" / "player_stats_2024.parquet"
        )
        assert "fantasy_points" not in written.columns
        assert "headshot_url" not in written.columns
        assert "passing_yards" in written.columns

    @patch("gridiron_edge.ingest.nflverse.player_stats.nfl")
    def test_skips_failed_season(self, mock_nfl: MagicMock, tmp_path: Path) -> None:
        """A season that fails to fetch should be skipped, not raise."""
        mock_nfl.load_player_stats.side_effect = Exception("404")

        paths = fetch_player_stats(seasons=[2099], repo=tmp_path)

        assert len(paths) == 0


class TestLoadPlayerStats:
    def test_raises_when_no_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No player stats found"):
            load_player_stats(repo=tmp_path)

    @patch("gridiron_edge.ingest.nflverse.player_stats.nfl")
    def test_loads_multiple_seasons(self, mock_nfl: MagicMock, tmp_path: Path) -> None:
        """load_player_stats combines multiple season files."""
        for season in [2023, 2024]:
            raw = _make_raw_player_stats(n=5, season=season)
            mock_nfl.load_player_stats.return_value = MagicMock(
                to_pandas=MagicMock(return_value=raw)
            )
            fetch_player_stats(seasons=[season], repo=tmp_path)

        combined = load_player_stats(repo=tmp_path)
        assert len(combined) == 10
        assert set(combined["season"].unique()) == {2023, 2024}

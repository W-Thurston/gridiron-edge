# tests/unit/api/test_loader_player_history.py

"""Unit tests for load_player_history."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _make_logs() -> pd.DataFrame:
    """Two players, one with 3 REG games in 2024 + 1 in 2023."""
    return pd.DataFrame(
        [
            {
                "player_id": "P1",
                "player_name": "D.Henry",
                "team": "BAL",
                "opponent_team": "KC",
                "season": 2024,
                "season_type": "REG",
                "week": 1,
                "rushing_yards": 100.0,
                "game_id": "2024_01_KC_BAL",
            },
            {
                "player_id": "P1",
                "player_name": "D.Henry",
                "team": "BAL",
                "opponent_team": "SF",
                "season": 2024,
                "season_type": "REG",
                "week": 2,
                "rushing_yards": 80.0,
                "game_id": "2024_02_BAL_SF",
            },
            {
                "player_id": "P1",
                "player_name": "D.Henry",
                "team": "BAL",
                "opponent_team": "CIN",
                "season": 2024,
                "season_type": "REG",
                "week": 3,
                "rushing_yards": 120.0,
                "game_id": "2024_03_CIN_BAL",
            },
            {
                "player_id": "P1",
                "player_name": "D.Henry",
                "team": "TEN",
                "opponent_team": "IND",
                "season": 2023,
                "season_type": "REG",
                "week": 1,
                "rushing_yards": 60.0,
                "game_id": "2023_01_IND_TEN",
            },
        ]
    )


def _write(tmp_path: Path, df: pd.DataFrame) -> None:
    d = tmp_path / "data" / "cleaned"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "player_game_logs.parquet", index=False)


class _Settings:
    def __init__(self, repo: Path) -> None:
        self.repo_root = repo


def test_unknown_stat_returns_none(tmp_path: Path) -> None:
    from gridiron_edge.api.loaders import load_player_history

    _write(tmp_path, _make_logs())
    result = load_player_history(_Settings(tmp_path), player_id="P1", stat="bogus_stat")
    assert result is None


def test_unknown_player_returns_none(tmp_path: Path) -> None:
    from gridiron_edge.api.loaders import load_player_history

    _write(tmp_path, _make_logs())
    result = load_player_history(_Settings(tmp_path), player_id="NOPE", stat="rush_yards")
    assert result is None


def test_defaults_to_latest_season(tmp_path: Path) -> None:
    from gridiron_edge.api.loaders import load_player_history

    _write(tmp_path, _make_logs())
    result = load_player_history(_Settings(tmp_path), player_id="P1", stat="rush_yards")
    assert result is not None
    assert result["season"] == 2024
    assert len(result["rows"]) == 3  # 2024 only, not 2023


def test_rows_sorted_by_week_with_values(tmp_path: Path) -> None:
    from gridiron_edge.api.loaders import load_player_history

    _write(tmp_path, _make_logs())
    result = load_player_history(_Settings(tmp_path), player_id="P1", stat="rush_yards")
    assert result is not None
    weeks = [r["week"] for r in result["rows"]]
    assert weeks == [1, 2, 3]
    values = [r["value"] for r in result["rows"]]
    assert values == [100.0, 80.0, 120.0]


def test_limit_returns_last_n(tmp_path: Path) -> None:
    from gridiron_edge.api.loaders import load_player_history

    _write(tmp_path, _make_logs())
    result = load_player_history(_Settings(tmp_path), player_id="P1", stat="rush_yards", limit=2)
    assert result is not None
    weeks = [r["week"] for r in result["rows"]]
    assert weeks == [2, 3]  # last 2 by week

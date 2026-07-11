# tests/unit/api/test_loader_players_list.py

"""Unit tests for load_players_list."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _logs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "P1",
                "player_name": "Z.Back",
                "position": "RB",
                "team": "SEA",
                "season": 2025,
                "season_type": "REG",
                "week": 1,
                "is_skill": True,
                "rushing_yards": 50.0,
            },
            {
                "player_id": "P1",
                "player_name": "Z.Back",
                "position": "RB",
                "team": "DAL",
                "season": 2025,
                "season_type": "REG",
                "week": 8,
                "is_skill": True,
                "rushing_yards": 70.0,
            },  # traded → DAL latest
            {
                "player_id": "P2",
                "player_name": "A.Arm",
                "position": "QB",
                "team": "KC",
                "season": 2025,
                "season_type": "REG",
                "week": 1,
                "is_skill": True,
                "passing_yards": 300.0,
            },
            {
                "player_id": "P3",
                "player_name": "O.Line",
                "position": "OL",
                "team": "KC",
                "season": 2025,
                "season_type": "REG",
                "week": 1,
                "is_skill": False,
                "rushing_yards": 0.0,
            },  # non-skill, excluded
        ]
    )


def _write(tmp: Path, df: pd.DataFrame) -> None:
    d = tmp / "data" / "cleaned"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "player_game_logs.parquet", index=False)


class _S:
    def __init__(self, repo: Path) -> None:
        self.repo_root = repo


def test_skill_only_deduped_latest_team(tmp_path: Path) -> None:
    from gridiron_edge.api.loaders import load_players_list

    _write(tmp_path, _logs())
    result = load_players_list(_S(tmp_path))
    assert result is not None
    rows = result["rows"]
    # OL excluded; P1 + P2 remain
    ids = {r["player_id"] for r in rows}
    assert ids == {"P1", "P2"}
    # P1 deduped to latest team (DAL, week 8)
    p1 = next(r for r in rows if r["player_id"] == "P1")
    assert p1["team"] == "DAL"
    # sorted by name → A.Arm before Z.Back
    assert [r["player_name"] for r in rows] == ["A.Arm", "Z.Back"]


def test_missing_logs_returns_none(tmp_path: Path) -> None:
    from gridiron_edge.api.loaders import load_players_list

    assert load_players_list(_S(tmp_path)) is None

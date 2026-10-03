# tests/unit/cli/test_evaluate_starting_qb_projection.py

"""Tests for the starting-QB projection evaluation command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from gridiron_edge.cli.evaluate import evaluate_app
from gridiron_edge.datasets.registry import dataset_path

pytestmark = pytest.mark.unit

runner = CliRunner()


@dataclass(frozen=True)
class _Settings:
    repo_root: Path


def _snapshot(tree: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(tree)): p.read_bytes() for p in sorted(tree.rglob("*")) if p.is_file()
    }


def _write_sources(repo: Path) -> None:
    games = pd.DataFrame(
        {
            "game_id": [f"2020_{w:02d}_DEN_KC" for w in (1, 2)],
            "season": [2020, 2020],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "home_team": ["KC", "KC"],
            "away_team": ["DEN", "DEN"],
            "home_qb_id": ["A", "A"],
            "away_qb_id": ["X", "X"],
        }
    )
    games_path = dataset_path(repo, "games_raw_nflverse")
    games_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(games_path, index=False)

    charts = pd.DataFrame(
        {
            "season": [2020] * 4,
            "club_code": ["KC", "DEN", "KC", "DEN"],
            "week": [1.0, 1.0, 2.0, 2.0],
            "game_type": ["REG"] * 4,
            "formation": ["Offense"] * 4,
            "depth_position": ["QB"] * 4,
            "depth_team": ["1"] * 4,
            "gsis_id": ["A", "X", "A", "X"],
        }
    )
    injuries = pd.DataFrame(
        {
            "season": [2020],
            "game_type": ["REG"],
            "team": ["KC"],
            "week": [2.0],
            "gsis_id": ["B"],
            "report_status": ["Out"],
        }
    )
    for name, frame in (("depth_charts", charts), ("injuries", injuries)):
        directory = repo / "data" / "raw" / name
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / f"{name}_2020.parquet", index=False)


def test_missing_caches_name_the_ingest_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings", lambda: _Settings(repo_root=tmp_path)
    )
    games_path = dataset_path(tmp_path, "games_raw_nflverse")
    games_path.parent.mkdir(parents=True)
    pd.DataFrame({"game_id": []}).to_parquet(games_path, index=False)

    result = runner.invoke(evaluate_app, ["starting-qb-projection"])

    assert result.exit_code != 0
    assert "gridiron ingest depth-charts" in result.output


def test_command_writes_only_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings", lambda: _Settings(repo_root=tmp_path)
    )
    _write_sources(tmp_path)
    before = _snapshot(tmp_path / "data")
    report_dir = tmp_path / "reports"

    result = runner.invoke(
        evaluate_app, ["starting-qb-projection", "--report-dir", str(report_dir)]
    )

    assert result.exit_code == 0, result.output
    assert _snapshot(tmp_path / "data") == before
    assert [p.name for p in report_dir.iterdir()] == ["starting-qb-projection.md"]


def test_command_refuses_to_report_when_nothing_is_scorable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gridiron_edge.core.settings.get_settings", lambda: _Settings(repo_root=tmp_path)
    )
    _write_sources(tmp_path)
    charts_dir = tmp_path / "data" / "raw" / "depth_charts"
    (charts_dir / "depth_charts_2020.parquet").rename(charts_dir / "depth_charts_2005.parquet")
    frame = pd.read_parquet(charts_dir / "depth_charts_2005.parquet").assign(season=2005)
    frame.to_parquet(charts_dir / "depth_charts_2005.parquet", index=False)

    result = runner.invoke(
        evaluate_app,
        ["starting-qb-projection", "--report-dir", str(tmp_path / "reports")],
    )

    assert result.exit_code != 0
    assert "No scorable team-games" in result.output
    assert not (tmp_path / "reports").exists()

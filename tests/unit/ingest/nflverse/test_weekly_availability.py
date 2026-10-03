# tests/unit/ingest/nflverse/test_weekly_availability.py

"""Tests for the cached nflverse depth chart and injury report ingestion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest
from typer.testing import CliRunner

from gridiron_edge.cli.ingest import ingest_app
from gridiron_edge.ingest.nflverse.weekly_availability import (
    DEPTH_CHARTS_FIRST_SEASON,
    INJURIES_FIRST_SEASON,
    fetch_depth_charts,
    fetch_injuries,
    load_depth_charts,
    load_injuries,
)

pytestmark = pytest.mark.unit

runner = CliRunner()
_MODULE = "gridiron_edge.ingest.nflverse.weekly_availability"


def _frame(season: int) -> pl.DataFrame:
    return pl.DataFrame({"season": [season, season], "gsis_id": ["00-1", "00-2"]})


def test_fetch_writes_the_frame_unchanged_per_season(tmp_path: Path) -> None:
    with patch(f"{_MODULE}.nfl.load_injuries", side_effect=lambda s: _frame(s[0])):
        paths = fetch_injuries(seasons=[2023, 2024], repo=tmp_path)

    assert [p.name for p in paths] == ["injuries_2023.parquet", "injuries_2024.parquet"]
    assert paths[0].parent == tmp_path / "data" / "raw" / "injuries"
    pd.testing.assert_frame_equal(pd.read_parquet(paths[0]), _frame(2023).to_pandas())


def test_depth_charts_use_their_own_directory(tmp_path: Path) -> None:
    with patch(f"{_MODULE}.nfl.load_depth_charts", side_effect=lambda s: _frame(s[0])):
        paths = fetch_depth_charts(seasons=[2024], repo=tmp_path)

    assert paths == [tmp_path / "data" / "raw" / "depth_charts" / "depth_charts_2024.parquet"]


def test_cached_season_is_not_refetched_without_force(tmp_path: Path) -> None:
    loader = MagicMock(side_effect=lambda s: _frame(s[0]))
    with patch(f"{_MODULE}.nfl.load_injuries", loader):
        fetch_injuries(seasons=[2024], repo=tmp_path)
        fetch_injuries(seasons=[2024], repo=tmp_path)
        fetch_injuries(seasons=[2024], force=True, repo=tmp_path)

    assert loader.call_count == 2


def test_failed_or_empty_fetch_never_replaces_the_cache(tmp_path: Path) -> None:
    with patch(f"{_MODULE}.nfl.load_injuries", side_effect=lambda s: _frame(s[0])):
        (path,) = fetch_injuries(seasons=[2024], repo=tmp_path)
    original = path.read_bytes()

    with patch(f"{_MODULE}.nfl.load_injuries", side_effect=RuntimeError("network")):
        assert fetch_injuries(seasons=[2024], force=True, repo=tmp_path) == []
    with patch(f"{_MODULE}.nfl.load_injuries", return_value=pl.DataFrame()):
        assert fetch_injuries(seasons=[2024], force=True, repo=tmp_path) == []

    assert path.read_bytes() == original


def test_all_years_starts_at_each_datasets_first_season(tmp_path: Path) -> None:
    depth = MagicMock(side_effect=lambda s: _frame(s[0]))
    injuries = MagicMock(side_effect=lambda s: _frame(s[0]))
    with (
        patch(f"{_MODULE}.nfl.load_depth_charts", depth),
        patch(f"{_MODULE}.nfl.load_injuries", injuries),
    ):
        fetch_depth_charts(all_years=True, repo=tmp_path)
        fetch_injuries(all_years=True, repo=tmp_path)

    assert depth.call_args_list[0].args[0] == [DEPTH_CHARTS_FIRST_SEASON]
    assert injuries.call_args_list[0].args[0] == [INJURIES_FIRST_SEASON]
    assert DEPTH_CHARTS_FIRST_SEASON == 2001
    assert INJURIES_FIRST_SEASON == 2009


def test_load_returns_cached_seasons_and_fails_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="gridiron ingest injuries"):
        load_injuries(repo=tmp_path)
    with patch(f"{_MODULE}.nfl.load_depth_charts", side_effect=lambda s: _frame(s[0])):
        fetch_depth_charts(seasons=[2023, 2024], repo=tmp_path)

    loaded = load_depth_charts(repo=tmp_path)
    assert sorted(loaded) == [2023, 2024]
    assert list(load_depth_charts(seasons=[2024], repo=tmp_path)) == [2024]


@patch("gridiron_edge.core.settings.current_nfl_season", return_value=2026)
@patch(f"{_MODULE}.fetch_injuries")
def test_injuries_command_defaults_to_refreshing_the_current_season(
    fetch: MagicMock, _season: MagicMock
) -> None:
    fetch.return_value = [Path("injuries_2026.parquet")]

    result = runner.invoke(ingest_app, ["injuries"])

    assert result.exit_code == 0, result.output
    fetch.assert_called_once_with(seasons=[2026], force=True)


@patch(f"{_MODULE}.fetch_depth_charts")
def test_depth_chart_command_flags(fetch: MagicMock) -> None:
    fetch.return_value = []

    assert runner.invoke(ingest_app, ["depth-charts", "--all-years"]).exit_code == 0
    fetch.assert_called_with(all_years=True)

    result = runner.invoke(ingest_app, ["depth-charts", "--season", "2025", "--force"])
    assert result.exit_code == 0, result.output
    fetch.assert_called_with(seasons=[2025], force=True)

# tests/integration/api/test_teams_routes.py

"""Integration tests for /teams routes.

W8 Tier 3 Substep 2b — verifies percentile fields populate end-to-end
when the artifact exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency


@dataclass
class _FakeSettings:
    repo_root: Path


def _write_percentiles(tmp_path: Path, rows: list[dict]) -> None:
    """Write a percentiles artifact directly."""
    pct_dir = tmp_path / "data" / "output" / "rankings" / "percentiles"
    pct_dir.mkdir(parents=True, exist_ok=True)
    # Derive filename from first row.
    season = rows[0]["season"]
    week = int(rows[0]["week"])
    filename = f"percentiles_{season}_wk{week:02d}.parquet"
    pd.DataFrame(rows).to_parquet(pct_dir / filename, index=False)


def _make_games_df() -> pd.DataFrame:
    """Minimal games DataFrame for team routes."""
    return pd.DataFrame(
        [
            {
                "GAME_ID": "2026_01_KC_LAC",
                "YEAR": "2026-2027",
                "WEEK_NUM": 1,
                "GAME_DATE": "2026-09-05",
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Los Angeles Chargers",
                "GAME_LOCATION": "H",
                "PTS_WINNER": 27,
                "PTS_LOSER": 20,
            },
        ]
    )


def _make_elo_df() -> pd.DataFrame:
    """Minimal Elo state DataFrame."""
    return pd.DataFrame(
        [
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1620.0,
            },
            {
                "NFL_TEAM": "Los Angeles Chargers",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1520.0,
            },
        ]
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """TestClient with settings pointing at tmp_path."""
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


class TestTeamRankingsPercentiles:
    def test_percentiles_populate_when_artifact_exists(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )

        _write_percentiles(
            tmp_path,
            [
                {
                    "team_abbr": "KC",
                    "season": "2026-2027",
                    "week": 1,
                    "rating_pct": 0.75,
                    "avg_wins_pct": 0.75,
                    "make_playoffs_pct": 0.75,
                    "win_sb_pct": 0.75,
                },
                {
                    "team_abbr": "LAC",
                    "season": "2026-2027",
                    "week": 1,
                    "rating_pct": 0.25,
                    "avg_wins_pct": 0.25,
                    "make_playoffs_pct": 0.25,
                    "win_sb_pct": 0.25,
                },
            ],
        )

        response = client.get("/teams?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        by_abbr = {item["abbr"]: item for item in body["items"]}
        assert by_abbr["KC"]["rating_pct"] == 0.75
        assert by_abbr["KC"]["make_playoffs_pct"] == 0.75
        assert by_abbr["LAC"]["rating_pct"] == 0.25

    def test_no_percentile_artifact_leaves_fields_null(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )
        # No percentile artifact written.

        response = client.get("/teams?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        by_abbr = {item["abbr"]: item for item in body["items"]}
        assert by_abbr["KC"]["rating_pct"] is None
        assert by_abbr["KC"]["avg_wins_pct"] is None


class TestTeamProfilePercentiles:
    def test_percentiles_populate_on_profile(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )

        _write_percentiles(
            tmp_path,
            [
                {
                    "team_abbr": "KC",
                    "season": "2026-2027",
                    "week": 1,
                    "rating_pct": 0.75,
                    "avg_wins_pct": 0.80,
                    "make_playoffs_pct": 0.85,
                    "win_sb_pct": 0.90,
                },
            ],
        )

        response = client.get("/teams/KC?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["rating_pct"] == 0.75
        assert body["avg_wins_pct"] == 0.80
        assert body["make_playoffs_pct"] == 0.85
        assert body["win_sb_pct"] == 0.90

    def test_no_percentile_artifact_leaves_fields_null_on_profile(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )

        response = client.get("/teams/KC?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["rating_pct"] is None
        assert body["avg_wins_pct"] is None

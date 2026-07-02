"""Integration tests for /compare/teams route.

W8 Tier 2 Step 8a. Exercises loader → serializer → route stack
end-to-end via MiniRepoBuilder + FastAPI dependency_overrides.
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


def _make_games_for_teams() -> pd.DataFrame:
    """Games DataFrame for two-team comparison tests."""
    return pd.DataFrame(
        [
            {
                "GAME_ID": "2026_01_KC_LAC",
                "YEAR": "2026-2027",
                "WEEK_NUM": 1,
                "GAME_DATE": "2026-09-05",
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Los Angeles Chargers",
                "WIN_OR_TIE": 1,
                "PTS_WINNER": 27,
                "PTS_LOSER": 20,
                "GAME_LOCATION": "H",
                "STADIUM": "Arrowhead Stadium",
            },
            {
                "GAME_ID": "2026_02_LAC_KC",
                "YEAR": "2026-2027",
                "WEEK_NUM": 2,
                "GAME_DATE": "2026-09-12",
                "WINNER": "Los Angeles Chargers",
                "LOSER": "Kansas City Chiefs",
                "WIN_OR_TIE": 1,
                "PTS_WINNER": 24,
                "PTS_LOSER": 21,
                "GAME_LOCATION": "H",
                "STADIUM": "SoFi Stadium",
            },
        ]
    )


def _make_elo_state_for_teams() -> pd.DataFrame:
    """Elo state for two teams across weeks."""
    return pd.DataFrame(
        [
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1580.0,
            },
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 2,
                "ELO": 1600.0,
            },
            {
                "NFL_TEAM": "Los Angeles Chargers",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1520.0,
            },
            {
                "NFL_TEAM": "Los Angeles Chargers",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 2,
                "ELO": 1540.0,
            },
        ]
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """TestClient with settings pointing at tmp_path.

    Teams reference must be populated per-test via
    MiniRepoBuilder.with_teams_reference().
    """
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


class TestCompareTeamsRoute:
    def test_returns_comparison_for_valid_pair(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_for_teams())
            .with_elo_state(_make_elo_state_for_teams())
            .with_teams_reference()
        )

        response = client.get("/compare/teams?team_a=KC&team_b=LAC&season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["season"] == "2026-2027"
        assert body["team_a"] == "KC"
        assert body["team_b"] == "LAC"
        assert len(body["stats"]) == 10  # 3 populated + 7 scaffolded

    def test_populated_stats_have_values(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_for_teams())
            .with_elo_state(_make_elo_state_for_teams())
            .with_teams_reference()
        )

        response = client.get("/compare/teams?team_a=KC&team_b=LAC&season=2026-2027")
        body = response.json()
        by_key = {row["key"]: row for row in body["stats"]}

        # Rating (latest Elo per team)
        assert by_key["rating"]["team_a_value"] == 1600.0
        assert by_key["rating"]["team_b_value"] == 1540.0

        # Rank (KC higher = rank 1)
        assert by_key["rank"]["team_a_value"] == 1
        assert by_key["rank"]["team_b_value"] == 2

        # Record (1-1-0 for both teams; each won and lost once)
        assert by_key["record"]["team_a_value"] == "1-1-0"
        assert by_key["record"]["team_b_value"] == "1-1-0"

    def test_scaffolded_stats_have_null_values(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_for_teams())
            .with_elo_state(_make_elo_state_for_teams())
            .with_teams_reference()
        )

        response = client.get("/compare/teams?team_a=KC&team_b=LAC&season=2026-2027")
        body = response.json()
        by_key = {row["key"]: row for row in body["stats"]}

        for scaffolded_key in (
            "off_rating",
            "def_rating",
            "trend",
            "schedule_difficulty",
            "playoff_probability",
            "cohort_splits",
            "percentile_ranks",
        ):
            row = by_key[scaffolded_key]
            assert row["team_a_value"] is None
            assert row["team_b_value"] is None

    def test_field_status_marks_scaffolded_stats(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_for_teams())
            .with_elo_state(_make_elo_state_for_teams())
            .with_teams_reference()
        )

        response = client.get("/compare/teams?team_a=KC&team_b=LAC&season=2026-2027")
        body = response.json()
        status = body["_meta"]["field_status"]

        # Blocked
        assert status["off_rating"]["blocker"] == "off_def_decomposition"
        assert status["def_rating"]["blocker"] == "off_def_decomposition"
        assert status["trend"]["blocker"] == "no_prior_snapshot"

        # Pending
        assert status["schedule_difficulty"] == "pending"
        assert status["playoff_probability"] == "pending"
        assert status["cohort_splits"] == "pending"
        assert status["percentile_ranks"] == "pending"

    def test_unknown_team_a_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_for_teams())
            .with_elo_state(_make_elo_state_for_teams())
            .with_teams_reference()
        )
        response = client.get("/compare/teams?team_a=XYZ&team_b=LAC&season=2026-2027")
        assert response.status_code == 404
        assert "team_a" in response.json()["detail"]

    def test_unknown_team_b_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_for_teams())
            .with_elo_state(_make_elo_state_for_teams())
            .with_teams_reference()
        )
        response = client.get("/compare/teams?team_a=KC&team_b=XYZ&season=2026-2027")
        assert response.status_code == 404
        assert "team_b" in response.json()["detail"]

    def test_missing_team_a_returns_422(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # FastAPI requires team_a and team_b query params.
        response = client.get("/compare/teams?team_b=LAC")
        assert response.status_code == 422

    def test_default_season_uses_current(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # No season passed — hits _resolve_scope which reads games CSV.
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_for_teams())
            .with_elo_state(_make_elo_state_for_teams())
            .with_teams_reference()
        )

        response = client.get("/compare/teams?team_a=KC&team_b=LAC")

        assert response.status_code == 200
        body = response.json()
        # resolve_current_season_week returns latest season from games.
        assert body["season"] == "2026-2027"

# tests/integration/api/test_projections_route.py

"""Integration tests for the projections route and Elo-delta enrichment."""

from __future__ import annotations

from dataclasses import dataclass
import json
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


def _write_projections_summary(tmp_path: Path, rows: list[dict]) -> None:
    """Write the projections summary CSV."""
    csv_dir = tmp_path / "data" / "output" / "temp"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        csv_dir / "projections_summary.csv",
        index=False,
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """TestClient with settings pointing at tmp_path."""
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


def _make_minimal_games_df() -> pd.DataFrame:
    """Games DataFrame sufficient for resolve_current_season_week."""
    return pd.DataFrame(
        [
            {
                "GAME_ID": "2026_01_KAN_LAC",
                "YEAR": "2026-2027",
                "WEEK_NUM": 1,
                "GAME_DATE": "2026-09-05",
            },
        ]
    )


class TestProjectionsEloDelta:
    """Cover Elo-delta enrichment from the latest two weeks of Elo state."""

    def test_populates_delta_from_elo_state(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # Elo state with two weeks so delta is computable.
        elo_state = pd.DataFrame(
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
                    "ELO": 1595.0,
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
                    "ELO": 1512.0,
                },
            ]
        )

        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_minimal_games_df())
            .with_elo_state(elo_state)
            .with_teams_reference()
        )

        _write_projections_summary(
            tmp_path,
            [
                {
                    "TEAM": "KAN",
                    "AVG_WINS": 11.5,
                    "P_MAKE_PLAYOFFS": 0.85,
                    "P_REACH_DIV": 0.65,
                    "P_REACH_CONF": 0.42,
                    "P_REACH_SB": 0.24,
                    "P_WIN_SB": 0.13,
                },
                {
                    "TEAM": "LAC",
                    "AVG_WINS": 9.2,
                    "P_MAKE_PLAYOFFS": 0.62,
                    "P_REACH_DIV": 0.35,
                    "P_REACH_CONF": 0.18,
                    "P_REACH_SB": 0.09,
                    "P_WIN_SB": 0.04,
                },
            ],
        )

        response = client.get("/projections")

        assert response.status_code == 200
        body = response.json()
        by_team = {item["abbr"]: item for item in body["items"]}
        assert by_team["KAN"]["elo_delta"] == 15.0
        assert by_team["LAC"]["elo_delta"] == -8.0
        assert "items.elo_delta" not in body["_meta"]["field_status"]

    def test_week_1_returns_null_deltas(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        elo_state = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1580.0,
                },
                {
                    "NFL_TEAM": "Los Angeles Chargers",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1520.0,
                },
            ]
        )

        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_minimal_games_df())
            .with_elo_state(elo_state)
            .with_teams_reference()
        )

        _write_projections_summary(
            tmp_path,
            [
                {
                    "TEAM": "KAN",
                    "AVG_WINS": 11.5,
                    "P_MAKE_PLAYOFFS": 0.85,
                    "P_REACH_DIV": 0.65,
                    "P_REACH_CONF": 0.42,
                    "P_REACH_SB": 0.24,
                    "P_WIN_SB": 0.13,
                },
            ],
        )

        response = client.get("/projections")
        body = response.json()

        for item in body["items"]:
            assert item["elo_delta"] is None

        status = body["_meta"]["field_status"]["items.elo_delta"]
        assert status == {
            "status": "blocked",
            "blocker": "no_prior_snapshot",
            "roadmap": "data",
        }

    def test_populated_delta_has_no_unavailable_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        """A usable prior-week snapshot needs no unavailable marker."""
        elo_state = pd.DataFrame(
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
                    "ELO": 1595.0,
                },
            ]
        )

        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_minimal_games_df())
            .with_elo_state(elo_state)
            .with_teams_reference()
        )

        _write_projections_summary(
            tmp_path,
            [
                {
                    "TEAM": "KAN",
                    "AVG_WINS": 11.5,
                    "P_MAKE_PLAYOFFS": 0.85,
                    "P_REACH_DIV": 0.65,
                    "P_REACH_CONF": 0.42,
                    "P_REACH_SB": 0.24,
                    "P_WIN_SB": 0.13,
                },
            ],
        )

        response = client.get("/projections")
        body = response.json()
        status = body["_meta"]["field_status"]

        # A usable prior-week snapshot needs no unavailable marker.
        assert "items.elo_delta" not in status


class TestNSimulationsMetadata:
    def test_populates_from_metadata_json(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        elo_state = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1580.0,
                },
            ]
        )
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_minimal_games_df())
            .with_elo_state(elo_state)
            .with_teams_reference()
        )

        _write_projections_summary(
            tmp_path,
            [
                {
                    "TEAM": "KAN",
                    "AVG_WINS": 11.5,
                    "P_MAKE_PLAYOFFS": 0.85,
                    "P_REACH_DIV": 0.65,
                    "P_REACH_CONF": 0.42,
                    "P_REACH_SB": 0.24,
                    "P_WIN_SB": 0.13,
                },
            ],
        )

        # Write metadata sidecar.
        metadata_path = tmp_path / "data" / "output" / "temp" / "projections_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "n_simulations": 5000,
                    "computed_at": "2026-07-03T14:00:00+00:00",
                }
            )
        )

        response = client.get("/projections")
        body = response.json()
        assert body["n_simulations"] == 5000

    def test_no_metadata_leaves_field_null(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        elo_state = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1580.0,
                },
            ]
        )
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_minimal_games_df())
            .with_elo_state(elo_state)
            .with_teams_reference()
        )

        _write_projections_summary(
            tmp_path,
            [
                {
                    "TEAM": "KAN",
                    "AVG_WINS": 11.5,
                    "P_MAKE_PLAYOFFS": 0.85,
                    "P_REACH_DIV": 0.65,
                    "P_REACH_CONF": 0.42,
                    "P_REACH_SB": 0.24,
                    "P_WIN_SB": 0.13,
                },
            ],
        )
        # No metadata sidecar written.

        response = client.get("/projections")
        body = response.json()
        assert body["n_simulations"] is None

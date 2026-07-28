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
    csv_dir: Path = tmp_path / "data" / "output" / "temp"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        csv_dir / "projections_summary.csv",
        index=False,
    )


def _write_season_grid(
    tmp_path: Path,
    rows: list[dict],
) -> None:
    """Write the weekly season-grid artifact."""
    output_dir: Path = tmp_path / "data" / "output" / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_rows: list[dict] = []
    for source_row in rows:
        row = dict(source_row)
        for week in range(1, 19):
            row.setdefault(f"W{week:02d}_WIN_P", 0.0)
        completed_rows.append(row)

    pd.DataFrame(completed_rows).to_csv(
        output_dir / "season_grid.csv",
        index=False,
    )


def _write_schedule(
    tmp_path: Path,
    rows: list[dict],
) -> None:
    """Write the cleaned upcoming schedule artifact."""
    schedule_path: Path = tmp_path / "data" / "cleaned" / "NFL_upcoming_schedule_cleaned.csv"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(schedule_path, index=False)


def _empty_games_df() -> pd.DataFrame:
    """Return an empty games artifact with the expected source schema."""
    return pd.DataFrame(
        columns=[
            "GAME_ID",
            "YEAR",
            "WEEK_NUM",
            "WINNER",
            "LOSER",
            "WIN_OR_TIE",
            "GAME_DATE",
            "GAMETIME",
        ]
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


class TestProjectionGridRoute:
    def _schedule_rows(self) -> list[dict]:
        return [
            {
                "WEEK_NUM": 1,
                "GAME_DAY_OF_WEEK": "Wednesday",
                "GAME_DATE": "2026-09-09",
                "AWAY_TEAM": "Buffalo Bills",
                "HOME_TEAM": "Seattle Seahawks",
                "GAMETIME": "20:20:00",
                "YEAR": "2026-2027",
                "GAME_ID": "2026_01_BUF_SEA",
            },
            {
                "WEEK_NUM": 2,
                "GAME_DAY_OF_WEEK": "Sunday",
                "GAME_DATE": "2026-09-20",
                "AWAY_TEAM": "Seattle Seahawks",
                "HOME_TEAM": "Buffalo Bills",
                "GAMETIME": "16:25:00",
                "YEAR": "2026-2027",
                "GAME_ID": "2026_02_SEA_BUF",
            },
        ]

    def _grid_rows(self) -> list[dict]:
        return [
            {
                "TEAM": "SEA",
                "W01_WIN_P": 1.0,
                "W02_WIN_P": 0.64,
            },
            {
                "TEAM": "BUF",
                "W01_WIN_P": 0.0,
                "W02_WIN_P": 0.36,
            },
        ]

    def test_returns_preseason_projected_grid(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_empty_games_df())
            .with_teams_reference(
                {
                    "Seattle Seahawks": "SEA",
                    "Buffalo Bills": "BUF",
                }
            )
        )
        _write_schedule(tmp_path, self._schedule_rows())
        _write_season_grid(tmp_path, self._grid_rows())

        response = client.get("/projections/grid")

        assert response.status_code == 200
        body = response.json()

        assert body["season"] == "2026-2027"
        assert body["completed_through_week"] == 0
        assert body["regular_season_weeks"] == 18
        assert body["total"] == 2

        by_team = {item["abbr"]: item for item in body["items"]}

        sea_week_1 = by_team["SEA"]["weeks"][0]
        assert sea_week_1 == {
            "week": 1,
            "state": "projected",
            "opponent": "BUF",
            "is_home": True,
            "game_id": "2026_01_BUF_SEA",
            "game_date": "2026-09-09",
            "game_time": "20:20:00",
            "win_probability": 1.0,
            "actual_result": None,
        }

        sea_week_2 = by_team["SEA"]["weeks"][1]
        assert sea_week_2["state"] == "projected"
        assert sea_week_2["opponent"] == "BUF"
        assert sea_week_2["is_home"] is False
        assert sea_week_2["win_probability"] == 0.64

        # No scheduled Week 3 game means a confirmed bye.
        assert by_team["SEA"]["weeks"][2]["state"] == "bye"
        assert by_team["SEA"]["weeks"][2]["win_probability"] is None

    def test_returns_played_results_and_completed_boundary(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        games = pd.DataFrame(
            [
                {
                    "GAME_ID": "2026_01_BUF_SEA",
                    "YEAR": "2026-2027",
                    "WEEK_NUM": 1,
                    "WINNER": "Seattle Seahawks",
                    "LOSER": "Buffalo Bills",
                    "WIN_OR_TIE": 1.0,
                    "GAME_DATE": "2026-09-09",
                    "GAMETIME": "20:20:00",
                },
            ]
        )

        (
            MiniRepoBuilder(tmp_path)
            .with_games(games)
            .with_teams_reference(
                {
                    "Seattle Seahawks": "SEA",
                    "Buffalo Bills": "BUF",
                }
            )
        )
        _write_schedule(tmp_path, self._schedule_rows())
        _write_season_grid(tmp_path, self._grid_rows())

        response = client.get("/projections/grid")

        assert response.status_code == 200
        body = response.json()

        assert body["completed_through_week"] == 1

        by_team = {item["abbr"]: item for item in body["items"]}

        sea = by_team["SEA"]["weeks"][0]
        buf = by_team["BUF"]["weeks"][0]

        assert sea["state"] == "played"
        assert sea["actual_result"] == "W"
        assert sea["win_probability"] == 1.0

        assert buf["state"] == "played"
        assert buf["actual_result"] == "L"
        assert buf["win_probability"] == 0.0

        assert by_team["SEA"]["weeks"][1]["state"] == "projected"

    def test_missing_grid_marks_items_unavailable(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_empty_games_df())
            .with_teams_reference(
                {
                    "Seattle Seahawks": "SEA",
                    "Buffalo Bills": "BUF",
                }
            )
        )
        _write_schedule(tmp_path, self._schedule_rows())

        response = client.get("/projections/grid")

        assert response.status_code == 200
        body = response.json()

        assert body["items"] == []
        assert body["total"] == 0

        status = body["_meta"]["field_status"]["items"]
        assert status == {
            "status": "blocked",
            "blocker": "no_projections_data",
            "roadmap": "data",
        }

    def test_missing_schedule_marks_weeks_unavailable(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        (
            MiniRepoBuilder(tmp_path)
            .with_games(
                pd.DataFrame(
                    [
                        {
                            "GAME_ID": "2025_22_SEA_BUF",
                            "YEAR": "2025-2026",
                            "WEEK_NUM": 22,
                            "WINNER": "Seattle Seahawks",
                            "LOSER": "Buffalo Bills",
                            "WIN_OR_TIE": 1.0,
                        },
                    ]
                )
            )
            .with_teams_reference(
                {
                    "Seattle Seahawks": "SEA",
                    "Buffalo Bills": "BUF",
                }
            )
        )
        _write_season_grid(tmp_path, self._grid_rows())

        response = client.get("/projections/grid")

        assert response.status_code == 200
        body = response.json()

        assert all(
            week["state"] == "unavailable" for item in body["items"] for week in item["weeks"]
        )

        status = body["_meta"]["field_status"]["items.weeks"]
        assert status == {
            "status": "blocked",
            "blocker": "no_schedule_data",
            "roadmap": "data",
        }


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

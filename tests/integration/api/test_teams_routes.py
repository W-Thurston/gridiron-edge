# tests/integration/api/test_teams_routes.py

"""Integration tests for /teams routes.

Verifies percentile fields populate end-to-end
when the artifact exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient
import pandas as pd
import pytest
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency


@dataclass
class _FakeSettings:
    repo_root: Path


def _write_weekly_elo_forecast(tmp_path: Path) -> None:
    path = tmp_path / "data" / "output" / "temp" / "weekly_elo_forecast.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "season": "2026-2027",
            "forecast_origin_week": 1,
            "team": team,
            "week": week,
            "elo_p10": 1500.0,
            "elo_median": 1550.0,
            "elo_p90": 1600.0,
            "win_out_elo_median": 1650.0,
            "lose_out_elo_median": 1450.0,
            "simulation_count": 100,
            "lower_quantile": 0.1,
            "center_quantile": 0.5,
            "upper_quantile": 0.9,
            "quantile_method": "linear",
            "computed_at": "2026-08-19T00:00:00+00:00",
        }
        for team in ("KC", "LAC")
        for week in range(1, 19)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


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
    """Minimal canonical games frame for team routes."""
    return pd.DataFrame(
        [
            {
                "GAME_ID": "2026_01_KC_LAC",
                "YEAR": "2026-2027",
                "WEEK_NUM": 1,
                "GAME_DATE": "2026-09-05",
                "AWAY_TEAM": "Los Angeles Chargers",
                "HOME_TEAM": "Kansas City Chiefs",
                "AWAY_SCORE": 20,
                "HOME_SCORE": 27,
                "IS_NEUTRAL_SITE": 0,
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


class TestPreseasonEloScope:
    def test_rankings_use_week_one_elo_without_completed_games(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        empty_games = cast(pd.DataFrame, _make_games_df().iloc[0:0].copy())
        (
            MiniRepoBuilder(tmp_path)
            .with_games(empty_games)
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )

        response = client.get("/teams?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["season"] == "2026-2027"
        assert body["as_of_week"] == 1
        assert body["total"] == 2
        by_abbr = {item["abbr"]: item for item in body["items"]}
        assert by_abbr["KC"]["rating"] == 1620.0
        assert by_abbr["KC"]["rank"] == 1
        assert by_abbr["KC"]["record"] == {"wins": 0, "losses": 0, "ties": 0}
        assert by_abbr["LAC"]["rating"] == 1520.0
        assert "items" not in body["_meta"]["field_status"]

    def test_profile_uses_week_one_elo_without_completed_games(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        empty_games = cast(pd.DataFrame, _make_games_df().iloc[0:0].copy())
        (
            MiniRepoBuilder(tmp_path)
            .with_games(empty_games)
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )

        response = client.get("/teams/KC?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["as_of_week"] == 1
        assert body["rating"] == 1620.0
        assert body["rank"] == 1
        assert body["record"] == {"wins": 0, "losses": 0, "ties": 0}
        assert body["rating_history"] == [{"week": 1, "rating": 1620.0}]
        assert body["rating_timeline"]["completed_through_week"] == 0
        assert body["rating_timeline"]["current_rating_week"] == 1
        current = next(
            point for point in body["rating_timeline"]["points"] if point["state"] == "current"
        )
        assert current["week"] == 1
        assert current["rating"] == 1620.0

    def test_unknown_season_remains_blocked(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        empty_games = cast(pd.DataFrame, _make_games_df().iloc[0:0].copy())
        (
            MiniRepoBuilder(tmp_path)
            .with_games(empty_games)
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )

        response = client.get("/teams?season=2030-2031")

        assert response.status_code == 200
        body = response.json()
        assert body["as_of_week"] == 0
        assert body["items"] == []
        assert body["_meta"]["field_status"]["items"]["blocker"] == "no_evaluation_data"

    def test_profile_uses_entering_next_week_rating_after_completed_game(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        elo = pd.concat(
            [
                _make_elo_df(),
                pd.DataFrame(
                    [
                        {
                            "NFL_TEAM": "Kansas City Chiefs",
                            "NFL_YEAR": "2026-2027",
                            "NFL_WEEK": 2,
                            "ELO": 1640.0,
                        },
                        {
                            "NFL_TEAM": "Los Angeles Chargers",
                            "NFL_YEAR": "2026-2027",
                            "NFL_WEEK": 2,
                            "ELO": 1500.0,
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_elo_state(elo)
            .with_teams_reference()
        )

        response = client.get("/teams/KC?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["rating_timeline"]["completed_through_week"] == 1
        assert body["rating_timeline"]["current_rating_week"] == 2
        current = next(
            point for point in body["rating_timeline"]["points"] if point["state"] == "current"
        )
        assert current["week"] == 2
        assert current["rating"] == 1640.0

    def test_completed_game_week_takes_precedence_over_later_elo_state(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        elo = pd.concat(
            [
                _make_elo_df(),
                pd.DataFrame(
                    [
                        {
                            "NFL_TEAM": "Kansas City Chiefs",
                            "NFL_YEAR": "2026-2027",
                            "NFL_WEEK": 2,
                            "ELO": 1640.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_elo_state(elo)
            .with_teams_reference()
        )

        response = client.get("/teams?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["as_of_week"] == 1
        by_abbr = {item["abbr"]: item for item in body["items"]}
        assert by_abbr["KC"]["rating"] == 1620.0


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


class TestTrendPopulation:
    def test_trend_populated_from_elo_delta(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # Elo state with two weeks so delta is computable
        elo_state = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1600.0,
                },
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 2,
                    "ELO": 1620.0,
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
                    "ELO": 1510.0,
                },
            ]
        )
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_elo_state(elo_state)
            .with_teams_reference()
        )

        response = client.get("/teams?season=2026-2027")
        body = response.json()
        by_abbr = {item["abbr"]: item for item in body["items"]}
        assert by_abbr["KC"]["trend"] == 20.0
        assert by_abbr["LAC"]["trend"] == -10.0

    def test_week_1_returns_null_trend(
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
            .with_games(_make_games_df())
            .with_elo_state(elo_state)
            .with_teams_reference()
        )

        response = client.get("/teams?season=2026-2027")
        body = response.json()
        for item in body["items"]:
            assert item["trend"] is None


class TestTeamProfileCohortSplits:
    def test_cohort_splits_populated_from_artifact(
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

        # Write cohort splits artifact.
        cohort_dir = tmp_path / "data" / "output" / "rankings"
        cohort_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "team_abbr": "KC",
                    "cohort": "season",
                    "off_epa_per_play": 0.15,
                    "def_epa_per_play": -0.10,
                    "sample_size": 4,
                    "rank_off_epa_per_play": 1,
                },
            ]
        ).to_parquet(cohort_dir / "team_cohort_splits.parquet", index=False)

        response = client.get("/teams/KC?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["cohort_splits"] is not None
        assert body["cohort_splits"]["season"]["off_epa_per_play"] == 0.15

    def test_missing_artifact_leaves_field_null_pending(
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
        body = response.json()
        assert body["cohort_splits"] is None
        status = body["_meta"]["field_status"]
        assert status.get("cohort_splits") == "pending"


class TestTeamRatingForecastIntegration:
    def test_valid_forecast_populates_future_regular_season_points(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        empty_games = cast(pd.DataFrame, _make_games_df().iloc[0:0].copy())
        (
            MiniRepoBuilder(tmp_path)
            .with_games(empty_games)
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )
        _write_weekly_elo_forecast(tmp_path)

        response = client.get("/teams/KC?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        timeline = body["rating_timeline"]
        by_week = {point["week"]: point for point in timeline["points"]}
        assert by_week[1]["state"] == "current"
        assert by_week[2]["state"] == "forecast"
        assert by_week[2]["rating"] == 1550.0
        assert by_week[18]["state"] == "forecast"
        assert by_week[19]["state"] == "unavailable"
        assert timeline["forecast_simulation_count"] == 100
        assert "rating_timeline.forecast" not in body["_meta"]["field_status"]

    def test_missing_forecast_keeps_history_available_and_marks_blocker(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        empty_games = cast(pd.DataFrame, _make_games_df().iloc[0:0].copy())
        (
            MiniRepoBuilder(tmp_path)
            .with_games(empty_games)
            .with_elo_state(_make_elo_df())
            .with_teams_reference()
        )

        response = client.get("/teams/KC?season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        assert body["rating_timeline"]["points"][0]["state"] == "current"
        assert (
            body["_meta"]["field_status"]["rating_timeline.forecast"]["blocker"]
            == "no_elo_forecast"
        )

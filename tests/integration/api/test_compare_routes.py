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


def _write_prop_manifest(
    tmp_path: Path,
    families: dict[str, str],
) -> None:
    """Write a champion manifest for the given families."""
    import json

    manifest_dir = tmp_path / "data" / "output" / "champions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "updated_at": "2026-07-04T00:00:00+00:00",
        "models": {
            family: {
                "model_type": model_type,
                "promoted_at": "2026-07-04T00:00:00",
                "source_run_id": "TEST_RUN",
                "metrics": {"mae": 60.0},
            }
            for family, model_type in families.items()
        },
    }
    (manifest_dir / "champions.json").write_text(json.dumps(manifest))


def _write_prop_archive(tmp_path: Path, rows: list[dict]) -> None:
    """Write a prop predictions archive with the given rows."""
    archive_dir = tmp_path / "data" / "output" / "props"
    archive_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        archive_dir / "prop_predictions_log.parquet",
        index=False,
    )


def _make_prop_archive_row(
    game_id: str = "2026_01_KC_LAC",
    player_id: str = "00-0033873",
    stat_type: str = "qb_pass_yards",
    model_type: str = "elasticnet",
    team: str = "KC",
) -> dict:
    """Build a canonical valid prop archive row."""
    return {
        "predicted_at": "2026-08-01T00:00:00+00:00",
        "is_backfilled": True,
        "season": 2026,
        "week": 1,
        "game_id": game_id,
        "player_id": player_id,
        "player_name": "P.Mahomes",
        "position": "QB",
        "team": team,
        "stat_type": stat_type,
        "model_name": stat_type,
        "model_type": model_type,
        "predicted_mean": 275.0,
        "predicted_std": 45.0,
        "lo_90": 200.0,
        "hi_90": 350.0,
        "line": None,
        "p_over": float("nan"),
        "lean": float("nan"),
        "confidence_tier": float("nan"),
    }


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
        assert (
            len(body["stats"]) == 11
        )  # 3 populated + 3 rankable (avg_wins/make_playoffs/win_sb) + 5 scaffolded

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
            "cohort_splits",
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
        assert status["cohort_splits"] == "pending"

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


class TestComparePlayerRoute:
    def _write_prop_manifest(
        self,
        tmp_path: Path,
        families: dict[str, str],
    ) -> None:
        import json

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "champions.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-07-01T14:00:00+00:00",
                    "models": {
                        family: {
                            "model_type": model_type,
                            "promoted_at": "2026-07-01T14:00:00",
                            "source_run_id": "RUN_X",
                            "metrics": {"mae": 63.0},
                        }
                        for family, model_type in families.items()
                    },
                }
            )
        )

    def _write_prop_archive(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "data" / "output" / "props"
        archive_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "predicted_at": "2026-08-01T00:00:00+00:00",
                    "is_backfilled": True,
                    "season": 2026,
                    "week": 1,
                    "game_id": "2026_01_KC_LAC",
                    "player_id": "00-0033873",
                    "player_name": "P.Mahomes",
                    "position": "QB",
                    "team": "KC",
                    "stat_type": "qb_pass_yards",
                    "model_name": "qb_pass_yards",
                    "model_type": "elasticnet",
                    "predicted_mean": 265.0,
                    "predicted_std": 45.0,
                    "lo_90": 190.0,
                    "hi_90": 340.0,
                    "line": None,
                    "p_over": float("nan"),
                    "lean": float("nan"),
                    "confidence_tier": float("nan"),
                }
            ]
        ).to_parquet(
            archive_dir / "prop_predictions_log.parquet",
            index=False,
        )

    def test_returns_comparison_for_known_prop(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        self._write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        self._write_prop_archive(tmp_path)

        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/compare/player/{prop_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["prop_id"] == prop_id
        assert body["player_name"] == "P.Mahomes"
        assert body["stat_type"] == "qb_pass_yards"
        assert len(body["stats"]) == 8  # 4 projection + 4 defense

    def test_projection_stats_populated(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        self._write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        self._write_prop_archive(tmp_path)

        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/compare/player/{prop_id}")

        body = response.json()
        by_key = {row["key"]: row for row in body["stats"]}

        assert by_key["mean"]["projection_value"] == 265.0
        assert by_key["std"]["projection_value"] == 45.0
        assert by_key["lo_90"]["projection_value"] == 190.0
        assert by_key["hi_90"]["projection_value"] == 340.0

        # Defense side null for all four projection rows.
        assert by_key["mean"]["defense_value"] is None

    def test_defense_stats_all_null(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        self._write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        self._write_prop_archive(tmp_path)

        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/compare/player/{prop_id}")

        body = response.json()
        by_key = {row["key"]: row for row in body["stats"]}

        for defense_key in (
            "avg_allowed",
            "rank_against_position",
            "last_4_games_avg",
            "red_zone_rate_allowed",
        ):
            row = by_key[defense_key]
            assert row["projection_value"] is None
            assert row["defense_value"] is None

    def test_field_status_marks_defense_blocked(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        self._write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        self._write_prop_archive(tmp_path)

        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/compare/player/{prop_id}")

        body = response.json()
        status = body["_meta"]["field_status"]

        for defense_key in (
            "avg_allowed",
            "rank_against_position",
            "last_4_games_avg",
            "red_zone_rate_allowed",
        ):
            assert status[defense_key]["blocker"] == "opponent_allowed_by_position"

    def test_malformed_prop_id_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        response = client.get("/compare/player/malformed_prop_id")

        assert response.status_code == 404
        assert "Malformed prop_id" in response.json()["detail"]

    def test_unknown_stat_type_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        prop_id = "2026_01_KC_LAC__00-0033873__bogus_stat"
        response = client.get(f"/compare/player/{prop_id}")

        assert response.status_code == 404
        assert "Unknown stat_type" in response.json()["detail"]

    def test_unknown_prop_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        self._write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        self._write_prop_archive(tmp_path)

        prop_id = "2026_01_KC_LAC__00-0000000__qb_pass_yards"  # unknown player_id
        response = client.get(f"/compare/player/{prop_id}")

        assert response.status_code == 404
        assert "Prop not found" in response.json()["detail"]

    def test_missing_manifest_returns_null_projection(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # No manifest at all.
        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/compare/player/{prop_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["prop_id"] == prop_id
        by_key = {row["key"]: row for row in body["stats"]}
        assert by_key["mean"]["projection_value"] is None
        assert by_key["std"]["projection_value"] is None


class TestCompareTeamsPercentiles:
    def test_percentiles_populate_on_rating_row(
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

        # Write percentile artifact
        pct_dir = tmp_path / "data" / "output" / "rankings" / "percentiles"
        pct_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
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
            ]
        ).to_parquet(pct_dir / "percentiles_2026-2027_wk01.parquet", index=False)

        response = client.get("/compare/teams?team_a=KC&team_b=LAC&season=2026-2027")

        assert response.status_code == 200
        body = response.json()
        by_key = {row["key"]: row for row in body["stats"]}
        assert by_key["rating"]["team_a_pct"] == 0.75
        assert by_key["rating"]["team_b_pct"] == 0.25

    def test_percentile_ranks_row_removed(
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
        keys = {row["key"] for row in body["stats"]}
        assert "percentile_ranks" not in keys

    def test_no_percentile_artifact_leaves_pct_fields_null(
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
        # No percentile artifact written.

        response = client.get("/compare/teams?team_a=KC&team_b=LAC&season=2026-2027")

        body = response.json()
        by_key = {row["key"]: row for row in body["stats"]}
        assert by_key["rating"]["team_a_pct"] is None
        assert by_key["rating"]["team_b_pct"] is None


class TestComparePlayerOpponentAllowed:
    def test_populates_defense_rows(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        _write_prop_archive(tmp_path, [_make_prop_archive_row()])

        # Write opponent-allowed artifact.
        oppdir = tmp_path / "data" / "output" / "props"
        oppdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "opponent_team": "LAC",
                    "position": "QB",
                    "stat_type": "qb_pass_yards",
                    "cohort": "season",
                    "avg_allowed": 275.0,
                    "sample_size": 5,
                    "rank_against_position": 3,
                },
                {
                    "opponent_team": "LAC",
                    "position": "QB",
                    "stat_type": "qb_pass_yards",
                    "cohort": "l4",
                    "avg_allowed": 265.0,
                    "sample_size": 5,
                    "rank_against_position": 2,
                },
            ]
        ).to_parquet(oppdir / "opponent_allowed.parquet", index=False)

        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/compare/player/{prop_id}")

        assert response.status_code == 200
        body = response.json()
        by_key = {row["key"]: row for row in body["stats"]}
        assert by_key["avg_allowed"]["defense_value"] == 275.0
        assert by_key["rank_against_position"]["defense_value"] == 3
        assert by_key["last_4_games_avg"]["defense_value"] == 265.0
        assert by_key["red_zone_rate_allowed"]["defense_value"] is None

    def test_missing_artifact_leaves_rows_blocked(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        _write_prop_archive(tmp_path, [_make_prop_archive_row()])
        # No opponent-allowed artifact.

        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/compare/player/{prop_id}")

        body = response.json()
        status = body["_meta"]["field_status"]
        assert status.get("avg_allowed") is not None
        assert status.get("rank_against_position") is not None
        assert status.get("last_4_games_avg") is not None


class TestCompareTeamsCohortSplits:
    def test_cohort_splits_populated_from_artifact(
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

        # Write cohort splits artifact.
        cohort_dir = tmp_path / "data" / "output" / "rankings"
        cohort_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "team_abbr": "KC",
                    "cohort": "season",
                    "off_epa_per_play": 0.15,
                    "sample_size": 4,
                },
                {
                    "team_abbr": "LAC",
                    "cohort": "season",
                    "off_epa_per_play": 0.10,
                    "sample_size": 4,
                },
            ]
        ).to_parquet(cohort_dir / "team_cohort_splits.parquet", index=False)

        response = client.get("/compare/teams?team_a=KC&team_b=LAC&season=2026-2027")

        body = response.json()
        assert body["cohort_splits"] is not None
        assert "KC" in body["cohort_splits"]
        assert "LAC" in body["cohort_splits"]

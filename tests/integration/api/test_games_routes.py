# tests/integration/api/test_games_routes.py

"""Integration tests for /games and /games/{game_id} routes.

W8 Tier 2 Step 5d. Exercises the loader → serializer → route stack
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


def _make_prediction(
    game_id: str = "2026_01_KC_LAC",
    model_type: str = "elo",
    home_team: str = "Los Angeles Chargers",
    away_team: str = "Kansas City Chiefs",
) -> dict:
    return {
        "predicted_at": pd.Timestamp("2026-08-01"),
        "is_backfilled": False,
        "model_name": "win_prob",
        "model_type": model_type,
        "season": "2026-2027",
        "week": 1,
        "game_id": game_id,
        "game_date": "2026-09-05",
        "away_team": away_team,
        "home_team": home_team,
        "away_elo": 1550.0,
        "home_elo": 1520.0,
        "away_win_prob": 0.45,
        "home_win_prob": 0.55,
        "model_total": 47.5,
        "model_spread": -2.5,
        "margin_std": 13.5,
        "win_prob_lo": 0.42,
        "win_prob_hi": 0.68,
        "confidence_tier": "Moderate",
        "projected_home_score": 25.0,
        "projected_away_score": 22.5,
    }


def _make_games_df(game_ids: list[str] | None = None) -> pd.DataFrame:
    """Build a games DataFrame with rows keyed by GAME_ID."""
    if game_ids is None:
        game_ids = ["2026_01_KC_LAC"]
    return pd.DataFrame(
        [
            {
                "GAME_ID": gid,
                "YEAR": "2026-2027",
                "WEEK_NUM": 1,
                "GAME_DATE": "2026-09-05",
            }
            for gid in game_ids
        ]
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with settings pointing at tmp_path via dependency_overrides.

    Callers populate tmp_path with MiniRepoBuilder in each test — this
    fixture only wires the settings redirect.

    load_team_name_map is monkeypatched because MiniRepoBuilder doesn't
    yet have a with_teams_reference method. Tracked as a follow-up.
    """
    monkeypatch.setattr(
        "gridiron_edge.api.loaders.load_team_name_map",
        lambda _settings: {
            "Kansas City Chiefs": "KC",
            "Los Angeles Chargers": "LAC",
            "Buffalo Bills": "BUF",
            "Miami Dolphins": "MIA",
        },
    )

    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


class TestListGamesRoute:
    def test_returns_games_for_week(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame(
            [
                _make_prediction("2026_01_KC_LAC"),
                _make_prediction(
                    "2026_01_BUF_MIA",
                    home_team="Miami Dolphins",
                    away_team="Buffalo Bills",
                ),
            ]
        )
        games = _make_games_df(["2026_01_KC_LAC", "2026_01_BUF_MIA"])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(games)
            .with_champion_manifest()
            .with_predictions_archive(predictions)
        )

        response = client.get("/games?season=2026-2027&week=1")

        assert response.status_code == 200
        body = response.json()
        assert body["season"] == "2026-2027"
        assert body["week"] == 1
        assert body["total"] == 2
        assert len(body["items"]) == 2
        game_ids = {item["game_id"] for item in body["items"]}
        assert game_ids == {"2026_01_KC_LAC", "2026_01_BUF_MIA"}
        first = next(i for i in body["items"] if i["game_id"] == "2026_01_KC_LAC")
        assert first["home_team"] == "LAC"
        assert first["prediction"]["home_win_prob"] == 0.55

    def test_missing_manifest_returns_empty_with_field_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction()])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_predictions_archive(predictions)
            # no with_champion_manifest
        )

        response = client.get("/games?season=2026-2027&week=1")

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0
        status = body["_meta"]["field_status"]["items"]
        assert status["status"] == "blocked"
        assert status["blocker"] == "no_champion_manifest"


class TestGetGameRoute:
    def test_returns_detail_for_known_game(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction("2026_01_KC_LAC")])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(predictions)
        )

        response = client.get("/games/2026_01_KC_LAC")

        assert response.status_code == 200
        body = response.json()
        assert body["game_id"] == "2026_01_KC_LAC"
        assert body["home_team"] == "LAC"
        assert body["away_team"] == "KC"
        assert body["day_of_week"] == "Saturday"
        assert body["prediction"]["home_win_prob"] == 0.55
        assert body["prediction"]["confidence_tier"] == "Moderate"

    def test_field_status_marks_pending_and_blocked(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction("2026_01_KC_LAC")])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(predictions)
        )

        response = client.get("/games/2026_01_KC_LAC")

        body = response.json()
        status = body["_meta"]["field_status"]
        assert status["kick"] == "pending"
        assert status["venue"] == "pending"
        assert status["weather"] == "pending"
        assert status["team_comparison"] == "pending"
        assert status["top_prop_edges"] == "pending"
        assert status["swing_factors"]["blocker"] == "feature_attribution"
        assert status["injuries"]["blocker"] == "injury_data_source"

    def test_unknown_game_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction("2026_01_KC_LAC")])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(predictions)
        )

        response = client.get("/games/bogus_game_id")

        assert response.status_code == 404
        assert "Unknown game_id" in response.json()["detail"]

    def test_missing_manifest_returns_200_with_field_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction("2026_01_KC_LAC")])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_predictions_archive(predictions)
            # no with_champion_manifest
        )

        response = client.get("/games/2026_01_KC_LAC")

        assert response.status_code == 200
        body = response.json()
        assert body["game_id"] == "2026_01_KC_LAC"
        assert body["prediction"] is None
        status = body["_meta"]["field_status"]["prediction"]
        assert status["status"] == "blocked"
        assert status["blocker"] == "no_champion_manifest"


class TestGameDetailTeamComparison:
    def test_team_comparison_populated_when_artifact_exists(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction("2026_01_KC_LAC")])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(predictions)
        )

        # Write cohort splits artifact for both teams
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

        response = client.get("/games/2026_01_KC_LAC")

        assert response.status_code == 200
        body = response.json()
        assert body["team_comparison"] is not None
        assert "KC" in body["team_comparison"]
        assert "LAC" in body["team_comparison"]
        assert body["team_comparison"]["KC"]["season"]["off_epa_per_play"] == 0.15
        # Marker removed
        assert "team_comparison" not in body["_meta"]["field_status"]

    def test_team_comparison_pending_when_artifact_missing(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction("2026_01_KC_LAC")])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(predictions)
        )
        # No cohort splits artifact.

        response = client.get("/games/2026_01_KC_LAC")

        body = response.json()
        assert body["team_comparison"] is None
        assert body["_meta"]["field_status"]["team_comparison"] == "pending"

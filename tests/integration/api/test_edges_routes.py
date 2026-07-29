# tests/integration/api/test_edges_routes.py

"""Integration tests for /edges route.

W8 Tier 2 Step 6d. Exercises loader → serializer → route stack
end-to-end via MiniRepoBuilder + FastAPI dependency_overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import Response
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
    *,
    game_id: str = "2026_01_KC_LAC",
    model_type: str = "elo",
    home_team: str = "Los Angeles Chargers",
    away_team: str = "Kansas City Chiefs",
    home_win_prob: float = 0.70,
    model_spread: float = -7.0,
    model_total: float = 50.0,
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
        "away_win_prob": 1.0 - home_win_prob,
        "home_win_prob": home_win_prob,
        "model_total": model_total,
        "model_spread": model_spread,
        "margin_std": 13.5,
        "win_prob_lo": 0.55,
        "win_prob_hi": 0.85,
        "confidence_tier": "High",
        "projected_home_score": 28.0,
        "projected_away_score": 22.0,
    }


def _make_odds_snapshot(
    game_id: str = "2026_01_KC_LAC",
) -> pd.DataFrame:
    """Long-format odds for one game, all three markets."""
    ts = pd.Timestamp("2026-09-05 12:00:00")
    return pd.DataFrame(
        [
            {
                "fetched_at": ts,
                "sportsbook": "draftkings",
                "season": "2026-2027",
                "week": 1,
                "game_id": game_id,
                "game_date": "2026-09-05",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market": "moneyline",
                "side": "home",
                "odds": -200.0,
                "line": float("nan"),
            },
            {
                "fetched_at": ts,
                "sportsbook": "draftkings",
                "season": "2026-2027",
                "week": 1,
                "game_id": game_id,
                "game_date": "2026-09-05",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market": "moneyline",
                "side": "away",
                "odds": 170.0,
                "line": float("nan"),
            },
            {
                "fetched_at": ts,
                "sportsbook": "draftkings",
                "season": "2026-2027",
                "week": 1,
                "game_id": game_id,
                "game_date": "2026-09-05",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market": "spread",
                "side": "home",
                "odds": -110.0,
                "line": -3.5,
            },
            {
                "fetched_at": ts,
                "sportsbook": "draftkings",
                "season": "2026-2027",
                "week": 1,
                "game_id": game_id,
                "game_date": "2026-09-05",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market": "spread",
                "side": "away",
                "odds": -110.0,
                "line": 3.5,
            },
            {
                "fetched_at": ts,
                "sportsbook": "draftkings",
                "season": "2026-2027",
                "week": 1,
                "game_id": game_id,
                "game_date": "2026-09-05",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market": "total",
                "side": "over",
                "odds": -110.0,
                "line": 44.0,
            },
            {
                "fetched_at": ts,
                "sportsbook": "draftkings",
                "season": "2026-2027",
                "week": 1,
                "game_id": game_id,
                "game_date": "2026-09-05",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
                "market": "total",
                "side": "under",
                "odds": -110.0,
                "line": 44.0,
            },
        ]
    )


def _make_games_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GAME_ID": "2026_01_KC_LAC",
                "YEAR": "2026-2027",
                "WEEK_NUM": 1,
                "GAME_DATE": "2026-09-05",
            }
        ]
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with settings pointing at tmp_path.

    Populated per-test via MiniRepoBuilder.
    """
    monkeypatch.setattr(
        "gridiron_edge.api.loaders.load_team_name_map",
        lambda _settings: {
            "Kansas City Chiefs": "KC",
            "Los Angeles Chargers": "LAC",
        },
    )

    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


class TestListEdgesRoute:
    def test_returns_ranked_edges_for_week(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction(game_id="2026_01_KC_LAC")])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(predictions)
            .with_odds_snapshot(_make_odds_snapshot())
        )

        response: Response = client.get(
            "/edges",
            params={
                "season": "2026-2027",
                "week": 1,
                "bankroll": 2500.0,
                "kelly_multiplier": 0.1,
            },
        )

        assert response.status_code == 200
        body = response.json()

        assert body["bankroll"] == 2500.0
        assert body["kelly_multiplier"] == 0.1
        assert body["items"][0]["american_odds"] != 0

        assert body["season"] == "2026-2027"
        assert body["week"] == 1
        assert body["min_ev"] == 0.0
        assert body["total"] > 0
        assert len(body["items"]) > 0

        # Every item conforms to schema shape.
        first = body["items"][0]
        assert first["game_id"] == "2026_01_KC_LAC"
        assert first["home_team"] == "LAC"
        assert first["away_team"] == "KC"
        assert first["model_key"] == "win_prob_elo"
        assert first["market_type"] in {"moneyline", "spread", "total"}
        assert first["side"] in {"home", "away", "over", "under"}

    def test_omitted_bankroll_leaves_dollar_stake_unavailable(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        predictions = pd.DataFrame([_make_prediction()])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(predictions)
            .with_odds_snapshot(_make_odds_snapshot())
        )

        response = client.get(
            "/edges",
            params={
                "season": "2026-2027",
                "week": 1,
                "kelly_multiplier": 0.25,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["bankroll"] is None
        assert body["kelly_multiplier"] == 0.25
        assert body["items"]

        for item in body["items"]:
            assert item["kelly_frac"] is not None
            assert item["kelly_stake"] is None

    @pytest.mark.parametrize(
        "invalid_params",
        [
            {"bankroll": -1.0},
            {"kelly_multiplier": -0.01},
            {"kelly_multiplier": 1.01},
        ],
    )
    def test_rejects_invalid_sizing_query(
        self,
        client: TestClient,
        invalid_params: dict[str, float],
    ) -> None:
        response = client.get(
            "/edges",
            params={
                "season": "2026-2027",
                "week": 1,
                **invalid_params,
            },
        )

        assert response.status_code == 422

    def test_missing_manifest_returns_field_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # Predictions + odds + games, but no manifest.
        predictions = pd.DataFrame([_make_prediction()])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_predictions_archive(predictions)
            .with_odds_snapshot(_make_odds_snapshot())
        )

        response: Response = client.get(
            "/edges",
            params={
                "season": "2026-2027",
                "week": 1,
                "bankroll": 2500.0,
                "kelly_multiplier": 0.1,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0
        status = body["_meta"]["field_status"]["items"]
        assert status["status"] == "blocked"
        assert status["blocker"] == "no_champion_manifest"
        assert body["items"] == []
        assert body["bankroll"] == 2500.0
        assert body["kelly_multiplier"] == 0.1

    def test_missing_odds_returns_field_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # Manifest + predictions + games, but no odds snapshot.
        predictions = pd.DataFrame([_make_prediction()])
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(predictions)
            # no with_odds_snapshot
        )

        response: Response = client.get(
            "/edges",
            params={
                "season": "2026-2027",
                "week": 1,
                "bankroll": 2500.0,
                "kelly_multiplier": 0.1,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0
        status = body["_meta"]["field_status"]["items"]
        assert status["status"] == "blocked"
        assert status["blocker"] == "no_odds_available"
        assert body["items"] == []
        assert body["bankroll"] == 2500.0
        assert body["kelly_multiplier"] == 0.1

    def test_empty_predictions_returns_empty_no_field_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # Everything present, but archive has no rows for this week.
        # Empty archive is a legitimate state (no predictions written yet),
        # so no field_status marker.
        (
            MiniRepoBuilder(tmp_path)
            .with_games(_make_games_df())
            .with_champion_manifest()
            .with_predictions_archive(
                pd.DataFrame(
                    [
                        _make_prediction(
                            game_id="2026_01_KC_LAC",
                            # week 2 — different from what we query for
                        )
                    ]
                ).assign(week=2)
            )
            .with_odds_snapshot(_make_odds_snapshot())
        )

        response: Response = client.get(
            "/edges",
            params={
                "season": "2026-2027",
                "week": 1,
                "bankroll": 2500.0,
                "kelly_multiplier": 0.1,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0
        # No field_status when empty is a legitimate state.
        assert (
            "_meta" not in body
            or body.get("_meta") is None
            or "field_status" not in body["_meta"]
            or "items" not in body["_meta"].get("field_status", {})
        )
        assert body["items"] == []
        assert body["bankroll"] == 2500.0
        assert body["kelly_multiplier"] == 0.1

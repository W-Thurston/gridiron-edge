"""Integration tests for schedule-complete /games routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.repos import MiniRepoBuilder

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency
from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.models.game_prediction.weekly_game_product import (
    build_weekly_game_product,
)

SEASON = "2026-2027"
WEEK = 1
GENERATED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)


@dataclass
class _FakeSettings:
    repo_root: Path


def _selected_product() -> DataFrame:
    """Return one available and one forecast-missing scheduled game."""
    base = DataFrame(
        {
            "season": [SEASON, SEASON],
            "week": [WEEK, WEEK],
            "game_id": ["2026_01_KC_LAC", "2026_01_BUF_MIA"],
            "game_day_of_week": ["Saturday", "Sunday"],
            "game_date": ["2026-09-05", "2026-09-06"],
            "game_time": ["18:00:00", "13:00:00"],
            "away_team": ["Kansas City Chiefs", "Buffalo Bills"],
            "home_team": ["Los Angeles Chargers", "Miami Dolphins"],
            "neutral_site": [False, False],
            "stadium": ["SoFi Stadium", "Hard Rock Stadium"],
            "win_status": ["available", "forecast_missing"],
            "win_selection_status": ["selected", "no_eligible_candidate"],
            "away_win_prob": [0.45, pd.NA],
            "home_win_prob": [0.55, pd.NA],
            "win_model_name": ["win_prob", pd.NA],
            "win_model_type": ["elo", pd.NA],
            "win_event_id": ["win-event-1", pd.NA],
            "win_run_id": ["api-games-run", pd.NA],
            "win_generated_at": [GENERATED_AT, pd.NaT],
            "win_role": ["live", pd.NA],
            "spread_status": ["available", "win_unavailable"],
            "model_spread": [-2.5, pd.NA],
            "spread_uncertainty": [13.0, pd.NA],
            "spread_source_event_id": ["win-event-1", pd.NA],
            "spread_model_name": ["win_prob", pd.NA],
            "spread_model_type": ["elo", pd.NA],
            "spread_calibration_key": ["win_prob_elo", pd.NA],
            "spread_calibration_updated_at": [
                "2026-09-01T12:00:00+00:00",
                pd.NA,
            ],
            "total_status": ["available", "forecast_missing"],
            "total_selection_status": ["selected", "no_eligible_candidate"],
            "model_total": [47.5, pd.NA],
            "total_uncertainty": [12.0, pd.NA],
            "total_model_name": ["total", pd.NA],
            "total_model_type": ["random_forest", pd.NA],
            "total_event_id": ["total-event-1", pd.NA],
            "total_run_id": ["api-games-run", pd.NA],
            "total_generated_at": [GENERATED_AT, pd.NaT],
            "total_role": ["live", pd.NA],
            "total_uncertainty_trained_at": [
                "2026-08-31T12:00:00+00:00",
                pd.NA,
            ],
        }
    )
    return build_weekly_game_product(base)


def _persist_selected_product(repo: Path) -> None:
    """Persist and explicitly select the schedule-complete product."""
    identity = WeeklyProductIdentity(
        product_id="api-games-product",
        run_id="api-games-run",
        season=SEASON,
        week=WEEK,
        generated_at=GENERATED_AT,
    )
    write_weekly_product(repo, _selected_product(), identity=identity)
    select_current_weekly_product(
        repo,
        identity.product_id,
        season=SEASON,
        week=WEEK,
        selected_at=datetime(2026, 9, 1, 13, tzinfo=UTC),
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a client whose API repository is the test repository."""
    MiniRepoBuilder(tmp_path).with_games()
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


def test_list_returns_every_selected_scheduled_game(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _persist_selected_product(tmp_path)

    response = client.get(f"/games?season={SEASON}&week={WEEK}")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [item["game_id"] for item in body["items"]] == [
        "2026_01_KC_LAC",
        "2026_01_BUF_MIA",
    ]


def test_list_keeps_missing_prediction_components_visible(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _persist_selected_product(tmp_path)

    body = client.get(f"/games?season={SEASON}&week={WEEK}").json()
    missing = next(item for item in body["items"] if item["game_id"] == "2026_01_BUF_MIA")

    assert missing["win"]["status"] == "forecast_missing"
    assert missing["win"]["home_win_prob"] is None
    assert missing["spread"]["status"] == "win_unavailable"
    assert missing["total"]["status"] == "forecast_missing"
    assert missing["total"]["model_total"] is None
    assert missing["projected_score"]["status"] == "spread_and_total_unavailable"


def test_list_serializes_separate_win_spread_and_total_provenance(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _persist_selected_product(tmp_path)

    body = client.get(f"/games?season={SEASON}&week={WEEK}").json()
    available = next(item for item in body["items"] if item["game_id"] == "2026_01_KC_LAC")

    assert available["away_team"] == "Kansas City Chiefs"
    assert available["home_team"] == "Los Angeles Chargers"
    assert available["win"]["event_id"] == "win-event-1"
    assert available["win"]["run_id"] == "api-games-run"
    assert available["win"]["model_type"] == "elo"
    assert available["spread"]["source_event_id"] == "win-event-1"
    assert available["spread"]["model_spread"] == pytest.approx(-2.5)
    assert available["total"]["event_id"] == "total-event-1"
    assert available["total"]["run_id"] == "api-games-run"
    assert available["total"]["model_type"] == "random_forest"


def test_scheduled_game_detail_is_200_when_predictions_are_missing(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _persist_selected_product(tmp_path)

    response = client.get("/games/2026_01_BUF_MIA")

    assert response.status_code == 200
    body = response.json()
    assert body["game_id"] == "2026_01_BUF_MIA"
    assert body["away_team"] == "Buffalo Bills"
    assert body["home_team"] == "Miami Dolphins"
    assert body["win"]["status"] == "forecast_missing"
    assert body["total"]["status"] == "forecast_missing"
    assert body["projected_score"]["home"] is None


def test_game_detail_uses_persisted_schedule_metadata(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _persist_selected_product(tmp_path)

    body = client.get("/games/2026_01_KC_LAC").json()

    assert body["day_of_week"] == "Saturday"
    assert body["kick"] == "18:00:00"
    assert body["venue"] == "SoFi Stadium"
    assert "kick" not in body["_meta"]["field_status"]
    assert "venue" not in body["_meta"]["field_status"]


def test_unknown_game_returns_404(
    client: TestClient,
    tmp_path: Path,
) -> None:
    _persist_selected_product(tmp_path)

    response = client.get("/games/2026_01_NYJ_NE")

    assert response.status_code == 404
    assert "Unknown game_id" in response.json()["detail"]


def test_missing_selected_product_returns_empty_list(
    client: TestClient,
) -> None:
    response = client.get(f"/games?season={SEASON}&week={WEEK}")

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0

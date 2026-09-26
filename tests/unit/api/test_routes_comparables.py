# tests/unit/api/test_routes_comparables.py

"""Tests for GET /games/{game_id}/comparables."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.evaluation.comparable_games_evidence import (
    EUCLIDEAN_STANDARDIZED_METRIC,
    EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
    ComparableFeatureContribution,
    ComparableGameMatch,
    create_comparable_games_batch,
)
from gridiron_edge.evaluation.comparable_games_evidence_store import (
    AmbiguousComparableGamesError,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _row(
    *,
    win_status: str = "available",
    win_model_type: str | None = "logistic",
    win_event_id: str = "event-1",
) -> dict:
    return {
        "game_id": "2026_02_CAR_ATL",
        "win_status": win_status,
        "win_model_type": win_model_type,
        "win_event_id": win_event_id,
    }


def _batch():
    feature_schema = create_prediction_feature_schema(
        model_name="win_prob",
        model_type="logistic",
        task="classification",
        modeling_schema_version=5,
        epa_window=6,
        feature_set_name="combined_111",
        ordered_columns=("ELO_DIFF",),
    )
    match = ComparableGameMatch(
        game_id="2006_19_DAL_SEA",
        rank=1,
        distance=8.59,
        season="2006-2007",
        week=19,
        game_date="2007-01-06",
        away_team="Dallas Cowboys",
        home_team="Seattle Seahawks",
        away_score=20,
        home_score=21,
        favorite_team="Seattle Seahawks",
        spread_magnitude=1.0,
        favorite_won=True,
        favorite_covered=False,
        top_contributing_features=(
            ComparableFeatureContribution(
                feature_name="ELO_DIFF",
                query_value=-2.3,
                candidate_value=0.15,
                squared_difference=6.13,
            ),
        ),
    )
    return create_comparable_games_batch(
        event_id="event-1",
        game_id="2026_02_CAR_ATL",
        corpus_id="d" * 64,
        model_name="win_prob",
        model_type="logistic",
        feature_schema=feature_schema,
        metric=EUCLIDEAN_STANDARDIZED_METRIC,
        metric_version=EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
        k_requested=20,
        distance_threshold=9.28,
        generated_at=datetime(2026, 9, 26, tzinfo=UTC),
        matches=(match,),
        sample_size=1,
        favorite_win_rate=1.0,
        favorite_cover_rate=0.0,
    )


class TestUnknownGame:
    def test_returns_404(self, client: TestClient) -> None:
        with patch("gridiron_edge.api.routes.comparables.load_game", return_value=None):
            response = client.get("/games/does-not-exist/comparables")
        assert response.status_code == 404


class TestLogisticWithEvidence:
    def test_returns_real_comparables(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.comparables.load_game",
                return_value=_row(),
            ),
            patch(
                "gridiron_edge.api.routes.comparables.load_comparable_games_for_event",
                return_value=_batch(),
            ) as mocked_loader,
        ):
            response = client.get("/games/2026_02_CAR_ATL/comparables")

        assert response.status_code == 200
        body = response.json()
        assert body["sample_size"] == 1
        assert body["comparables"][0]["game_id"] == "2006_19_DAL_SEA"
        assert body["comparables"][0]["favorite_won"] is True
        assert "comparables" not in body.get("_meta", {}).get("field_status", {})
        mocked_loader.assert_called_once()
        assert mocked_loader.call_args.kwargs == {"event_id": "event-1"}


class TestNonLogisticChampion:
    def test_comparables_blocked(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.comparables.load_game",
                return_value=_row(win_model_type="xgboost"),
            ),
            patch(
                "gridiron_edge.api.routes.comparables.load_comparable_games_for_event",
            ) as mocked_loader,
        ):
            response = client.get("/games/2026_02_CAR_ATL/comparables")

        assert response.status_code == 200
        body = response.json()
        assert body["_meta"]["field_status"]["comparables"]["blocker"] == "comparables_retrieval"
        mocked_loader.assert_not_called()


class TestAmbiguousEvidence:
    def test_comparables_blocked_not_500(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.comparables.load_game",
                return_value=_row(),
            ),
            patch(
                "gridiron_edge.api.routes.comparables.load_comparable_games_for_event",
                side_effect=AmbiguousComparableGamesError("multiple batches"),
            ),
        ):
            response = client.get("/games/2026_02_CAR_ATL/comparables")

        assert response.status_code == 200
        body = response.json()
        assert (
            body["_meta"]["field_status"]["comparables"]["blocker"]
            == "ambiguous_comparable_evidence"
        )


class TestWinUnavailable:
    def test_all_fields_blocked(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.comparables.load_game",
                return_value=_row(win_status="forecast_missing"),
            ),
            patch(
                "gridiron_edge.api.routes.comparables.load_comparable_games_for_event",
            ) as mocked_loader,
        ):
            response = client.get("/games/2026_02_CAR_ATL/comparables")

        assert response.status_code == 200
        body = response.json()
        assert body["_meta"]["field_status"]["comparables"]["blocker"] == "no_win_forecast"
        assert body["_meta"]["field_status"]["sample_size"]["blocker"] == "no_win_forecast"
        mocked_loader.assert_not_called()


class TestNoEvidenceYet:
    def test_comparables_blocked_no_evidence(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.comparables.load_game",
                return_value=_row(),
            ),
            patch(
                "gridiron_edge.api.routes.comparables.load_comparable_games_for_event",
                return_value=None,
            ),
        ):
            response = client.get("/games/2026_02_CAR_ATL/comparables")

        assert response.status_code == 200
        body = response.json()
        assert body["_meta"]["field_status"]["comparables"]["blocker"] == "no_comparable_evidence"

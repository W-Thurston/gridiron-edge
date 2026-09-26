# tests/unit/api/test_routes_explain.py

"""Tests for GET /games/{game_id}/explain."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.evaluation.logistic_explanation_evidence import (
    LogisticExplanationEvent,
    LogisticFeatureContribution,
    sigmoid,
)
from gridiron_edge.evaluation.logistic_explanation_evidence_store import (
    AmbiguousLogisticExplanationError,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _row(
    *,
    win_status: str = "available",
    win_model_type: str | None = "logistic",
    home_win_prob: float | None = 0.62,
    win_event_id: str = "event-1",
) -> dict:
    return {
        "game_id": "2026_02_CAR_ATL",
        "win_status": win_status,
        "win_model_type": win_model_type,
        "home_win_prob": home_win_prob,
        "win_event_id": win_event_id,
    }


def _explanation_event() -> LogisticExplanationEvent:
    contribution = LogisticFeatureContribution(
        feature_name="ELO_DIFF",
        transformed_value=0.5,
        coefficient=0.8,
        contribution=0.4,
    )
    reconstructed_log_odds = 0.1 + contribution.contribution
    reconstructed_probability = sigmoid(reconstructed_log_odds)
    return LogisticExplanationEvent(
        event_id="event-1",
        game_id="2026_02_CAR_ATL",
        intercept=0.1,
        contributions=(contribution,),
        reconstructed_log_odds=reconstructed_log_odds,
        reconstructed_probability=reconstructed_probability,
        raw_estimator_output=reconstructed_probability,
        tolerance=1e-9,
    )


class TestUnknownGame:
    def test_returns_404(self, client: TestClient) -> None:
        with patch("gridiron_edge.api.routes.explain.load_game", return_value=None):
            response = client.get("/games/does-not-exist/explain")
        assert response.status_code == 404


class TestLogisticWithEvidence:
    def test_returns_real_factors(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.explain.load_game",
                return_value=_row(),
            ),
            patch(
                "gridiron_edge.api.routes.explain.load_logistic_explanation_for_event",
                return_value=_explanation_event(),
            ) as mocked_loader,
        ):
            response = client.get("/games/2026_02_CAR_ATL/explain")

        assert response.status_code == 200
        body = response.json()
        assert body["headline_win_prob"] == 0.62
        assert body["factors"][0]["key"] == "intercept"
        assert body["factors"][1]["key"] == "ELO_DIFF"
        assert body["factors"][1]["log_odds_contribution"] == 0.4
        assert "factors" not in body.get("_meta", {}).get("field_status", {})
        mocked_loader.assert_called_once()
        assert mocked_loader.call_args.kwargs == {"event_id": "event-1"}


class TestNonLogisticChampion:
    def test_factors_blocked(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.explain.load_game",
                return_value=_row(win_model_type="xgboost"),
            ),
            patch(
                "gridiron_edge.api.routes.explain.load_logistic_explanation_for_event",
            ) as mocked_loader,
        ):
            response = client.get("/games/2026_02_CAR_ATL/explain")

        assert response.status_code == 200
        body = response.json()
        assert body["_meta"]["field_status"]["factors"]["blocker"] == "feature_attribution"
        mocked_loader.assert_not_called()


class TestAmbiguousEvidence:
    def test_factors_blocked_not_500(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.explain.load_game",
                return_value=_row(),
            ),
            patch(
                "gridiron_edge.api.routes.explain.load_logistic_explanation_for_event",
                side_effect=AmbiguousLogisticExplanationError("multiple batches"),
            ),
        ):
            response = client.get("/games/2026_02_CAR_ATL/explain")

        assert response.status_code == 200
        body = response.json()
        assert body["headline_win_prob"] == 0.62
        assert (
            body["_meta"]["field_status"]["factors"]["blocker"] == "ambiguous_explanation_evidence"
        )


class TestWinUnavailable:
    def test_headline_and_factors_blocked(self, client: TestClient) -> None:
        with (
            patch(
                "gridiron_edge.api.routes.explain.load_game",
                return_value=_row(win_status="forecast_missing", home_win_prob=None),
            ),
            patch(
                "gridiron_edge.api.routes.explain.load_logistic_explanation_for_event",
            ) as mocked_loader,
        ):
            response = client.get("/games/2026_02_CAR_ATL/explain")

        assert response.status_code == 200
        body = response.json()
        assert body["_meta"]["field_status"]["headline_win_prob"]["blocker"] == "no_win_forecast"
        assert body["_meta"]["field_status"]["factors"]["blocker"] == "no_win_forecast"
        mocked_loader.assert_not_called()

# tests/unit/api/test_routes_recaps.py

"""Tests for GET /recaps and GET /recaps/{season}/{week} (``DECISIONS.md`` D86)."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from tests.fixtures.weekly_recap import (
    BUILT_AT,
    GAME_A,
    GAME_B,
    GAME_C,
    GAME_D,
    SEASON,
    WEEK,
    recap_week,
)

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency
from gridiron_edge.evaluation.weekly_recap import BuiltWeeklyRecap, build_weekly_recap
from gridiron_edge.evaluation.weekly_recap_store import (
    select_current_weekly_recap,
    weekly_recap_manifest_path,
    weekly_recap_root,
    write_weekly_recap,
)

pytestmark = pytest.mark.unit

SELECTED_AT = datetime(2026, 10, 7, 13, 0, tzinfo=UTC)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: SimpleNamespace(repo_root=tmp_path)
    return TestClient(app)


def _selected(tmp_path: Path) -> BuiltWeeklyRecap:
    week = recap_week()
    built = build_weekly_recap(
        season=SEASON,
        week=WEEK,
        schedule=week.schedule,
        product=week.product,
        forecast_events=week.events,
        games=week.games,
        built_at=BUILT_AT,
    )
    write_weekly_recap(built, repo=tmp_path)
    select_current_weekly_recap(built.recap.recap_id, selected_at=SELECTED_AT, repo=tmp_path)
    return built


def _game(body: dict, game_id: str) -> dict:
    return next(game for game in body["games"] if game["game_id"] == game_id)


class TestList:
    def test_no_selection_is_an_empty_list(self, client: TestClient) -> None:
        response = client.get("/recaps")

        assert response.status_code == 200
        assert response.json()["items"] == []

    def test_lists_the_selected_recap_with_family_availability(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        built = _selected(tmp_path)

        body = client.get("/recaps").json()

        assert body["items"] == [
            {
                "season": SEASON,
                "week": WEEK,
                "recap_id": built.recap.recap_id,
                "built_at": "2026-10-07T12:00:00Z",
                "selected_at": "2026-10-07T13:00:00Z",
                "product_id": built.recap.product_id,
                "forecast_role": "live",
                "row_forecast_roles": ["live"],
                "scheduled_games": 4,
                "in_product": 3,
                # A available; B tied (outcome_tie); C not in product; D total missing.
                "available": {"win": 2, "spread": 3, "total": 2, "projected_score": 2},
            }
        ]

    def test_a_tampered_recap_is_a_server_error(self, client: TestClient, tmp_path: Path) -> None:
        built = _selected(tmp_path)
        frame = weekly_recap_root(tmp_path) / built.recap.games.artifact
        tampered = built.games.copy()
        tampered.loc[0, "home_win_prob"] = 0.99
        tampered.to_parquet(frame, index=False)

        response = client.get("/recaps")

        assert response.status_code == 500
        assert response.json()["detail"] == "Weekly recap artifacts failed verification."


class TestDetail:
    def test_an_unselected_week_is_a_404_naming_the_command(self, client: TestClient) -> None:
        response = client.get(f"/recaps/{SEASON}/{WEEK}")

        assert response.status_code == 404
        assert (
            f"gridiron evaluate weekly-recap --season {SEASON} --week {WEEK}"
            in response.json()["detail"]
        )

    def test_a_week_below_one_is_rejected(self, client: TestClient) -> None:
        assert client.get(f"/recaps/{SEASON}/0").status_code == 422

    def test_a_selected_manifest_that_no_longer_matches_its_id_is_a_server_error(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        built = _selected(tmp_path)
        path = weekly_recap_manifest_path(built.recap.recap_id, repo=tmp_path)
        payload = json.loads(path.read_text())
        payload["product_id"] = "another_product"
        path.write_text(json.dumps(payload))

        response = client.get(f"/recaps/{SEASON}/{WEEK}")

        assert response.status_code == 500

    def test_a_selected_recap_whose_files_are_gone_is_a_server_error(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        built = _selected(tmp_path)
        weekly_recap_manifest_path(built.recap.recap_id, repo=tmp_path).unlink()

        assert client.get(f"/recaps/{SEASON}/{WEEK}").status_code == 500

    def test_serves_the_manifest_metrics_and_every_scheduled_game(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        built = _selected(tmp_path)

        response = client.get(f"/recaps/{SEASON}/{WEEK}")

        assert response.status_code == 200
        body = response.json()
        assert body["recap_id"] == built.recap.recap_id
        assert body["selected_at"] == "2026-10-07T13:00:00Z"
        assert body["forecast_role"] == "live"
        assert body["carried_run_ids"] == list(built.recap.carried_run_ids)
        assert body["counts"] == {
            "scheduled_games": 4,
            "in_product": 3,
            "not_in_product": 1,
            "carried_rows": 1,
            "outcome_ties": 1,
        }
        win = built.recap.metrics.win
        assert body["metrics"]["win"]["model"]["brier"] == pytest.approx(win.model.brier)
        assert body["metrics"]["win"]["market"]["count"] == win.market.count
        assert body["metrics"]["spread"]["closer"] == {
            "model": built.recap.metrics.spread.closer.model,
            "market": built.recap.metrics.spread.closer.market,
            "tie": built.recap.metrics.spread.closer.tie,
        }
        assert [game["game_id"] for game in body["games"]] == list(built.games["game_id"])
        assert {GAME_A, GAME_B, GAME_C, GAME_D} == {game["game_id"] for game in body["games"]}

    def test_unbuilt_sections_are_null_and_pending(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _selected(tmp_path)

        body = client.get(f"/recaps/{SEASON}/{WEEK}").json()

        for section in ("starting_qbs", "edges", "wagers", "season_to_date"):
            assert body[section] is None
            assert body["_meta"]["field_status"][section] == "pending"

    def test_game_statuses_explain_every_missing_value(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _selected(tmp_path)

        body = client.get(f"/recaps/{SEASON}/{WEEK}").json()

        not_in_product = _game(body, GAME_C)
        assert not_in_product["product_row_status"] == "not_in_product"
        assert not_in_product["forecast"] == {
            "role": None,
            "run_id": None,
            "generated_at": None,
            "carried": None,
        }
        assert not_in_product["win"]["status"] == "not_in_product"
        assert not_in_product["win"]["home_win_prob"] is None
        assert not_in_product["market_at_forecast"]["status"] == "not_in_product"
        assert not_in_product["closing_market"]["home_margin"] == -1.0

        tied = _game(body, GAME_B)
        assert tied["forecast"]["carried"] is True
        assert tied["result"]["home_win"] is None
        assert tied["win"]["status"] == "outcome_tie"
        assert tied["win"]["home_win_prob"] is not None
        assert tied["win"]["model_brier"] is None

        missing = _game(body, GAME_D)
        assert missing["total"]["status"] == "forecast_missing"
        assert missing["total"]["value"] is None
        assert missing["projected_score"]["status"] == "forecast_missing"
        assert missing["closing_market"]["spread_status"] == "closing_line_missing"
        assert missing["closing_market"]["total_status"] == "closing_line_missing"
        assert missing["closing_market"]["home_win_prob"] is None
        assert missing["spread"]["closer"] is None


class TestOpenApi:
    def test_statuses_are_schema_enums(self, client: TestClient) -> None:
        schemas = client.get("/openapi.json").json()["components"]["schemas"]

        assert set(schemas["ForecastStatus"]["enum"]) == {
            "available",
            "forecast_missing",
            "not_in_product",
            "outcome_tie",
        }
        assert set(schemas["ProductRowStatus"]["enum"]) == {"in_product", "not_in_product"}
        assert set(schemas["ForecastMarketStatus"]["enum"]) == {
            "available",
            "not_recorded",
            "not_in_product",
        }
        assert set(schemas["ClosingLineStatus"]["enum"]) == {
            "available",
            "closing_line_missing",
        }
        assert set(schemas["Closer"]["enum"]) == {"model", "market", "tie"}
        assert "development" in schemas["ForecastRole"]["enum"]

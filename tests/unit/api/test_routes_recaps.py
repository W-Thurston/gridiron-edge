# tests/unit/api/test_routes_recaps.py

"""Tests for GET /recaps, /recaps/{season}/{week} and /recaps/{season}/summary (D86)."""

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
    ledger,
    recap_week,
)

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency
from gridiron_edge.evaluation.weekly_recap import BuiltWeeklyRecap, build_weekly_recap
from gridiron_edge.evaluation.weekly_recap_season import SeasonRecap, build_season_recap
from gridiron_edge.evaluation.weekly_recap_store import (
    season_recap_manifest_path,
    select_current_season_recap,
    select_current_weekly_recap,
    weekly_recap_manifest_path,
    weekly_recap_root,
    write_season_recap,
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
        player_game_logs=week.logs,
        sidecar_captures=week.captures,
        markets=week.markets,
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

    def test_serves_the_edge_and_wager_sections(self, client: TestClient, tmp_path: Path) -> None:
        built = _selected(tmp_path)

        body = client.get(f"/recaps/{SEASON}/{WEEK}").json()

        edges = body["edges"]
        assert edges["status"] == "withheld_unvalidated"
        assert edges["chain"] is None
        assert edges["receipts_through"] == "2026-10-06T00:15:00Z"
        assert [(item["outcome"], item["product_id"]) for item in edges["receipts"]] == [
            ("done", "weekly_product"),
            ("blocked", "weekly_product"),
        ]
        wagers = body["wagers"]
        assert wagers["ledger_digest"] == built.recap.ledger_digest
        total = wagers["total"]
        assert (total["bets"], total["settled"], total["unsettled"]) == (4, 3, 1)
        assert (total["wins"], total["losses"], total["bonus_funded"]) == (1, 2, 1)
        assert total["pnl"] == pytest.approx(0.66 - 1.0 - 0.1)
        assert total["clv_unavailable"] == 4
        assert [item["market_type"] for item in wagers["markets"]] == [
            "moneyline",
            "same_game_parlay",
        ]
        assert body["_meta"] is None  # nothing is pending
        assert "season_to_date" not in body

    def test_serves_the_starter_summary_and_every_side(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        built = _selected(tmp_path)

        body = client.get(f"/recaps/{SEASON}/{WEEK}").json()

        starters = body["starting_qbs"]
        assert starters["slots"] == built.recap.starters.slots == 8
        assert starters["scored"] == 4
        assert starters["assumed"] == {"count": 4, "correct": 3, "accuracy": 0.75}
        assert starters["carry_forward"] == {"count": 4, "correct": 2, "accuracy": 0.5}
        assert starters["capture_digests"] == list(built.recap.starter_capture_digests)
        assert body["conventions"]["actual_starter"].startswith("nflverse *_qb_id")
        carried = _game(body, GAME_B)["starting_qbs"]["away"]
        assert carried["status"] == "available"
        assert carried["record_source"] == "sidecar_capture"
        assert (carried["assumed_id"], carried["actual_id"]) == ("qc1", "qc1")
        assert carried["actual_source"] == "game_log_top_passer"
        assert (carried["assumed_correct"], carried["nflverse_correct"]) == (True, False)
        not_in_product = _game(body, GAME_C)["starting_qbs"]["home"]
        assert not_in_product["status"] == "not_in_product"
        assert not_in_product["assumed_id"] is None
        assert not_in_product["actual_name"] == "Unforecast Home"
        assert not_in_product["assumed_correct"] is None
        unresolved = _game(body, GAME_D)["starting_qbs"]
        assert unresolved["home"]["status"] == "starter_unresolved"
        assert unresolved["away"]["status"] == "actual_missing"

    def test_starter_statuses_are_schema_enums(self, client: TestClient) -> None:
        schemas = client.get("/openapi.json").json()["components"]["schemas"]

        assert set(schemas["StarterStatus"]["enum"]) == {
            "available",
            "not_recorded",
            "starter_unresolved",
            "not_in_product",
            "actual_missing",
        }
        assert set(schemas["StarterRecordSource"]["enum"]) == {"product_record", "sidecar_capture"}
        assert set(schemas["ActualStarterSource"]["enum"]) == {
            "nflverse_schedule",
            "game_log_top_passer",
        }

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


def _season_selected(tmp_path: Path) -> SeasonRecap:
    built = _selected(tmp_path)
    recap = build_season_recap(
        bets=ledger(),
        season=SEASON,
        recaps=[(built.recap, built.games)],
        finished_weeks=(WEEK - 1, WEEK),
        built_at=BUILT_AT,
    )
    write_season_recap(recap, repo=tmp_path)
    select_current_season_recap(recap.season_recap_id, selected_at=SELECTED_AT, repo=tmp_path)
    return recap


class TestSeasonSummary:
    def test_an_unselected_season_is_a_404_naming_the_command(self, client: TestClient) -> None:
        response = client.get(f"/recaps/{SEASON}/summary")

        assert response.status_code == 404
        assert "--season-only" in response.json()["detail"]

    def test_serves_every_week_the_season_and_calibration(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        recap = _season_selected(tmp_path)

        response = client.get(f"/recaps/{SEASON}/summary")

        assert response.status_code == 200
        body = response.json()
        assert body["season_recap_id"] == recap.season_recap_id
        assert body["selected_at"] == "2026-10-07T13:00:00Z"
        assert body["weekly_recap_ids"] == list(recap.weekly_recap_ids)
        not_built, recapped = body["weeks"]
        assert (not_built["week"], not_built["status"]) == (WEEK - 1, "not_built")
        assert not_built["metrics"] is None and not_built["to_date"] is None
        assert recapped["status"] == "recapped"
        assert recapped["forecast_role"] == "live"
        assert recapped["edge_status"] == "withheld_unvalidated"
        assert recapped["wagers"]["bets"] == 4
        assert not_built["edge_status"] is None and not_built["wagers"] is None
        assert body["wagers"]["recapped_weeks"]["bets"] == 4
        assert body["wagers"]["unattributed"]["bets"] == 1
        assert body["wagers"]["unattributed"]["losses"] == 1
        assert (
            recapped["metrics"]["win"]["model"]["brier"] == recap.all_games.metrics.win.model.brier
        )
        season = body["all_games"]
        assert season["games"] == 4
        assert len(season["calibration"]["model"]["bins"]) == 10
        assert season["calibration"]["model"]["ece"] == recap.all_games.calibration.model.ece
        assert season["paired"]["win"]["count"] == recap.all_games.paired.win.count
        assert season["starting_qbs"]["slots"] == 8
        assert "capture_digests" not in season["starting_qbs"]
        assert body["conventions"]["interval_z"] == 1.96

    def test_a_tampered_season_recap_is_a_server_error(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        recap = _season_selected(tmp_path)
        path = season_recap_manifest_path(recap.season_recap_id, repo=tmp_path)
        payload = json.loads(path.read_text())
        payload["season"] = "2027-2028"
        path.write_text(json.dumps(payload))

        response = client.get(f"/recaps/{SEASON}/summary")

        assert response.status_code == 500
        assert response.json()["detail"] == "Weekly recap artifacts failed verification."

    def test_a_week_route_still_parses_weeks(self, client: TestClient, tmp_path: Path) -> None:
        _season_selected(tmp_path)

        assert client.get(f"/recaps/{SEASON}/{WEEK}").status_code == 200
        assert client.get(f"/recaps/{SEASON}/0").status_code == 422


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
        assert set(schemas["SeasonWeekStatus"]["enum"]) == {"recapped", "not_built"}
        assert set(schemas["EdgeEvidenceStatus"]["enum"]) == {
            "production_chain",
            "withheld_unvalidated",
            "not_recorded",
        }
        assert "not_evaluated" in schemas["CandidateRecommendationState"]["enum"]
        assert "kickoff_conflict" in schemas["MarketCloseoutStatus"]["enum"]
        assert "development" in schemas["ForecastRole"]["enum"]

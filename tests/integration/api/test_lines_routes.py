"""Integration coverage for the current Line Shopping list route."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency
from gridiron_edge.api.routes import lines as lines_route
from gridiron_edge.ingest.odds.store import (
    QUOTE_COLUMNS,
    write_current_odds_snapshot,
)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: SimpleNamespace(repo_root=tmp_path)
    return TestClient(app)


def quote_rows() -> pd.DataFrame:
    common = {
        "fetched_at": pd.Timestamp("2026-08-05T22:05:33Z"),
        "provider": "the_odds_api",
        "provider_event_id": "event-1",
        "sportsbook_updated_at": pd.Timestamp("2026-08-05T22:05:03Z"),
        "commence_time": pd.Timestamp("2026-09-10T00:15:00Z"),
        "is_live": False,
        "season": "2026-2027",
        "week": 1,
        "game_id": "2026_01_NE_SEA",
        "game_date": "2026-09-09",
        "away_team": "New England Patriots",
        "home_team": "Seattle Seahawks",
    }
    rows = [
        {
            **common,
            "sportsbook": "draftkings",
            "market": "spread",
            "side": "away",
            "odds": -110.0,
            "line": 3.5,
        },
        {
            **common,
            "sportsbook": "betrivers",
            "market": "spread",
            "side": "away",
            "odds": -114.0,
            "line": 4.5,
        },
        {
            **common,
            "sportsbook": "betmgm",
            "market": "moneyline",
            "side": "away",
            "odds": 175.0,
            "line": None,
        },
    ]
    return pd.DataFrame(rows, columns=list(QUOTE_COLUMNS))


def test_missing_snapshot_is_unavailable(client: TestClient) -> None:
    body = client.get("/lines?season=2026-2027&week=1").json()
    assert body["items"] == []
    assert body["total"] == 0
    status = body["_meta"]["field_status"]["items"]
    assert status["status"] == "blocked"
    assert status["blocker"] == "no_odds_available"
    assert status["roadmap"] == "data"


def test_returns_classified_current_quotes(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lines_route,
        "load_games_for_week",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "game_id": "2026_01_NE_SEA",
                    "win_status": "available",
                    "away_win_prob": 0.48,
                    "home_win_prob": 0.52,
                    "spread_status": "available",
                    "model_spread": -1.0,
                    "spread_uncertainty": 13.0,
                    "total_status": "available",
                    "model_total": 47.0,
                    "total_uncertainty": 13.0,
                    "product_id": "weekly-product",
                    "product_run_id": "weekly-run",
                }
            ]
        ),
    )
    write_current_odds_snapshot(quote_rows(), repo=tmp_path)
    response = client.get("/lines?season=2026-2027&week=1&market=spread")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["sportsbooks"] == ["betmgm", "betrivers", "draftkings"]
    offers = body["items"][0]["offers"]
    assert len(offers) == 2
    best_line = next(row for row in offers if row["is_best_line"])
    assert best_line["sportsbook"] == "betrivers"
    assert best_line["line"] == 4.5
    assert best_line["provider_event_id"] == "event-1"
    assert best_line["model_status"] == "available"
    assert best_line["expected_value"] is not None
    assert best_line["product_id"] == "weekly-product"
    guidance = body["items"][0]["guidance"]
    assert len(guidance) == 1
    assert guidance[0]["side"] == "away"
    assert guidance[0]["reference_odds"] == -110


def test_scope_without_rows_is_analytical_empty(
    client: TestClient,
    tmp_path: Path,
) -> None:
    write_current_odds_snapshot(quote_rows(), repo=tmp_path)
    body = client.get("/lines?season=2026-2027&week=2").json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body.get("_meta") is None


def test_missing_selected_product_preserves_raw_quotes(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_current_odds_snapshot(quote_rows(), repo=tmp_path)

    def missing_product(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise FileNotFoundError("No current weekly product selected")

    monkeypatch.setattr(lines_route, "load_games_for_week", missing_product)
    body = client.get("/lines?season=2026-2027&week=1&market=spread").json()

    assert len(body["items"][0]["offers"]) == 2
    assert body["items"][0]["guidance"] == []
    assert {offer["model_status"] for offer in body["items"][0]["offers"]} == {"model_unavailable"}

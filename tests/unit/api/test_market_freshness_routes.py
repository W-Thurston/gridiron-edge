# tests/unit/api/test_market_freshness_routes.py

"""Tests for the market freshness context served with edges and lines (``DECISIONS.md`` D80)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import clock_dependency, settings_dependency
from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS
from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeDiagnostics,
    EdgeResultState,
)
from gridiron_edge.market.edge_report import EdgeResult
from gridiron_edge.market.line_shopping import classify_line_shopping_offers

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: SimpleNamespace(repo_root=tmp_path)
    app.dependency_overrides[clock_dependency] = lambda: NOW
    return TestClient(app)


def _edge_result(blockers: tuple[EdgeDiagnosticBlocker, ...] = ()) -> EdgeResult:
    return EdgeResult(
        rows=DataFrame(),
        diagnostics=EdgeDiagnostics(
            season="2026-2027",
            week=4,
            prediction_game_count=0,
            market_game_count=0,
            matched_game_count=0,
            complete_moneyline_count=0,
            complete_spread_count=0,
            complete_total_count=0,
            eligible_market_count=0,
            calculated_edge_count=0,
            positive_edge_count=0,
            filtered_edge_count=0,
            state=EdgeResultState.BLOCKED if blockers else EdgeResultState.NO_CALCULABLE_EDGES,
            blockers=blockers,
        ),
    )


def test_edges_serve_the_clock_and_the_limit_and_pass_the_clock_to_the_service(
    client: TestClient,
) -> None:
    with patch(
        "gridiron_edge.api.routes.edges.load_edges_for_week", return_value=_edge_result()
    ) as service:
        body = client.get("/edges", params={"season": "2026-2027", "week": 4}).json()

    assert service.call_args.kwargs["now"] == NOW
    assert body["freshness"] == {"evaluated_at": "2026-10-04T12:00:00Z", "max_age_seconds": 86400}


def test_stale_market_data_is_a_blocked_state_with_its_own_reason(client: TestClient) -> None:
    result = _edge_result((EdgeDiagnosticBlocker.MARKET_STALE,))
    with patch("gridiron_edge.api.routes.edges.load_edges_for_week", return_value=result):
        body = client.get("/edges", params={"season": "2026-2027", "week": 4}).json()

    assert body["diagnostics"]["blockers"] == ["market_stale"]
    assert body["_meta"]["field_status"]["items"]["blocker"] == "stale_market_data"
    assert body["items"] == []


def _moneyline_quotes(fetched_at: str) -> DataFrame:
    rows = []
    for side, odds in (("home", -120.0), ("away", 110.0)):
        rows.append(
            {
                "fetched_at": pd.Timestamp(fetched_at, tz="UTC"),
                "provider": "the_odds_api",
                "provider_event_id": "evt",
                "sportsbook": "book",
                "sportsbook_updated_at": pd.Timestamp(fetched_at, tz="UTC"),
                "commence_time": pd.Timestamp("2026-10-10T17:00:00Z"),
                "is_live": False,
                "season": "2026-2027",
                "week": 4,
                "game_id": "2026_04_AAA_BBB",
                "game_date": "2026-10-10",
                "away_team": "A",
                "home_team": "B",
                "market": "moneyline",
                "side": side,
                "odds": odds,
                "line": float("nan"),
            }
        )
    return DataFrame(rows, columns=list(QUOTE_COLUMNS))


def _lines(client: TestClient, fetched_at: str) -> tuple[dict, MagicMock]:
    guidance = MagicMock()
    classified = classify_line_shopping_offers(_moneyline_quotes(fetched_at))
    guidance.return_value = SimpleNamespace(offers=classified, guidance=None)
    with (
        patch("gridiron_edge.api.routes.lines.load_current_odds") as quotes,
        patch("gridiron_edge.api.routes.lines.load_games_for_week") as games,
        patch("gridiron_edge.api.routes.lines.load_recommended_bet_results_for_week") as recs,
        patch("gridiron_edge.api.routes.lines.evaluate_line_shopping_guidance", guidance),
    ):
        quotes.return_value = _moneyline_quotes(fetched_at)
        games.return_value = DataFrame({"game_id": ["2026_04_AAA_BBB"]})
        recs.return_value = ()
        body = client.get("/lines", params={"season": "2026-2027", "week": 4}).json()
    return body, guidance


def test_fresh_lines_get_price_guidance_and_the_context(client: TestClient) -> None:
    body, guidance = _lines(client, "2026-10-04T09:00:00")

    assert guidance.called
    assert body["freshness"] == {"evaluated_at": "2026-10-04T12:00:00Z", "max_age_seconds": 86400}
    assert "guidance" not in (body.get("_meta") or {}).get("field_status", {})
    assert body["items"], "the quotes themselves are always shown"


def test_stale_lines_keep_the_quotes_but_withhold_price_guidance(client: TestClient) -> None:
    body, guidance = _lines(client, "2026-09-30T12:38:29")

    assert not guidance.called
    assert body["_meta"]["field_status"]["guidance"]["blocker"] == "stale_market_data"
    assert body["items"], "stale quotes are still shown, with their age"
    assert body["market_fetched_at"] == ["2026-09-30T12:38:29Z"]


def test_the_limit_boundary_is_inclusive(client: TestClient) -> None:
    body, guidance = _lines(client, "2026-10-03T12:00:00")

    assert guidance.called
    assert "guidance" not in (body.get("_meta") or {}).get("field_status", {})

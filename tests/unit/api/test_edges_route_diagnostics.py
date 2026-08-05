# tests/unit/api/test_edges_route_diagnostics.py

"""Tests for /edges diagnostic translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pandas import DataFrame
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency
from gridiron_edge.api.routes import edges as edges_route
from gridiron_edge.market.edge_diagnostics import (
    EdgeDiagnosticBlocker,
    EdgeDiagnostics,
    EdgeResultState,
)
from gridiron_edge.market.recommendations import EdgeResult


@dataclass
class _FakeSettings:
    repo_root: Path


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


def _diagnostics(
    *,
    state: EdgeResultState,
    blockers: tuple[EdgeDiagnosticBlocker, ...] = (),
    calculated: int = 0,
    positive: int = 0,
) -> EdgeDiagnostics:
    return EdgeDiagnostics(
        season="2026-2027",
        week=1,
        prediction_game_count=0,
        market_game_count=0,
        matched_game_count=0,
        complete_moneyline_count=0,
        complete_spread_count=0,
        complete_total_count=0,
        eligible_market_count=0,
        calculated_edge_count=calculated,
        positive_edge_count=positive,
        filtered_edge_count=0,
        state=state,
        blockers=blockers,
    )


def _empty_result(
    *,
    state: EdgeResultState = EdgeResultState.BLOCKED,
    blockers: tuple[EdgeDiagnosticBlocker, ...] = (),
    calculated: int = 0,
    positive: int = 0,
) -> EdgeResult:
    return EdgeResult(
        rows=DataFrame(),
        diagnostics=_diagnostics(
            state=state,
            blockers=blockers,
            calculated=calculated,
            positive=positive,
        ),
    )


@pytest.mark.parametrize(
    ("blocker", "slug"),
    [
        (EdgeDiagnosticBlocker.NO_PREDICTIONS, "no_weekly_product"),
        (EdgeDiagnosticBlocker.NO_MARKET_DATA, "no_odds_available"),
        (EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE, "market_scope_mismatch"),
        (EdgeDiagnosticBlocker.MARKET_STALE, "stale_market_data"),
        (EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES, "zero_edge_game_matches"),
        (EdgeDiagnosticBlocker.INCOMPLETE_MARKETS, "incomplete_market_data"),
    ],
)
def test_blocker_maps_to_items_field_status(
    client: TestClient,
    blocker: EdgeDiagnosticBlocker,
    slug: str,
) -> None:
    result = _empty_result(
        blockers=(blocker,),
    )
    with patch(
        "gridiron_edge.api.routes.edges.load_edges_for_week",
        return_value=result,
    ):
        response = client.get(
            "/edges",
            params={"season": "2026-2027", "week": 1},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0
    status = body["_meta"]["field_status"]["items"]
    assert status["status"] == "blocked"
    assert status["blocker"] == slug
    assert status["roadmap"] == "data"
    assert body["diagnostics"]["state"] == "blocked"
    assert body["diagnostics"]["blockers"] == [blocker.value]


@pytest.mark.parametrize(
    ("state", "calculated", "positive"),
    [
        (EdgeResultState.NO_CALCULABLE_EDGES, 0, 0),
        (EdgeResultState.NO_POSITIVE_EDGES, 1, 0),
        (EdgeResultState.POSITIVE_EDGES, 2, 1),
    ],
)
def test_analytical_empty_state_has_no_blocked_field_status(
    client: TestClient,
    state: EdgeResultState,
    calculated: int,
    positive: int,
) -> None:
    result = _empty_result(
        state=state,
        calculated=calculated,
        positive=positive,
    )
    with patch(
        "gridiron_edge.api.routes.edges.load_edges_for_week",
        return_value=result,
    ):
        response = client.get(
            "/edges",
            params={"season": "2026-2027", "week": 1},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body.get("_meta") is None
    assert body["diagnostics"]["state"] == state.value
    assert body["diagnostics"]["calculated_edge_count"] == calculated
    assert body["diagnostics"]["positive_edge_count"] == positive
    assert body["diagnostics"]["blockers"] == []


def test_first_diagnostic_blocker_controls_current_field_status(
    client: TestClient,
) -> None:
    result = _empty_result(
        blockers=(
            EdgeDiagnosticBlocker.NO_PREDICTIONS,
            EdgeDiagnosticBlocker.NO_MARKET_DATA,
        ),
    )
    with patch(
        "gridiron_edge.api.routes.edges.load_edges_for_week",
        return_value=result,
    ):
        body = client.get(
            "/edges",
            params={"season": "2026-2027", "week": 1},
        ).json()

    status = body["_meta"]["field_status"]["items"]
    assert status["blocker"] == "no_weekly_product"
    assert body["diagnostics"]["blockers"] == [
        "no_predictions",
        "no_market_data",
    ]


def test_explicit_scope_does_not_load_current_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _FakeSettings(repo_root=tmp_path)

    def fail_if_called(_settings):
        raise AssertionError("Explicit season and week must not resolve current scope.")

    monkeypatch.setattr(
        edges_route,
        "resolve_current_season_week",
        fail_if_called,
    )

    assert edges_route._resolve_scope(
        settings,
        "2026-2027",
        1,
    ) == ("2026-2027", 1)


@pytest.mark.parametrize(
    ("season", "week", "expected"),
    [
        ("2026-2027", None, ("2026-2027", 8)),
        (None, 3, ("2025-2026", 3)),
        (None, None, ("2025-2026", 8)),
    ],
)
def test_missing_scope_values_use_current_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    season: str | None,
    week: int | None,
    expected: tuple[str, int],
) -> None:
    settings = _FakeSettings(repo_root=tmp_path)

    monkeypatch.setattr(
        edges_route,
        "resolve_current_season_week",
        lambda _settings: ("2025-2026", 8),
    )

    assert (
        edges_route._resolve_scope(
            settings,
            season,
            week,
        )
        == expected
    )

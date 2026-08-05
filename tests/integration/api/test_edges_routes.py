# tests/integration/api/test_edges_routes.py
"""Integration tests for /edges through persisted weekly edge inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Response
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency
from gridiron_edge.api.schemas.edges import EdgeDiagnosticsResponse
from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.ingest.odds.store import write_current_odds_snapshot
from gridiron_edge.market.weekly_edge_service import build_weekly_edge_result

SEASON = "2026-2027"
WEEK = 1
GAME_ID = "2026_01_KC_LAC"


@dataclass
class _FakeSettings:
    repo_root: Path


def _weekly_product(
    *,
    home_win_prob: float = 0.70,
    model_spread: float = -7.0,
    model_total: float = 50.0,
) -> pd.DataFrame:
    """Return one valid weekly product with independent model identities."""
    away_win_prob = 1.0 - home_win_prob
    projected_home = (model_total - model_spread) / 2.0
    projected_away = (model_total + model_spread) / 2.0
    return pd.DataFrame(
        {
            "season": [SEASON],
            "week": [WEEK],
            "game_id": [GAME_ID],
            "game_date": ["2026-09-05"],
            "away_team": ["Kansas City Chiefs"],
            "home_team": ["Los Angeles Chargers"],
            "neutral_site": [False],
            "win_status": ["available"],
            "away_win_prob": [away_win_prob],
            "home_win_prob": [home_win_prob],
            "win_model_name": ["win_prob"],
            "win_model_type": ["elo"],
            "win_event_id": ["win-event-1"],
            "win_run_id": ["api-weekly-run"],
            "win_generated_at": ["2026-09-04T12:00:00+00:00"],
            "win_role": ["live"],
            "win_selection_status": ["selected"],
            "spread_status": ["available"],
            "model_spread": [model_spread],
            "spread_uncertainty": [13.5],
            "spread_source_event_id": ["win-event-1"],
            "spread_model_name": ["win_prob"],
            "spread_model_type": ["elo"],
            "spread_calibration_key": ["win_prob_elo"],
            "spread_calibration_updated_at": ["2026-07-30T12:00:00+00:00"],
            "total_status": ["available"],
            "model_total": [model_total],
            "total_uncertainty": [12.8],
            "total_model_name": ["total"],
            "total_model_type": ["xgboost"],
            "total_event_id": ["total-event-1"],
            "total_run_id": ["api-weekly-run"],
            "total_generated_at": ["2026-09-04T12:00:00+00:00"],
            "total_role": ["live"],
            "total_selection_status": ["selected"],
            "total_uncertainty_trained_at": ["2026-07-01T14:20:00"],
            "projected_score_status": ["available"],
            "projected_home_score": [projected_home],
            "projected_away_score": [projected_away],
        }
    )


def _odds_snapshot() -> pd.DataFrame:
    """Return one complete source-labeled market snapshot."""
    timestamp = pd.Timestamp("2026-09-05T12:00:00Z")
    base: dict[str, object] = {
        "fetched_at": timestamp,
        "provider": "nflverse",
        "provider_event_id": None,
        "sportsbook": None,
        "sportsbook_updated_at": pd.NaT,
        "commence_time": pd.NaT,
        "is_live": False,
        "season": SEASON,
        "week": WEEK,
        "game_id": GAME_ID,
        "game_date": "2026-09-05",
        "away_team": "Kansas City Chiefs",
        "home_team": "Los Angeles Chargers",
    }
    return pd.DataFrame(
        [
            {
                **base,
                "market": "moneyline",
                "side": "home",
                "odds": -200.0,
                "line": float("nan"),
            },
            {
                **base,
                "market": "moneyline",
                "side": "away",
                "odds": 170.0,
                "line": float("nan"),
            },
            {
                **base,
                "market": "spread",
                "side": "home",
                "odds": -110.0,
                "line": -3.5,
            },
            {
                **base,
                "market": "spread",
                "side": "away",
                "odds": -110.0,
                "line": 3.5,
            },
            {
                **base,
                "market": "total",
                "side": "over",
                "odds": -110.0,
                "line": 44.0,
            },
            {
                **base,
                "market": "total",
                "side": "under",
                "odds": -110.0,
                "line": 44.0,
            },
        ]
    )


def _persist_selected_product(
    repo: Path,
    *,
    product: pd.DataFrame | None = None,
) -> None:
    """Write and explicitly select one immutable weekly product."""
    identity = WeeklyProductIdentity(
        product_id="api-weekly-product",
        run_id="api-weekly-run",
        season=SEASON,
        week=WEEK,
        generated_at=datetime(2026, 9, 4, 12, tzinfo=UTC),
    )
    write_weekly_product(
        repo,
        _weekly_product() if product is None else product,
        identity=identity,
    )
    select_current_weekly_product(
        repo,
        identity.product_id,
        season=SEASON,
        week=WEEK,
        selected_at=datetime(2026, 9, 4, 13, tzinfo=UTC),
    )


def _persist_markets(repo: Path, markets: pd.DataFrame | None = None) -> None:
    """Write the current source-labeled market snapshot."""
    write_current_odds_snapshot(
        _odds_snapshot() if markets is None else markets,
        repo=repo,
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a client whose settings use the temporary repository."""
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


class TestListEdgesRoute:
    """Exercise the complete selected-product API edge path."""

    def test_returns_ranked_edges_from_selected_product(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _persist_selected_product(tmp_path)
        _persist_markets(tmp_path)
        direct_result = build_weekly_edge_result(
            season=SEASON,
            week=WEEK,
            bankroll=2500.0,
            kelly_multiplier=0.10,
            min_ev=0.0,
            repo=tmp_path,
        )

        response: Response = client.get(
            "/edges",
            params={
                "season": SEASON,
                "week": WEEK,
                "bankroll": 2500.0,
                "kelly_multiplier": 0.10,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["season"] == SEASON
        assert body["week"] == WEEK
        assert body["min_ev"] == 0.0
        assert body["bankroll"] == 2500.0
        assert body["kelly_multiplier"] == 0.10
        assert body["total"] > 0
        assert body["items"]

        first = body["items"][0]
        assert first["game_id"] == GAME_ID
        assert first["home_team"] == "Los Angeles Chargers"
        assert first["away_team"] == "Kansas City Chiefs"
        assert first["model_key"] == "win_prob_elo"
        assert first["market_type"] in {"moneyline", "spread", "total"}
        assert first["side"] in {"home", "away", "over", "under"}
        assert first["american_odds"] != 0
        assert first["kelly_stake"] is not None

        expected_diagnostics = EdgeDiagnosticsResponse.model_validate(
            direct_result.diagnostics.to_dict()
        ).model_dump(mode="json")
        assert body["diagnostics"] == expected_diagnostics
        assert body["total"] == len(direct_result.rows)
        assert [item["game_id"] for item in body["items"]] == direct_result.rows["game_id"].astype(
            str
        ).tolist()
        provenance = body["diagnostics"]["provenance"]
        assert provenance["win_event_ids"] == ["win-event-1"]
        assert provenance["win_run_ids"] == ["api-weekly-run"]
        assert provenance["win_model_names"] == ["win_prob"]
        assert provenance["win_model_types"] == ["elo"]
        assert provenance["total_event_ids"] == ["total-event-1"]
        assert provenance["total_run_ids"] == ["api-weekly-run"]
        assert provenance["total_model_names"] == ["total"]
        assert provenance["total_model_types"] == ["xgboost"]
        assert provenance["product_ids"] == ["api-weekly-product"]
        assert provenance["product_run_ids"] == ["api-weekly-run"]
        assert provenance["market_providers"] == ["nflverse"]
        assert provenance["market_sportsbooks"] == []
        assert provenance["market_fetched_at"] == ["2026-09-05T12:00:00Z"]

    def test_omitted_bankroll_keeps_dollar_stake_unavailable(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _persist_selected_product(tmp_path)
        _persist_markets(tmp_path)

        response = client.get(
            "/edges",
            params={"season": SEASON, "week": WEEK},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["bankroll"] is None
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
            {"min_ev": -0.01},
        ],
    )
    def test_rejects_invalid_query_values(
        self,
        client: TestClient,
        invalid_params: dict[str, float],
    ) -> None:
        response = client.get(
            "/edges",
            params={"season": SEASON, "week": WEEK, **invalid_params},
        )
        assert response.status_code == 422

    def test_missing_current_product_returns_field_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _persist_markets(tmp_path)

        response = client.get(
            "/edges",
            params={"season": SEASON, "week": WEEK},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0
        status = body["_meta"]["field_status"]["items"]
        assert status["status"] == "blocked"
        assert status["blocker"] == "no_weekly_product"
        assert body["diagnostics"]["state"] == "blocked"
        assert body["diagnostics"]["blockers"] == ["no_predictions"]

    def test_missing_market_snapshot_returns_field_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _persist_selected_product(tmp_path)

        response = client.get(
            "/edges",
            params={"season": SEASON, "week": WEEK},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0
        status = body["_meta"]["field_status"]["items"]
        assert status["status"] == "blocked"
        assert status["blocker"] == "no_odds_available"
        assert body["diagnostics"]["state"] == "blocked"
        assert body["diagnostics"]["blockers"] == ["no_market_data"]

    def test_threshold_filtered_result_is_empty_without_blocker(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _persist_selected_product(tmp_path)
        _persist_markets(tmp_path)

        response = client.get(
            "/edges",
            params={
                "season": SEASON,
                "week": WEEK,
                "min_ev": 1.0,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0
        assert body["min_ev"] == 1.0
        assert body.get("_meta") is None
        assert body["diagnostics"]["state"] == "positive_edges"
        assert body["diagnostics"]["positive_edge_count"] > 0
        assert body["diagnostics"]["filtered_edge_count"] == 0

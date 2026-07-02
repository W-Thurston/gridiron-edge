# tests/integration/api/test_props_routes.py

"""Integration tests for /props and /props/{prop_id} routes.

W8 Tier 2 Step 7d. Exercises loader → serializer → route stack
end-to-end via MiniRepoBuilder + FastAPI dependency_overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.api.deps import settings_dependency


@dataclass
class _FakeSettings:
    repo_root: Path


def _make_prop_archive_row(
    game_id: str = "2026_01_KC_LAC",
    player_id: str = "00-0033873",
    player_name: str = "P.Mahomes",
    position: str = "QB",
    team: str = "KC",
    stat_type: str = "qb_pass_yards",
    model_type: str = "elasticnet",
    week: int = 1,
    season: int = 2026,
) -> dict:
    return {
        "predicted_at": "2026-08-01T00:00:00+00:00",
        "is_backfilled": True,
        "season": season,
        "week": week,
        "game_id": game_id,
        "player_id": player_id,
        "player_name": player_name,
        "position": position,
        "team": team,
        "stat_type": stat_type,
        "model_name": stat_type,
        "model_type": model_type,
        "predicted_mean": 265.0,
        "predicted_std": 45.0,
        "lo_90": 190.0,
        "hi_90": 340.0,
        "line": None,
        "p_over": float("nan"),
        "lean": float("nan"),
        "confidence_tier": float("nan"),
    }


def _write_prop_archive(tmp_path: Path, rows: list[dict]) -> None:
    """Write the prop archive directly (MiniRepoBuilder doesn't yet have
    a with_prop_archive helper; simple path)."""
    archive_dir = tmp_path / "data" / "output" / "props"
    archive_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        archive_dir / "prop_predictions_log.parquet",
        index=False,
    )


def _write_prop_manifest(
    tmp_path: Path,
    families: dict[str, str],
) -> None:
    """Write a champions manifest with the given family → model_type entries."""
    manifest_dir = tmp_path / "data" / "output" / "champions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "champions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": "2026-07-01T14:00:00+00:00",
                "models": {
                    family: {
                        "model_type": model_type,
                        "promoted_at": "2026-07-01T14:00:00",
                        "source_run_id": "RUN_X",
                        "metrics": {"mae": 63.0},
                    }
                    for family, model_type in families.items()
                },
            }
        )
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with settings pointing at tmp_path via dependency_overrides."""
    app = create_app()
    app.dependency_overrides[settings_dependency] = lambda: _FakeSettings(repo_root=tmp_path)
    return TestClient(app)


class TestListPropsRoute:
    def test_returns_props_for_week(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        _write_prop_archive(tmp_path, [_make_prop_archive_row()])

        response = client.get("/props?season=2026-2027&week=1")

        assert response.status_code == 200
        body = response.json()
        assert body["season"] == "2026-2027"
        assert body["week"] == 1
        assert body["total"] == 1
        assert len(body["items"]) == 1

        item = body["items"][0]
        assert item["prop_id"] == ("2026_01_KC_LAC__00-0033873__qb_pass_yards")
        assert item["player_name"] == "P.Mahomes"
        assert item["model_key"] == "qb_pass_yards_elasticnet"
        assert item["projection"]["predicted_mean"] == 265.0
        # Line context always emitted, but all null in T2.
        assert item["line_context"]["line"] is None

    def test_field_status_marks_line_context_pending(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        _write_prop_archive(tmp_path, [_make_prop_archive_row()])

        response = client.get("/props?season=2026-2027&week=1")

        body = response.json()
        status = body["_meta"]["field_status"]
        assert status["items.line_context.line"] == "pending"
        assert status["items.line_context.p_over"] == "pending"

    def test_stat_type_filter(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _write_prop_manifest(
            tmp_path,
            {
                "qb_pass_yards": "elasticnet",
                "rb_rush_yards": "random_forest",
            },
        )
        _write_prop_archive(
            tmp_path,
            [
                _make_prop_archive_row(stat_type="qb_pass_yards"),
                _make_prop_archive_row(
                    game_id="2026_01_BUF_MIA",
                    player_id="00-0035700",
                    stat_type="rb_rush_yards",
                    model_type="random_forest",
                    position="RB",
                ),
            ],
        )

        response = client.get("/props?season=2026-2027&week=1&stat_type=qb_pass_yards")

        body = response.json()
        assert body["total"] == 1
        assert body["items"][0]["stat_type"] == "qb_pass_yards"

    def test_missing_manifest_returns_field_status(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # No manifest, no archive.
        response = client.get("/props?season=2026-2027&week=1")

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0
        status = body["_meta"]["field_status"]["items"]
        assert status["status"] == "blocked"
        assert status["blocker"] == "no_champion_manifest"

    def test_family_resolved_but_empty_archive(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # Manifest present, archive empty. Legitimate empty state.
        _write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        # No archive file.

        response = client.get("/props?season=2026-2027&week=1")

        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["total"] == 0


class TestGetPropRoute:
    def test_returns_detail_for_known_prop(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        _write_prop_archive(tmp_path, [_make_prop_archive_row()])

        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/props/{prop_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["prop_id"] == prop_id
        assert body["player_name"] == "P.Mahomes"
        assert body["projection"]["predicted_mean"] == 265.0

    def test_field_status_marks_pending_and_blocked(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        _write_prop_archive(tmp_path, [_make_prop_archive_row()])

        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/props/{prop_id}")

        body = response.json()
        status = body["_meta"]["field_status"]
        # Pending
        assert status["line_context.line"] == "pending"
        assert status["historical_vs_opponent"] == "pending"
        assert status["situational_splits"] == "pending"
        assert status["recent_form"] == "pending"
        # Blocked
        assert status["prop_reasoning"]["blocker"] == "feature_attribution"
        assert status["injury_status"]["blocker"] == "injury_data_source"
        assert status["multi_book_shopping"]["blocker"] == "multi_book_ingest"

    def test_malformed_prop_id_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        response = client.get("/props/malformed_prop_id")

        assert response.status_code == 404
        assert "Malformed prop_id" in response.json()["detail"]

    def test_unknown_stat_type_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        prop_id = "2026_01_KC_LAC__00-0033873__bogus_stat"
        response = client.get(f"/props/{prop_id}")

        assert response.status_code == 404
        assert "Unknown stat_type" in response.json()["detail"]

    def test_unknown_prop_returns_404(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        _write_prop_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        _write_prop_archive(tmp_path, [_make_prop_archive_row()])

        prop_id = "2026_01_KC_LAC__00-0000000__qb_pass_yards"  # unknown player_id
        response = client.get(f"/props/{prop_id}")

        assert response.status_code == 404
        assert "Prop not found" in response.json()["detail"]

    def test_missing_manifest_returns_null_projection(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        # No manifest at all, so champion resolution fails.
        prop_id = "2026_01_KC_LAC__00-0033873__qb_pass_yards"
        response = client.get(f"/props/{prop_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["prop_id"] == prop_id
        assert body["projection"] is None
        assert body["line_context"] is None
        status = body["_meta"]["field_status"]
        assert status["projection"]["status"] == "blocked"
        assert status["projection"]["blocker"] == "no_champion_manifest"
        assert status["line_context"]["status"] == "blocked"

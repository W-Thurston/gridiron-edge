"""End-to-end smoke tests for the API surface.

Verifies that every registered endpoint resolves through TestClient,
returns 200, and carries a valid response shape with `_meta.field_status`
entries that reference a registered blocker slug.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from gridiron_edge.api.app import create_app
from gridiron_edge.api.meta import Blocker, Unavailable


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# Endpoints where the path identifier is the only required input.
# Each entry is (method, path, expected_blocker_slug).
LIST_ENDPOINTS: list[tuple[str, str, str]] = [
    ("GET", "/live", "live_state_ingest"),
    ("GET", "/news", "news_ingest"),
    ("GET", "/news/alerts", "news_ingest"),
]

DETAIL_ENDPOINTS: list[tuple[str, str, str]] = [
    ("GET", "/live/sf-bal", "live_state_ingest"),
    ("GET", "/games/sf-bal/injuries", "injury_data_source"),
    ("GET", "/games/sf-bal/explain", "scenario_engine"),
    ("GET", "/games/sf-bal/swing-factors", "feature_attribution"),
    ("GET", "/games/sf-bal/comparables", "comparables_retrieval"),
    ("GET", "/props/lamar-rush/shop", "multi_book_ingest"),
    ("GET", "/props/lamar-rush/reasoning", "feature_attribution"),
]

ALL_ENDPOINTS = LIST_ENDPOINTS + DETAIL_ENDPOINTS


class TestAllEndpointsReachable:
    @pytest.mark.parametrize("method,path,_blocker", ALL_ENDPOINTS)
    def test_returns_200(
        self,
        client: TestClient,
        method: str,
        path: str,
        _blocker: str,
    ) -> None:
        response = client.request(method, path)
        assert response.status_code == 200, response.text


class TestListEndpointShape:
    @pytest.mark.parametrize("method,path,blocker", LIST_ENDPOINTS)
    def test_list_shape(
        self,
        client: TestClient,
        method: str,
        path: str,
        blocker: str,
    ) -> None:
        body = client.request(method, path).json()
        assert body["items"] == []
        assert body["total"] == 0
        assert "_meta" in body
        items_status = body["_meta"]["field_status"]["items"]
        assert items_status["status"] == "blocked"
        assert items_status["blocker"] == blocker


class TestDetailEndpointShape:
    @pytest.mark.parametrize("method,path,blocker", DETAIL_ENDPOINTS)
    def test_detail_shape(
        self,
        client: TestClient,
        method: str,
        path: str,
        blocker: str,
    ) -> None:
        body = client.request(method, path).json()
        assert "_meta" in body
        field_status = body["_meta"]["field_status"]
        # At least one field must be blocked on the expected blocker.
        blocked_slugs = {
            v["blocker"]
            for v in field_status.values()
            if isinstance(v, dict) and v.get("status") == "blocked"
        }
        assert blocker in blocked_slugs, (
            f"Expected blocker '{blocker}' on {path}, got {blocked_slugs}"
        )


class TestEveryBlockerSlugIsRegistered:
    """The slug consistency contract from D16.

    Walks every endpoint's `_meta.field_status` and asserts every blocker
    slug appears in `Blocker.all_slugs()`. Catches typos that would only
    surface to consumers otherwise.
    """

    def test_all_slugs_are_registered(self, client: TestClient) -> None:
        registered = Blocker.all_slugs() | Unavailable.all_slugs()
        seen_slugs: set[str] = set()

        for _method, path, _blocker in ALL_ENDPOINTS:
            body = client.request("GET", path).json()
            field_status = body.get("_meta", {}).get("field_status", {})
            for value in field_status.values():
                if isinstance(value, dict) and value.get("status") == "blocked":
                    seen_slugs.add(value["blocker"])

        unregistered = seen_slugs - registered
        assert not unregistered, f"Unregistered blocker slugs in API responses: {unregistered}"

    def test_at_least_one_slug_appears(self, client: TestClient) -> None:
        """Sanity check — if this fails, the loop above is silently empty."""
        body = client.request("GET", "/live").json()
        assert body["_meta"]["field_status"]


class TestOpenApiPathInventory:
    """Every route file's endpoints must appear in /openapi.json."""

    def test_all_paths_documented(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        documented = set(schema["paths"].keys())

        expected_paths = {
            "/lines",
            "/live",
            "/live/{game_id}",
            "/news",
            "/news/alerts",
            "/games/{game_id}/injuries",
            "/games/{game_id}/explain",
            "/games/{game_id}/swing-factors",
            "/games/{game_id}/comparables",
            "/props/{prop_id}/shop",
            "/props/{prop_id}/reasoning",
        }

        missing = expected_paths - documented
        assert not missing, f"Missing OpenAPI paths: {missing}"

# tests/unit/api/test_app.py
"""Unit tests for api/app.py.

Covers:
- create_app() returns a configured FastAPI instance.
- OpenAPI metadata renders without error.
- All declared tags appear in the OpenAPI schema.
- CORS middleware is attached.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gridiron_edge.api.app import _OPENAPI_TAGS, create_app


class TestCreateApp:
    def test_returns_fastapi_instance(self) -> None:
        app: FastAPI = create_app()
        assert isinstance(app, FastAPI)

    def test_metadata_set(self) -> None:
        app: FastAPI = create_app()
        assert app.title == "Gridiron Edge API"
        assert app.version == "0.1.0"
        assert "read-only" in (app.description or "").lower()

    def test_factory_returns_fresh_instances(self) -> None:
        """Each create_app() call must produce an isolated FastAPI instance.

        Required so tests can build apps with different dependency
        overrides without bleeding state across cases.
        """
        a: FastAPI = create_app()
        b: FastAPI = create_app()
        assert a is not b


class TestOpenApiTags:
    def test_all_declared_tags_have_descriptions(self) -> None:
        for tag in _OPENAPI_TAGS:
            assert tag.get("name")
            assert tag.get("description")

    def test_tags_cover_known_route_domains(self) -> None:
        names: set[str] = {tag["name"] for tag in _OPENAPI_TAGS}
        # Populated route domains.
        assert {
            "weeks",
            "games",
            "edges",
            "teams",
            "projections",
            "props",
            "portfolio",
            "compare",
            "model",
        } <= names
        # Blocked route domains.
        assert {
            "lines",
            "live",
            "news",
            "injuries",
            "explain",
            "swing-factors",
            "comparables",
            "prop-shop",
            "prop-reasoning",
        } <= names

    def test_openapi_schema_renders(self) -> None:
        app: FastAPI = create_app()
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Gridiron Edge API"
        # Tags should round-trip into the schema.
        schema_tags: set = {tag["name"] for tag in schema.get("tags", [])}
        assert "games" in schema_tags
        assert "live" in schema_tags


class TestCorsConfiguration:
    def test_cors_preflight_passes(self) -> None:
        app: FastAPI = create_app()
        client = TestClient(app)
        # Add a probe route so there's something to preflight against.

        @app.get("/probe")
        def probe() -> dict[str, str]:
            return {"ok": "yes"}

        response = client.options(
            "/probe",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"

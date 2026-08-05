# tests/unit/api/test_deps.py
"""Unit tests for api/deps.py.

Covers:
- Settings caching at the FastAPI dependency layer.
- DataPathResolver closes over the resolved repo root.
- Dependency override mechanism works for tests.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gridiron_edge.api.deps import (
    DataPathResolver,
    DataPathResolverDep,
    SettingsDep,
    _settings_singleton,
    data_path_resolver_dependency,
    settings_dependency,
)
from gridiron_edge.core.settings import Settings
from gridiron_edge.datasets.registry import DatasetKey


def _make_settings(repo_root: Path) -> Settings:
    return Settings(
        repo_root=repo_root,
        owm_api_key=None,
        odds_api_key=None,
        data_raw=repo_root / "data" / "raw",
        data_cleaned=repo_root / "data" / "cleaned",
        data_modeling=repo_root / "data" / "modeling",
        data_output=repo_root / "data" / "output",
    )


class TestSettingsDependency:
    def test_returns_a_settings_instance(self) -> None:
        _settings_singleton.cache_clear()
        settings: Settings = settings_dependency()
        assert isinstance(settings, Settings)
        assert settings.repo_root.is_absolute()

    def test_is_cached_across_calls(self) -> None:
        _settings_singleton.cache_clear()
        a: Settings = settings_dependency()
        b: Settings = settings_dependency()
        assert a is b

    def test_override_substitutes_a_test_double(self, tmp_path: Path) -> None:
        app = FastAPI()
        stub: Settings = _make_settings(tmp_path)

        @app.get("/probe")
        def probe(settings: SettingsDep) -> dict[str, str]:
            return {"repo_root": str(settings.repo_root)}

        app.dependency_overrides[settings_dependency] = lambda: stub

        client = TestClient(app)
        response = client.get("/probe")
        assert response.status_code == 200
        assert response.json() == {"repo_root": str(tmp_path)}


class TestDataPathResolverDependency:
    def test_resolver_closes_over_repo_root(self, tmp_path: Path) -> None:
        stub: Settings = _make_settings(tmp_path)
        resolver: DataPathResolver = data_path_resolver_dependency(stub)

        key: DatasetKey = "modeling_full"
        resolved: Path = resolver(key)

        assert isinstance(resolved, Path)
        assert resolved.is_absolute()
        assert (
            tmp_path in resolved.parents
            or resolved.parent == tmp_path
            or str(resolved).startswith(str(tmp_path))
        )

    def test_resolver_handles_multiple_keys(self, tmp_path: Path) -> None:
        stub: Settings = _make_settings(tmp_path)
        resolver: DataPathResolver = data_path_resolver_dependency(stub)

        modeling: Path = resolver("modeling_full")
        ledger: Path = resolver("bet_ledger")

        assert modeling != ledger
        assert str(modeling).startswith(str(tmp_path))
        assert str(ledger).startswith(str(tmp_path))

    def test_override_substitutes_a_resolver(self, tmp_path: Path) -> None:
        app = FastAPI()
        stub_path: Path = tmp_path / "fixtures" / "modeling.parquet"

        @app.get("/probe")
        def probe(resolver: DataPathResolverDep) -> dict[str, str]:
            return {"path": str(resolver("modeling_full"))}

        app.dependency_overrides[data_path_resolver_dependency] = lambda: lambda _key: stub_path

        client = TestClient(app)
        response = client.get("/probe")
        assert response.status_code == 200
        assert response.json() == {"path": str(stub_path)}

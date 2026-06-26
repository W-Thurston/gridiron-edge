# src/gridiron_edge/api/deps.py
"""Shared FastAPI dependencies.

Bridges FastAPI's dependency-injection system to the existing
`gridiron_edge.core.settings.get_settings()` and
`gridiron_edge.datasets.registry.dataset_path()` facilities. Routes
type their parameters as `SettingsDep` / `DataPathResolverDep` rather
than importing those facilities directly, which gives us a single seam
to override in tests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import Depends

from gridiron_edge.core.settings import Settings, get_settings
from gridiron_edge.datasets.registry import DatasetKey, dataset_path


@lru_cache(maxsize=1)
def _settings_singleton() -> Settings:
    """Cache `Settings` once per process.

    `get_settings()` is itself cheap, but caching here lets the FastAPI
    dependency layer share one instance across handlers within and
    between requests. This also gives us a single point to clear in
    tests via `_settings_singleton.cache_clear()`.
    """
    return get_settings()


def settings_dependency() -> Settings:
    """FastAPI dependency that resolves the process-wide `Settings`."""
    return _settings_singleton()


SettingsDep = Annotated[Settings, Depends(settings_dependency)]


class DataPathResolver(Protocol):
    """Resolves dataset keys to absolute filesystem paths.

    Wraps `dataset_path` so routes don't import the registry module
    directly. Override in tests by providing a stub via
    `app.dependency_overrides[data_path_resolver_dependency]`.
    """

    def __call__(self, key: DatasetKey) -> Path:
        """Resolve `key` to an absolute filesystem path."""
        ...


def data_path_resolver_dependency(
    settings: SettingsDep,
) -> DataPathResolver:
    """FastAPI dependency that produces a `DataPathResolver`.

    The returned callable closes over the resolved `Settings.repo_root`,
    so routes can call `resolver("modeling_full")` without threading the
    repo root themselves.
    """

    def resolve(key: DatasetKey) -> Path:
        return dataset_path(settings.repo_root, key)

    return resolve


DataPathResolverDep = Annotated[
    DataPathResolver,
    Depends(data_path_resolver_dependency),
]

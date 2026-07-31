# tests/unit/features/test_pipeline.py

"""Tests for modeling-artifact compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path

from gridiron_edge.features.manifest import (
    CURRENT_DATA_VERSION,
    CURRENT_SCHEMA_VERSION,
)
from gridiron_edge.features.pipeline import (
    _modeling_artifact_is_stale,
)


def _write_manifest(
    path: Path,
    *,
    schema_version: object = CURRENT_SCHEMA_VERSION,
    data_version: object = CURRENT_DATA_VERSION,
    include_schema: bool = True,
    include_data: bool = True,
) -> None:
    """Write a minimal modeling manifest."""
    manifest: dict[str, object] = {
        "feature_names": [],
        "feature_columns": [],
        "all_columns": [],
        "row_count": 0,
    }

    if include_schema:
        manifest["schema_version"] = schema_version

    if include_data:
        manifest["data_version"] = data_version

    (path / "modeling_file_manifest.json").write_text(json.dumps(manifest))


def test_missing_manifest_is_stale(
    tmp_path: Path,
) -> None:
    assert _modeling_artifact_is_stale(tmp_path) is True


def test_old_schema_version_is_stale(
    tmp_path: Path,
) -> None:
    _write_manifest(
        tmp_path,
        schema_version=(CURRENT_SCHEMA_VERSION - 1),
    )

    assert _modeling_artifact_is_stale(tmp_path) is True


def test_old_data_version_is_stale(
    tmp_path: Path,
) -> None:
    _write_manifest(
        tmp_path,
        data_version=(CURRENT_DATA_VERSION - 1),
    )

    assert _modeling_artifact_is_stale(tmp_path) is True


def test_missing_schema_version_is_stale(
    tmp_path: Path,
) -> None:
    _write_manifest(
        tmp_path,
        include_schema=False,
    )

    assert _modeling_artifact_is_stale(tmp_path) is True


def test_missing_data_version_is_stale(
    tmp_path: Path,
) -> None:
    _write_manifest(
        tmp_path,
        include_data=False,
    )

    assert _modeling_artifact_is_stale(tmp_path) is True


def test_matching_versions_are_not_stale(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)

    assert _modeling_artifact_is_stale(tmp_path) is False

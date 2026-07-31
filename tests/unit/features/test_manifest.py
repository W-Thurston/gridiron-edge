# tests/unit/features/test_manifest.py

"""Tests for feature-manifest version metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridiron_edge.features.manifest import (
    CURRENT_DATA_VERSION,
    CURRENT_SCHEMA_VERSION,
    read_manifest,
    write_manifest,
)


def _canonical_frame() -> pd.DataFrame:
    """Return one canonical modeling row."""
    return pd.DataFrame(
        {
            "GAME_ID": ["x"],
            "AWAY_TEAM": ["Away"],
            "HOME_TEAM": ["Home"],
            "HOME_WIN": [1],
        }
    )


def test_manifest_includes_current_versions(
    tmp_path: Path,
) -> None:
    write_manifest(
        _canonical_frame(),
        feature_names=[
            "home_away_elo",
        ],
        feature_columns=[
            "AWAY_ELO",
            "HOME_ELO",
        ],
        modeling_dir=tmp_path,
    )

    manifest = read_manifest(tmp_path)

    assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION
    assert manifest["data_version"] == CURRENT_DATA_VERSION


def test_manifest_accepts_custom_versions(
    tmp_path: Path,
) -> None:
    write_manifest(
        _canonical_frame(),
        feature_names=[],
        feature_columns=[],
        modeling_dir=tmp_path,
        schema_version=41,
        data_version=42,
    )

    manifest = read_manifest(tmp_path)

    assert manifest["schema_version"] == 41
    assert manifest["data_version"] == 42

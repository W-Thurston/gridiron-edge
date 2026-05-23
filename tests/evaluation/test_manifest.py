# tests/evaluation/test_manifest.py

"""Tests for the feature set manifest writer and validator."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron_edge.features.manifest import (
    CURRENT_SCHEMA_VERSION,
    read_manifest,
    validate_columns,
    validate_schema_version,
    write_manifest,
)


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "TEAM_A": ["KC", "PHI"],
            "TEAM_B": ["LAC", "DAL"],
            "HOME_FIELD": [1, 0],
            "TEAM_A_ELO": [1520.0, 1490.0],
        }
    )


# ---------------------------------------------------------------------------
# write_manifest / read_manifest round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_manifest(tmp_path: pytest.FixtureValue) -> None:
    df = _make_df()
    write_manifest(
        df,
        feature_names=["home_field", "team_elo"],
        feature_columns=["HOME_FIELD", "TEAM_A_ELO"],
        modeling_dir=tmp_path,
    )
    manifest = read_manifest(tmp_path)

    assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION
    assert manifest["feature_names"] == ["home_field", "team_elo"]
    assert manifest["feature_columns"] == ["HOME_FIELD", "TEAM_A_ELO"]
    assert manifest["all_columns"] == list(df.columns)
    assert manifest["row_count"] == 2
    assert "created_at" in manifest


def test_read_manifest_raises_if_missing(tmp_path: pytest.FixtureValue) -> None:
    with pytest.raises(FileNotFoundError, match="No feature manifest"):
        read_manifest(tmp_path)


def test_write_manifest_creates_json_file(tmp_path: pytest.FixtureValue) -> None:
    write_manifest(
        _make_df(),
        feature_names=["home_field"],
        feature_columns=["HOME_FIELD"],
        modeling_dir=tmp_path,
    )
    manifest_file = tmp_path / "modeling_file_manifest.json"
    assert manifest_file.exists()
    import json

    data = json.loads(manifest_file.read_text())
    assert data["schema_version"] == CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# validate_columns
# ---------------------------------------------------------------------------


def test_validate_columns_passes_when_all_present() -> None:
    df = _make_df()
    # Should not raise
    validate_columns(df, expected_columns=["GAME_ID", "HOME_FIELD"])


def test_validate_columns_raises_on_missing() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_columns(df, expected_columns=["GAME_ID", "NONEXISTENT_COL"])


def test_validate_columns_includes_context_in_error() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match="logistic_v1"):
        validate_columns(
            df,
            expected_columns=["MISSING"],
            context="logistic_v1",
        )


def test_validate_columns_raises_on_retrain_hint() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match="retrain"):
        validate_columns(df, expected_columns=["EPA_OFF"])


# ---------------------------------------------------------------------------
# validate_schema_version
# ---------------------------------------------------------------------------


def test_validate_schema_version_passes_on_match() -> None:
    manifest = {"schema_version": 1}
    # Should not raise
    validate_schema_version(manifest, required_version=1)


def test_validate_schema_version_raises_on_mismatch() -> None:
    manifest = {"schema_version": 2}
    with pytest.raises(ValueError, match="schema version mismatch"):
        validate_schema_version(manifest, required_version=1)


def test_validate_schema_version_includes_context() -> None:
    manifest = {"schema_version": 2}
    with pytest.raises(ValueError, match="logistic_v1"):
        validate_schema_version(manifest, required_version=1, context="logistic_v1")


def test_validate_schema_version_raises_on_retrain_hint() -> None:
    manifest = {"schema_version": 99}
    with pytest.raises(ValueError, match="Retrain"):
        validate_schema_version(manifest, required_version=1)

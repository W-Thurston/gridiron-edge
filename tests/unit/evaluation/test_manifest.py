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
    """Return a minimal canonical modeling frame."""
    return pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "AWAY_TEAM": ["KC", "PHI"],
            "HOME_TEAM": ["LAC", "DAL"],
            "AWAY_ELO": [1520.0, 1490.0],
            "HOME_ELO": [1480.0, 1510.0],
            "HOME_WIN": [0, 1],
        }
    )


# ---------------------------------------------------------------------------
# write_manifest / read_manifest round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_manifest(tmp_path: pytest.FixtureValue) -> None:
    df = _make_df()
    write_manifest(
        df,
        feature_names=["home_away_elo"],
        feature_columns=[
            "AWAY_ELO",
            "HOME_ELO",
        ],
        modeling_dir=tmp_path,
    )
    manifest = read_manifest(tmp_path)

    assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION
    assert manifest["feature_names"] == ["home_away_elo"]
    assert manifest["feature_columns"] == [
        "AWAY_ELO",
        "HOME_ELO",
    ]
    assert manifest["all_columns"] == list(df.columns)
    assert manifest["row_count"] == 2
    assert "created_at" in manifest


def test_read_manifest_raises_if_missing(tmp_path: pytest.FixtureValue) -> None:
    with pytest.raises(FileNotFoundError, match="No feature manifest"):
        read_manifest(tmp_path)


def test_write_manifest_creates_json_file(tmp_path: pytest.FixtureValue) -> None:
    write_manifest(
        _make_df(),
        feature_names=["home_away_elo"],
        feature_columns=[
            "AWAY_ELO",
            "HOME_ELO",
        ],
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
    validate_columns(
        df,
        expected_columns=[
            "GAME_ID",
            "AWAY_TEAM",
            "HOME_TEAM",
            "AWAY_ELO",
            "HOME_ELO",
        ],
    )


def test_validate_columns_raises_on_missing() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_columns(df, expected_columns=["GAME_ID", "NONEXISTENT_COL"])


def test_validate_columns_includes_context_in_error() -> None:
    df = _make_df()
    with pytest.raises(ValueError, match="logistic"):
        validate_columns(
            df,
            expected_columns=["MISSING"],
            context="logistic",
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
    with pytest.raises(ValueError, match="logistic"):
        validate_schema_version(manifest, required_version=1, context="logistic")


def test_validate_schema_version_raises_on_retrain_hint() -> None:
    manifest = {"schema_version": 99}
    with pytest.raises(ValueError, match="Retrain"):
        validate_schema_version(manifest, required_version=1)


def test_manifest_examples_exclude_retired_orientation() -> None:
    df = _make_df()

    retired = {
        "TEAM_A",
        "TEAM_B",
        "HOME_FIELD",
        "RESULT",
    }

    assert not (retired & set(df.columns))

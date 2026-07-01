# tests/unit/evaluation/test_champion_resolver.py
"""Unit tests for the champion manifest read API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridiron_edge.evaluation.champion_resolver import (
    CURRENT_SCHEMA_VERSION,
    ChampionNotFoundError,
    list_current_champions,
    read_manifest,
    resolve_current_champion,
    resolve_current_champion_with_metadata,
)


def _write_fixture_manifest(repo: Path, entries: dict[str, dict]) -> Path:
    """Helper: write a fixture manifest to the standard location under ``repo``."""
    manifest = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "updated_at": "2026-07-01T14:23:00Z",
        "models": entries,
    }
    directory = repo / "data" / "output" / "champions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "champions.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


class TestReadManifest:
    def test_raises_when_manifest_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ChampionNotFoundError) as exc_info:
            read_manifest(repo=tmp_path)
        assert "not found" in str(exc_info.value).lower()

    def test_loads_valid_manifest(self, tmp_path: Path) -> None:
        _write_fixture_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"brier": 0.213},
                },
            },
        )
        manifest = read_manifest(repo=tmp_path)
        assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "win_prob" in manifest["models"]

    def test_manifest_carries_updated_at(self, tmp_path: Path) -> None:
        _write_fixture_manifest(tmp_path, {})
        manifest = read_manifest(repo=tmp_path)
        assert "updated_at" in manifest
        assert manifest["updated_at"].endswith("Z")


class TestResolveCurrentChampion:
    def test_happy_path(self, tmp_path: Path) -> None:
        _write_fixture_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"brier": 0.213},
                },
            },
        )
        result = resolve_current_champion("win_prob", repo=tmp_path)
        assert result == ("win_prob", "random_forest")

    def test_multiple_models_resolve_independently(self, tmp_path: Path) -> None:
        _write_fixture_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"brier": 0.213},
                },
                "total": {
                    "model_type": "xgboost",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"mae": 10.24},
                },
                "qb_pass_yards": {
                    "model_type": "elasticnet",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"mae": 63.4},
                },
            },
        )
        assert resolve_current_champion("win_prob", repo=tmp_path) == (
            "win_prob",
            "random_forest",
        )
        assert resolve_current_champion("total", repo=tmp_path) == ("total", "xgboost")
        assert resolve_current_champion("qb_pass_yards", repo=tmp_path) == (
            "qb_pass_yards",
            "elasticnet",
        )

    def test_raises_when_manifest_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ChampionNotFoundError):
            resolve_current_champion("win_prob", repo=tmp_path)

    def test_raises_when_model_name_missing(self, tmp_path: Path) -> None:
        _write_fixture_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"brier": 0.213},
                },
            },
        )
        with pytest.raises(ChampionNotFoundError) as exc_info:
            resolve_current_champion("nonexistent", repo=tmp_path)
        # Error message should list available options
        assert "win_prob" in str(exc_info.value)

    def test_raises_when_models_field_missing(self, tmp_path: Path) -> None:
        # Manifest exists but has no "models" key at all
        directory = tmp_path / "data" / "output" / "champions"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "champions.json"
        path.write_text(json.dumps({"schema_version": 1, "updated_at": "2026-07-01T00:00:00Z"}))

        with pytest.raises(ChampionNotFoundError):
            resolve_current_champion("win_prob", repo=tmp_path)


class TestResolveCurrentChampionWithMetadata:
    def test_returns_full_entry(self, tmp_path: Path) -> None:
        _write_fixture_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
                },
            },
        )
        entry = resolve_current_champion_with_metadata("win_prob", repo=tmp_path)
        assert entry["model_type"] == "random_forest"
        assert entry["promoted_at"] == "2026-07-01T14:20:00Z"
        assert entry["source_run_id"] == "20260701_142000"
        assert entry["metrics"]["brier"] == 0.213

    def test_returns_defensive_copy(self, tmp_path: Path) -> None:
        _write_fixture_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"brier": 0.213},
                },
            },
        )
        entry = resolve_current_champion_with_metadata("win_prob", repo=tmp_path)
        entry["model_type"] = "MUTATED"
        # Second call should return original value
        entry2 = resolve_current_champion_with_metadata("win_prob", repo=tmp_path)
        assert entry2["model_type"] == "random_forest"

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ChampionNotFoundError):
            resolve_current_champion_with_metadata("win_prob", repo=tmp_path)


class TestListCurrentChampions:
    def test_returns_empty_when_manifest_missing(self, tmp_path: Path) -> None:
        # No manifest exists — should return empty dict without raising
        result = list_current_champions(repo=tmp_path)
        assert result == {}

    def test_lists_all_registered(self, tmp_path: Path) -> None:
        _write_fixture_manifest(
            tmp_path,
            {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"brier": 0.213},
                },
                "total": {
                    "model_type": "xgboost",
                    "promoted_at": "2026-07-01T14:20:00Z",
                    "source_run_id": "20260701_142000",
                    "metrics": {"mae": 10.24},
                },
            },
        )
        result = list_current_champions(repo=tmp_path)
        assert result == {
            "win_prob": ("win_prob", "random_forest"),
            "total": ("total", "xgboost"),
        }

    def test_returns_empty_when_manifest_has_no_models(self, tmp_path: Path) -> None:
        _write_fixture_manifest(tmp_path, {})
        result = list_current_champions(repo=tmp_path)
        assert result == {}

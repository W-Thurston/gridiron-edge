# tests/unit/evaluation/test_champion_resolver.py
"""Unit tests for the champion manifest read API."""

from __future__ import annotations

from datetime import UTC, datetime
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
    write_manifest,
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


class TestWriteManifest:
    def test_writes_manifest_with_correct_schema(self, tmp_path):
        """write_manifest produces a manifest that read_manifest can parse."""
        entries = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:20:00",
                "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
            },
            "total": {
                "model_type": "xgboost",
                "promoted_at": "2026-07-01T14:20:05",
                "metrics": {"mae": 10.24, "rmse": 12.87, "r2": 0.18},
            },
        }
        write_manifest(entries, source_run_id="20260701_142000", repo=tmp_path)
        manifest = read_manifest(repo=tmp_path)

        assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "updated_at" in manifest
        assert set(manifest["models"].keys()) == {"win_prob", "total"}

    def test_stamps_source_run_id_on_every_entry(self, tmp_path):
        entries = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:20:00",
                "metrics": {"brier": 0.213},
            },
            "qb_pass_yards": {
                "model_type": "elasticnet",
                "promoted_at": "2026-07-01T14:20:05",
                "metrics": {"mae": 63.4},
            },
        }
        write_manifest(entries, source_run_id="RUN_XYZ", repo=tmp_path)
        manifest = read_manifest(repo=tmp_path)

        assert manifest["models"]["win_prob"]["source_run_id"] == "RUN_XYZ"
        assert manifest["models"]["qb_pass_yards"]["source_run_id"] == "RUN_XYZ"

    def test_preserves_existing_source_run_id_when_present(self, tmp_path):
        """Entries that already carry source_run_id keep it (preservation semantics)."""
        entries = {
            "win_prob": {  # fresh — no source_run_id
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:20:00",
                "metrics": {"brier": 0.213},
            },
            "rb_rush_yards": {  # preserved — carries its own source_run_id
                "model_type": "elasticnet",
                "promoted_at": "2026-06-01T00:00:00",
                "source_run_id": "OLD_RUN",
                "metrics": {"mae": 25.0},
            },
        }
        write_manifest(entries, source_run_id="NEW_RUN", repo=tmp_path)
        manifest = read_manifest(repo=tmp_path)

        assert manifest["models"]["win_prob"]["source_run_id"] == "NEW_RUN"
        assert manifest["models"]["rb_rush_yards"]["source_run_id"] == "OLD_RUN"

    def test_updated_at_is_recent_utc(self, tmp_path):
        before = datetime.now(UTC)
        write_manifest({}, source_run_id="R", repo=tmp_path)
        after = datetime.now(UTC)

        manifest = read_manifest(repo=tmp_path)
        updated_at = datetime.fromisoformat(manifest["updated_at"])
        assert before <= updated_at <= after

    def test_roundtrip_preserves_entry_shape(self, tmp_path):
        entries = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:20:00",
                "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
            },
        }
        write_manifest(entries, source_run_id="R", repo=tmp_path)

        # resolve_current_champion should return the identity
        assert resolve_current_champion("win_prob", repo=tmp_path) == (
            "win_prob",
            "random_forest",
        )

        # resolve_current_champion_with_metadata should return the full entry
        full = resolve_current_champion_with_metadata("win_prob", repo=tmp_path)
        assert full["model_type"] == "random_forest"
        assert full["promoted_at"] == "2026-07-01T14:20:00"
        assert full["source_run_id"] == "R"
        assert full["metrics"] == {"brier": 0.213, "ece": 0.041, "auc": 0.721}

    def test_overwrites_existing_manifest(self, tmp_path):
        entries_v1 = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:00:00",
                "metrics": {"brier": 0.25},
            },
        }
        write_manifest(entries_v1, source_run_id="V1", repo=tmp_path)

        entries_v2 = {
            "win_prob": {
                "model_type": "xgboost",
                "promoted_at": "2026-07-02T14:00:00",
                "metrics": {"brier": 0.21},
            },
        }
        write_manifest(entries_v2, source_run_id="V2", repo=tmp_path)

        assert resolve_current_champion("win_prob", repo=tmp_path) == (
            "win_prob",
            "xgboost",
        )
        full = resolve_current_champion_with_metadata("win_prob", repo=tmp_path)
        assert full["source_run_id"] == "V2"

    def test_atomic_write_uses_tmp_then_rename(self, tmp_path, monkeypatch):
        """Verify tmp file is written and renamed, not written in place."""
        tmp_files_seen: list[Path] = []
        original_replace = Path.replace

        def spy_replace(self, target):
            tmp_files_seen.append(self)
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", spy_replace)

        write_manifest(
            {"win_prob": {"model_type": "rf", "promoted_at": "x", "metrics": {}}},
            source_run_id="R",
            repo=tmp_path,
        )

        assert len(tmp_files_seen) == 1
        assert tmp_files_seen[0].suffix == ".tmp"

    def test_empty_entries_writes_valid_manifest(self, tmp_path):
        """An empty write is still a valid manifest — used for full-retrain
        subsetting cold-start when no families were touched."""
        write_manifest({}, source_run_id="R", repo=tmp_path)
        manifest = read_manifest(repo=tmp_path)
        assert manifest["models"] == {}
        assert manifest["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_defensive_copy_of_metrics(self, tmp_path):
        """Post-write mutation of caller's metrics dict does not affect the file."""
        metrics = {"brier": 0.213}
        entries = {
            "win_prob": {
                "model_type": "rf",
                "promoted_at": "x",
                "metrics": metrics,
            },
        }
        write_manifest(entries, source_run_id="R", repo=tmp_path)
        metrics["brier"] = 999.0  # mutate after write

        manifest = read_manifest(repo=tmp_path)
        assert manifest["models"]["win_prob"]["metrics"]["brier"] == 0.213

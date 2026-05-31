# tests/unit/models/test_artifact.py
"""Tests for gridiron_edge.models.artifact — ArtifactStore and ModelMetadata."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gridiron_edge.models.artifact import ArtifactStore, ModelMetadata


def _make_metadata(model_version: str = "test_v1") -> ModelMetadata:
    """Build a minimal ModelMetadata for testing."""
    return ModelMetadata(
        model_version=model_version,
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=1,
        training_seasons=["2020-2021", "2021-2022"],
        holdout_seasons=["2023-2024"],
        holdout_brier=0.230,
        parameters={"n_estimators": 100},
        feature_columns=["HOME_FIELD", "TEAM_A_ELO"],
    )


class TestModelMetadata:
    def test_has_expected_fields(self) -> None:
        md: ModelMetadata = _make_metadata()
        assert md.model_version == "test_v1"
        assert md.holdout_brier == 0.230
        assert md.schema_version == 1
        assert "HOME_FIELD" in md.feature_columns

    def test_training_seasons_is_list(self) -> None:
        md: ModelMetadata = _make_metadata()
        assert isinstance(md.training_seasons, list)
        assert len(md.training_seasons) == 2

    def test_parameters_is_dict(self) -> None:
        md: ModelMetadata = _make_metadata()
        assert isinstance(md.parameters, dict)


class TestArtifactStoreInit:
    def test_accepts_repo_path(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store is not None


class TestArtifactStoreSaveLoad:
    def test_save_creates_directory(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: ModelMetadata = _make_metadata()
        dummy_model: dict[str, list[int] | str] = {"type": "dummy", "weights": [1, 2, 3]}
        store.save(model_version="test_v1", model_obj=dummy_model, metadata=md)
        assert (tmp_path / "data" / "models" / "test_v1").is_dir()

    def test_save_creates_metadata_json(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: ModelMetadata = _make_metadata()
        store.save(model_version="test_v1", model_obj={"w": 1}, metadata=md)
        assert (tmp_path / "data" / "models" / "test_v1" / "metadata.json").is_file()

    def test_save_creates_model_file(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: ModelMetadata = _make_metadata()
        store.save(model_version="test_v1", model_obj={"w": 1}, metadata=md)
        assert (tmp_path / "data" / "models" / "test_v1" / "model.joblib").is_file()

    def test_load_returns_saved_object(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: ModelMetadata = _make_metadata()
        original: dict[str, list[int] | str] = {"type": "dummy", "weights": [1, 2, 3]}
        store.save(model_version="test_v1", model_obj=original, metadata=md)
        loaded = store.load("test_v1")
        assert loaded == original

    def test_read_metadata_roundtrip(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: ModelMetadata = _make_metadata()
        store.save(model_version="test_v1", model_obj={"w": 1}, metadata=md)
        loaded_md: ModelMetadata = store.read_metadata("test_v1")
        assert loaded_md.model_version == md.model_version
        assert loaded_md.holdout_brier == md.holdout_brier
        assert loaded_md.feature_columns == md.feature_columns

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises((FileNotFoundError, KeyError)):
            store.load("nonexistent_v1")

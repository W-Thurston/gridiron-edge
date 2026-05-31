# tests/integration/test_model_train_predict.py
"""Integration: artifact save → load → predict roundtrip."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from gridiron_edge.models.artifact import ArtifactStore, ModelMetadata


def _make_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_version="integration_test_v1",
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        schema_version=1,
        training_seasons=["2020-2021", "2021-2022"],
        holdout_seasons=["2023-2024"],
        holdout_brier=0.230,
        parameters={"n_estimators": 10, "max_depth": 3},
        feature_columns=["HOME_FIELD", "TEAM_A_ELO", "TEAM_B_ELO"],
    )


class TestArtifactRoundtrip:
    """Save a model artifact, load it back, verify metadata integrity."""

    def test_save_load_model_object(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: ModelMetadata = _make_metadata()
        model_obj: dict[str, list[float] | str] = {"type": "mock_rf", "weights": [1.0, 2.0, 3.0]}

        store.save(model_version="integration_test_v1", model_obj=model_obj, metadata=md)
        loaded = store.load("integration_test_v1")

        assert loaded == model_obj

    def test_save_load_metadata(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: ModelMetadata = _make_metadata()
        store.save(model_version="integration_test_v1", model_obj={"w": 1}, metadata=md)

        loaded_md: ModelMetadata = store.read_metadata("integration_test_v1")
        assert loaded_md.model_version == "integration_test_v1"
        assert loaded_md.holdout_brier == 0.230
        assert loaded_md.feature_columns == ["HOME_FIELD", "TEAM_A_ELO", "TEAM_B_ELO"]
        assert loaded_md.training_seasons == ["2020-2021", "2021-2022"]

    def test_multiple_versions_coexist(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)

        for version in ("model_a_v1", "model_b_v1"):
            md = ModelMetadata(
                model_version=version,
                trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                schema_version=1,
                training_seasons=["2023-2024"],
                holdout_seasons=["2024-2025"],
                holdout_brier=0.25,
                parameters={},
                feature_columns=["HOME_FIELD"],
            )
            store.save(model_version=version, model_obj={"v": version}, metadata=md)

        assert store.load("model_a_v1") == {"v": "model_a_v1"}
        assert store.load("model_b_v1") == {"v": "model_b_v1"}

    def test_metadata_json_is_valid(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: ModelMetadata = _make_metadata()
        store.save(model_version="integration_test_v1", model_obj={"w": 1}, metadata=md)

        json_path: Path = tmp_path / "data" / "models" / "integration_test_v1" / "metadata.json"
        with open(json_path) as f:
            data = json.load(f)

        assert data["model_version"] == "integration_test_v1"
        assert isinstance(data["feature_columns"], list)

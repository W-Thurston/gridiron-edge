# tests/integration/test_model_train_predict.py
"""Integration: artifact save → load → predict roundtrip (Workstream 2)."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from gridiron_edge.models.artifact import ArtifactStore
from gridiron_edge.models.game_prediction.base import GameModelMetadata


def _make_metadata(
    model_name: str = "win_prob",
    model_type: str = "random_forest",
) -> GameModelMetadata:
    return GameModelMetadata(
        model_name=model_name,
        model_type=model_type,
        task="classification",
        trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        training_seasons=["2020-2021", "2021-2022"],
        holdout_seasons=["2023-2024"],
        parameters={"n_estimators": 10, "max_depth": 3},
        feature_columns=["HOME_FIELD", "TEAM_A_ELO", "TEAM_B_ELO"],
        n_train_rows=100,
        n_holdout_rows=20,
        holdout_brier=0.230,
    )


class TestArtifactRoundtrip:
    """Save a model artifact, load it back, verify metadata integrity."""

    def test_save_load_model_object(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: GameModelMetadata = _make_metadata()
        model_obj: dict[str, list[float] | str] = {"type": "mock_rf", "weights": [1.0, 2.0, 3.0]}

        store.save(metadata=md, model_obj=model_obj)
        loaded = store.load("win_prob", "random_forest")

        assert loaded == model_obj

    def test_save_load_metadata(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: GameModelMetadata = _make_metadata()
        store.save(metadata=md, model_obj={"w": 1})

        loaded_md = store.read_metadata("win_prob", "random_forest")
        assert isinstance(loaded_md, GameModelMetadata)
        assert loaded_md.model_name == "win_prob"
        assert loaded_md.model_type == "random_forest"
        assert loaded_md.task == "classification"
        assert loaded_md.holdout_brier == 0.230
        assert loaded_md.feature_columns == ["HOME_FIELD", "TEAM_A_ELO", "TEAM_B_ELO"]
        assert loaded_md.training_seasons == ["2020-2021", "2021-2022"]
        assert loaded_md.holdout_seasons == ["2023-2024"]

    def test_multiple_pairs_coexist(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)

        pairs: list[tuple[str, str]] = [
            ("win_prob", "random_forest"),
            ("win_prob", "xgboost"),
        ]
        for model_name, model_type in pairs:
            md = GameModelMetadata(
                model_name=model_name,
                model_type=model_type,
                task="classification",
                trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                training_seasons=["2023-2024"],
                holdout_seasons=["2024-2025"],
                parameters={},
                feature_columns=["HOME_FIELD"],
                n_train_rows=10,
                n_holdout_rows=2,
                holdout_brier=0.25,
            )
            store.save(metadata=md, model_obj={"pair": f"{model_name}_{model_type}"})

        assert store.load("win_prob", "random_forest") == {"pair": "win_prob_random_forest"}
        assert store.load("win_prob", "xgboost") == {"pair": "win_prob_xgboost"}

    def test_metadata_json_is_valid(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        md: GameModelMetadata = _make_metadata()
        store.save(metadata=md, model_obj={"w": 1})

        json_path: Path = (
            tmp_path / "data" / "models" / "win_prob" / "random_forest" / "metadata.json"
        )
        with open(json_path) as f:
            data = json.load(f)

        assert data["model_name"] == "win_prob"
        assert data["model_type"] == "random_forest"
        assert data["task"] == "classification"
        assert isinstance(data["feature_columns"], list)

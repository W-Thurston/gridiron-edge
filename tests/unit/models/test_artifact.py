# tests/unit/models/test_artifact.py

"""Tests for gridiron_edge.models.artifact - ArtifactStore.

Covers the (model_name, model_type) API and nested path scheme, plus the
metadata-subclass discrimination on read.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from gridiron_edge.models.artifact import ArtifactStore, BaseModelMetadata
from gridiron_edge.models.game_prediction.base import GameModelMetadata
from gridiron_edge.models.prop_prediction.base import PropModelMetadata

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_game_meta(
    model_name: str = "win_prob",
    model_type: str = "random_forest",
) -> GameModelMetadata:
    return GameModelMetadata(
        model_name=model_name,
        model_type=model_type,
        task="classification",
        trained_at=datetime.now(UTC).isoformat(),
        training_seasons=["2020-2021", "2021-2022"],
        holdout_seasons=["2023-2024"],
        parameters={"n_estimators": 100},
        feature_columns=["AWAY_ELO", "HOME_ELO"],
        n_train_rows=5000,
        n_holdout_rows=500,
        metrics={
            "brier": 0.220,
            "ece": 0.018,
            "auc": 0.762,
            "log_loss": 0.628,
            "accuracy": 0.681,
        },
    )


def _make_prop_meta(
    model_name: str = "qb_pass_yards",
    model_type: str = "elasticnet",
) -> PropModelMetadata:
    return PropModelMetadata(
        model_name=model_name,
        model_type=model_type,
        task="regression",
        trained_at=datetime.now(UTC).isoformat(),
        training_seasons=["2020-2021", "2021-2022"],
        holdout_seasons=["2023-2024"],
        parameters={"alpha": 0.1, "l1_ratio": 0.5},
        feature_columns=["a", "b", "c"],
        n_train_rows=5706,
        n_holdout_rows=1367,
        target_col="passing_yards",
        metrics={
            "mae": 58.0,
            "rmse": 72.6,
            "r2": 0.071,
        },
    )


# ---------------------------------------------------------------------------
# Init + paths
# ---------------------------------------------------------------------------


class TestArtifactStoreInit:
    def test_accepts_repo_path(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store is not None

    def test_artifact_dir_is_nested(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        path: Path = store.artifact_dir("win_prob", "random_forest")
        assert path == tmp_path / "data" / "models" / "win_prob" / "random_forest"

    def test_is_trained_false_when_missing(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store.is_trained("win_prob", "random_forest") is False


# ---------------------------------------------------------------------------
# Save / load round-trip - game
# ---------------------------------------------------------------------------


class TestSaveLoadGame:
    def test_save_writes_nested_path(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        meta = _make_game_meta()
        dummy = {"type": "dummy", "weights": [1, 2, 3]}

        store.save(metadata=meta, model_obj=dummy)

        artifact_dir: Path = tmp_path / "data" / "models" / "win_prob" / "random_forest"
        assert artifact_dir.is_dir()
        assert (artifact_dir / "model.joblib").exists()
        assert (artifact_dir / "metadata.json").exists()

    def test_save_then_load_returns_object(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        meta = _make_game_meta()
        dummy = {"type": "dummy", "weights": [1, 2, 3]}

        store.save(metadata=meta, model_obj=dummy)
        loaded = store.load("win_prob", "random_forest")
        assert loaded == dummy

    def test_save_then_read_metadata_returns_game_subclass(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        meta = _make_game_meta()
        store.save(metadata=meta, model_obj={"x": 1})

        out: BaseModelMetadata = store.read_metadata("win_prob", "random_forest")
        assert isinstance(out, GameModelMetadata)
        assert out.model_name == "win_prob"
        assert out.model_type == "random_forest"
        assert out.task == "classification"
        assert out.feature_columns == [
            "AWAY_ELO",
            "HOME_ELO",
        ]
        assert out.metrics["brier"] == pytest.approx(0.220)
        # Regression-side metrics default to NaN
        assert "mae" not in out.metrics

    def test_is_trained_true_after_save(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(metadata=_make_game_meta(), model_obj={"x": 1})
        assert store.is_trained("win_prob", "random_forest") is True

        def test_current_game_metadata_excludes_legacy_fields(
            self,
            tmp_path: Path,
        ) -> None:
            store = ArtifactStore(tmp_path)
            store.save(
                metadata=_make_game_meta(),
                model_obj={
                    "x": 1,
                },
            )

            metadata_path = (
                tmp_path / "data" / "models" / "win_prob" / "random_forest" / "metadata.json"
            )
            payload = json.loads(metadata_path.read_text())

            assert payload["feature_columns"] == [
                "AWAY_ELO",
                "HOME_ELO",
            ]
            assert payload["metrics"]["brier"] == pytest.approx(0.220)

            assert "holdout_brier" not in payload
            assert "holdout_mae" not in payload
            assert "holdout_rmse" not in payload
            assert "holdout_r2" not in payload


# ---------------------------------------------------------------------------
# Save / load round-trip - prop (with scaler)
# ---------------------------------------------------------------------------


class TestSaveLoadProp:
    def test_save_with_scaler_writes_both_files(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        meta = _make_prop_meta()
        store.save(
            metadata=meta,
            model_obj={"model": "dummy"},
            scaler={"scaler": "dummy"},
        )

        artifact_dir: Path = tmp_path / "data" / "models" / "qb_pass_yards" / "elasticnet"
        assert (artifact_dir / "model.joblib").exists()
        assert (artifact_dir / "scaler.joblib").exists()
        assert (artifact_dir / "metadata.json").exists()

    def test_load_scaler_returns_none_when_absent(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(metadata=_make_prop_meta(), model_obj={"x": 1})
        assert store.load_scaler("qb_pass_yards", "elasticnet") is None

    def test_load_scaler_round_trip(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(
            metadata=_make_prop_meta(),
            model_obj={"x": 1},
            scaler={"scaler": "dummy"},
        )
        loaded = store.load_scaler("qb_pass_yards", "elasticnet")
        assert loaded == {"scaler": "dummy"}

    def test_read_metadata_returns_prop_subclass(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        meta = _make_prop_meta()
        store.save(metadata=meta, model_obj={"x": 1})

        out: BaseModelMetadata = store.read_metadata("qb_pass_yards", "elasticnet")
        assert isinstance(out, PropModelMetadata)
        assert out.model_name == "qb_pass_yards"
        assert out.target_col == "passing_yards"
        assert out.metrics["mae"] == pytest.approx(58.0)


# ---------------------------------------------------------------------------
# Overwrite behavior
# ---------------------------------------------------------------------------


class TestOverwrite:
    def test_save_raises_when_artifact_exists(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(metadata=_make_game_meta(), model_obj={"v": 1})
        with pytest.raises(FileExistsError):
            store.save(metadata=_make_game_meta(), model_obj={"v": 2})

    def test_save_overwrite_replaces_artifact(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(metadata=_make_game_meta(), model_obj={"v": 1})
        store.save(metadata=_make_game_meta(), model_obj={"v": 2}, overwrite=True)
        loaded = store.load("win_prob", "random_forest")
        assert loaded == {"v": 2}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_read_metadata_missing_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.read_metadata("win_prob", "random_forest")

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load("win_prob", "random_forest")


# ---------------------------------------------------------------------------
# list_trained
# ---------------------------------------------------------------------------


class TestListTrained:
    def test_empty_when_no_models_dir(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store.list_trained() == []

    def test_lists_all_two_levels_deep(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(metadata=_make_game_meta("win_prob", "random_forest"), model_obj={"x": 1})
        store.save(metadata=_make_game_meta("win_prob", "xgboost"), model_obj={"x": 2})
        store.save(metadata=_make_prop_meta("qb_pass_yards", "elasticnet"), model_obj={"x": 3})

        out: list[BaseModelMetadata] = store.list_trained()
        assert len(out) == 3
        pairs = sorted((m.model_name, m.model_type) for m in out)
        assert pairs == [
            ("qb_pass_yards", "elasticnet"),
            ("win_prob", "random_forest"),
            ("win_prob", "xgboost"),
        ]

    def test_list_trained_returns_correct_subclasses(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(metadata=_make_game_meta(), model_obj={"x": 1})
        store.save(metadata=_make_prop_meta(), model_obj={"x": 2})

        out: list[BaseModelMetadata] = store.list_trained()
        by_name: dict[str, BaseModelMetadata] = {m.model_name: m for m in out}
        assert isinstance(by_name["win_prob"], GameModelMetadata)
        assert isinstance(by_name["qb_pass_yards"], PropModelMetadata)


# ---------------------------------------------------------------------------
# Metadata-only persistence
# ---------------------------------------------------------------------------


class TestSaveMetadataOnly:
    def test_save_metadata_writes_json(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save_metadata(_make_game_meta())
        path: Path = tmp_path / "data" / "models" / "win_prob" / "random_forest" / "metadata.json"
        assert path.exists()


class TestKindInvariant:
    def test_game_meta_has_kind_game(self) -> None:
        meta = _make_game_meta()
        assert meta.kind == "game"

    def test_prop_meta_has_kind_prop(self) -> None:
        meta = _make_prop_meta()
        assert meta.kind == "prop"


class TestExplicitKindDiscriminator:
    def test_game_metadata_roundtrips_as_game(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        meta = _make_game_meta()
        store.save(metadata=meta, model_obj={"x": 1})

        out = store.read_metadata("win_prob", "random_forest")
        assert isinstance(out, GameModelMetadata)
        assert out.kind == "game"

    def test_prop_metadata_roundtrips_as_prop(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        meta = _make_prop_meta()
        store.save(metadata=meta, model_obj={"x": 1})

        out = store.read_metadata("qb_pass_yards", "elasticnet")
        assert isinstance(out, PropModelMetadata)
        assert out.kind == "prop"


class TestBackwardCompatNoKind:
    def test_legacy_prop_metadata_detected_via_target_col(self, tmp_path: Path) -> None:
        """Old artifacts without `kind` should still discriminate correctly."""
        import json

        artifact_dir = tmp_path / "data" / "models" / "qb_pass_yards" / "elasticnet"
        artifact_dir.mkdir(parents=True)
        legacy: dict[str, object] = {
            "model_name": "qb_pass_yards",
            "model_type": "elasticnet",
            "task": "regression",
            "trained_at": "2025-01-01T00:00:00",
            "schema_version": 2,
            "training_seasons": [],
            "holdout_seasons": [],
            "parameters": {},
            "feature_columns": [],
            "n_train_rows": 0,
            "n_holdout_rows": 0,
            "notes": "",
            "target_col": "passing_yards",
            "holdout_mae": 0.0,
            "holdout_rmse": 0.0,
            "holdout_r2": 0.0,
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(legacy))

        store = ArtifactStore(tmp_path)
        out: BaseModelMetadata = store.read_metadata("qb_pass_yards", "elasticnet")
        assert isinstance(out, PropModelMetadata)
        assert out.target_col == "passing_yards"
        # Legacy holdout fields migrated into metrics. Zeros from the legacy
        # fixture survive (only NaNs are dropped).
        assert out.metrics.get("mae") == pytest.approx(0.0)

    def test_legacy_game_metadata_defaults_to_game(self, tmp_path: Path) -> None:
        import json

        artifact_dir = tmp_path / "data" / "models" / "win_prob" / "random_forest"
        artifact_dir.mkdir(parents=True)
        legacy: dict[str, object] = {
            "model_name": "win_prob",
            "model_type": "random_forest",
            "task": "classification",
            "trained_at": "2025-01-01T00:00:00",
            "schema_version": 2,
            "training_seasons": [],
            "holdout_seasons": [],
            "parameters": {},
            "feature_columns": [],
            "n_train_rows": 0,
            "n_holdout_rows": 0,
            "notes": "",
            "holdout_brier": 0.5,
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(legacy))

        store = ArtifactStore(tmp_path)
        out: BaseModelMetadata = store.read_metadata("win_prob", "random_forest")
        assert isinstance(out, GameModelMetadata)
        assert out.metrics.get("brier") == pytest.approx(0.5)


class TestLegacyMetricMigration:
    """Pre-Unit-9 metadata fields should fold into the metrics dict."""

    def test_legacy_classification_fields_migrate(self, tmp_path: Path) -> None:
        import json

        artifact_dir = tmp_path / "data" / "models" / "win_prob" / "logistic"
        artifact_dir.mkdir(parents=True)
        legacy = {
            "model_name": "win_prob",
            "model_type": "logistic",
            "task": "classification",
            "trained_at": "2025-01-01T00:00:00",
            "schema_version": 2,
            "kind": "game",
            "training_seasons": [],
            "holdout_seasons": [],
            "parameters": {},
            "feature_columns": [],
            "n_train_rows": 0,
            "n_holdout_rows": 0,
            "notes": "",
            "holdout_brier": 0.22,
            "holdout_ece": 0.02,
            "holdout_auc": 0.76,
            "holdout_log_loss": 0.63,
            "holdout_accuracy": 0.68,
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(legacy))

        store = ArtifactStore(tmp_path)
        out = store.read_metadata("win_prob", "logistic")

        assert out.metrics["brier"] == pytest.approx(0.22)
        assert out.metrics["accuracy"] == pytest.approx(0.68)
        # Legacy fields no longer present on the dataclass.
        assert not hasattr(out, "holdout_brier")

    def test_legacy_nan_metrics_are_dropped(self, tmp_path: Path) -> None:
        import json

        artifact_dir = tmp_path / "data" / "models" / "total" / "xgboost"
        artifact_dir.mkdir(parents=True)
        legacy = {
            "model_name": "total",
            "model_type": "xgboost",
            "task": "regression",
            "trained_at": "2025-01-01T00:00:00",
            "schema_version": 2,
            "kind": "game",
            "training_seasons": [],
            "holdout_seasons": [],
            "parameters": {},
            "feature_columns": [],
            "n_train_rows": 0,
            "n_holdout_rows": 0,
            "notes": "",
            # Classification metrics that should be ignored because the
            # task is regression and they're NaN.
            "holdout_brier": float("nan"),
            "holdout_ece": float("nan"),
            "holdout_auc": float("nan"),
            "holdout_log_loss": float("nan"),
            "holdout_accuracy": float("nan"),
            "holdout_mae": 8.2,
            "holdout_rmse": 10.5,
            "holdout_r2": 0.31,
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(legacy))

        store = ArtifactStore(tmp_path)
        out = store.read_metadata("total", "xgboost")

        assert out.metrics["mae"] == pytest.approx(8.2)
        assert "brier" not in out.metrics

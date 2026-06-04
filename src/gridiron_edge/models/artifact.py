# src/gridiron_edge/models/artifact.py

"""Model artifact store.

Handles reading and writing of trained model artifacts and their metadata.
Each model version gets its own directory under ``data/models/`` containing
the serialised model file and a ``metadata.json`` describing when it was
trained, what feature schema it was trained on, and its holdout performance.

Directory layout::

    data/models/
        random_forest/
            model.joblib
            metadata.json
        neural_v1/
            model.pt
            metadata.json

Artifacts are immutable once written. A new training run produces a new
model. Use ``--overwrite`` to retrain and replace the current champion.
This ensures that evaluation comparisons between versions remain valid.

Typical usage::

    store = ArtifactStore(repo_root)

    # Training: save the artifact
    from datetime import UTC, datetime

    store.save(
        model_version="random_forest",
        model_obj=fitted_pipeline,
        metadata=ModelMetadata(
            model_version="random_forest",
            trained_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
            schema_version=1,
            training_seasons=["1999-2000", ..., "2022-2023"],
            holdout_seasons=["2023-2024", "2024-2025", "2025-2026"],
            holdout_brier=0.221,
            parameters={"C": 1.0, "max_iter": 1000},
            feature_columns=["HOME_FIELD", "TEAM_A_ELO", ...],
        ),
    )

    # Prediction: load the artifact
    model = store.load("random_forest")
    metadata = store.read_metadata("random_forest")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"
_MODELS_DIR = Path("data") / "models"


@dataclass
class ModelMetadata:
    """Metadata recorded alongside every trained model artifact.

    Attributes:
        model_version: Registered model version string (e.g. ``"logistic_v1"``).
        trained_at: ISO-format UTC timestamp of when training completed.
        schema_version: Feature set schema version the model was trained on.
            Must match ``CURRENT_SCHEMA_VERSION`` in ``features/manifest.py``
            at prediction time, or predictions will be rejected.
        training_seasons: Season labels used for training.
        holdout_seasons: Season labels held out from training for evaluation.
        holdout_brier: Brier score on the holdout set. Primary quality signal.
        parameters: Hyperparameters used during training (free-form dict).
        feature_columns: Ordered list of feature columns the model expects.
            Used to validate the feature matrix at prediction time.
        notes: Optional free-text notes about this training run.
    """

    model_version: str
    trained_at: str
    schema_version: int
    training_seasons: list[str]
    holdout_seasons: list[str]
    holdout_brier: float
    parameters: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    notes: str = ""


class ArtifactStore:
    """Filesystem store for trained model artifacts and metadata.

    Args:
        repo: Repository root path. Artifacts are stored under
            ``{repo}/data/models/{model_version}/``.
    """

    def __init__(self, repo: Path) -> None:
        self._root = repo / _MODELS_DIR

    def artifact_dir(self, model_version: str) -> Path:
        """Return the artifact directory for a model version.

        Args:
            model_version: Registered model version string.

        Returns:
            Path to ``data/models/{model_version}/``. Not guaranteed to exist.
        """
        return self._root / model_version

    def is_trained(self, model_version: str) -> bool:
        """Return whether a trained artifact exists for this model version.

        Args:
            model_version: Registered model version string.

        Returns:
            ``True`` if ``metadata.json`` exists in the artifact directory.
        """
        return (self.artifact_dir(model_version) / _METADATA_FILENAME).exists()

    def read_metadata(self, model_version: str) -> ModelMetadata:
        """Load metadata for a trained model artifact.

        Args:
            model_version: Registered model version string.

        Returns:
            ``ModelMetadata`` for the artifact.

        Raises:
            FileNotFoundError: If no artifact exists for this model version.
        """
        path = self.artifact_dir(model_version) / _METADATA_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No trained artifact found for '{model_version}'. "
                f"Run 'gridiron models train {model_version}' first."
            )
        data = json.loads(path.read_text())
        return ModelMetadata(**data)

    def save_metadata(self, metadata: ModelMetadata) -> Path:
        """Write metadata to the artifact directory.

        Args:
            metadata: Metadata to write.

        Returns:
            Path to the written ``metadata.json`` file.
        """
        directory = self.artifact_dir(metadata.model_version)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _METADATA_FILENAME
        path.write_text(json.dumps(asdict(metadata), indent=2))
        logger.debug("Model metadata written to %s", path)
        return path

    def save(
        self,
        model_version: str,
        model_obj: object,
        *,
        metadata: ModelMetadata,
        filename: str = "model.joblib",
    ) -> Path:
        """Serialise a model object and write metadata to the artifact store.

        Uses ``joblib`` for serialisation — suitable for sklearn pipelines
        and most Python objects. For PyTorch models, call ``save_metadata``
        directly and handle serialisation with ``torch.save``.

        Args:
            model_version: Registered model version string.
            model_obj: The fitted model object to serialise.
            metadata: Metadata to write alongside the artifact.
            filename: Filename for the serialised model. Defaults to
                ``"model.joblib"``.

        Returns:
            Path to the written model file.

        Raises:
            FileExistsError: If an artifact already exists for this version.
                Artifacts are immutable — use a new version string instead.
        """
        directory = self.artifact_dir(model_version)
        model_path = directory / filename

        if model_path.exists():
            raise FileExistsError(
                f"Artifact already exists for '{model_version}' at {model_path}. "
                "Artifacts are immutable. Use a new version string "
                "(e.g. 'logistic_v2') rather than overwriting."
            )

        try:
            import joblib  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "joblib is required to save model artifacts. Add it to pyproject.toml dependencies."
            ) from e

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_obj, model_path)
        self.save_metadata(metadata)

        logger.info(
            "Artifact saved: %s (%.1f KB)",
            model_path,
            model_path.stat().st_size / 1024,
        )
        return model_path

    def load(
        self,
        model_version: str,
        *,
        filename: str = "model.joblib",
    ) -> Any:  # noqa: ANN401
        """Load a serialised model object from the artifact store.

        Args:
            model_version: Registered model version string.
            filename: Filename of the serialised model. Defaults to
                ``"model.joblib"``.

        Returns:
            The deserialised model object.

        Raises:
            FileNotFoundError: If no artifact exists for this version.
        """
        path = self.artifact_dir(model_version) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"No model artifact found at {path}. "
                f"Run 'gridiron models train {model_version}' first."
            )

        try:
            import joblib  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("joblib is required to load model artifacts.") from e

        return joblib.load(path)

    def list_trained(self) -> list[ModelMetadata]:
        """Return metadata for all trained artifacts in the store.

        Returns:
            List of ``ModelMetadata`` objects, sorted by ``model_version``.
            Empty list if no artifacts exist.
        """
        if not self._root.exists():
            return []

        results: list[ModelMetadata] = []
        for version_dir in sorted(self._root.iterdir()):
            if not version_dir.is_dir():
                continue
            meta_path = version_dir / _METADATA_FILENAME
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text())
                    results.append(ModelMetadata(**data))
                except Exception:
                    logger.warning("Could not read metadata from %s", meta_path)

        return results

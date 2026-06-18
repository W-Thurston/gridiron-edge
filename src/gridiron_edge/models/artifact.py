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
from logging import Logger
from pathlib import Path
from typing import Any

logger: Logger = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"
_MODELS_DIR: Path = Path("data") / "models"


@dataclass(kw_only=True)
class BaseModelMetadata:
    """Shared metadata for all trained model artifacts.

    Both :class:`gridiron_edge.models.game_prediction.base.GameModelMetadata`
    and :class:`gridiron_edge.models.prop_prediction.base.PropModelMetadata`
    inherit from this class. Subclasses add task-specific holdout metrics.

    Field naming convention (Workstream 2):
        - ``model_name``: model purpose (e.g. ``"win_prob"``, ``"total"``,
          ``"qb_pass_yards"``).
        - ``model_type``: algorithm (e.g. ``"random_forest"``, ``"xgboost"``,
          ``"logistic"``, ``"elasticnet"``).
        - ``task``: ``"classification"`` or ``"regression"``.

    Construction is keyword-only (``kw_only=True``) so that subclasses can
    add required fields without dataclass field-ordering errors.

    Attributes:
        model_name: Model purpose.
        model_type: Model algorithm.
        task: ``"classification"`` or ``"regression"``.
        trained_at: ISO-format UTC timestamp of when training completed.
        schema_version: Feature set schema version. Bumped to ``2`` for the
            WS2 metadata break.
        training_seasons: Season labels used for training, e.g.
            ``["1999-2000", ..., "2022-2023"]``.
        holdout_seasons: Season labels held out from training.
        parameters: Hyperparameters used during training.
        feature_columns: Ordered list of feature columns the model expects.
        n_train_rows: Number of training rows.
        n_holdout_rows: Number of holdout rows.
        notes: Optional free-text notes about this training run.
    """

    model_name: str
    model_type: str
    task: str
    trained_at: str
    schema_version: int = 2
    training_seasons: list[str] = field(default_factory=list)
    holdout_seasons: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    n_train_rows: int = 0
    n_holdout_rows: int = 0
    notes: str = ""


@dataclass
class ModelMetadata:
    """Metadata recorded alongside every trained model artifact.

    Attributes:
        model_version: Registered model version string (e.g. ``"logistic"``).
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
        self._root: Path = repo / _MODELS_DIR

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
        path: Path = self.artifact_dir(model_version) / _METADATA_FILENAME
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
        directory: Path = self.artifact_dir(metadata.model_version)
        directory.mkdir(parents=True, exist_ok=True)
        path: Path = directory / _METADATA_FILENAME
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
        directory: Path = self.artifact_dir(model_version)
        model_path: Path = directory / filename

        if model_path.exists():
            raise FileExistsError(
                f"Artifact already exists for '{model_version}' at {model_path}. "
                "Artifacts are immutable. Use a new version string "
                "(e.g. 'logistic') rather than overwriting."
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
        path: Path = self.artifact_dir(model_version) / filename
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
            meta_path: Path = version_dir / _METADATA_FILENAME
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text())
                    results.append(ModelMetadata(**data))
                except Exception:
                    logger.warning("Could not read metadata from %s", meta_path)

        return results

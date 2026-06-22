# src/gridiron_edge/models/artifact.py

"""Model artifact store.

Handles reading and writing of trained model artifacts and their metadata.
Each trained model is identified by the pair (model_name, model_type) and
lives under ``data/models/{model_name}/{model_type}/`` containing the
serialised model, an optional scaler, and a ``metadata.json``.

Directory layout (Workstream 2)::

    data/models/
        win_prob/
            random_forest/
                model.joblib
                metadata.json
            xgboost/
                model.joblib
                metadata.json
            logistic/
                model.joblib
                scaler.joblib
                metadata.json
        total/
            random_forest/
                model.joblib
                metadata.json
        qb_pass_yards/
            elasticnet/
                model.joblib
                scaler.joblib
                metadata.json

Artifacts are immutable once written. A new training run replaces the
existing artifact for that (model_name, model_type) pair.

Field naming convention:
    - ``model_name``: purpose (``"win_prob"``, ``"total"``, ``"qb_pass_yards"``)
    - ``model_type``: algorithm (``"random_forest"``, ``"xgboost"``, ``"logistic"``,
      ``"elasticnet"``)

Typical usage::

    store = ArtifactStore(repo_root)

    # Save (Game)
    meta = GameModelMetadata(
        model_name="win_prob",
        model_type="random_forest",
        task="classification",
        trained_at=datetime.now(UTC).isoformat(),
        ...,
        holdout_brier=0.220,
    )
    store.save(metadata=meta, model_obj=fitted_pipeline)

    # Save (Prop, with scaler)
    meta = PropModelMetadata(
        model_name="qb_pass_yards",
        model_type="elasticnet",
        task="regression",
        ...,
    )
    store.save(metadata=meta, model_obj=model, scaler=scaler)

    # Load
    model = store.load("win_prob", "random_forest")
    meta = store.read_metadata("win_prob", "random_forest")  # → GameModelMetadata
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"
_MODEL_FILENAME = "model.joblib"
_SCALER_FILENAME = "scaler.joblib"
_MODELS_DIR = Path("data") / "models"


@dataclass(kw_only=True)
class BaseModelMetadata:
    """Shared metadata for all trained model artifacts.

    Both :class:`gridiron_edge.models.game_prediction.base.GameModelMetadata`
    and :class:`gridiron_edge.models.prop_prediction.base.PropModelMetadata`
    inherit from this class. Subclasses add task-specific identity fields
    (e.g. ``target_col`` for props) but no longer carry their own
    holdout-metric fields - those live in :attr:`metrics`.

    Construction is keyword-only (``kw_only=True``) so subclasses can add
    required fields without dataclass field-ordering errors.

    Field naming convention (Workstream 2):
        - ``model_name``: model purpose (e.g. ``"win_prob"``, ``"total"``,
          ``"qb_pass_yards"``).
        - ``model_type``: algorithm (e.g. ``"random_forest"``, ``"xgboost"``,
          ``"logistic"``, ``"elasticnet"``).
        - ``task``: ``"classification"`` or ``"regression"``.

    The :attr:`metrics` dict carries task-appropriate holdout metrics:

    classification:
        ``brier``, ``ece``, ``auc``, ``log_loss``, ``accuracy``

    regression:
        ``mae``, ``rmse``, ``r2``

    Display surfaces dispatch on :attr:`task` to select the right keys.
    """

    model_name: str
    model_type: str
    task: str
    trained_at: str
    schema_version: int = 3
    kind: str = "game"
    training_seasons: list[str] = field(default_factory=list)
    holdout_seasons: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    n_train_rows: int = 0
    n_holdout_rows: int = 0
    notes: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def _read_metadata_subclass(data: dict[str, Any]) -> BaseModelMetadata:
    """Discriminate metadata subclass from on-disk JSON.

    Preferred discriminator: explicit ``kind`` field. Backward-compat:
    artifacts written before Unit 6b stored no ``kind`` field; fall back
    to detecting prop metadata by the presence of ``target_col``.

    Backward-compat also handles the Unit 9 metric migration: artifacts
    written before Unit 9 stored each holdout metric as a top-level
    field (``holdout_brier``, ``holdout_mae``, etc.). On read those
    legacy fields are folded into the :attr:`BaseModelMetadata.metrics`
    dict and the original keys are stripped so the dataclass constructor
    does not see them.

    Both branches strip unknown keys defensively so additions to the
    *other* subclass do not crash on this load.
    """
    from gridiron_edge.models.game_prediction.base import GameModelMetadata
    from gridiron_edge.models.prop_prediction.base import PropModelMetadata

    data = _migrate_legacy_metrics(dict(data))

    kind: str | None = data.get("kind")
    if kind == "prop":
        cls: type[BaseModelMetadata] = PropModelMetadata
    elif kind == "game":
        cls = GameModelMetadata
    else:
        cls = PropModelMetadata if "target_col" in data else GameModelMetadata

    known: set[str] = {f.name for f in fields(cls)}
    filtered: dict[str, Any] = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)


_LEGACY_CLASSIFICATION_METRICS: dict[str, str] = {
    "holdout_brier": "brier",
    "holdout_ece": "ece",
    "holdout_auc": "auc",
    "holdout_log_loss": "log_loss",
    "holdout_accuracy": "accuracy",
}

_LEGACY_REGRESSION_METRICS: dict[str, str] = {
    "holdout_mae": "mae",
    "holdout_rmse": "rmse",
    "holdout_r2": "r2",
}


def _migrate_legacy_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Fold legacy top-level metric fields into the new metrics dict.

    Pre-Unit-9 artifacts persisted each metric as its own top-level
    field. This helper drains those legacy keys into a single
    ``metrics`` dict so the dataclass constructor only receives the new
    schema.

    NaN metrics from the legacy schema are dropped. The new schema does
    not store NaNs - absence means "not recorded".
    """
    import math

    metrics: dict[str, float] = dict(data.get("metrics", {}))
    legacy_keys: set[str] = set(_LEGACY_CLASSIFICATION_METRICS) | set(_LEGACY_REGRESSION_METRICS)

    for legacy_key in list(legacy_keys):
        if legacy_key not in data:
            continue
        value = data.pop(legacy_key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        new_key = _LEGACY_CLASSIFICATION_METRICS.get(legacy_key) or _LEGACY_REGRESSION_METRICS.get(
            legacy_key
        )
        if new_key is not None and new_key not in metrics:
            metrics[new_key] = float(value)

    if metrics:
        data["metrics"] = metrics
    return data


class ArtifactStore:
    """Filesystem store for trained model artifacts and metadata.

    Artifacts are identified by ``(model_name, model_type)`` and stored
    under ``{repo}/data/models/{model_name}/{model_type}/``. The store
    is metadata-class agnostic: it accepts any :class:`BaseModelMetadata`
    subclass and discriminates on read.
    """

    def __init__(self, repo: Path) -> None:
        self._root: Path = repo / _MODELS_DIR

    def artifact_dir(self, model_name: str, model_type: str) -> Path:
        """Return the artifact directory for a (model_name, model_type) pair.

        The directory is not guaranteed to exist.
        """
        return self._root / model_name / model_type

    def is_trained(self, model_name: str, model_type: str) -> bool:
        """Return whether a trained artifact exists for this pair."""
        return (self.artifact_dir(model_name, model_type) / _METADATA_FILENAME).exists()

    def read_metadata(self, model_name: str, model_type: str) -> BaseModelMetadata:
        """Load metadata for a trained artifact.

        Returns a :class:`GameModelMetadata` or :class:`PropModelMetadata`
        instance based on the JSON shape on disk.

        Raises:
            FileNotFoundError: If no artifact exists for this pair.
        """
        path: Path = self.artifact_dir(model_name, model_type) / _METADATA_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No trained artifact found for ({model_name!r}, {model_type!r}). "
                f"Expected metadata at {path}."
            )
        data: dict[str, Any] = json.loads(path.read_text())
        return _read_metadata_subclass(data)

    def save_metadata(self, metadata: BaseModelMetadata) -> Path:
        """Write metadata to the artifact directory.

        Returns the path to the written ``metadata.json``.
        """
        directory: Path = self.artifact_dir(metadata.model_name, metadata.model_type)
        directory.mkdir(parents=True, exist_ok=True)
        path: Path = directory / _METADATA_FILENAME
        path.write_text(json.dumps(asdict(metadata), indent=2))
        logger.debug("Model metadata written to %s", path)
        return path

    def save(
        self,
        *,
        metadata: BaseModelMetadata,
        model_obj: object,
        scaler: object | None = None,
        filename: str = _MODEL_FILENAME,
        scaler_filename: str = _SCALER_FILENAME,
        overwrite: bool = False,
    ) -> Path:
        """Serialise a model object (and optional scaler) and write metadata.

        Args:
            metadata: Model metadata. Drives the storage path via
                ``metadata.model_name`` and ``metadata.model_type``.
            model_obj: Fitted model object to serialise.
            scaler: Optional fitted scaler to serialise alongside the model.
                Used by logistic / elasticnet models that require feature
                standardisation at predict time.
            filename: Filename for the serialised model.
            scaler_filename: Filename for the serialised scaler.
            overwrite: If False (default), raises if an artifact already
                exists. Champion retrains pass ``overwrite=True``.

        Returns:
            Path to the written model file.

        Raises:
            FileExistsError: If ``overwrite=False`` and an artifact already
                exists for this pair.
        """
        directory: Path = self.artifact_dir(metadata.model_name, metadata.model_type)
        model_path: Path = directory / filename

        if model_path.exists() and not overwrite:
            raise FileExistsError(
                f"Artifact already exists for "
                f"({metadata.model_name!r}, {metadata.model_type!r}) at {model_path}. "
                f"Pass overwrite=True to replace."
            )

        try:
            import joblib  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "joblib is required to save model artifacts. Add it to pyproject.toml dependencies."
            ) from e

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_obj, model_path)

        if scaler is not None:
            scaler_path: Path = directory / scaler_filename
            joblib.dump(scaler, scaler_path)
            logger.debug("Scaler written to %s", scaler_path)

        self.save_metadata(metadata)

        logger.info(
            "Artifact saved: %s (%.1f KB)",
            model_path,
            model_path.stat().st_size / 1024,
        )
        return model_path

    def load(
        self,
        model_name: str,
        model_type: str,
        *,
        filename: str = _MODEL_FILENAME,
    ) -> Any:  # noqa: ANN401
        """Load a serialised model object."""
        path: Path = self.artifact_dir(model_name, model_type) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"No model artifact found at {path}. Train ({model_name!r}, {model_type!r}) first."
            )

        try:
            import joblib  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("joblib is required to load model artifacts.") from e

        return joblib.load(path)

    def load_scaler(
        self,
        model_name: str,
        model_type: str,
        *,
        filename: str = _SCALER_FILENAME,
    ) -> Any | None:  # noqa: ANN401
        """Load a serialised scaler if one was saved alongside the model.

        Returns ``None`` if no scaler file is present (e.g. tree models).
        """
        path: Path = self.artifact_dir(model_name, model_type) / filename
        if not path.exists():
            return None

        try:
            import joblib  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("joblib is required to load scaler artifacts.") from e

        return joblib.load(path)

    def list_trained(self) -> list[BaseModelMetadata]:
        """Return metadata for all trained artifacts in the store.

        Walks ``data/models/*/*/`` two levels deep. Returns the list sorted
        by ``(model_name, model_type)``.
        """
        if not self._root.exists():
            return []

        results: list[BaseModelMetadata] = []
        for name_dir in sorted(self._root.iterdir()):
            if not name_dir.is_dir():
                continue
            for type_dir in sorted(name_dir.iterdir()):
                if not type_dir.is_dir():
                    continue
                meta_path: Path = type_dir / _METADATA_FILENAME
                if not meta_path.exists():
                    continue
                try:
                    data: dict[str, Any] = json.loads(meta_path.read_text())
                    results.append(_read_metadata_subclass(data))
                except Exception:
                    logger.warning("Could not read metadata from %s", meta_path)

        return results

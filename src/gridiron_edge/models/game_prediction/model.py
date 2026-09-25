# src/gridiron_edge/models/game_prediction/model.py

"""Game prediction model registry entry point.

This module is the single import that callers use to ensure all game
prediction models are registered with ``ModelRegistry``. It contains:

- The :class:`GamesModel` base class (GameModel + Trainable protocols).
- Five composite-key subclasses registered with ``ModelRegistry``:
    * ``"win_prob_logistic"`` / ``"win_prob_random_forest"`` / ``"win_prob_xgboost"``
    * ``"total_random_forest"`` / ``"total_xgboost"``
- Pure helpers that assemble canonical game-level classification and
  regression prediction rows.

All game-side training and prediction flows through :class:`GamesTrainer`
and this module's :class:`GamesModel`. ``ModelRegistry`` keys use
the composite ``{model_name}_{model_type}`` convention (e.g.
``"win_prob_random_forest"``).
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np
import pandas as pd

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.datasets.loaders import load_modeling_file
from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    CalibrationResolutionSource,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionFeatureSchema,
    PredictionPostProcessingEvidence,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    create_prediction_feature_schema,
)
from gridiron_edge.evaluation.prediction_input_sources import (
    file_identity,
)
from gridiron_edge.features.pipeline import (
    CANONICAL_FEATURES,
)
from gridiron_edge.features.registry import run_features
from gridiron_edge.models.artifact import (
    ArtifactStore,
    BaseModelMetadata,
)
from gridiron_edge.models.base import ModelSpec
from gridiron_edge.models.game_prediction._columns import _SCHEMA_VERSION, FeatureSet
from gridiron_edge.models.game_prediction._epa_window import _rebuild_features_with_window
from gridiron_edge.models.game_prediction.base import (
    GameModelMetadata,
    GameModelSpec,
    GameModelType,
    GamesTrainer,
)
from gridiron_edge.models.game_prediction.post_process import (
    PredictionPostProcessingResolution,
    enrich_predictions,
    enrich_predictions_with_resolution,
    resolve_prediction_post_processing,
)
from gridiron_edge.models.game_prediction.prediction_execution import (
    StatisticalPredictionExecution,
    build_statistical_prediction_execution,
)
from gridiron_edge.models.game_prediction.total import TotalTrainer
from gridiron_edge.models.game_prediction.win_prob import WinProbTrainer
from gridiron_edge.models.registry import ModelRegistry

if TYPE_CHECKING:
    from pandas import DataFrame


logger: Logger = logging.getLogger(__name__)
_CALIBRATOR_FILENAME: Final[str] = "calibrator.joblib"
_CALIBRATION_REGISTRY_PATH: Final[str] = "data/output/calibration/game_model_calibration.json"

# ---------------------------------------------------------------------------
# Trainer dispatch - maps model_name → GamesTrainer subclass
# ---------------------------------------------------------------------------


_TRAINER_FOR_NAME: dict[str, type[GamesTrainer]] = {
    "win_prob": WinProbTrainer,
    "total": TotalTrainer,
}


def get_known_model_names() -> tuple[str, ...]:
    """Return the model_names recognized by ``GamesModel``.

    Used by composite-key parsing in other modules (e.g.
    :mod:`evaluation.select`, :mod:`cli.models`, :mod:`cli.evaluate`)
    to split keys of the form ``f"{model_name}_{model_type}"`` correctly
    when ``model_name`` itself contains underscores.

    Returns:
        Tuple of registered model_names, sorted longest-first so that
        prefix matching against ambiguous keys is deterministic.
    """
    return tuple(sorted(_TRAINER_FOR_NAME.keys(), key=len, reverse=True))


# ---------------------------------------------------------------------------
# Canonical historical prediction rows
# ---------------------------------------------------------------------------


def build_game_predictions(
    df: pd.DataFrame,
    home_win_probs: np.ndarray,
) -> pd.DataFrame:
    """Map canonical Home-win probabilities to one row per game.

    Args:
        df: Canonical one-row-per-game modeling DataFrame.
        home_win_probs: Probability that the designated Home team wins,
            aligned one-to-one with ``df``.

    Returns:
        Canonical game prediction rows with Home-win probability stored
        directly and Away-win probability derived as its complement.

    Raises:
        ValueError: If prediction count does not match the input rows or
            canonical Game IDs are duplicated.
    """
    if len(home_win_probs) != len(df):
        raise ValueError("Home-win probability count must match canonical game rows.")

    if df["GAME_ID"].duplicated().any():
        raise ValueError("Canonical prediction input contains duplicate game IDs.")

    work = df.copy()
    work["_HOME_WIN_PROB"] = home_win_probs

    work = work.sort_values(
        [
            "YEAR",
            "WEEK_NUM",
            "GAME_ID",
        ],
        kind="stable",
    )

    home_probabilities = work["_HOME_WIN_PROB"].to_numpy(dtype=float)

    return pd.DataFrame(
        {
            "season": work["YEAR"].values,
            "week": work["WEEK_NUM"].astype(int).values,
            "game_id": work["GAME_ID"].values,
            "game_date": work.get(
                "GAME_DATE",
                pd.Series(
                    [None] * len(work),
                    index=work.index,
                    dtype=object,
                ),
            ).values,
            "away_team": work["AWAY_TEAM"].values,
            "home_team": work["HOME_TEAM"].values,
            "away_elo": work.get(
                "AWAY_ELO",
                pd.Series(
                    [float("nan")] * len(work),
                    index=work.index,
                    dtype=float,
                ),
            ).values,
            "home_elo": work.get(
                "HOME_ELO",
                pd.Series(
                    [float("nan")] * len(work),
                    index=work.index,
                    dtype=float,
                ),
            ).values,
            "away_win_prob": (1.0 - home_probabilities),
            "home_win_prob": home_probabilities,
        }
    ).reset_index(drop=True)


def build_regression_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Map Total predictions directly to canonical game rows.

    Args:
        df: Canonical one-row-per-game modeling DataFrame.
        predictions: Predicted combined scores aligned one-to-one with
            the input rows.

    Returns:
        One Total prediction row per canonical game.

    Raises:
        ValueError: If prediction count does not match the input rows or
            canonical Game IDs are duplicated.
    """
    if len(predictions) != len(df):
        raise ValueError("Total prediction count must match canonical game rows.")

    if df["GAME_ID"].duplicated().any():
        raise ValueError("Canonical Total prediction input contains duplicate game IDs.")

    work: DataFrame = df.copy()
    work["_MODEL_TOTAL"] = predictions

    work = work.sort_values(
        [
            "YEAR",
            "WEEK_NUM",
            "GAME_ID",
        ],
        kind="stable",
    )

    return pd.DataFrame(
        {
            "season": work["YEAR"].values,
            "week": work["WEEK_NUM"].astype(int).values,
            "game_id": work["GAME_ID"].values,
            "game_date": work.get(
                "GAME_DATE",
                pd.Series(
                    [None] * len(work),
                    index=work.index,
                    dtype=object,
                ),
            ).values,
            "away_team": work["AWAY_TEAM"].values,
            "home_team": work["HOME_TEAM"].values,
            "model_total": work["_MODEL_TOTAL"].to_numpy(dtype=float),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical prediction evidence helpers
# ---------------------------------------------------------------------------


def _present_binary_reference(
    kind: PredictionArtifactKind,
    *,
    path: Path,
    repo: Path,
) -> BinaryArtifactReference:
    """Capture one required present binary artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Required prediction binary artifact is missing: {path}")

    digest, size_bytes = file_identity(path)
    return BinaryArtifactReference(
        kind=kind,
        source_relative_path=path.relative_to(repo).as_posix(),
        state=PredictionArtifactState.PRESENT,
        content_digest=digest,
        size_bytes=size_bytes,
    )


def _optional_binary_reference(
    kind: PredictionArtifactKind,
    *,
    path: Path,
    repo: Path,
) -> BinaryArtifactReference:
    """Capture one optional binary artifact or explicit absence."""
    relative_path = path.relative_to(repo).as_posix()

    if not path.is_file():
        return BinaryArtifactReference(
            kind=kind,
            source_relative_path=relative_path,
            state=PredictionArtifactState.ABSENT,
            content_digest=None,
            size_bytes=None,
        )

    digest, size_bytes = file_identity(path)
    return BinaryArtifactReference(
        kind=kind,
        source_relative_path=relative_path,
        state=PredictionArtifactState.PRESENT,
        content_digest=digest,
        size_bytes=size_bytes,
    )


def _statistical_artifact_references(
    store: ArtifactStore,
    *,
    model_name: str,
    model_type: str,
    repo: Path,
) -> tuple[BinaryArtifactReference, ...]:
    """Capture the exact statistical model artifact inventory."""
    artifact_directory = store.artifact_dir(
        model_name,
        model_type,
    )
    references = (
        _optional_binary_reference(
            PredictionArtifactKind.EXTERNAL_CALIBRATOR,
            path=artifact_directory / _CALIBRATOR_FILENAME,
            repo=repo,
        ),
        _present_binary_reference(
            PredictionArtifactKind.MODEL,
            path=store.model_path(model_name, model_type),
            repo=repo,
        ),
        _present_binary_reference(
            PredictionArtifactKind.MODEL_METADATA,
            path=store.metadata_path(model_name, model_type),
            repo=repo,
        ),
        _optional_binary_reference(
            PredictionArtifactKind.SCALER,
            path=store.scaler_path(model_name, model_type),
            repo=repo,
        ),
    )

    return tuple(
        sorted(
            references,
            key=lambda reference: reference.kind.value,
        )
    )


def _validated_epa_window(parameters: dict[str, object]) -> int:
    """Validate and return one artifact's persisted EPA rolling window."""
    epa_window = parameters.get("epa_window")
    if isinstance(epa_window, bool) or not isinstance(epa_window, int) or epa_window < 1:
        raise ValueError("Prediction metadata epa_window must be a positive integer.")
    return epa_window


def _validate_prediction_metadata(
    metadata: BaseModelMetadata,
    *,
    model_name: str,
    model_type: str,
    task: str,
    feature_set: FeatureSet,
    features: pd.DataFrame,
) -> GameModelMetadata:
    """Validate persisted metadata against runtime prediction inputs."""
    if not isinstance(metadata, GameModelMetadata):
        raise TypeError("Game prediction requires GameModelMetadata.")

    if metadata.model_name != model_name:
        raise ValueError("Prediction metadata model_name does not match the selected model.")

    if metadata.model_type != model_type:
        raise ValueError("Prediction metadata model_type does not match the selected model.")

    if metadata.task != task:
        raise ValueError("Prediction metadata task does not match the selected model.")

    feature_set_name = metadata.parameters.get("feature_set")
    if feature_set_name != feature_set.name:
        raise ValueError(
            "Prediction metadata feature_set does not match the current model feature contract."
        )

    modeling_schema_version = metadata.parameters.get("modeling_schema_version")
    if (
        isinstance(modeling_schema_version, bool)
        or not isinstance(modeling_schema_version, int)
        or modeling_schema_version != _SCHEMA_VERSION
    ):
        raise ValueError(
            "Prediction metadata modeling_schema_version does not "
            "match the current modeling schema."
        )

    _validated_epa_window(metadata.parameters)

    persisted_columns = tuple(metadata.feature_columns)
    declared_columns = tuple(feature_set.feature_names)
    runtime_columns = tuple(str(column) for column in features.columns)

    if persisted_columns != declared_columns:
        raise ValueError(
            "Prediction metadata feature_columns do not match the current model feature contract."
        )

    if runtime_columns != declared_columns:
        raise ValueError(
            "Runtime prediction feature columns do not match the current model feature contract."
        )

    return metadata


def _prediction_feature_schema(
    metadata: GameModelMetadata,
    *,
    feature_set: FeatureSet,
) -> PredictionFeatureSchema:
    """Create the authenticated ordered prediction feature schema."""
    modeling_schema_version = metadata.parameters["modeling_schema_version"]

    if isinstance(modeling_schema_version, bool) or not isinstance(modeling_schema_version, int):
        raise ValueError("Prediction metadata modeling_schema_version must be an integer.")

    return create_prediction_feature_schema(
        model_name=metadata.model_name,
        model_type=metadata.model_type,
        task=metadata.task,
        modeling_schema_version=modeling_schema_version,
        epa_window=_validated_epa_window(metadata.parameters),
        feature_set_name=feature_set.name,
        ordered_columns=tuple(metadata.feature_columns),
    )


def _source_reference(
    source_artifacts: tuple[SourceArtifactReference, ...],
    *,
    relative_path: str,
) -> SourceArtifactReference:
    """Return one exact source reference from the captured inventory."""
    matches = tuple(
        reference for reference in source_artifacts if reference.relative_path == relative_path
    )

    if len(matches) != 1:
        raise ValueError(
            f"Prediction source inventory must contain exactly one reference for {relative_path}."
        )

    return matches[0]


def _embedded_estimator_calibration(
    metadata: GameModelMetadata,
) -> bool:
    """Return whether calibration is embedded in the persisted estimator."""
    if metadata.task != "classification":
        return False

    if metadata.model_type == "random_forest":
        return True

    if metadata.model_type == "xgboost":
        value = metadata.parameters.get("calibration_applied")
        if not isinstance(value, bool):
            raise ValueError(
                "XGBoost classification metadata requires a boolean calibration_applied value."
            )
        return value

    return False


def _post_processing_evidence(
    resolution: PredictionPostProcessingResolution,
    *,
    metadata: GameModelMetadata,
    source_artifacts: tuple[SourceArtifactReference, ...],
    binary_artifacts: tuple[BinaryArtifactReference, ...],
) -> PredictionPostProcessingEvidence:
    """Build exact Win post-processing execution evidence."""
    registry_reference = _source_reference(
        source_artifacts,
        relative_path=_CALIBRATION_REGISTRY_PATH,
    )

    external_references = tuple(
        reference
        for reference in binary_artifacts
        if (reference.kind is PredictionArtifactKind.EXTERNAL_CALIBRATOR)
    )
    if len(external_references) != 1:
        raise ValueError(
            "Statistical artifact inventory must contain exactly one external calibrator reference."
        )

    external_state = external_references[0].state
    expected_state = (
        PredictionArtifactState.PRESENT
        if resolution.calibrator is not None
        else PredictionArtifactState.ABSENT
    )
    if external_state is not expected_state:
        raise ValueError(
            "Loaded external calibrator state does not match the captured artifact inventory."
        )

    if (
        resolution.sigma_source is CalibrationResolutionSource.PERSISTED_REGISTRY
        or resolution.margin_std_source is CalibrationResolutionSource.PERSISTED_REGISTRY
    ):
        if registry_reference.state is not PredictionSourceState.PRESENT:
            raise ValueError(
                "Persisted post-processing calibration requires "
                "a present calibration registry source."
            )
    elif resolution.registry_entry_updated_at is not None:
        raise ValueError(
            "Unused calibration registry entry must not carry an updated_at timestamp."
        )

    return PredictionPostProcessingEvidence(
        registry_reference=registry_reference,
        registry_entry_updated_at=(resolution.registry_entry_updated_at),
        sigma=resolution.sigma,
        sigma_source=resolution.sigma_source,
        margin_std=resolution.margin_std,
        margin_std_source=resolution.margin_std_source,
        external_calibrator_state=external_state,
        embedded_estimator_calibration=(_embedded_estimator_calibration(metadata)),
    )


# ---------------------------------------------------------------------------
# GamesModel base
# ---------------------------------------------------------------------------


class GamesModel:
    """Base class for game prediction models.

    Each composite ``(model_name, model_type)`` pair has a thin subclass
    that sets ``model_name``, ``model_type``, and ``spec`` at class scope
    and is registered with :class:`ModelRegistry`. All logic lives
    here - subclasses are spec-only.

    The class implements both :class:`Model` (via ``predict_historical``
    / ``predict_upcoming``) and :class:`Trainable` (via ``train`` /
    ``is_trained``). Dispatch on classification vs regression happens
    internally based on the trainer's :attr:`GameModelSpec.task`.
    """

    # Set by subclasses.
    model_name: ClassVar[str] = ""
    model_type: ClassVar[str] = ""
    spec: ModelSpec

    # ------------------------------------------------------------------
    # Trainer / spec accessors
    # ------------------------------------------------------------------

    def _trainer(self) -> GamesTrainer:
        """Return a fresh :class:`GamesTrainer` instance for this model_name."""
        trainer_cls: type[GamesTrainer] = _TRAINER_FOR_NAME[self.model_name]
        return trainer_cls()

    def _game_model_spec(self) -> GameModelSpec:
        """Return the underlying :class:`GameModelSpec` from the trainer."""
        return self._trainer().spec

    def _task(self) -> str:
        """Return ``"classification"`` or ``"regression"`` for this model."""
        return self._game_model_spec().task

    def prediction_feature_set(self) -> FeatureSet:
        """Return the exact current feature contract for prediction."""
        gm_spec: GameModelSpec = self._game_model_spec()
        return gm_spec.feature_set[GameModelType(self.model_type)]

    def _feature_fn(self):  # noqa: ANN202 - return type is a Callable
        """Return the feature engineering function for this model_type."""
        return self.prediction_feature_set().feature_fn

    # ------------------------------------------------------------------
    # Trainable protocol
    # ------------------------------------------------------------------

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for this (model_name, model_type) pair."""
        resolved_repo: Path = repo or get_settings().repo_root
        return ArtifactStore(resolved_repo).is_trained(self.model_name, self.model_type)

    def train(
        self,
        df: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> GameModelMetadata:
        """Train the underlying model and save its artifact.

        Delegates to :meth:`GamesTrainer.train` with the appropriate
        :class:`GameModelType`. Returns the produced metadata.
        """
        trainer = self._trainer()
        return trainer.train(
            df,
            model_type=GameModelType(self.model_type),
            repo=repo,
        )

    # ------------------------------------------------------------------
    # Model protocol
    # ------------------------------------------------------------------

    def predict_historical(
        self,
        games: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for all historical games.

        Args:
            games: Canonical games DataFrame (unused - the modeling file
                is loaded internally). Kept for :class:`Model`
                protocol compatibility.
            repo: Repository root path.

        Returns:
            DataFrame in prediction archive schema. Empty if the model
            artifact has not been trained.
        """
        resolved_repo: Path = repo or get_settings().repo_root
        if self._task() == "classification":
            return self._predict_historical_classification(repo=resolved_repo)
        return self._predict_historical_regression(repo=resolved_repo)

    def predict_upcoming(
        self,
        schedule: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for upcoming (unplayed) games.

        Args:
            schedule: Canonical upcoming schedule DataFrame.
            repo: Repository root path.

        Returns:
            Enriched prediction DataFrame. Empty if the model artifact
            has not been trained or no rows have complete features.
        """
        resolved_repo: Path = repo or get_settings().repo_root
        if self._task() == "classification":
            return self._predict_upcoming_classification(schedule, repo=resolved_repo)
        return self._predict_upcoming_regression(schedule, repo=resolved_repo)

    def predict_upcoming_with_evidence(
        self,
        schedule: pd.DataFrame,
        *,
        source_revision: SourceRevision,
        source_artifacts: tuple[SourceArtifactReference, ...],
        repo: Path | None = None,
    ) -> StatisticalPredictionExecution:
        """Generate upcoming predictions and exact execution evidence."""
        resolved_repo = repo or get_settings().repo_root

        if self._task() == "classification":
            return self._predict_upcoming_classification_with_evidence(
                schedule,
                source_revision=source_revision,
                source_artifacts=source_artifacts,
                repo=resolved_repo,
            )

        return self._predict_upcoming_regression_with_evidence(
            schedule,
            source_revision=source_revision,
            source_artifacts=source_artifacts,
            repo=resolved_repo,
        )

    # ------------------------------------------------------------------
    # Classification (win_prob) prediction
    # ------------------------------------------------------------------

    def _predict_historical_classification(self, *, repo: Path) -> pd.DataFrame:
        """Historical prediction lifecycle for classification (win_prob).

        Loads the modeling file, applies the feature function, runs
        ``predict_proba``, optionally attaches totals, builds game
        predictions, and enriches.
        """
        store = ArtifactStore(repo)

        if not store.is_trained(self.model_name, self.model_type):
            logger.warning(
                "predict_historical: (%s, %s) not trained.",
                self.model_name,
                self.model_type,
            )
            return pd.DataFrame()

        df: DataFrame = load_modeling_file(repo, required_schema_version=_SCHEMA_VERSION)
        feature_fn = self._feature_fn()
        features = feature_fn(df)
        valid = features.notna().all(axis=1)
        df_valid = df.loc[valid].copy()
        x_feat = features.loc[valid]

        if x_feat.empty:
            return pd.DataFrame()

        pipeline = store.load(self.model_name, self.model_type)
        scaler = store.load_scaler(self.model_name, self.model_type)
        x_feat_arr = scaler.transform(x_feat) if scaler is not None else x_feat.values
        probs = pipeline.predict_proba(x_feat_arr)[:, 1]

        result: DataFrame = build_game_predictions(
            df_valid,
            probs,
        )

        return enrich_predictions(
            result,
            model_name=self.model_name,
            model_type=self.model_type,
            recalibrate=True,
            repo=repo,
        )

    def _predict_upcoming_classification(
        self, schedule: pd.DataFrame, *, repo: Path
    ) -> pd.DataFrame:
        """Upcoming prediction lifecycle for classification (win_prob).

        Builds features on the schedule, runs ``predict_proba``,
        attaches totals when available, and enriches.
        """
        store = ArtifactStore(repo)

        if not store.is_trained(self.model_name, self.model_type):
            logger.warning(
                "predict_upcoming: (%s, %s) not trained.",
                self.model_name,
                self.model_type,
            )
            return pd.DataFrame()

        datasets = DatasetAccessor(repo=repo)

        upcoming_df: DataFrame = run_features(
            df=schedule,
            feature_names=CANONICAL_FEATURES,
            datasets=datasets,
        )
        feature_fn = self._feature_fn()
        features = feature_fn(upcoming_df)
        valid = features.notna().all(axis=1)
        upcoming_valid = upcoming_df.loc[valid].copy()
        x_feat = features.loc[valid]

        if x_feat.empty:
            return pd.DataFrame()

        pipeline = store.load(self.model_name, self.model_type)
        scaler = store.load_scaler(self.model_name, self.model_type)
        x_feat_arr = scaler.transform(x_feat) if scaler is not None else x_feat.values
        probs = pipeline.predict_proba(x_feat_arr)[:, 1]
        result = upcoming_valid[["GAME_ID", "AWAY_TEAM", "HOME_TEAM", "WEEK_NUM"]].copy()
        result["HOME_WIN_PROB"] = probs
        result["AWAY_WIN_PROB"] = 1.0 - probs
        home_probabilities = pd.Series(
            probs,
            index=upcoming_valid.index,
            dtype=float,
        )
        away_probabilities = 1.0 - home_probabilities

        result["HOME_TEAM_WIN_PROB"] = (
            home_probabilities.mul(100).map(lambda value: f"{value:.1f} %").to_numpy()
        )
        result["AWAY_TEAM_WIN_PROB"] = (
            away_probabilities.mul(100).map(lambda value: f"{value:.1f} %").to_numpy()
        )
        result["AWAY_TEAM_ELO"] = upcoming_valid.get(
            "AWAY_ELO",
            float("nan"),
        )
        result["HOME_TEAM_ELO"] = upcoming_valid.get(
            "HOME_ELO",
            float("nan"),
        )

        result = enrich_predictions(
            result,
            model_name=self.model_name,
            model_type=self.model_type,
            recalibrate=True,
            repo=repo,
        )
        return result.reset_index(drop=True)

    def _predict_upcoming_classification_with_evidence(
        self,
        schedule: pd.DataFrame,
        *,
        source_revision: SourceRevision,
        source_artifacts: tuple[SourceArtifactReference, ...],
        repo: Path,
    ) -> StatisticalPredictionExecution:
        """Execute one Win estimator with exact input evidence."""
        store = ArtifactStore(repo)

        if not store.is_trained(
            self.model_name,
            self.model_type,
        ):
            raise ValueError(
                "Evidence-aware Win prediction requires "
                f"a trained ({self.model_name}, {self.model_type}) "
                "artifact."
            )

        metadata = store.read_metadata(
            self.model_name,
            self.model_type,
        )
        binary_artifacts = _statistical_artifact_references(
            store,
            model_name=self.model_name,
            model_type=self.model_type,
            repo=repo,
        )

        datasets = DatasetAccessor(repo=repo)
        upcoming_df: DataFrame = run_features(
            df=schedule,
            feature_names=CANONICAL_FEATURES,
            datasets=datasets,
        )
        upcoming_df = _rebuild_features_with_window(
            upcoming_df,
            window=_validated_epa_window(metadata.parameters),
            repo=repo,
        )

        feature_set = self.prediction_feature_set()
        features: DataFrame = feature_set.feature_fn(upcoming_df)

        validated_metadata = _validate_prediction_metadata(
            metadata,
            model_name=self.model_name,
            model_type=self.model_type,
            task="classification",
            feature_set=feature_set,
            features=features,
        )
        feature_schema = _prediction_feature_schema(
            validated_metadata,
            feature_set=feature_set,
        )

        valid = features.notna().all(axis=1)
        upcoming_valid = upcoming_df.loc[valid].copy()
        x_feat = features.loc[valid].copy()

        if x_feat.empty:
            raise ValueError(
                "Evidence-aware Win prediction has no rows with complete model features."
            )

        pipeline = store.load(
            self.model_name,
            self.model_type,
        )
        scaler = store.load_scaler(
            self.model_name,
            self.model_type,
        )

        if self.model_type == "logistic":
            if scaler is None:
                raise ValueError("Logistic Win prediction requires a persisted scaler.")
        elif scaler is not None:
            raise ValueError("Non-logistic Win prediction requires an absent scaler.")

        raw_matrix = x_feat.to_numpy(dtype=float)
        transformed_matrix = (
            np.asarray(
                scaler.transform(x_feat),
                dtype=float,
            )
            if scaler is not None
            else raw_matrix.copy()
        )

        raw_probabilities = np.asarray(
            pipeline.predict_proba(transformed_matrix)[:, 1],
            dtype=float,
        )

        if len(raw_probabilities) != len(upcoming_valid):
            raise ValueError("Win estimator output count does not match the valid upcoming rows.")

        resolution = resolve_prediction_post_processing(
            model_name=self.model_name,
            model_type=self.model_type,
            repo=repo,
        )
        post_processing = _post_processing_evidence(
            resolution,
            metadata=validated_metadata,
            source_artifacts=source_artifacts,
            binary_artifacts=binary_artifacts,
        )

        result = upcoming_valid[
            [
                "GAME_ID",
                "AWAY_TEAM",
                "HOME_TEAM",
                "WEEK_NUM",
            ]
        ].copy()
        result["HOME_WIN_PROB"] = raw_probabilities
        result["AWAY_WIN_PROB"] = 1.0 - raw_probabilities
        result["AWAY_TEAM_ELO"] = upcoming_valid.get(
            "AWAY_ELO",
            float("nan"),
        )
        result["HOME_TEAM_ELO"] = upcoming_valid.get(
            "HOME_ELO",
            float("nan"),
        )

        result = enrich_predictions_with_resolution(
            result,
            resolution=resolution,
        ).reset_index(drop=True)

        post_probabilities = result["HOME_WIN_PROB"].to_numpy(dtype=float)

        result["HOME_TEAM_WIN_PROB"] = (
            pd.Series(
                post_probabilities,
                index=result.index,
                dtype=float,
            )
            .mul(100)
            .map(lambda value: f"{value:.1f} %")
        )
        result["AWAY_TEAM_WIN_PROB"] = (
            pd.Series(
                1.0 - post_probabilities,
                index=result.index,
                dtype=float,
            )
            .mul(100)
            .map(lambda value: f"{value:.1f} %")
        )

        return build_statistical_prediction_execution(
            result,
            game_ids=tuple(str(value) for value in upcoming_valid["GAME_ID"].tolist()),
            raw_feature_values=tuple(tuple(float(value) for value in row) for row in raw_matrix),
            transformed_feature_values=tuple(
                tuple(float(value) for value in row) for row in transformed_matrix
            ),
            raw_estimator_outputs=tuple(float(value) for value in raw_probabilities),
            post_estimator_outputs=tuple(float(value) for value in post_probabilities),
            source_revision=source_revision,
            source_artifacts=source_artifacts,
            binary_artifacts=binary_artifacts,
            feature_schema=feature_schema,
            post_processing=post_processing,
        )

    # ------------------------------------------------------------------
    # Regression (total) prediction
    # ------------------------------------------------------------------

    def _predict_historical_regression(self, *, repo: Path) -> pd.DataFrame:
        """Historical prediction lifecycle for regression (total).

        Returns canonical total prediction rows for historical games whose
        required model features are complete.
        """
        store = ArtifactStore(repo)

        if not store.is_trained(self.model_name, self.model_type):
            logger.warning(
                "predict_historical: (%s, %s) not trained.",
                self.model_name,
                self.model_type,
            )
            return pd.DataFrame()

        df: DataFrame = load_modeling_file(repo, required_schema_version=_SCHEMA_VERSION)
        feature_fn = self._feature_fn()
        features = feature_fn(df)
        valid = features.notna().all(axis=1)
        df_valid = df.loc[valid].copy()
        x_feat = features.loc[valid]

        if x_feat.empty:
            return pd.DataFrame()

        model = store.load(self.model_name, self.model_type)
        scaler = store.load_scaler(self.model_name, self.model_type)
        x_feat_arr = scaler.transform(x_feat) if scaler is not None else x_feat.values
        preds: np.ndarray = model.predict(x_feat_arr)

        return build_regression_predictions(
            df_valid,
            preds,
        )

    def _predict_upcoming_regression(self, schedule: pd.DataFrame, *, repo: Path) -> pd.DataFrame:
        """Upcoming prediction lifecycle for regression (total)."""
        store = ArtifactStore(repo)

        if not store.is_trained(self.model_name, self.model_type):
            logger.warning(
                "predict_upcoming: (%s, %s) not trained.",
                self.model_name,
                self.model_type,
            )
            return pd.DataFrame()

        model = store.load(self.model_name, self.model_type)
        datasets = DatasetAccessor(repo=repo)

        upcoming_df: DataFrame = run_features(
            df=schedule,
            feature_names=CANONICAL_FEATURES,
            datasets=datasets,
        )
        feature_fn = self._feature_fn()
        features = feature_fn(upcoming_df)
        valid = features.notna().all(axis=1)
        upcoming_valid = upcoming_df.loc[valid].copy()
        x_feat = features.loc[valid]

        if x_feat.empty:
            return pd.DataFrame()

        scaler = store.load_scaler(self.model_name, self.model_type)
        x_feat_arr = scaler.transform(x_feat) if scaler is not None else x_feat.values
        preds: np.ndarray = model.predict(x_feat_arr)
        result = upcoming_valid[["GAME_ID", "AWAY_TEAM", "HOME_TEAM", "WEEK_NUM"]].copy()
        result["model_total"] = preds
        result["model_name"] = self.model_name
        result["model_type"] = self.model_type
        return result.reset_index(drop=True)

    def _predict_upcoming_regression_with_evidence(
        self,
        schedule: pd.DataFrame,
        *,
        source_revision: SourceRevision,
        source_artifacts: tuple[SourceArtifactReference, ...],
        repo: Path,
    ) -> StatisticalPredictionExecution:
        """Execute one Total estimator with exact input evidence."""
        store = ArtifactStore(repo)

        if not store.is_trained(
            self.model_name,
            self.model_type,
        ):
            raise ValueError(
                "Evidence-aware Total prediction requires "
                f"a trained ({self.model_name}, {self.model_type}) "
                "artifact."
            )

        metadata = store.read_metadata(
            self.model_name,
            self.model_type,
        )
        binary_artifacts = _statistical_artifact_references(
            store,
            model_name=self.model_name,
            model_type=self.model_type,
            repo=repo,
        )

        datasets = DatasetAccessor(repo=repo)
        upcoming_df: DataFrame = run_features(
            df=schedule,
            feature_names=CANONICAL_FEATURES,
            datasets=datasets,
        )
        upcoming_df = _rebuild_features_with_window(
            upcoming_df,
            window=_validated_epa_window(metadata.parameters),
            repo=repo,
        )

        feature_set = self.prediction_feature_set()
        features: DataFrame = feature_set.feature_fn(upcoming_df)

        validated_metadata = _validate_prediction_metadata(
            metadata,
            model_name=self.model_name,
            model_type=self.model_type,
            task="regression",
            feature_set=feature_set,
            features=features,
        )
        feature_schema = _prediction_feature_schema(
            validated_metadata,
            feature_set=feature_set,
        )

        valid = features.notna().all(axis=1)
        upcoming_valid = upcoming_df.loc[valid].copy()
        x_feat = features.loc[valid].copy()

        if x_feat.empty:
            raise ValueError(
                "Evidence-aware Total prediction has no rows with complete model features."
            )

        model = store.load(
            self.model_name,
            self.model_type,
        )
        scaler = store.load_scaler(
            self.model_name,
            self.model_type,
        )

        if scaler is not None:
            raise ValueError("Total prediction requires an absent scaler.")

        raw_matrix = x_feat.to_numpy(dtype=float)
        transformed_matrix = raw_matrix.copy()

        predictions = np.asarray(
            model.predict(transformed_matrix),
            dtype=float,
        )

        if len(predictions) != len(upcoming_valid):
            raise ValueError("Total estimator output count does not match the valid upcoming rows.")

        result = upcoming_valid[
            [
                "GAME_ID",
                "AWAY_TEAM",
                "HOME_TEAM",
                "WEEK_NUM",
            ]
        ].copy()
        result["model_total"] = predictions
        result["model_name"] = self.model_name
        result["model_type"] = self.model_type
        result = result.reset_index(drop=True)

        return build_statistical_prediction_execution(
            result,
            game_ids=tuple(str(value) for value in upcoming_valid["GAME_ID"].tolist()),
            raw_feature_values=tuple(tuple(float(value) for value in row) for row in raw_matrix),
            transformed_feature_values=tuple(
                tuple(float(value) for value in row) for row in transformed_matrix
            ),
            raw_estimator_outputs=tuple(float(value) for value in predictions),
            post_estimator_outputs=tuple(float(value) for value in predictions),
            source_revision=source_revision,
            source_artifacts=source_artifacts,
            binary_artifacts=binary_artifacts,
            feature_schema=feature_schema,
            post_processing=None,
        )


# ---------------------------------------------------------------------------
# Composite-key registrations
# ---------------------------------------------------------------------------


@ModelRegistry.register
class WinProbLogisticModel(GamesModel):
    """Win probability - logistic regression."""

    model_name = "win_prob"
    model_type = "logistic"
    spec = ModelSpec(
        name="win_prob_logistic",
        description=(
            "Win probability - logistic regression (combined features, TimeSeriesSplit CV)."
        ),
        trainable=True,
    )


@ModelRegistry.register
class WinProbRandomForestModel(GamesModel):
    """Win probability - Random Forest with isotonic calibration."""

    model_name = "win_prob"
    model_type = "random_forest"
    spec = ModelSpec(
        name="win_prob_random_forest",
        description=(
            "Win probability - Random Forest (expanded features, "
            "isotonic calibration, TimeSeriesSplit CV)."
        ),
        trainable=True,
    )


@ModelRegistry.register
class WinProbXGBoostModel(GamesModel):
    """Win probability - XGBoost with conditional isotonic calibration."""

    model_name = "win_prob"
    model_type = "xgboost"
    spec = ModelSpec(
        name="win_prob_xgboost",
        description=(
            "Win probability - XGBoost (expanded features, "
            "conditional isotonic calibration, TimeSeriesSplit CV)."
        ),
        trainable=True,
    )


@ModelRegistry.register
class TotalRandomForestModel(GamesModel):
    """Total points - Random Forest regression."""

    model_name = "total"
    model_type = "random_forest"
    spec = ModelSpec(
        name="total_random_forest",
        description=(
            "Total points - Random Forest regression (expanded features, randomized HP search)."
        ),
        trainable=True,
    )


@ModelRegistry.register
class TotalXGBoostModel(GamesModel):
    """Total points - XGBoost regression."""

    model_name = "total"
    model_type = "xgboost"
    spec = ModelSpec(
        name="total_xgboost",
        description=(
            "Total points - XGBoost regression (expanded features, randomized HP search)."
        ),
        trainable=True,
    )

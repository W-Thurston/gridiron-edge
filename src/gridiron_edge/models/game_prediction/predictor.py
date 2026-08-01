# src/gridiron_edge/models/game_prediction/predictor.py

"""Game prediction model registry entry point.

This module is the single import that callers use to ensure all game
prediction models are registered with ``ModelRegistry``. It contains:

- The :class:`GamesPredictor` base class (Predictor + Trainable protocols).
- Five composite-key subclasses registered with ``ModelRegistry``:
    * ``"win_prob_logistic"`` / ``"win_prob_random_forest"`` / ``"win_prob_xgboost"``
    * ``"total_random_forest"`` / ``"total_xgboost"``
- Pure helpers that assemble canonical game-level classification and
  regression prediction rows.

All game-side training and prediction flows through :class:`GamesTrainer`
and this module's :class:`GamesPredictor`. ``ModelRegistry`` keys use
the composite ``{model_name}_{model_type}`` convention (e.g.
``"win_prob_random_forest"``).
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.datasets.loaders import load_modeling_file
from gridiron_edge.features.pipeline import (
    CANONICAL_FEATURES,
    FEATURES,
)
from gridiron_edge.features.registry import run_features
from gridiron_edge.models.artifact import ArtifactStore
from gridiron_edge.models.base import ModelSpec
from gridiron_edge.models.game_prediction._columns import _SCHEMA_VERSION
from gridiron_edge.models.game_prediction.base import (
    GameModelMetadata,
    GameModelSpec,
    GameModelType,
    GamesTrainer,
)
from gridiron_edge.models.game_prediction.post_process import enrich_predictions
from gridiron_edge.models.game_prediction.total import TotalTrainer
from gridiron_edge.models.game_prediction.win_prob import WinProbTrainer
from gridiron_edge.models.registry import ModelRegistry

if TYPE_CHECKING:
    from pandas import DataFrame, Series

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer dispatch - maps model_name → GamesTrainer subclass
# ---------------------------------------------------------------------------


_TRAINER_FOR_NAME: dict[str, type[GamesTrainer]] = {
    "win_prob": WinProbTrainer,
    "total": TotalTrainer,
}


def get_known_model_names() -> tuple[str, ...]:
    """Return the model_names recognized by ``GamesPredictor``.

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
    *,
    totals: pd.Series | None = None,
) -> pd.DataFrame:
    """Map canonical Home-win probabilities to one row per game.

    Args:
        df: Canonical one-row-per-game modeling DataFrame.
        home_win_probs: Probability that the designated Home team wins,
            aligned one-to-one with ``df``.
        totals: Optional predicted game totals aligned with ``df``.

    Returns:
        Canonical game prediction rows with Home-win probability stored
        directly and Away-win probability derived as its complement.

    Raises:
        ValueError: If prediction lengths do not match the input rows or
            canonical game identities are duplicated.
    """
    if len(home_win_probs) != len(df):
        raise ValueError("Home-win probability count must match canonical game rows.")

    if df["GAME_ID"].duplicated().any():
        raise ValueError("Canonical prediction input contains duplicate game IDs.")

    if totals is not None and len(totals) != len(df):
        raise ValueError("Total prediction count must match canonical game rows.")

    work: DataFrame = df.copy()
    work["_HOME_WIN_PROB"] = home_win_probs

    if totals is not None:
        work["_MODEL_TOTAL"] = totals.reindex(work.index).to_numpy(dtype=float)

    work = work.sort_values(
        [
            "YEAR",
            "WEEK_NUM",
            "GAME_ID",
        ],
        kind="stable",
    )

    home_probabilities = work["_HOME_WIN_PROB"].to_numpy(dtype=float)

    result = pd.DataFrame(
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
    )

    if "_MODEL_TOTAL" in work.columns:
        result["model_total"] = work["_MODEL_TOTAL"].to_numpy(dtype=float)

    return result.reset_index(drop=True)


def build_regression_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
) -> pd.DataFrame:
    """Map regression outputs onto canonical game prediction rows.

    Standard and neutral-site games use the same deterministic team
    orientation as classification predictions.

    Args:
        df: Modeling DataFrame containing game, team, season, week, and
            home-field identity.
        preds: Predicted game totals aligned with ``df``.

    Returns:
        Canonical game-level total prediction rows with one row per game.
    """
    work: DataFrame = df.copy()
    work["_total"] = preds

    has_home: pd.Series = work.groupby("GAME_ID")["HOME_FIELD"].transform("max") == 1

    standard_rows: DataFrame | Series = work.loc[has_home & (work["HOME_FIELD"] == 0)]

    neutral_rows: DataFrame = (
        work.loc[~has_home]
        # pyrefly: ignore [no-matching-overload]
        .sort_values(
            ["GAME_ID", "TEAM_A"],
            kind="stable",
        )
        .drop_duplicates(
            subset=["GAME_ID"],
            keep="first",
        )
    )

    away: DataFrame = (
        pd.concat(
            [standard_rows, neutral_rows],
            ignore_index=False,
        )
        .drop_duplicates(
            subset=["GAME_ID"],
            keep="first",
        )
        .sort_values(
            ["YEAR", "WEEK_NUM", "GAME_ID"],
        )
    )

    return pd.DataFrame(
        {
            "season": away["YEAR"].values,
            "week": away["WEEK_NUM"].astype(int).values,
            "game_id": away["GAME_ID"].values,
            "game_date": away.get(
                "GAME_DATE",
                pd.Series(
                    [None] * len(away),
                    index=away.index,
                    dtype=object,
                ),
            ).values,
            "away_team": away["TEAM_A"].values,
            "home_team": away["TEAM_B"].values,
            "model_total": away["_total"].to_numpy(
                dtype=float,
            ),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# GamesPredictor base
# ---------------------------------------------------------------------------


class GamesPredictor:
    """Base class for game prediction model predictors.

    Each composite ``(model_name, model_type)`` pair has a thin subclass
    that sets ``model_name``, ``model_type``, and ``spec`` at class scope
    and is registered with :class:`ModelRegistry`. All logic lives
    here - subclasses are spec-only.

    The class implements both :class:`Predictor` (via ``predict_historical``
    / ``predict_upcoming``) and :class:`Trainable` (via ``train`` /
    ``is_trained``). Dispatch on classification vs regression happens
    internally based on the trainer's :attr:`GameModelSpec.task`.

    For win_prob predictions, totals are attached via an internal call to
    the predictor identified by :attr:`default_total_model_type`. If the
    total model is not yet trained, totals are silently omitted.
    """

    # Set by subclasses.
    model_name: ClassVar[str] = ""
    model_type: ClassVar[str] = ""
    spec: ClassVar[ModelSpec]

    #: Which total model to attach to win_prob predictions. Subclasses for
    #: ``win_prob`` predictors can override; total predictors ignore this.
    default_total_model_type: ClassVar[str] = "random_forest"

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
        """Return ``"classification"`` or ``"regression"`` for this predictor."""
        return self._game_model_spec().task

    def _feature_fn(self):  # noqa: ANN202 - return type is a Callable
        """Return the feature engineering function for this model_type."""
        gm_spec: GameModelSpec = self._game_model_spec()
        return gm_spec.feature_set[GameModelType(self.model_type)].feature_fn

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
    # Predictor protocol
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
                is loaded internally). Kept for :class:`Predictor`
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

        # Attach totals via the configured total model. Best-effort -
        # totals are silently omitted if the total model isn't trained.
        totals: Series | None = self._maybe_predict_totals(df_valid, repo=repo)

        result: DataFrame = build_game_predictions(
            df_valid,
            probs,
            totals=totals,
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

        # Attach total point estimates if available.
        totals: Series | None = self._maybe_predict_totals(
            upcoming_valid,
            repo=repo,
        )
        if totals is not None:
            result["model_total"] = totals.reindex(upcoming_valid.index).to_numpy(dtype=float)

        result = enrich_predictions(
            result,
            model_name=self.model_name,
            model_type=self.model_type,
            recalibrate=True,
            repo=repo,
        )
        return result.reset_index(drop=True)

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
            df=schedule, feature_names=FEATURES, datasets=datasets
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

    # ------------------------------------------------------------------
    # Internal helper: attach totals for win_prob predictions
    # ------------------------------------------------------------------

    def _maybe_predict_totals(self, df: pd.DataFrame, *, repo: Path) -> Series | None:
        """Return predicted totals for *df* using ``default_total_model_type``.

        Returns ``None`` when:
            - This predictor is itself a total predictor (no recursion).
            - The configured total model artifact is not trained.

        Best-effort - any other failure is logged at DEBUG and treated
        as ``None`` so callers can attach totals optionally.
        """
        if self.model_name == "total":
            return None

        store = ArtifactStore(repo)
        total_model_name: str = "total"
        total_model_type: str = self.default_total_model_type

        if not store.is_trained(total_model_name, total_model_type):
            logger.debug(
                "_maybe_predict_totals: (%s, %s) not trained - totals omitted.",
                total_model_name,
                total_model_type,
            )
            return None

        # Build the total model's features for this DataFrame.
        total_spec: GameModelSpec = TotalTrainer().spec
        total_feature_fn = total_spec.feature_set[GameModelType(total_model_type)].feature_fn

        features = total_feature_fn(df)
        valid = features.notna().all(axis=1)

        model = store.load(total_model_name, total_model_type)
        scaler = store.load_scaler(total_model_name, total_model_type)
        preds: Series[float] = pd.Series(np.nan, index=df.index, dtype=float)
        if valid.sum() > 0:
            features_valid = features.loc[valid]
            features_arr = (
                scaler.transform(features_valid) if scaler is not None else features_valid.values
            )
            preds.loc[valid] = model.predict(features_arr)
        return preds


# ---------------------------------------------------------------------------
# Composite-key registrations
# ---------------------------------------------------------------------------


@ModelRegistry.register
class WinProbLogisticPredictor(GamesPredictor):
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
class WinProbRandomForestPredictor(GamesPredictor):
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
class WinProbXGBoostPredictor(GamesPredictor):
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
class TotalRandomForestPredictor(GamesPredictor):
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
class TotalXGBoostPredictor(GamesPredictor):
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

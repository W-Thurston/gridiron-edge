# src/gridiron_edge/models/game_prediction/predictor.py

"""Game prediction model registry entry point.

This module is the single import that callers use to ensure all game
prediction models are registered with ``PredictorRegistry``. It contains:

- The :class:`GamesPredictor` base class (Predictor + Trainable protocols).
- Five composite-key subclasses registered with ``PredictorRegistry``:
    * ``"win_prob_logistic"`` / ``"win_prob_random_forest"`` / ``"win_prob_xgboost"``
    * ``"total_random_forest"`` / ``"total_xgboost"``
- The :func:`build_game_predictions` helper used internally by
  classification predict_historical to assemble game-level rows.

All game-side training and prediction flows through :class:`GamesTrainer`
and this module's :class:`GamesPredictor`. ``PredictorRegistry`` keys use
the composite ``{model_name}_{model_type}`` convention (e.g.
``"win_prob_random_forest"``).
"""

from __future__ import annotations

import datetime as dt
import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.datasets.loaders import load_modeling_file
from gridiron_edge.features.pipeline import FEATURES
from gridiron_edge.features.registry import run_features
from gridiron_edge.models.artifact import ArtifactStore
from gridiron_edge.models.base import PredictorSpec
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
from gridiron_edge.models.registry import PredictorRegistry

if TYPE_CHECKING:
    from pandas import DataFrame, Series

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer dispatch — maps model_name → GamesTrainer subclass
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
# build_game_predictions — internal helper for assembling archive rows
# ---------------------------------------------------------------------------


def build_game_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    *,
    model_name: str,
    model_type: str,
    is_backfilled: bool = True,
    totals: pd.Series | None = None,
) -> pd.DataFrame:
    """Map raw model outputs onto game-level prediction rows.

    The modeling DataFrame has one row per team-game (two rows per game).
    This function picks exactly one row per game and constructs the
    standard archive schema.

    For standard games (one team has HOME_FIELD=1, the other HOME_FIELD=0),
    the HOME_FIELD=0 row is kept and labeled as the away perspective.

    For neutral-site games (both rows have HOME_FIELD=0 — e.g. London,
    Mexico City), there is no canonical away/home distinction. We pick
    the row whose TEAM_A name is alphabetically first, then label that
    team as "away" purely for archive-schema compatibility. The actual
    win probability assignment remains correct because TEAM_A is the
    perspective for which ``probs`` was computed.

    Args:
        df: Modeling DataFrame (must include GAME_ID, TEAM_A, TEAM_B,
            YEAR, WEEK_NUM, HOME_FIELD). Aligned with *probs*.
        probs: Predicted probability that TEAM_A wins, aligned with *df*.
        model_name: Win-probability model purpose (e.g. ``"win_prob"``).
        model_type: Win-probability model algorithm (e.g. ``"random_forest"``).
        is_backfilled: Whether these are historical backfill predictions.
        totals: Optional predicted game totals, aligned with *df*.

    Returns:
        Game-level predictions DataFrame with exactly one row per game.
    """
    work = df.copy()
    work["_prob"] = probs
    if totals is not None:
        work["_total"] = totals

    # Identify games that have a "true home" (one HOME_FIELD=1 row).
    has_home: pd.Series = work.groupby("GAME_ID")["HOME_FIELD"].transform("max") == 1

    # Pick one row per game:
    # - Standard games (has_home == True): keep the HOME_FIELD=0 row (away).
    # - Neutral games (has_home == False): keep the row where TEAM_A is
    #   alphabetically smaller. Stable tiebreaker; ensures the same game
    #   gets the same away/home labeling across runs.
    standard_rows = work.loc[has_home & (work["HOME_FIELD"] == 0)]

    # For neutral games: sort by GAME_ID then TEAM_A, take the first row
    # per GAME_ID. This deterministic picking closes the neutral-site
    # arbitrary-labeling bug (predictor/C1 from audit_2026_06_18.md).
    neutral_rows = (
        work.loc[~has_home]
        # pyrefly: ignore [no-matching-overload]
        .sort_values(["GAME_ID", "TEAM_A"], kind="stable")
        .drop_duplicates(subset=["GAME_ID"], keep="first")
    )

    away = (
        pd.concat([standard_rows, neutral_rows], ignore_index=False)
        .drop_duplicates(subset=["GAME_ID"], keep="first")
        .sort_values(["YEAR", "WEEK_NUM", "GAME_ID"])
    )

    ts = dt.datetime.now(tz=dt.UTC).replace(tzinfo=None)

    result = pd.DataFrame(
        {
            "predicted_at": ts,
            "is_backfilled": is_backfilled,
            "model_name": model_name,
            "model_type": model_type,
            "season": away["YEAR"].values,
            "week": away["WEEK_NUM"].astype(int).values,
            "game_id": away["GAME_ID"].values,
            "game_date": "",
            "away_team": away["TEAM_A"].values,
            "home_team": away["TEAM_B"].values,
            "away_elo": float("nan"),
            "home_elo": float("nan"),
            "away_win_prob": away["_prob"].to_numpy(dtype=float),
            "home_win_prob": 1.0 - away["_prob"].to_numpy(dtype=float),
        }
    )

    if "_total" in away.columns:
        result["model_total"] = away["_total"].to_numpy(dtype=float)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# GamesPredictor base
# ---------------------------------------------------------------------------


class GamesPredictor:
    """Base class for game prediction model predictors.

    Each composite ``(model_name, model_type)`` pair has a thin subclass
    that sets ``model_name``, ``model_type``, and ``spec`` at class scope
    and is registered with :class:`PredictorRegistry`. All logic lives
    here — subclasses are spec-only.

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
    spec: ClassVar[PredictorSpec]

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

    def _feature_fn(self):  # noqa: ANN202 — return type is a Callable
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
            games: Canonical games DataFrame (unused — the modeling file
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

        # Attach totals via the configured total model. Best-effort —
        # totals are silently omitted if the total model isn't trained.
        totals: Series | None = self._maybe_predict_totals(df_valid, repo=repo)

        result = build_game_predictions(
            df_valid,
            probs,
            model_name=self.model_name,
            model_type=self.model_type,
            is_backfilled=True,
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
            df=schedule, feature_names=FEATURES, datasets=datasets
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
        result["AWAY_WIN_PROB"] = probs
        result["HOME_WIN_PROB"] = 1.0 - probs
        result["AWAY_TEAM_WIN_PROB"] = (pd.Series(probs) * 100).map(lambda x: f"{x:.1f} %").values
        result["HOME_TEAM_WIN_PROB"] = (
            ((1.0 - pd.Series(probs)) * 100).map(lambda x: f"{x:.1f} %").values
        )
        result["AWAY_TEAM_ELO"] = upcoming_valid.get("TEAM_A_ELO", float("nan"))
        result["HOME_TEAM_ELO"] = upcoming_valid.get("TEAM_B_ELO", float("nan"))

        # Attach total point estimates if available.
        totals: Series | None = self._maybe_predict_totals(upcoming_valid, repo=repo)
        if totals is not None:
            result["model_total"] = totals.loc[upcoming_valid.loc[valid].index].values

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

        Returns a DataFrame with point estimates for every modeling-file
        row whose features are complete. Schema parallels the
        classification archive: ``predicted_at``, ``is_backfilled``,
        ``model_name``, ``model_type``, ``season``, ``week``, ``game_id``,
        ``model_total``.
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

        # One row per game (away-team perspective, dedup on GAME_ID).
        df_valid["_total"] = preds
        away = df_valid.loc[df_valid["HOME_FIELD"] == 0].drop_duplicates(subset=["GAME_ID"])

        ts = dt.datetime.now(tz=dt.UTC).replace(tzinfo=None)
        result = pd.DataFrame(
            {
                "predicted_at": ts,
                "is_backfilled": True,
                "model_name": self.model_name,
                "model_type": self.model_type,
                "season": away["YEAR"],
                "week": away["WEEK_NUM"].astype(int),
                "game_id": away["GAME_ID"],
                "model_total": away["_total"],
            }
        )
        return result.reset_index(drop=True)

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

        Best-effort — any other failure is logged at DEBUG and treated
        as ``None`` so callers can attach totals optionally.
        """
        if self.model_name == "total":
            return None

        store = ArtifactStore(repo)
        total_model_name: str = "total"
        total_model_type: str = self.default_total_model_type

        if not store.is_trained(total_model_name, total_model_type):
            logger.debug(
                "_maybe_predict_totals: (%s, %s) not trained — totals omitted.",
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


@PredictorRegistry.register
class WinProbLogisticPredictor(GamesPredictor):
    """Win probability — logistic regression."""

    model_name = "win_prob"
    model_type = "logistic"
    spec = PredictorSpec(
        name="win_prob_logistic",
        description=(
            "Win probability — logistic regression (combined features, TimeSeriesSplit CV)."
        ),
        trainable=True,
    )


@PredictorRegistry.register
class WinProbRandomForestPredictor(GamesPredictor):
    """Win probability — Random Forest with isotonic calibration."""

    model_name = "win_prob"
    model_type = "random_forest"
    spec = PredictorSpec(
        name="win_prob_random_forest",
        description=(
            "Win probability — Random Forest (expanded features, "
            "isotonic calibration, TimeSeriesSplit CV)."
        ),
        trainable=True,
    )


@PredictorRegistry.register
class WinProbXGBoostPredictor(GamesPredictor):
    """Win probability — XGBoost with conditional isotonic calibration."""

    model_name = "win_prob"
    model_type = "xgboost"
    spec = PredictorSpec(
        name="win_prob_xgboost",
        description=(
            "Win probability — XGBoost (expanded features, "
            "conditional isotonic calibration, TimeSeriesSplit CV)."
        ),
        trainable=True,
    )


@PredictorRegistry.register
class TotalRandomForestPredictor(GamesPredictor):
    """Total points — Random Forest regression."""

    model_name = "total"
    model_type = "random_forest"
    spec = PredictorSpec(
        name="total_random_forest",
        description=(
            "Total points — Random Forest regression (expanded features, randomized HP search)."
        ),
        trainable=True,
    )


@PredictorRegistry.register
class TotalXGBoostPredictor(GamesPredictor):
    """Total points — XGBoost regression."""

    model_name = "total"
    model_type = "xgboost"
    spec = PredictorSpec(
        name="total_xgboost",
        description=(
            "Total points — XGBoost regression (expanded features, randomized HP search)."
        ),
        trainable=True,
    )

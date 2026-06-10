# src/gridiron_edge/models/prop_prediction/base.py

"""Base infrastructure for prop prediction models.

Prop models predict continuous player stats (passing yards, rushing yards,
etc.) rather than binary game outcomes. They share the player feature
pipeline (rolling stats + matchup features) but have their own training,
evaluation, and prediction interfaces.

Architecture:
    - ``PropModelSpec`` — metadata describing a prop model
    - ``PropModelResult`` — standardized prediction output
    - ``PropTrainer`` — base class for training prop models
    - Evaluation uses MAE/RMSE/R² instead of Brier/AUC/ECE

Adding a new prop model:
    1. Subclass ``PropTrainer``
    2. Implement ``_feature_columns()``, ``_build_features()``, ``_fit()``
    3. Register in the prop model registry
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
from sklearn.model_selection import TimeSeriesSplit

from gridiron_edge.core.settings import get_settings
from gridiron_edge.features.player.matchup import build_matchup_features
from gridiron_edge.features.player.rolling import build_player_rolling_features

logger: Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropModelSpec:
    """Metadata describing a prop model's identity.

    Attributes:
        name: Unique key (e.g. ``"qb_pass_yards"``).
        target_col: Column in player game logs that is the prediction target.
        position_filter: Position(s) this model applies to.
        description: Human-readable description.
    """

    name: str
    target_col: str
    position_filter: list[str]
    description: str = ""


@dataclass
class PropModelMetadata:
    """Metadata recorded alongside a trained prop model artifact.

    Attributes:
        model_name: Registered model name.
        trained_at: ISO-format UTC timestamp.
        target_col: Target column name.
        holdout_mae: MAE on holdout set (primary metric).
        holdout_rmse: RMSE on holdout set.
        holdout_r2: R² on holdout set.
        training_seasons: Seasons used for training.
        holdout_seasons: Seasons used for evaluation.
        parameters: Hyperparameters.
        feature_columns: Ordered feature columns the model expects.
        n_train_rows: Number of training rows.
        n_holdout_rows: Number of holdout rows.
        notes: Free-text notes.
    """

    model_name: str
    trained_at: str
    target_col: str
    holdout_mae: float
    holdout_rmse: float
    holdout_r2: float
    training_seasons: list[int] = field(default_factory=list)
    holdout_seasons: list[int] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    n_train_rows: int = 0
    n_holdout_rows: int = 0
    notes: str = ""


@dataclass
class PropPrediction:
    """A single prop prediction for a player-game.

    Attributes:
        player_id: nflverse player ID.
        player_name: Display name.
        game_id: nflverse game ID.
        season: Season year.
        week: Week number.
        predicted: Model's point estimate.
        actual: Actual value (None for upcoming games).
    """

    player_id: str
    player_name: str
    game_id: str
    season: int
    week: int
    predicted: float
    actual: float | None = None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_props(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute prop model evaluation metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dict with MAE, RMSE, R², and median absolute error.
    """
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2: float = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    median_ae = float(np.median(np.abs(residuals)))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": median_ae,
    }


# ---------------------------------------------------------------------------
# Minimum attempts filter — prevents modeling garbage-time / low-usage games
# ---------------------------------------------------------------------------

_MIN_ATTEMPTS: Final[dict[str, tuple[str, int]]] = {
    "passing_yards": ("attempts", 10),
    "rushing_yards": ("carries", 5),
    "receiving_yards": ("targets", 2),
}

# ---------------------------------------------------------------------------
# Universal feature columns — shared by all prop models.
# Built programmatically so they stay in sync with rolling + matchup modules.
# ElasticNet handles feature selection; no manual per-model curation needed.
# ---------------------------------------------------------------------------


def _build_universal_features() -> list[str]:
    """Build the universal feature column list from rolling + matchup + context."""
    from gridiron_edge.features.player.matchup import _MATCHUP_STATS
    from gridiron_edge.features.player.rolling import DEFAULT_WINDOWS, ROLLING_STAT_COLS

    cols: list[str] = []

    # Rolling features: {stat}_L{WIND_SPEED_MPHow}_{agg}
    for stat in ROLLING_STAT_COLS:
        for w in DEFAULT_WINDOWS:
            cols.append(f"{stat}_L{w}_mean")
            cols.append(f"{stat}_L{w}_std")

    # Matchup features: opp_{name}_allowed_L6 + rank
    for _, _, name in _MATCHUP_STATS:
        cols.append(f"opp_{name}_allowed_L6")
        cols.append(f"opp_{name}_allowed_rank_L6")

    # Game context features
    cols.extend(
        [
            "implied_team_total",
            "spread_line",
            "OVER_UNDER",
            "is_home",
            "roof_dome",
            "surface_turf",
            "TEMP_F",
            "WIND_SPEED_MPH",
            "rest_days",
            "opp_rest_days",
            "rest_diff",
            "DIV_GAME",
        ]
    )

    return cols


UNIVERSAL_FEATURE_COLS: Final[list[str]] = _build_universal_features()

# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------


class PropTrainer(ABC):
    """Abstract base class for prop model trainers.

    Subclasses implement:
        - ``spec`` — PropModelSpec describing the model
        - ``_feature_columns()`` — which feature columns to use
        - ``_build_features()`` — assemble the feature matrix
        - ``_fit()`` — train the underlying sklearn/xgb model

    The base class handles:
        - Data loading and filtering
        - Train/holdout splitting (TimeSeriesSplit)
        - Evaluation
        - Artifact persistence
    """

    @property
    @abstractmethod
    def spec(self) -> PropModelSpec:
        """Return the model specification."""
        ...

    def _feature_columns(self) -> list[str]:
        """Return the universal feature column list.

        All prop models use the same 132 features. ElasticNet handles
        feature selection — no manual per-model curation needed.
        Subclasses may override if they have a reason to diverge.
        """
        return UNIVERSAL_FEATURE_COLS

    @abstractmethod
    def _build_features(self, df: DataFrame) -> DataFrame:
        """Build the feature matrix from enriched player game logs.

        Args:
            df: Player game logs with rolling + matchup features.

        Returns:
            DataFrame with feature columns and target column.
        """
        ...

    @abstractmethod
    def _fit(
        self,
        x_train: DataFrame,
        y_train: pd.Series,
        x_val: DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Fit the model and return hyperparameters used.

        Args:
            x_train: Training features.
            y_train: Training target.
            x_val: Validation features.
            y_val: Validation target.

        Returns:
            Dict of hyperparameters for metadata recording.
        """
        ...

    @abstractmethod
    def _predict(self, x: DataFrame) -> np.ndarray:
        """Generate predictions from fitted model.

        Args:
            x: Feature matrix.

        Returns:
            Array of predicted values.
        """
        ...

    def _join_game_context(self, df: DataFrame, repo: Path) -> DataFrame:
        """Join cleaned games data for market/venue context."""
        games_path: Path = repo / "data" / "cleaned" / "NFL_wk_by_wk_cleaned.csv"
        if not games_path.exists():
            logger.warning("Cleaned games not found at %s — skipping game context", games_path)
            return df

        games: DataFrame = pd.read_csv(games_path)
        game_cols: list[str] = [
            "GAME_ID",
            "VEGAS_LINE",
            "OVER_UNDER",
            "ROOF",
            "SURFACE",
            "DIV_GAME",
        ]
        df = df.merge(
            games[game_cols],
            left_on="game_id",
            right_on="GAME_ID",
            how="left",
        ).drop(columns=["GAME_ID"])

        # Derive is_home from game_id format: {season}_{week}_{away}_{home}
        df["is_home"] = (df["game_id"].str.split("_").str[3] == df["team"]).astype(int)

        # Naive implied team total (refined below with spread from schedule)
        df["implied_team_total"] = np.where(
            df["OVER_UNDER"].notna(),
            df["OVER_UNDER"] / 2,
            np.nan,
        )

        # Roof → dome flag
        df["roof_dome"] = df["ROOF"].str.lower().isin(["dome", "closed"]).astype(int)

        # Surface → turf flag
        df["surface_turf"] = (~df["SURFACE"].str.lower().isin(["grass", "dessograss"])).astype(int)

        df = df.drop(columns=["ROOF", "SURFACE"], errors="ignore")
        logger.info("Joined game context: VEGAS_LINE, OVER_UNDER, roof, surface")
        return df

    def _join_schedule_context(self, df: DataFrame) -> DataFrame:
        """Join nflverse schedule for TEMP_F, WIND_SPEED_MPH, rest, and proper spread."""
        try:
            # pyrefly: ignore [missing-import]
            import nflreadpy as nflr

            seasons: list[str] = sorted(df["season"].unique().tolist())
            sched = nflr.load_schedules([int(s) for s in seasons]).to_pandas()

            sched_rows: list[dict] = []
            for _, g in sched.iterrows():
                common: dict = {
                    "game_id": g["game_id"],
                    "TEMP_F": g.get("TEMP_F"),
                    "WIND_SPEED_MPH": g.get("WIND_SPEED_MPH"),
                }
                spread = g.get("spread_line")
                # Home row
                sched_rows.append(
                    {
                        **common,
                        "team": g["home_team"],
                        "rest_days": g.get("home_rest"),
                        "opp_rest_days": g.get("away_rest"),
                        "spread_line": -spread if pd.notna(spread) else np.nan,
                    }
                )
                # Away row
                sched_rows.append(
                    {
                        **common,
                        "team": g["away_team"],
                        "rest_days": g.get("away_rest"),
                        "opp_rest_days": g.get("home_rest"),
                        "spread_line": spread if pd.notna(spread) else np.nan,
                    }
                )

            sched_ctx = pd.DataFrame(sched_rows)
            sched_ctx["team"] = sched_ctx["team"].replace(
                {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX"}
            )
            sched_ctx["rest_diff"] = sched_ctx["rest_days"] - sched_ctx["opp_rest_days"]

            df = df.merge(
                sched_ctx[
                    [
                        "game_id",
                        "team",
                        "spread_line",
                        "TEMP_F",
                        "WIND_SPEED_MPH",
                        "rest_days",
                        "opp_rest_days",
                        "rest_diff",
                    ]
                ],
                on=["game_id", "team"],
                how="left",
            )

            # Proper implied team total with spread
            if "OVER_UNDER" in df.columns:
                df["implied_team_total"] = np.where(
                    df["OVER_UNDER"].notna() & df["spread_line"].notna(),
                    (df["OVER_UNDER"] + df["spread_line"]) / 2,
                    df["implied_team_total"],
                )

            # Zero out TEMP_F/WIND_SPEED_MPH for domes
            if "roof_dome" in df.columns:
                df.loc[df["roof_dome"] == 1, "TEMP_F"] = np.nan
                df.loc[df["roof_dome"] == 1, "WIND_SPEED_MPH"] = np.nan

            logger.info("Joined schedule context: spread_line, TEMP_F, WIND_SPEED_MPH, rest")
        except Exception:  # nflreadpy may raise varied errors (network, parse, schema)
            logger.warning(
                "Failed to fetch nflverse schedules — skipping TEMP_F/WIND_SPEED_MPH/rest"
            )

        return df

    def _load_data(self, *, repo: Path | None = None) -> DataFrame:
        """Load player game logs with rolling + matchup + game context features.

        Chains the rolling and matchup feature builders, joins cleaned games
        and nflverse schedule for context, then filters to the relevant
        positions and applies minimum atTEMP_Ft thresholds.
        """
        resolved_repo: Path = repo or get_settings().repo_root

        # Build rolling features (includes skill position filter)
        df: DataFrame = build_player_rolling_features(repo=resolved_repo)

        # Build matchup features separately and join
        matchup_df: DataFrame = build_matchup_features(repo=resolved_repo)
        matchup_cols: list[str] = [c for c in matchup_df.columns if c.startswith("opp_")]
        join_keys: list[str] = ["player_id", "season", "week"]

        df = df.merge(
            # pyrefly: ignore [no-matching-overload]
            matchup_df[join_keys + matchup_cols].drop_duplicates(subset=join_keys),
            on=join_keys,
            how="left",
        )

        # Game context from cleaned games + nflverse schedule
        df = self._join_game_context(df, resolved_repo)
        df = self._join_schedule_context(df)

        # Filter to relevant positions
        df = df.loc[df["position"].isin(self.spec.position_filter), :].copy()

        # Apply minimum atTEMP_Ft threshold
        target: str = self.spec.target_col
        if target in _MIN_ATTEMPTS:
            volume_col, min_val = _MIN_ATTEMPTS[target]
            if volume_col in df.columns:
                before: int = len(df)
                df = df[df[volume_col] >= min_val].copy()
                logger.info(
                    "Filtered %s >= %d: %d → %d rows",
                    volume_col,
                    min_val,
                    before,
                    len(df),
                )

        # Drop rows where target is NaN
        df = df.dropna(subset=[target])

        logger.info(
            "Loaded %d rows for %s (%s)",
            len(df),
            self.spec.name,
            self.spec.target_col,
        )
        return df

    def train(self, *, repo: Path | None = None) -> PropModelMetadata:
        """Full training pipeline: load data, split, fit, evaluate.

        Uses TimeSeriesSplit for TEMP_Foral validation, consistent with
        the game prediction models.

        Returns:
            PropModelMetadata with evaluation metrics.
        """
        df: DataFrame = self._load_data(repo=repo)
        features_df: DataFrame = self._build_features(df)
        feature_cols: list[str] = self._feature_columns()

        # Ensure chronological order
        features_df = features_df.sort_values(["season", "week"]).reset_index(drop=True)

        # Drop rows with NaN in features or target
        target: str = self.spec.target_col
        required_cols: list[str] = [*feature_cols, target]
        features_df = features_df.dropna(subset=required_cols)

        x: DataFrame = features_df.loc[:, feature_cols]
        y: Series = features_df[target]

        logger.info(
            "Training %s: %d rows, %d features",
            self.spec.name,
            len(x),
            len(feature_cols),
        )

        # TimeSeriesSplit — same approach as game models
        tscv = TimeSeriesSplit(n_splits=5)
        splits: list = list(tscv.split(x))
        train_idx, val_idx = splits[-1]  # Use last split for final eval

        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit
        params: dict[str, Any] = self._fit(x_train, y_train, x_val, y_val)

        # Evaluate on holdout
        y_pred: ndarray = self._predict(x_val)
        metrics: dict[str, float] = evaluate_props(np.asarray(y_val), y_pred)

        logger.info(
            "%s holdout: MAE=%.1f, RMSE=%.1f, R²=%.3f (n=%d)",
            self.spec.name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            len(y_val),
        )

        # Determine season ranges
        train_seasons: list = sorted(features_df.iloc[train_idx]["season"].unique().tolist())
        holdout_seasons: list = sorted(features_df.iloc[val_idx]["season"].unique().tolist())

        return PropModelMetadata(
            model_name=self.spec.name,
            trained_at=datetime.now(UTC).isoformat(),
            target_col=self.spec.target_col,
            holdout_mae=metrics["mae"],
            holdout_rmse=metrics["rmse"],
            holdout_r2=metrics["r2"],
            training_seasons=train_seasons,
            holdout_seasons=holdout_seasons,
            parameters=params,
            feature_columns=feature_cols,
            n_train_rows=len(x_train),
            n_holdout_rows=len(x_val),
        )

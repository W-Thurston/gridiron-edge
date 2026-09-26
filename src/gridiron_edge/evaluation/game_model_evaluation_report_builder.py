# src/gridiron_edge/evaluation/game_model_evaluation_report_builder.py

"""Build and persist one cross-family game-model evaluation report.

Evaluates all six game-model families (Win: Elo, Logistic, Random Forest,
XGBoost; Total: Random Forest, XGBoost) against one exact, caller-supplied
backfill run per family, verifies every family covers the identical set of
games before computing any metric, and persists one immutable, run-bound
report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.datasets.loaders import load_modeling_file
from gridiron_edge.evaluation.game_model_evaluation_report import (
    TOTAL_MODEL_TYPES,
    WIN_MODEL_TYPES,
    GameModelEvaluationReport,
    TotalFamilyMetrics,
    WinFamilyMetrics,
    create_game_model_evaluation_report,
)
from gridiron_edge.evaluation.game_model_evaluation_report_store import (
    verify_game_model_evaluation_report,
    write_game_model_evaluation_report,
)
from gridiron_edge.evaluation.metrics import (
    accuracy,
    brier_score,
    build_forecast_run_evaluation_df,
    build_forecast_run_total_evaluation_df,
    calibration_slope_intercept,
    environment_slice_metrics,
    expected_calibration_error,
    interval_coverage,
    log_loss,
    median_absolute_error,
    roc_auc,
    season_stability,
    sharpness,
)

# Nominal coverage level used for the Total interval-coverage diagnostic.
_NOMINAL_COVERAGE: float = 0.90
_COLD_TEMP_F: float = 32.0
_WINDY_MPH: float = 15.0


@dataclass(frozen=True, slots=True)
class GameModelEvaluationReportBuildResult:
    """Accounting for one persisted and replay-verified report build."""

    report: GameModelEvaluationReport
    manifest_path: Path
    win_evidence_row_count: int
    total_evidence_row_count: int
    environment_slice_row_count: int


def build_and_write_game_model_evaluation_report(
    *,
    win_run_ids: dict[str, str],
    total_run_ids: dict[str, str],
    generated_at: datetime,
    repo: Path,
) -> GameModelEvaluationReportBuildResult:
    """Evaluate all six families on a verified common game set and persist the report.

    Args:
        win_run_ids: Exact backfill ``run_id`` for each Win model type; must
            supply exactly ``{"elo", "logistic", "random_forest", "xgboost"}``.
        total_run_ids: Exact backfill ``run_id`` for each Total model type;
            must supply exactly ``{"random_forest", "xgboost"}``.
        generated_at: Shared timezone-aware UTC generation timestamp.
        repo: Repository root.

    Raises:
        ValueError: If the six families do not evaluate the identical set of
            games (fails closed rather than building a report over
            inconsistent game sets).
    """
    _require_utc(generated_at)
    _require_exact_keys(win_run_ids, WIN_MODEL_TYPES, label="win_run_ids")
    _require_exact_keys(total_run_ids, TOTAL_MODEL_TYPES, label="total_run_ids")

    win_frames: dict[str, DataFrame] = {
        model_type: build_forecast_run_evaluation_df(
            run_id=win_run_ids[model_type],
            model_name="win_prob",
            model_type=model_type,
            repo=repo,
        )
        for model_type in WIN_MODEL_TYPES
    }
    total_frames: dict[str, DataFrame] = {
        model_type: build_forecast_run_total_evaluation_df(
            run_id=total_run_ids[model_type],
            model_type=model_type,
            repo=repo,
        )
        for model_type in TOTAL_MODEL_TYPES
    }

    game_id_sets: dict[str, frozenset[str]] = {
        f"win_prob/{model_type}": frozenset(str(game_id) for game_id in frame["game_id"])
        for model_type, frame in win_frames.items()
    } | {
        f"total/{model_type}": frozenset(str(game_id) for game_id in frame["game_id"])
        for model_type, frame in total_frames.items()
    }
    game_ids = _require_common_game_set(game_id_sets)

    win_metrics = tuple(
        _win_family_metrics(model_type, win_run_ids[model_type], win_frames[model_type])
        for model_type in WIN_MODEL_TYPES
    )
    total_metrics = tuple(
        _total_family_metrics(model_type, total_run_ids[model_type], total_frames[model_type])
        for model_type in TOTAL_MODEL_TYPES
    )

    win_evidence = pd.concat(
        [win_frames[model_type] for model_type in WIN_MODEL_TYPES],
        ignore_index=True,
    )
    total_evidence = pd.concat(
        [total_frames[model_type] for model_type in TOTAL_MODEL_TYPES],
        ignore_index=True,
    )
    environment_slices = _build_environment_slices(total_frames, repo=repo)

    token = _artifact_token(generated_at)
    report = create_game_model_evaluation_report(
        generated_at=generated_at,
        game_ids=game_ids,
        win_metrics=win_metrics,
        total_metrics=total_metrics,
        win_evidence=win_evidence,
        total_evidence=total_evidence,
        environment_slices=environment_slices,
        win_evidence_artifact=f"schema=1/win_evidence/{token}.parquet",
        total_evidence_artifact=f"schema=1/total_evidence/{token}.parquet",
        environment_slices_artifact=f"schema=1/environment_slices/{token}.parquet",
    )
    manifest_path = write_game_model_evaluation_report(
        report,
        win_evidence=win_evidence,
        total_evidence=total_evidence,
        environment_slices=environment_slices,
        repo=repo,
    )
    stored_win, stored_total, stored_slices = verify_game_model_evaluation_report(
        report,
        repo=repo,
    )
    _require_exact_replay(expected=win_evidence, stored=stored_win, label="win evidence")
    _require_exact_replay(expected=total_evidence, stored=stored_total, label="total evidence")
    _require_exact_replay(
        expected=environment_slices,
        stored=stored_slices,
        label="environment slices",
    )
    return GameModelEvaluationReportBuildResult(
        report=report,
        manifest_path=manifest_path,
        win_evidence_row_count=len(stored_win),
        total_evidence_row_count=len(stored_total),
        environment_slice_row_count=len(stored_slices),
    )


def _win_family_metrics(model_type: str, run_id: str, df: DataFrame) -> WinFamilyMetrics:
    p: Series = df["away_win_prob"]
    y: Series = df["away_team_won"]
    binary_mask: Series[bool] = y.isin([0.0, 1.0])
    slope, intercept = calibration_slope_intercept(p, y)
    auc = (
        roc_auc(p[binary_mask], y[binary_mask])
        if binary_mask.sum() >= 2 and y[binary_mask].nunique() >= 2
        else float("nan")
    )
    return WinFamilyMetrics(
        model_type=model_type,
        run_id=run_id,
        n_games=len(df),
        brier=round(brier_score(p, y), 6),
        log_loss=round(log_loss(p, y), 6),
        accuracy=round(accuracy(p, y), 6),
        auc=_none_if_nan(auc),
        ece=round(expected_calibration_error(p, y), 6),
        calibration_slope=_none_if_nan(slope),
        calibration_intercept=_none_if_nan(intercept),
        sharpness=round(sharpness(p), 6),
        season_stability=_none_if_nan(season_stability(df)),
    )


def _total_family_metrics(model_type: str, run_id: str, df: DataFrame) -> TotalFamilyMetrics:
    y_pred: Series = df["model_total"]
    y_true: Series = df["actual_total"]
    residuals: Series = y_pred - y_true
    residual_std = float(residuals.std())
    coverage = interval_coverage(
        y_true,
        y_pred,
        residual_std=residual_std,
        nominal=_NOMINAL_COVERAGE,
    )
    return TotalFamilyMetrics(
        model_type=model_type,
        run_id=run_id,
        n_games=len(df),
        mae=round(float(residuals.abs().mean()), 6),
        median_absolute_error=round(median_absolute_error(y_pred, y_true), 6),
        rmse=round(float(np.sqrt((residuals**2).mean())), 6),
        nominal_coverage=coverage["nominal_coverage"],
        actual_coverage=_none_if_nan(coverage["actual_coverage"]),
    )


def _build_environment_slices(
    total_frames: dict[str, DataFrame],
    *,
    repo: Path,
) -> DataFrame:
    """Build one combined environment-slice table across every Total family.

    Slices on dome/outdoor (``IS_DOME``), a cold-weather flag
    (``TEMP_F < 32``), and a windy flag (``WIND_SPEED_MPH >= 15``), all
    already-present columns in the canonical modeling file - no new feature
    construction.
    """
    environment = load_modeling_file(repo)
    environment_columns = environment.loc[
        :,
        ["GAME_ID", "IS_DOME", "TEMP_F", "WIND_SPEED_MPH"],
    ].copy()
    environment_columns["IS_DOME"] = environment_columns["IS_DOME"].astype(bool)
    environment_columns["IS_COLD"] = environment_columns["TEMP_F"] < _COLD_TEMP_F
    environment_columns["IS_WINDY"] = environment_columns["WIND_SPEED_MPH"] >= _WINDY_MPH

    dimensions: tuple[str, ...] = ("IS_DOME", "IS_COLD", "IS_WINDY")
    tables: list[DataFrame] = []
    for model_type, frame in total_frames.items():
        enriched = frame.merge(
            environment_columns,
            how="inner",
            left_on="game_id",
            right_on="GAME_ID",
            validate="one_to_one",
        )
        if len(enriched) != len(frame):
            raise ValueError(
                f"Environment columns did not match every game for Total/{model_type}."
            )
        for dimension in dimensions:
            table = environment_slice_metrics(enriched, dimension=dimension)
            table = table.rename(columns={dimension: "slice_label"})
            table.insert(0, "model_type", model_type)
            table.insert(1, "dimension", dimension)
            tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    combined["slice_label"] = combined["slice_label"].astype(str)
    return combined


def _require_common_game_set(game_id_sets: dict[str, frozenset[str]]) -> frozenset[str]:
    labels = list(game_id_sets)
    reference_label = labels[0]
    reference = game_id_sets[reference_label]
    mismatched = {
        label: sorted(game_ids.symmetric_difference(reference))
        for label, game_ids in game_id_sets.items()
        if game_ids != reference
    }
    if mismatched:
        raise ValueError(
            "Game-model families do not share a common game set; "
            f"reference={reference_label!r} ({len(reference)} games), "
            f"mismatched={mismatched!r}."
        )
    return reference


def _none_if_nan(value: float) -> float | None:
    return None if np.isnan(value) else value


def _artifact_token(generated_at: datetime) -> str:
    """Return one filename-safe invocation identity."""
    timestamp = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"game-model-evaluation-{timestamp}"


def _require_exact_keys(
    run_ids: dict[str, str],
    expected: tuple[str, ...],
    *,
    label: str,
) -> None:
    if set(run_ids) != set(expected):
        raise ValueError(f"{label} must supply exactly {set(expected)!r}, got {set(run_ids)!r}.")
    for model_type, run_id in run_ids.items():
        if not run_id.strip():
            raise ValueError(f"{label}[{model_type!r}] must not be empty.")


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("generated_at must be timezone-aware UTC.")


def _require_exact_replay(
    *,
    expected: DataFrame,
    stored: DataFrame,
    label: str,
) -> None:
    if not stored.equals(expected):
        raise ValueError(f"Stored {label} does not exactly replay input.")

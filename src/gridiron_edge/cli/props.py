# src/gridiron_edge/cli/props.py
"""CLI commands for player prop projections.

Provides four sub-commands:
    gridiron props projections   Prop projections table for a given week
    gridiron props evaluate      Holdout evaluation report for a prop model
    gridiron props backfill      Backfill historical predictions to archive
    gridiron props champion      Train all model types, compare, select champion
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

from numpy import ndarray
from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.core.console import console, step
from gridiron_edge.evaluation.prop_metrics import PropEvalReport
from gridiron_edge.models.prop_prediction.base import PropModelMetadata, PropModelType, PropTrainer

logger: Logger = logging.getLogger(__name__)

props_app = typer.Typer(help="Player prop projections.", no_args_is_help=True)

# ---------------------------------------------------------------------------
# Trainer registry — maps model name to trainer class
# ---------------------------------------------------------------------------


def _ensure_prop_models_registered() -> None:
    """Import prop model modules so ModelRegistry is populated."""
    import gridiron_edge.models.prop_prediction.qb_pass_yards
    import gridiron_edge.models.prop_prediction.qb_rush_yards
    import gridiron_edge.models.prop_prediction.rb_rush_yards
    import gridiron_edge.models.prop_prediction.te_rec_yards
    import gridiron_edge.models.prop_prediction.wr_rec_yards  # noqa: F401


def _all_prop_models() -> list[str]:
    """Return registered prop model family names."""
    from gridiron_edge.models.registry import ModelRegistry

    _ensure_prop_models_registered()

    names: list[str] = []
    for key, model_cls in ModelRegistry.all().items():
        instance = model_cls()
        if isinstance(instance, PropTrainer):
            names.append(key)

    return sorted(names)


def _get_trainer(model_name: str) -> PropTrainer:
    """Instantiate a registered prop trainer by model family name."""
    from gridiron_edge.models.registry import ModelRegistry

    _ensure_prop_models_registered()

    try:
        model_cls = ModelRegistry.get(model_name)
    except KeyError as exc:
        available = _all_prop_models()
        typer.echo(f"Unknown model: {model_name}. Available: {available}")
        raise typer.Exit(code=1) from exc

    trainer = model_cls()
    if not isinstance(trainer, PropTrainer):
        available = _all_prop_models()
        typer.echo(f"Registered model is not a prop trainer: {model_name}. Available: {available}")
        raise typer.Exit(code=1)

    return trainer


def _prepare_holdout_data(
    model_name: str,
) -> tuple[PropTrainer, DataFrame, list[str]]:
    """Build features once and prepare holdout-filtered, NaN-cleaned data.

    Shared by champion_cmd across model types so feature engineering
    runs once per stat family rather than once per (stat, model_type)
    combination.

    Args:
        model_name: Which stat family (e.g. "qb_pass_yards").

    Returns:
        Tuple of (trainer, holdout_df, usable_feature_columns).
    """
    from gridiron_edge.core.constants import HOLDOUT_SEASONS
    from gridiron_edge.features.player.builder import build_prop_features

    trainer: PropTrainer = _get_trainer(model_name)

    holdout_ints: set[int] = {int(s.split("-")[0]) for s in HOLDOUT_SEASONS}
    df: DataFrame = build_prop_features(position_filter=trainer.spec.position_filter)

    # Filter to holdout seasons
    df = df.loc[df["season"].isin(holdout_ints), :].copy()

    # Get feature columns and filter to usable ones
    feature_cols: list[str] = trainer._feature_columns()
    available: list[str] = [c for c in feature_cols if c in df.columns]
    nan_rates: Series = df[available].isna().mean()
    usable: list[str] = [c for c in available if nan_rates[c] < 0.5]

    # Drop NaN
    target: str = trainer.spec.target_col
    df = df.dropna(subset=[*usable, target])

    if len(df) == 0:
        typer.echo(f"No holdout data available for {model_name}")
        raise typer.Exit(code=1)

    return trainer, df, usable


def _enrich_predictions_for_holdout(
    trainer: PropTrainer,
    holdout_df: DataFrame,
    usable_features: list[str],
    model_rmse: float,
) -> DataFrame:
    """Predict on prepared holdout data and enrich with post-process columns.

    Args:
        trainer: Already-trained PropTrainer instance.
        holdout_df: Holdout DataFrame from _prepare_holdout_data().
        usable_features: Feature columns to use for prediction.
        model_rmse: Trained model's holdout RMSE for std computation.

    Returns:
        Enriched predictions DataFrame.
    """
    import numpy as np

    from gridiron_edge.models.prop_prediction.post_process import (
        TARGET_STD_MAP,
        enrich_prop_predictions,
    )

    # Generate predictions
    preds: ndarray = trainer._predict(holdout_df.loc[:, usable_features])
    df = holdout_df.copy()
    df["predicted_mean"] = preds
    df["stat_type"] = trainer.spec.name

    # Enrich
    target: str = trainer.spec.target_col
    std_col: str = TARGET_STD_MAP.get(trainer.spec.name, f"{target}_L3_std")
    if std_col not in df.columns:
        df[std_col] = np.nan

    enriched: DataFrame = enrich_prop_predictions(
        df=df,
        model_rmse=model_rmse,
        target_std_col=std_col,
    )

    return enriched


def _train_and_enrich(
    model_name: str,
    model_type: PropModelType = PropModelType.ELASTICNET,
) -> tuple[DataFrame, float]:
    """Train a model, generate holdout predictions, and enrich them.

    Convenience wrapper combining _prepare_holdout_data and training.
    Used by evaluate_cmd, backfill_cmd, and projections_cmd which only
    need to handle one model type. champion_cmd uses the split functions
    directly to share data prep across model types.

    Args:
        model_name: Which stat family (e.g. "qb_pass_yards").
        model_type: Algorithm to use.

    Returns:
        Tuple of (enriched predictions DataFrame, model RMSE).
    """
    trainer, holdout_df, usable = _prepare_holdout_data(model_name)
    meta: PropModelMetadata = trainer.train(model_type=model_type)
    enriched = _enrich_predictions_for_holdout(trainer, holdout_df, usable, meta.holdout_rmse)
    return enriched, meta.holdout_rmse


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@props_app.command("evaluate")
def evaluate_cmd(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Prop model family to evaluate, e.g. qb_pass_yards.",
    ),
    model_type: str = typer.Option(
        "elasticnet",
        "--model-type",
        "-t",
        help="Algorithm type: elasticnet, random_forest, xgboost",
    ),
) -> None:
    """Run holdout evaluation report for a prop model."""
    from gridiron_edge.evaluation.prop_metrics import evaluate_prop_model

    mt = PropModelType(model_type)
    console.header("props evaluate", subtitle=f"{model} · {mt}")

    with step(f"Train {model} ({mt})") as s:
        enriched, _rmse = _train_and_enrich(model, model_type=mt)
        s.set_rows(len(enriched))
        s.set_detail(f"RMSE={_rmse:.1f}")

    with step("Evaluate holdout") as s:
        target = _get_trainer(model).spec.target_col
        report: PropEvalReport = evaluate_prop_model(
            model_name=model,
            actual=enriched[target],
            predicted_mean=enriched["predicted_mean"],
            predicted_std=enriched.get("predicted_std"),
            lo_90=enriched.get("lo_90"),
            hi_90=enriched.get("hi_90"),
        )
        s.set_detail(f"MAE={report.accuracy.mae:.1f}  R²={report.accuracy.r2:.3f}")

    typer.echo(f"\n  MAE:       {report.accuracy.mae:.1f}")
    typer.echo(f"  RMSE:      {report.accuracy.rmse:.1f}")
    typer.echo(f"  R²:        {report.accuracy.r2:.3f}")
    typer.echo(f"  Median AE: {report.accuracy.median_ae:.1f}")
    typer.echo(f"  N:         {report.accuracy.n:,}")
    typer.echo(f"\n  Bias:      {report.bias.mean_error:+.1f}")
    typer.echo(f"  % Over:    {report.bias.pct_over_predicted:.1%}")
    if report.coverage is not None:
        typer.echo(
            f"\n  Coverage:  {report.coverage.actual_coverage:.1%}"
            f" (nominal {report.coverage.nominal_coverage:.0%})"
        )
        typer.echo(f"  Interval:  {report.coverage.mean_interval_width:.1f} avg width")

    console.summary()


@props_app.command("champion")
def champion_cmd(
    model: str = typer.Option(
        "all",
        "--model",
        "-m",
        help="Model to run champion selection on, e.g. qb_pass_yards. 'all' runs all.",
    ),
) -> None:
    """Train all model types for a stat family and select champion.

    Trains ElasticNet, RandomForest, and XGBoost for each specified model,
    runs guardrail checks, and selects the champion with lowest MAE.

    Data preparation (feature engineering, holdout filtering, NaN handling)
    runs ONCE per stat family and is shared across all three model types,
    rather than being repeated for each (stat, model_type) combination.
    """
    from gridiron_edge.evaluation.champion import (
        RegressionComparisonResult,
        RegressionModelResult,
        compare_regression_models,
        format_regression_comparison,
        select_prop_champion,
    )
    from gridiron_edge.evaluation.prop_metrics import evaluate_prop_model

    models: list[str] = _all_prop_models() if model == "all" else [model]
    model_types: list[PropModelType] = list(PropModelType)

    for m in models:
        console.header("props champion", subtitle=m)

        # Prepare data ONCE per stat family — shared across model types.
        with step(f"Prepare data for {m}") as s:
            trainer, holdout_df, usable = _prepare_holdout_data(m)
            s.set_rows(len(holdout_df))
            s.set_detail(f"{len(usable)} usable features")

        results: list[RegressionModelResult] = []

        for mt in model_types:
            with step(f"Train {m} ({mt})") as s:
                try:
                    meta: PropModelMetadata = trainer.train(model_type=mt)
                    enriched = _enrich_predictions_for_holdout(
                        trainer, holdout_df, usable, meta.holdout_rmse
                    )
                    target: str = trainer.spec.target_col

                    report: PropEvalReport = evaluate_prop_model(
                        model_name=m,
                        actual=enriched[target],
                        predicted_mean=enriched["predicted_mean"],
                        predicted_std=enriched.get("predicted_std"),
                        lo_90=enriched.get("lo_90"),
                        hi_90=enriched.get("hi_90"),
                    )

                    coverage: float = float("nan")
                    if report.coverage is not None:
                        coverage = report.coverage.actual_coverage

                    result = RegressionModelResult(
                        model_type=str(mt),
                        mae=report.accuracy.mae,
                        rmse=report.accuracy.rmse,
                        r2=report.accuracy.r2,
                        coverage=coverage,
                    )
                    results.append(result)

                    s.set_detail(
                        f"MAE={result.mae:.1f}  RMSE={result.rmse:.1f}  R²={result.r2:.3f}"
                    )
                    s.set_rows(len(enriched))

                except Exception as e:
                    typer.echo(f"    ⚠️  {mt} failed: {e}")
                    raise

        if not results:
            typer.echo(f"\n  ❌ No models trained successfully for {m}.")
            console.summary()
            continue

        # Select champion
        with step("Select champion") as s:
            champion, summary = select_prop_champion(results)
            s.set_detail(f"🏆 {champion.model_type} (MAE={champion.mae:.2f})")

        typer.echo(summary)

        # Show pairwise comparisons against champion
        challengers: list[RegressionModelResult] = [
            r for r in results if r.model_type != champion.model_type
        ]
        for challenger in challengers:
            comparison: RegressionComparisonResult = compare_regression_models(champion, challenger)
            typer.echo(format_regression_comparison(comparison))

        console.summary()

    if len(models) > 1:
        typer.echo("  Champion selection complete across all stat families.\n")


@props_app.command("backfill")
def backfill_cmd(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Prop model family to backfill, e.g. qb_pass_yards.",
    ),
    model_type: str = typer.Option(
        "elasticnet",
        "--model-type",
        "-t",
        help="Algorithm type: elasticnet, random_forest, xgboost",
    ),
) -> None:
    """Backfill historical predictions to the archive."""
    from gridiron_edge.evaluation.prop_archive import archive_prop_predictions

    mt = PropModelType(model_type)
    console.header(
        "props backfill",
        subtitle=f"{model} · {mt}",
    )

    with step(f"Train {model} ({mt})") as s:
        enriched, rmse = _train_and_enrich(model, model_type=mt)
        s.set_rows(len(enriched))
        s.set_detail(f"RMSE={rmse:.1f}")

    with step("Archive predictions") as s:
        path: Path = archive_prop_predictions(
            enriched,
            is_backfilled=True,
            model_name=model,
            model_type=mt.value,
        )
        s.set_rows(len(enriched))

    typer.echo(f"  Archived {len(enriched):,} predictions → {path}")


@props_app.command("projections")
def projections_cmd(
    model: str = typer.Option(
        "all",
        "--model",
        "-m",
        help="Prop model family to project, e.g. qb_pass_yards. 'all' runs all models",
    ),
    top: int = typer.Option(
        20,
        "--top",
        "-n",
        help="Number of top projections to display.",
    ),
) -> None:
    """Display prop projections table."""
    subtitle: str = "all models" if model == "all" else model

    console.header(
        "props projections",
        subtitle=subtitle,
    )

    models: list[str] = _all_prop_models() if model == "all" else [model]

    all_enriched: list[DataFrame] = []

    for m in models:
        with step(f"Train {m}") as s:
            try:
                enriched, rmse = _train_and_enrich(m)
                s.set_rows(len(enriched))
                s.set_detail(f"RMSE={rmse:.1f}")
                all_enriched.append(enriched)
            except Exception as e:
                typer.echo(f"  ⚠️  {m} failed: {e}")
                continue

    if not all_enriched:
        typer.echo("No models produced results.")
        raise typer.Exit(code=1)

    import pandas as pd

    combined: DataFrame = pd.concat(all_enriched, ignore_index=True)

    # Sort by predicted_mean descending
    combined = combined.sort_values("predicted_mean", ascending=False)

    # Format output table
    display_cols: list[str] = [
        "player_name",
        "position",
        "stat_type",
        "predicted_mean",
        "lo_90",
        "hi_90",
        "predicted_std",
    ]
    available_display: list[str] = [c for c in display_cols if c in combined.columns]
    display = combined[available_display].head(top).copy()

    # Round numeric columns
    for col in ["predicted_mean", "lo_90", "hi_90", "predicted_std"]:
        if col in display.columns:
            display[col] = display[col].round(1)

    # Rename for display
    rename_map: dict[str, str] = {
        "player_name": "Player",
        "position": "Pos",
        "stat_type": "Stat",
        "predicted_mean": "Proj",
        "lo_90": "Lo90",
        "hi_90": "Hi90",
        "predicted_std": "Std",
    }
    # pyrefly: ignore [no-matching-overload]
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})

    typer.echo()
    typer.echo(display.to_string(index=False))
    console.summary()

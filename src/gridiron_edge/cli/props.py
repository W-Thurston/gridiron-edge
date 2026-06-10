# src/gridiron_edge/cli/props.py
"""CLI commands for player prop projections.

Provides three sub-commands:

    gridiron props projections   Prop projections table for a given week
    gridiron props evaluate      Holdout evaluation report for a prop model
    gridiron props backfill      Backfill historical predictions to archive
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.evaluation.prop_metrics import PropEvalReport
from gridiron_edge.models.prop_prediction.base import PropTrainer

logger: Logger = logging.getLogger(__name__)

props_app = typer.Typer(help="Player prop projections.", no_args_is_help=True)

# ---------------------------------------------------------------------------
# Trainer registry — maps model name to trainer class
# ---------------------------------------------------------------------------

_TRAINER_MAP: Final[dict[str, str]] = {
    "qb_pass_yards": "gridiron_edge.models.prop_prediction.qb_pass_yards.QBPassYardsTrainer",
    "qb_rush_yards": "gridiron_edge.models.prop_prediction.qb_rush_yards.QBRushYardsTrainer",
    "rb_rush_yards": "gridiron_edge.models.prop_prediction.rb_rush_yards.RBRushYardsTrainer",
    "wr_rec_yards": "gridiron_edge.models.prop_prediction.wr_rec_yards.WRRecYardsTrainer",
    "te_rec_yards": "gridiron_edge.models.prop_prediction.te_rec_yards.TERecYardsTrainer",
}

_ALL_MODELS: Final[list[str]] = list(_TRAINER_MAP.keys())


def _get_trainer(model_name: str) -> PropTrainer:
    """Lazy-import and instantiate a trainer by model name."""
    if model_name not in _TRAINER_MAP:
        typer.echo(f"Unknown model: {model_name}. Available: {_ALL_MODELS}")
        raise typer.Exit(code=1)

    import importlib

    module_path, class_name = _TRAINER_MAP[model_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


def _train_and_enrich(model_name: str) -> tuple[DataFrame, float]:
    """Train a model, generate holdout predictions, and enrich them.

    Returns:
        Tuple of (enriched predictions DataFrame, model RMSE).
    """
    import numpy as np

    from gridiron_edge.features.player.builder import build_prop_features
    from gridiron_edge.models.prop_prediction.post_process import (
        TARGET_STD_MAP,
        enrich_prop_predictions,
    )

    trainer = _get_trainer(model_name)
    meta = trainer.train()

    # Build features for holdout predictions
    from gridiron_edge.core.constants import HOLDOUT_SEASONS

    holdout_ints: set[int] = {int(s.split("-")[0]) for s in HOLDOUT_SEASONS}
    df: DataFrame = build_prop_features(position_filter=trainer.spec.position_filter)

    # Filter to holdout seasons
    df = df.loc[df["season"].isin(holdout_ints), :].copy()

    # Get feature columns and filter to usable ones
    feature_cols = trainer._feature_columns()
    available: list[str] = [c for c in feature_cols if c in df.columns]
    nan_rates: Series = df[available].isna().mean()
    usable: list[str] = [c for c in available if nan_rates[c] < 0.5]

    # Drop NaN
    target = trainer.spec.target_col
    df = df.dropna(subset=[*usable, target])

    if len(df) == 0:
        typer.echo(f"No holdout data available for {model_name}")
        raise typer.Exit(code=1)

    # Generate predictions
    preds = trainer._predict(df[usable])
    df["predicted_mean"] = preds
    df["stat_type"] = model_name

    # Enrich
    std_col: str = TARGET_STD_MAP.get(model_name, f"{target}_L3_std")
    if std_col not in df.columns:
        df[std_col] = np.nan

    enriched: DataFrame = enrich_prop_predictions(
        df=df,
        model_rmse=meta.holdout_rmse,
        target_std_col=std_col,
    )

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
        help=f"Model to evaluate. Options: {_ALL_MODELS}",
    ),
) -> None:
    """Run holdout evaluation report for a prop model."""
    from gridiron_edge.evaluation.prop_metrics import evaluate_prop_model

    typer.echo(f"\n🏈 Evaluating {model}...\n")

    enriched, _rmse = _train_and_enrich(model)
    target = _get_trainer(model).spec.target_col

    report: PropEvalReport = evaluate_prop_model(
        model_name=model,
        actual=enriched[target],
        predicted_mean=enriched["predicted_mean"],
        predicted_std=enriched.get("predicted_std"),
        lo_90=enriched.get("lo_90"),
        hi_90=enriched.get("hi_90"),
    )

    # Print accuracy summary to console
    typer.echo(f"  MAE:       {report.accuracy.mae:.1f}")
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


@props_app.command("backfill")
def backfill_cmd(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help=f"Model to backfill. Options: {_ALL_MODELS}",
    ),
) -> None:
    """Backfill historical predictions to the archive."""
    from gridiron_edge.evaluation.prop_archive import archive_prop_predictions

    typer.echo(f"\n🏈 Backfilling {model}...\n")

    enriched, _rmse = _train_and_enrich(model)

    path: Path = archive_prop_predictions(
        enriched,
        is_backfilled=True,
        model_version="v1",
    )

    typer.echo(f"  Archived {len(enriched):,} predictions → {path}")


@props_app.command("projections")
def projections_cmd(
    model: str = typer.Option(
        "all",
        "--model",
        "-m",
        help=f"Model to project. 'all' runs all models. Options: {_ALL_MODELS}",
    ),
    top: int = typer.Option(
        20,
        "--top",
        "-n",
        help="Number of top projections to display.",
    ),
) -> None:
    """Display prop projections table."""
    models: list[str] = _ALL_MODELS if model == "all" else [model]

    all_enriched: list[DataFrame] = []
    for m in models:
        typer.echo(f"  Training {m}...")
        try:
            enriched, _ = _train_and_enrich(m)
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

    typer.echo(f"\n🏈 Prop Projections (top {top})\n")
    typer.echo(display.to_string(index=False))
    typer.echo(f"\n  Total: {len(combined):,} projections across {len(models)} models")

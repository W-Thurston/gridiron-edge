# src/gridiron_edge/cli/models.py
"""CLI commands for model training and champion/challenger management."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.evaluation.champion import ComparisonResult

if TYPE_CHECKING:
    from gridiron_edge.models.artifact import ModelMetadata
    from gridiron_edge.models.base import Predictor

models_app = typer.Typer(
    help="Train and manage prediction model artifacts.",
    no_args_is_help=True,
)


def _apply_promotion_decision(
    *,
    champion_meta: ModelMetadata | None,
    challenger_meta: ModelMetadata,
    champion_dir: Path,
    backup_dir: Path,
    force: bool,
    no_promote: bool,
) -> None:
    """Compare challenger to champion and handle promotion/rejection."""
    import shutil

    from gridiron_edge.evaluation.champion import compare_models, format_comparison

    if champion_meta is None:
        typer.echo("\nNo existing champion. Saved as champion.")
        typer.echo(f"  Brier: {challenger_meta.holdout_brier:.5f}")
        typer.echo(f"  Artifact: {champion_dir}")
        return

    result: ComparisonResult = compare_models(champion_meta, challenger_meta)
    typer.echo(format_comparison(result))

    promote: bool = (result.should_promote or force) and not no_promote

    if promote:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if force and not result.should_promote:
            typer.echo("  ⚠️  Force-promoted despite failed gates.")
        else:
            typer.echo("  ✅ New champion promoted.")
    else:
        if champion_dir.exists():
            shutil.rmtree(champion_dir)
        shutil.move(str(backup_dir), str(champion_dir))
        if no_promote:
            typer.echo("  Champion unchanged (--no-promote).")
        else:
            typer.echo("  ❌ Challenger rejected. Champion unchanged.")
            typer.echo("  Use --force to promote anyway.")


@models_app.command("train")
def models_train(
    model_name: str = typer.Argument(
        ...,
        help="Model to train (e.g. 'random_forest'). Must be registered and Trainable.",
    ),
    *,
    force: bool = typer.Option(
        False,
        "--force",
        help="Promote challenger even if promotion gates fail.",
    ),
    no_promote: bool = typer.Option(
        False,
        "--no-promote",
        help="Train and compare against champion but do not replace it.",
    ),
) -> None:
    r"""Train a model and compare against the current champion.

    If no champion exists, the new model is saved as champion.
    If a champion exists, the new model is compared using promotion
    gates (Brier improvement, ECE tolerance, AUC tolerance).
    The champion is replaced only if all gates pass (or --force is used).

    \b
    Examples:
      gridiron models train random_forest
      gridiron models train xgboost --force
      gridiron models train logistic --no-promote
    """
    import shutil

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets import loaders
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.base import Trainable
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import PredictorRegistry

    repo: Path = get_settings().repo_root
    store = ArtifactStore(repo)
    console.header("models train", subtitle=model_name)

    # ── Resolve model ──────────────────────────────────────────────
    with step("Resolve model") as s:
        predictor: Predictor = PredictorRegistry.get(model_name)()
        if not isinstance(predictor, Trainable):
            raise typer.BadParameter(
                f"'{model_name}' does not implement Trainable. "
                f"Trainable models: {PredictorRegistry.trainable_names()}"
            )
        s.set_detail(predictor.spec.description)

    # ── Check for existing champion ────────────────────────────────
    champion_meta: ModelMetadata | None = None
    champion_dir: Path = store.artifact_dir(model_name)
    backup_dir: Path = store.artifact_dir(f"{model_name}__backup")

    if store.is_trained(model_name):
        with step("Read champion metadata") as s:
            champion_meta = store.read_metadata(model_name)
            s.set_detail(f"Brier: {champion_meta.holdout_brier:.5f}")

        # Backup champion before training overwrites it
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(champion_dir, backup_dir)
        shutil.rmtree(champion_dir)

    # ── Train ──────────────────────────────────────────────────────
    with step("Load feature matrix") as s:
        df: DataFrame = loaders.load_modeling_file(repo)
        s.set_detail(f"{len(df):,} rows")

    with step(f"Train {model_name}") as s:
        challenger_meta: ModelMetadata = predictor.train(df, repo=repo)
        s.set_detail(f"holdout Brier: {challenger_meta.holdout_brier:.5f}")

    # ── Compare and decide ─────────────────────────────────────────
    _apply_promotion_decision(
        champion_meta=champion_meta,
        challenger_meta=challenger_meta,
        champion_dir=champion_dir,
        backup_dir=backup_dir,
        force=force,
        no_promote=no_promote,
    )

    console.summary()


@models_app.command("list")
def models_list() -> None:
    r"""List all registered models and their training status.

    \b
    Examples:
      gridiron models list
    """
    import pandas as pd

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.base import Trainable
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import PredictorRegistry

    repo: Path = get_settings().repo_root
    store = ArtifactStore(repo)

    rows: list[dict[str, str]] = []
    for name in PredictorRegistry.names():
        predictor: Predictor = PredictorRegistry.get(name)()
        is_trainable: bool = isinstance(predictor, Trainable)
        model_type: Literal["analytic", "trainable"] = "trainable" if is_trainable else "analytic"

        if is_trainable and store.is_trained(name):
            meta: ModelMetadata = store.read_metadata(name)
            trained_at: str = meta.trained_at
            brier: str = f"{meta.holdout_brier:.5f}"
        else:
            trained_at = "(not trained)" if is_trainable else "(no artifact)"
            brier = "-"

        rows.append(
            {
                "model": name,
                "type": model_type,
                "trained_at": trained_at,
                "holdout_brier": brier,
            }
        )

    df = pd.DataFrame(rows)
    typer.echo(df.to_string(index=False))


@models_app.command("info")
def models_info(
    model_name: str = typer.Argument(..., help="Model to inspect."),
) -> None:
    r"""Print detailed metadata for a trained model artifact.

    \b
    Examples:
      gridiron models info random_forest
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.champion import extract_metrics
    from gridiron_edge.models.artifact import ArtifactStore
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401

    repo: Path = get_settings().repo_root
    store = ArtifactStore(repo)

    if not store.is_trained(model_name):
        typer.echo(
            f"No trained artifact found for '{model_name}'. "
            f"Run 'gridiron models train {model_name}' first."
        )
        raise typer.Exit(code=1)

    meta: ModelMetadata = store.read_metadata(model_name)
    metrics: dict[str, float] = extract_metrics(meta)

    typer.echo(f"Model:           {meta.model_version}")
    typer.echo(f"Description:     {meta.notes or '(none)'}")
    typer.echo(f"Trained at:      {meta.trained_at}")
    typer.echo(f"Schema version:  {meta.schema_version}")
    typer.echo(f"Holdout Brier:   {meta.holdout_brier:.5f}")

    # Show all metrics if available
    from math import isnan

    for label, key in [
        ("ECE", "ece"),
        ("AUC", "auc"),
        ("Log Loss", "log_loss"),
        ("Accuracy", "accuracy"),
    ]:
        val: float = metrics[key]
        val_str: str = f"{val:.5f}" if not isnan(val) else "(not recorded)"
        typer.echo(f"{label + ':':17s}{val_str}")

    typer.echo(
        f"Training seasons: {', '.join(meta.training_seasons[:3])} ... {meta.training_seasons[-1]}"
    )
    typer.echo(f"Holdout seasons: {', '.join(meta.holdout_seasons)}")
    typer.echo(f"Features:        {len(meta.feature_columns)} columns")
    if meta.parameters:
        typer.echo(f"Parameters:      {meta.parameters}")

# src/gridiron_edge/cli/models.py
"""CLI commands for model training and artifact management."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.models.artifact import ModelMetadata
from gridiron_edge.models.base import Predictor

models_app = typer.Typer(
    help="Train and manage prediction model artifacts.",
    no_args_is_help=True,
)


@models_app.command("train")
def models_train(
    model_version: str = typer.Argument(
        ...,
        help="Model version to train (e.g. 'logistic_v1'). Must be registered and Trainable.",
    ),
    *,
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Overwrite an existing artifact. By default artifacts are immutable.",
    ),
) -> None:
    r"""Train a model and save its artifact to data/models/.

    Loads the current feature matrix, runs the model's train() method,
    saves the artifact, and prints the holdout Brier score.

    \b
    Examples:
      gridiron models train logistic_v1
      gridiron models train xgboost_v1 --overwrite
    """
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
    console.header("models train", subtitle=model_version)

    with step("Resolve model") as s:
        predictor: Predictor = PredictorRegistry.get(model_version)()
        if not isinstance(predictor, Trainable):
            raise typer.BadParameter(
                f"'{model_version}' does not implement Trainable. "
                "Elo models are computed analytically and do not require training. "
                f"Trainable models: {PredictorRegistry.trainable_names()}"
            )
        s.set_detail(predictor.spec.description)

    if store.is_trained(model_version) and not overwrite:
        typer.echo(
            f"Artifact already exists for '{model_version}'. "
            "Use --overwrite to retrain, or use a new version string."
        )
        raise typer.Exit(code=1)

    if store.is_trained(model_version) and overwrite:
        import shutil

        shutil.rmtree(store.artifact_dir(model_version))

    with step("Load feature matrix") as s:
        df: DataFrame = loaders.load_modeling_file(repo)
        s.set_detail(f"{len(df):,} rows")

    with step(f"Train {model_version}") as s:
        metadata: ModelMetadata = predictor.train(df, repo=repo)
        s.set_detail(f"holdout Brier: {metadata.holdout_brier:.5f}")

    typer.echo(
        f"\nTrained {model_version}: holdout Brier = {metadata.holdout_brier:.5f}"
        f"\nArtifact: {store.artifact_dir(model_version)}"
    )
    console.summary()


@models_app.command("list")
def models_list() -> None:
    r"""List all registered models and their training status.

    \b
    Examples:
      gridiron models list
    """
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
                "model_version": name,
                "type": model_type,
                "trained_at": trained_at,
                "holdout_brier": brier,
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    typer.echo(df.to_string(index=False))


@models_app.command("info")
def models_info(
    model_version: str = typer.Argument(..., help="Model version to inspect."),
) -> None:
    r"""Print detailed metadata for a trained model artifact.

    \b
    Examples:
      gridiron models info logistic_v1
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401

    repo: Path = get_settings().repo_root
    store = ArtifactStore(repo)

    if not store.is_trained(model_version):
        typer.echo(
            f"No trained artifact found for '{model_version}'. "
            f"Run 'gridiron models train {model_version}' first."
        )
        raise typer.Exit(code=1)

    meta: ModelMetadata = store.read_metadata(model_version)
    typer.echo(f"Model:           {meta.model_version}")
    typer.echo(f"Description:     {meta.notes or '(none)'}")
    typer.echo(f"Trained at:      {meta.trained_at}")
    typer.echo(f"Schema version:  {meta.schema_version}")
    typer.echo(f"Holdout Brier:   {meta.holdout_brier:.5f}")
    typer.echo(
        f"Training seasons: {', '.join(meta.training_seasons[:3])} ... {meta.training_seasons[-1]}"
    )
    typer.echo(f"Holdout seasons: {', '.join(meta.holdout_seasons)}")
    typer.echo(f"Feature columns: {', '.join(meta.feature_columns)}")
    if meta.parameters:
        typer.echo(f"Parameters:      {meta.parameters}")

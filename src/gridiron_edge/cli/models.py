# src/gridiron_edge/cli/models.py
"""CLI commands for model training and champion/challenger management.

Workstream 2 user surface:
    All commands that touch a specific model take two positional args —
    ``model_name`` (purpose, e.g. ``"win_prob"``) and ``model_type``
    (algorithm, e.g. ``"random_forest"``). This matches the ``ArtifactStore``
    storage scheme at ``data/models/{model_name}/{model_type}/`` and the
    composite ``PredictorRegistry`` keys (e.g. ``"win_prob_random_forest"``).

Examples:
    gridiron models train win_prob random_forest
    gridiron models info  win_prob xgboost
    gridiron models list

Workstream 2 D2b.3 status:
    The transitional ``_ARTIFACT_TO_REGISTRY`` map is gone. All
    ``PredictorRegistry`` keys are composite, so the resolution is a
    pure ``f"{model_name}_{model_type}"`` concatenation. Any pair that
    isn't a registered key surfaces as a ``KeyError`` from
    ``PredictorRegistry.get`` — clear and loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.evaluation.champion import ClassificationComparisonResult

if TYPE_CHECKING:
    from gridiron_edge.models.artifact import BaseModelMetadata
    from gridiron_edge.models.base import Predictor

models_app = typer.Typer(
    help="Train and manage prediction model artifacts.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_composite_key(key: str) -> tuple[str, str] | None:
    """Split a composite registry key into ``(model_name, model_type)``.

    Matches the key against the known model_name prefixes returned by
    :func:`gridiron_edge.models.game_prediction.predictor.get_known_model_names`.
    Returns ``None`` if the key doesn't match any known prefix (so
    ``models list`` can display ``—`` instead of crashing).

    Args:
        key: Composite registry key (e.g. ``"win_prob_random_forest"``).

    Returns:
        Tuple of ``(model_name, model_type)``, or ``None`` if no known
        prefix matches.
    """
    from gridiron_edge.models.game_prediction.predictor import get_known_model_names

    for model_name in get_known_model_names():
        prefix: str = f"{model_name}_"
        if key.startswith(prefix):
            return model_name, key[len(prefix) :]
    return None


def _apply_promotion_decision(
    *,
    champion_meta: BaseModelMetadata | None,
    challenger_meta: BaseModelMetadata,
    champion_dir: Path,
    backup_dir: Path,
    force: bool,
    no_promote: bool,
) -> None:
    """Compare challenger to champion and handle promotion/rejection.

    Champion / backup directories live under the nested scheme; backup
    is co-located with champion at
    ``data/models/{model_name}/{model_type}__backup/``.
    """
    import shutil

    from gridiron_edge.evaluation.champion import (
        compare_classification_models,
        format_classification_comparison,
    )

    if champion_meta is None:
        typer.echo("\nNo existing champion. Saved as champion.")
        typer.echo(f"  Brier: {challenger_meta.holdout_brier:.5f}")  # type: ignore[attr-defined]
        typer.echo(f"  Artifact: {champion_dir}")
        return

    result: ClassificationComparisonResult = compare_classification_models(
        # pyrefly: ignore [bad-argument-type]
        champion_meta,
        # pyrefly: ignore [bad-argument-type]
        challenger_meta,
    )
    typer.echo(format_classification_comparison(result))

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
        help="Model purpose (e.g. 'win_prob').",
    ),
    model_type: str = typer.Argument(
        ...,
        help="Model algorithm (e.g. 'random_forest', 'xgboost', 'logistic').",
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
      gridiron models train win_prob random_forest
      gridiron models train win_prob xgboost --force
      gridiron models train win_prob logistic --no-promote
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
    console.header("models train", subtitle=f"{model_name} {model_type}")

    # ── Resolve (name, type) → composite registry key ─────────────
    registry_key: str = f"{model_name}_{model_type}"

    # ── Resolve model ──────────────────────────────────────────────
    with step("Resolve model") as s:
        try:
            predictor: Predictor = PredictorRegistry.get(registry_key)()
        except KeyError as exc:
            raise typer.BadParameter(
                f"'{registry_key}' is not a registered predictor. "
                f"Available: {PredictorRegistry.trainable_names()}"
            ) from exc
        if not isinstance(predictor, Trainable):
            raise typer.BadParameter(
                f"'{registry_key}' does not implement Trainable. "
                f"Trainable registry keys: {PredictorRegistry.trainable_names()}"
            )
        s.set_detail(predictor.spec.description)

    # ── Check for existing champion ────────────────────────────────
    champion_meta: BaseModelMetadata | None = None
    champion_dir: Path = store.artifact_dir(model_name, model_type)
    # Sibling directory under data/models/{model_name}/.
    backup_dir: Path = champion_dir.parent / f"{model_type}__backup"

    if store.is_trained(model_name, model_type):
        with step("Read champion metadata") as s:
            champion_meta = store.read_metadata(model_name, model_type)
            s.set_detail(f"Brier: {champion_meta.holdout_brier:.5f}")  # type: ignore[attr-defined]

        # Backup champion before training overwrites it
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(champion_dir, backup_dir)
        shutil.rmtree(champion_dir)

    # ── Train ──────────────────────────────────────────────────────
    with step("Load feature matrix") as s:
        df: DataFrame = loaders.load_modeling_file(repo)
        s.set_detail(f"{len(df):,} rows")

    with step(f"Train {model_name} {model_type}") as s:
        challenger_meta: BaseModelMetadata = predictor.train(df, repo=repo)
        s.set_detail(f"holdout Brier: {challenger_meta.holdout_brier:.5f}")  # type: ignore[attr-defined]

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

    Shows the WS2 ``(model_name, model_type)`` pair for each registry
    entry. Every key is composite after D2b.3 so the pair is derived
    by splitting on the first underscore.

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
    for key in PredictorRegistry.names():
        predictor: Predictor = PredictorRegistry.get(key)()
        is_trainable: bool = isinstance(predictor, Trainable)
        kind: Literal["analytic", "trainable"] = "trainable" if is_trainable else "analytic"

        pair: tuple[str, str] | None = _split_composite_key(key)

        if is_trainable and pair is not None and store.is_trained(*pair):
            meta: BaseModelMetadata = store.read_metadata(*pair)
            trained_at: str = meta.trained_at
            brier: str = f"{meta.holdout_brier:.5f}"  # type: ignore[attr-defined]
        else:
            trained_at = "(not trained)" if is_trainable else "(no artifact)"
            brier = "-"

        model_name_disp: str = pair[0] if pair else "—"
        model_type_disp: str = pair[1] if pair else "—"

        rows.append(
            {
                "model_name": model_name_disp,
                "model_type": model_type_disp,
                "registry_key": key,
                "kind": kind,
                "trained_at": trained_at,
                "holdout_brier": brier,
            }
        )

    df = pd.DataFrame(rows)
    typer.echo(df.to_string(index=False))


@models_app.command("info")
def models_info(
    model_name: str = typer.Argument(..., help="Model purpose (e.g. 'win_prob')."),
    model_type: str = typer.Argument(..., help="Model algorithm (e.g. 'random_forest')."),
) -> None:
    r"""Print detailed metadata for a trained model artifact.

    \b
    Examples:
      gridiron models info win_prob random_forest
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.champion import extract_classification_metrics
    from gridiron_edge.models.artifact import ArtifactStore
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import PredictorRegistry

    repo: Path = get_settings().repo_root
    store = ArtifactStore(repo)
    registry_key: str = f"{model_name}_{model_type}"

    # Validate the pair is a registered key (gives a clean error message
    # before falling through to a generic FileNotFoundError from the store).
    try:
        PredictorRegistry.get(registry_key)
    except KeyError as exc:
        raise typer.BadParameter(
            f"'{registry_key}' is not a registered predictor. "
            f"Available: {sorted(PredictorRegistry.names())}"
        ) from exc

    if not store.is_trained(model_name, model_type):
        typer.echo(
            f"No trained artifact found for '{model_name} {model_type}'. "
            f"Run 'gridiron models train {model_name} {model_type}' first."
        )
        raise typer.Exit(code=1)

    meta: BaseModelMetadata = store.read_metadata(model_name, model_type)
    # pyrefly: ignore [bad-argument-type]
    metrics: dict[str, float] = extract_classification_metrics(meta)

    typer.echo(f"Model:           {meta.model_name} / {meta.model_type}")
    typer.echo(f"Task:            {meta.task}")
    typer.echo(f"Description:     {meta.notes or '(none)'}")
    typer.echo(f"Trained at:      {meta.trained_at}")
    typer.echo(f"Schema version:  {meta.schema_version}")
    typer.echo(f"Holdout Brier:   {meta.holdout_brier:.5f}")  # type: ignore[attr-defined]

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

    if meta.training_seasons:
        head: str = ", ".join(meta.training_seasons[:3])
        tail: str = meta.training_seasons[-1]
        typer.echo(f"Training seasons: {head} ... {tail}")
    typer.echo(f"Holdout seasons: {', '.join(meta.holdout_seasons)}")
    typer.echo(f"Features:        {len(meta.feature_columns)} columns")
    typer.echo(f"Rows:            train={meta.n_train_rows:,}  holdout={meta.n_holdout_rows:,}")
    if meta.parameters:
        typer.echo(f"Parameters:      {meta.parameters}")

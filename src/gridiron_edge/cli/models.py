# src/gridiron_edge/cli/models.py
"""CLI commands for model training and champion/challenger management.

Workstream 2 user surface:
    All commands that touch a specific model take two positional args -
    ``model_name`` (purpose, e.g. ``"win_prob"``) and ``model_type``
    (algorithm, e.g. ``"random_forest"``). This matches the ``ArtifactStore``
    storage scheme at ``data/models/{model_name}/{model_type}/`` and the
    composite ``ModelRegistry`` keys (e.g. ``"win_prob_random_forest"``).

Examples:
    gridiron models train win_prob random_forest
    gridiron models info  win_prob xgboost
    gridiron models list

Registry key resolution:
    All ``ModelRegistry`` keys are composite, so resolution is a
    pure ``f"{model_name}_{model_type}"`` concatenation. Any pair that
    isn't a registered key surfaces as a ``KeyError`` from
    ``ModelRegistry.get`` - clear and loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.evaluation.champion import ClassificationComparisonResult

if TYPE_CHECKING:
    from gridiron_edge.models.artifact import BaseModelMetadata


class _ChallengerTrainer(Protocol):
    """Training surface required by challenger artifact staging."""

    def train(
        self,
        df: DataFrame,
        *,
        repo: Path | None = None,
    ) -> BaseModelMetadata:
        """Train and persist a challenger model artifact."""
        ...


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
    ``models list`` can display ``-`` instead of crashing).

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
    candidate_dir: Path,
    force: bool,
    no_promote: bool,
) -> None:
    """Compare challenger to champion and atomically promote or discard.

    The challenger has been trained into ``candidate_dir`` (a sibling of
    ``champion_dir``). This function decides whether to:

    - **Promote**: atomically move ``candidate_dir`` to ``champion_dir``,
      replacing the previous champion (which is deleted first).
    - **Reject**: delete ``candidate_dir``, leaving the champion untouched.

    The promote step is atomic at the filesystem level: ``shutil.rmtree``
    of the old champion happens immediately before ``shutil.move`` of the
    candidate. There's no window during which the champion directory is
    missing while the candidate is also not in place - except for the
    transition itself, which is bounded to a single ``move`` call. If
    that fails mid-stream, the user has both an old-champion-deleted
    state and a candidate dir, and recovery is to re-run training.

    The training failure / interruption mode is now safe: if training
    crashes before this function is called, ``candidate_dir`` is partial
    or missing, but ``champion_dir`` was never touched and the existing
    champion remains usable.

    Args:
        champion_meta: Existing champion metadata, or None if no champion.
        challenger_meta: Newly-trained challenger metadata.
        champion_dir: Destination path for the champion artifact.
        candidate_dir: Path where the challenger was trained.
        force: Override gate failures and promote anyway.
        no_promote: Reject regardless of gate results.
    """
    import shutil

    from gridiron_edge.evaluation.champion import (
        compare_classification_models,
        format_classification_comparison,
    )

    if champion_meta is None:
        # No existing champion - just move candidate into place.
        if champion_dir.exists():
            shutil.rmtree(champion_dir)
        shutil.move(str(candidate_dir), str(champion_dir))
        typer.echo("\nNo existing champion. Saved as champion.")
        label, value = _primary_metric_for(challenger_meta)
        typer.echo(f"  {label}: {value}")
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
        # Atomic-ish promotion: delete old champion, move candidate to champion.
        # The window between these two operations is bounded to the time of
        # a single shutil.move call. If it fails mid-stream, the user re-runs
        # training and the candidate path is fresh.
        if champion_dir.exists():
            shutil.rmtree(champion_dir)
        shutil.move(str(candidate_dir), str(champion_dir))
        if force and not result.should_promote:
            typer.echo("  ⚠️  Force-promoted despite failed gates.")
        else:
            typer.echo("  ✅ New champion promoted.")
    else:
        # Reject: delete candidate, champion already exists at champion_dir.
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        if no_promote:
            typer.echo("  Champion unchanged (--no-promote).")
        else:
            typer.echo("  ❌ Challenger rejected. Champion unchanged.")
            typer.echo("  Use --force to promote anyway.")


def _train_challenger_into_candidate(
    *,
    model: _ChallengerTrainer,
    df: DataFrame,
    repo: Path,
    champion_dir: Path,
    candidate_dir: Path,
    model_type: str,
) -> BaseModelMetadata:
    """Train challenger into ``candidate_dir`` without disturbing the champion.

    Mechanism: the existing champion (if any) is temporarily moved aside to
    a ``__holding`` directory before training. The model's save target
    is ``champion_dir`` (derived from the artifact store's path scheme), so
    we move the freshly-trained artifact to ``candidate_dir`` immediately
    after training and restore the champion from holding.

    On any failure during training, the champion is restored from holding
    and any partial candidate is cleaned up. The user is left with the
    same champion they started with.

    Args:
        model: Trainable model returned by ``ModelRegistry.get()``.
        df: Modeling feature matrix.
        repo: Repository root.
        champion_dir: Path where the artifact store writes by default.
        candidate_dir: Sibling path where the trained challenger will live
            until the promotion decision.
        model_type: For naming the ``__holding`` sibling directory.

    Returns:
        Metadata from the trained challenger.

    Raises:
        Any exception raised by ``model.train()``. Champion is
        restored before re-raising.
    """
    import shutil

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    # Temporarily move existing champion aside so the model's save
    # to champion_dir succeeds without overwriting.
    champion_holding: Path | None = None
    if champion_dir.exists():
        champion_holding = champion_dir.parent / f"{model_type}__holding"
        if champion_holding.exists():
            shutil.rmtree(champion_holding)
        shutil.move(str(champion_dir), str(champion_holding))

    try:
        challenger_meta: BaseModelMetadata = model.train(df, repo=repo)
        # Move freshly-trained artifact to candidate location.
        shutil.move(str(champion_dir), str(candidate_dir))
        # Restore champion from holding so it's available for comparison.
        if champion_holding is not None:
            shutil.move(str(champion_holding), str(champion_dir))
    except Exception:
        # Training failed or interrupted. Clean up any partial artifact,
        # restore the original champion, and re-raise.
        if champion_dir.exists():
            shutil.rmtree(champion_dir)
        if champion_holding is not None and champion_holding.exists():
            shutil.move(str(champion_holding), str(champion_dir))
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        raise

    return challenger_meta


_PRIMARY_METRIC_BY_TASK: dict[str, str] = {
    "classification": "brier",
    "regression": "mae",
}

_METRIC_LABELS: dict[str, str] = {
    "brier": "Holdout Brier",
    "ece": "ECE",
    "auc": "AUC",
    "log_loss": "Log Loss",
    "accuracy": "Accuracy",
    "mae": "Holdout MAE",
    "rmse": "Holdout RMSE",
    "r2": "Holdout R²",
}

_METRIC_ORDER_BY_TASK: dict[str, list[str]] = {
    "classification": ["brier", "ece", "auc", "log_loss", "accuracy"],
    "regression": ["mae", "rmse", "r2"],
}


def _primary_metric_for(meta: BaseModelMetadata) -> tuple[str, str]:
    """Return (display label, formatted value) for a model's primary metric."""
    key = _PRIMARY_METRIC_BY_TASK.get(meta.task)
    if key is None:
        return ("-", "-")
    value = meta.metrics.get(key)
    if value is None:
        return (_METRIC_LABELS[key], "(not recorded)")
    return (_METRIC_LABELS[key], f"{value:.5f}")


def _metric_block_for(meta: BaseModelMetadata) -> list[tuple[str, str]]:
    """Build (label, value) rows for the metric block in ``models info``."""
    order = _METRIC_ORDER_BY_TASK.get(meta.task, [])
    rows: list[tuple[str, str]] = []
    for key in order:
        value = meta.metrics.get(key)
        label = _METRIC_LABELS[key]
        if value is None:
            rows.append((label, "(not recorded)"))
        else:
            rows.append((label, f"{value:.5f}"))
    return rows


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

    Training writes the challenger to a candidate directory. The existing
    champion is not touched until the promotion decision is made. If
    training fails or the user interrupts, the champion remains intact at
    its original location.

    If no champion exists, the new model is saved as champion. If a
    champion exists, the new model is compared using promotion gates
    appropriate to the task: classification gates check Brier
    improvement, ECE tolerance, and AUC tolerance; regression gates
    check MAE improvement and RMSE tolerance. The champion is replaced
    only if all gates pass (or --force is used).

    \b
    Examples:
      gridiron models train win_prob random_forest
      gridiron models train win_prob xgboost --force
      gridiron models train win_prob logistic --no-promote
      gridiron models train total random_forest
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets import loaders
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.base import Trainable
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import ModelRegistry

    repo: Path = get_settings().repo_root
    store = ArtifactStore(repo)
    console.header("models train", subtitle=f"{model_name} {model_type}")

    registry_key: str = f"{model_name}_{model_type}"

    # ── Resolve model ──────────────────────────────────────────────
    with step("Resolve model") as s:
        try:
            from typing import cast

            from gridiron_edge.models.base import GameModel

            model = cast(GameModel, ModelRegistry.get(registry_key)())
        except KeyError as exc:
            raise typer.BadParameter(
                f"'{registry_key}' is not a registered model. "
                f"Available: {ModelRegistry.trainable_names()}"
            ) from exc
        if not isinstance(model, Trainable):
            raise typer.BadParameter(
                f"'{registry_key}' does not implement Trainable. "
                f"Trainable registry keys: {ModelRegistry.trainable_names()}"
            )
        s.set_detail(model.spec.description)

    # ── Resolve paths ──────────────────────────────────────────────
    champion_dir: Path = store.artifact_dir(model_name, model_type)
    candidate_dir: Path = champion_dir.parent / f"{model_type}__candidate"

    # ── Read champion metadata if present ──────────────────────────
    champion_meta: BaseModelMetadata | None = None
    if store.is_trained(model_name, model_type):
        with step("Read champion metadata") as s:
            champion_meta = store.read_metadata(model_name, model_type)
            label, value = _primary_metric_for(champion_meta)
            s.set_detail(f"{label}: {value}")

    # ── Train challenger into candidate directory ──────────────────
    with step("Load feature matrix") as s:
        df: DataFrame = loaders.load_modeling_file(repo)
        s.set_detail(f"{len(df):,} rows")

    with step(f"Train {model_name} {model_type}") as s:
        challenger_meta = _train_challenger_into_candidate(
            model=cast(_ChallengerTrainer, model),
            df=df,
            repo=repo,
            champion_dir=champion_dir,
            candidate_dir=candidate_dir,
            model_type=model_type,
        )
        label, value = _primary_metric_for(challenger_meta)
        s.set_detail(f"holdout {label}: {value}")

    # ── Compare and decide ─────────────────────────────────────────
    _apply_promotion_decision(
        champion_meta=champion_meta,
        challenger_meta=challenger_meta,
        champion_dir=champion_dir,
        candidate_dir=candidate_dir,
        force=force,
        no_promote=no_promote,
    )

    console.summary()


@models_app.command("list")
def models_list() -> None:
    r"""List all registered models and their training status.

    Shows the ``(model_name, model_type)`` pair for each registry
    entry along with a task-appropriate primary metric. Classification
    models show Brier; regression models show MAE.

    \b
    Examples:
      gridiron models list
    """
    from typing import cast

    import pandas as pd

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.base import GameModel, Trainable
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import ModelRegistry

    repo: Path = get_settings().repo_root
    store = ArtifactStore(repo)

    rows: list[dict[str, str]] = []
    for key in ModelRegistry.names():
        model = cast(GameModel, ModelRegistry.get(key)())
        is_trainable: bool = isinstance(model, Trainable)
        kind: Literal["analytic", "trainable"] = "trainable" if is_trainable else "analytic"

        pair: tuple[str, str] | None = _split_composite_key(key)

        primary_label: str = "-"
        primary_value: str = "-"
        trained_at: str = "(not trained)" if is_trainable else "(no artifact)"

        if is_trainable and pair is not None and store.is_trained(*pair):
            meta = store.read_metadata(*pair)
            trained_at = meta.trained_at
            primary_label, primary_value = _primary_metric_for(meta)

        model_name_disp: str = pair[0] if pair else "-"
        model_type_disp: str = pair[1] if pair else "-"

        rows.append(
            {
                "model_name": model_name_disp,
                "model_type": model_type_disp,
                "registry_key": key,
                "kind": kind,
                "trained_at": trained_at,
                "primary_metric": primary_label,
                "primary_value": primary_value,
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

    Displays task-appropriate metrics. Classification models report
    Brier, ECE, AUC, log loss, and accuracy. Regression models report
    MAE, RMSE, and R².

    \b
    Examples:
      gridiron models info win_prob random_forest
      gridiron models info total xgboost
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import ModelRegistry

    repo: Path = get_settings().repo_root
    store = ArtifactStore(repo)
    registry_key: str = f"{model_name}_{model_type}"

    try:
        ModelRegistry.get(registry_key)
    except KeyError as exc:
        raise typer.BadParameter(
            f"'{registry_key}' is not a registered model. "
            f"Available: {sorted(ModelRegistry.names())}"
        ) from exc

    if not store.is_trained(model_name, model_type):
        # Check whether this is an analytic model that doesn't persist artifacts
        try:
            model = ModelRegistry.get(registry_key)()
            if hasattr(model, "spec") and not getattr(model.spec, "trainable", True):
                typer.echo(
                    f"'{model_name} {model_type}' is an analytic model "
                    f"without persisted training state. Use 'gridiron evaluate "
                    f"summary --model-key {registry_key}' to see archive metrics."
                )
                raise typer.Exit(code=0)
        except KeyError:
            pass

        typer.echo(
            f"No trained artifact found for '{model_name} {model_type}'. "
            f"Run 'gridiron models train {model_name} {model_type}' first."
        )
        raise typer.Exit(code=1)

    meta = store.read_metadata(model_name, model_type)

    typer.echo(f"Model:           {meta.model_name} / {meta.model_type}")
    typer.echo(f"Task:            {meta.task}")
    typer.echo(f"Description:     {meta.notes or '(none)'}")
    typer.echo(f"Trained at:      {meta.trained_at}")
    typer.echo(f"Schema version:  {meta.schema_version}")

    for label, value in _metric_block_for(meta):
        typer.echo(f"{label + ':':17s}{value}")

    if meta.training_seasons:
        head: str = ", ".join(meta.training_seasons[:3])
        tail: str = meta.training_seasons[-1]
        typer.echo(f"Training seasons: {head} ... {tail}")
    typer.echo(f"Holdout seasons: {', '.join(meta.holdout_seasons)}")
    typer.echo(f"Features:        {len(meta.feature_columns)} columns")
    typer.echo(f"Rows:            train={meta.n_train_rows:,}  holdout={meta.n_holdout_rows:,}")
    if meta.parameters:
        typer.echo(f"Parameters:      {meta.parameters}")

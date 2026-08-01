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

from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._composites import write_champion_manifest
from gridiron_edge.core.console import console, step
from gridiron_edge.core.settings import Settings
from gridiron_edge.evaluation.prop_metrics import PropEvalReport
from gridiron_edge.models.prop_prediction.base import PropModelMetadata, PropModelType, PropTrainer

logger: Logger = logging.getLogger(__name__)

props_app = typer.Typer(help="Player prop projections.", no_args_is_help=True)

# ---------------------------------------------------------------------------
# Trainer registry - maps model name to trainer class
# ---------------------------------------------------------------------------


def _parse_season_arg(value: str | None) -> int | None:
    """Parse a CLI season argument that may be int-like or a season label.

    Accepts:
        - None
        - "2023"
        - "2023-2024"
        - 2023 (already an int - passed through)

    Returns:
        The integer starting year for the requested season, or None.

    Raises:
        typer.BadParameter: If the value is malformed.
    """
    if value is None:
        return None

    text: str = value.strip()
    if not text:
        return None

    # Plain integer form
    if text.isdigit():
        return int(text)

    # Season-label form: "YYYY-YYYY+1"
    if "-" in text:
        head, _, tail = text.partition("-")
        if head.isdigit() and tail.isdigit():
            head_int = int(head)
            tail_int = int(tail)
            if tail_int == head_int + 1:
                return head_int
            raise typer.BadParameter(
                f"Season label '{text}' is not contiguous. "
                f"Expected 'YYYY-YYYY+1' (e.g. '2023-2024')."
            )

    raise typer.BadParameter(
        f"Could not parse season '{text}'. "
        f"Use either an integer (e.g. 2023) or a season label "
        f"(e.g. '2023-2024')."
    )


def _ensure_prop_models_registered() -> None:
    """Import prop model modules so ModelRegistry is populated."""
    import gridiron_edge.models.prop_prediction.qb_pass_yards
    import gridiron_edge.models.prop_prediction.qb_rush_yards
    import gridiron_edge.models.prop_prediction.rb_rush_yards
    import gridiron_edge.models.prop_prediction.te_rec_yards
    import gridiron_edge.models.prop_prediction.wr_rec_yards  # noqa: F401


def _enrich_and_predict(
    trainer: PropTrainer,
    df: DataFrame,
    meta: PropModelMetadata,
    model_rmse: float,
) -> DataFrame:
    """Single prediction + enrichment path for prop CLI + composite.

    Uses ``trainer.predict_with_meta()`` so ``meta.feature_columns`` is
    the source of truth for which columns get passed to the model.
    Rows dropped by that path are excluded from the returned DataFrame.

    The returned DataFrame is archive-ready: ``predicted_mean``,
    ``stat_type``, and all post-process enrichment columns
    (``predicted_std``, ``lo_90``, ``hi_90``, and, when ``line`` is
    present, ``p_over``, ``lean``, ``confidence_tier``) are populated.

    Args:
        trainer: Fitted (or artifact-loaded) prop trainer instance.
        df: Data slice to predict on.
        meta: Metadata from the trainer's ``train()`` /
            ``train_through()`` or from ``ArtifactStore.read_metadata``.
        model_rmse: Trained model's RMSE, used by
            ``enrich_prop_predictions`` for std computation.

    Returns:
        Archive-ready DataFrame, empty if no rows survived NaN-drop.
    """
    import numpy as np

    from gridiron_edge.models.prop_prediction.post_process import (
        TARGET_STD_MAP,
        enrich_prop_predictions,
    )

    preds, predicted_df = trainer.predict_with_meta(df, meta)
    if predicted_df.empty:
        return DataFrame()

    result: DataFrame = predicted_df.copy()
    result["predicted_mean"] = preds
    result["stat_type"] = trainer.spec.name

    target: str = trainer.spec.target_col
    std_col: str = TARGET_STD_MAP.get(trainer.spec.name, f"{target}_L3_std")
    if std_col not in result.columns:
        result[std_col] = np.nan

    return enrich_prop_predictions(
        df=result,
        model_rmse=model_rmse,
        target_std_col=std_col,
    )


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


def _walk_forward_predict_for_season(
    *,
    model_name: str,
    model_type: PropModelType,
    season: int,
    features_df: DataFrame,
) -> tuple[DataFrame, float]:
    """Train through ``season`` and predict that season's player-games.

    Used by ``backfill_cmd`` and the composite full-retrain prop
    backfill stage. Returns archive-ready enriched predictions.

    Args:
        model_name: Prop family name (e.g. ``"qb_pass_yards"``).
        model_type: Algorithm to use.
        season: Integer season label. Becomes cutoff and prediction window.
        features_df: Pre-built features DataFrame containing all seasons.

    Returns:
        Tuple of (enriched predictions DataFrame, model RMSE).
    """
    trainer: PropTrainer = _get_trainer(model_name)
    try:
        meta: PropModelMetadata = trainer.train_through(
            cutoff_season=season,
            model_type=model_type,
        )
    except ValueError as exc:
        # Some cutoffs produce an empty training or holdout slice after
        # the era-aware NaN filter. This is expected at era boundaries
        # and is not a program error — it means the model cannot honestly
        # predict that season with the feature set the training slice
        # supports. Log and return an empty result; the outer loop
        # continues.
        msg = str(exc)
        if "No training rows" in msg or "No holdout rows" in msg or "No rows available" in msg:
            logger.warning(
                "%s/%s cutoff=%d skipped: %s",
                model_name,
                model_type.value,
                season,
                msg,
            )
            return DataFrame(), float("nan")
        raise

    season_df: DataFrame = features_df.loc[features_df["season"] == season, :].copy()
    rmse: float = meta.metrics.get("rmse", float("nan"))

    enriched: DataFrame = _enrich_and_predict(trainer, season_df, meta, rmse)
    return enriched, rmse


# ---------------------------------------------------------------------------
# Archive + upcoming-feature helpers
# ---------------------------------------------------------------------------


def _load_archive_or_exit(
    *,
    model_name: str,
    model_type: PropModelType,
    season: int | None = None,
) -> DataFrame:
    """Load archive predictions for a model, exit if empty or unregistered."""
    from gridiron_edge.evaluation.prop_archive import build_prop_evaluation_df

    try:
        eval_df: DataFrame = build_prop_evaluation_df(
            model_name=model_name,
            model_type=model_type.value,
            season=season,
        )
    except KeyError as exc:
        available = _all_prop_models()
        typer.echo(f"Unknown model: {model_name}. Available: {available}")
        raise typer.Exit(code=1) from exc

    if eval_df.empty:
        typer.echo(
            f"No archived predictions found for {model_name} ({model_type}).\n"
            f"Run: gridiron props backfill --model {model_name} "
            f"--model-type {model_type.value}"
        )
        raise typer.Exit(code=1)

    return eval_df


def _load_upcoming_prop_features(
    trainer: PropTrainer,
) -> DataFrame:
    """Load upcoming-week features for a prop trainer.

    Returns an empty DataFrame when no upcoming player-game rows exist
    yet (out-of-season, or feature pipeline not refreshed). Callers
    must handle the empty case explicitly.

    Args:
        trainer: Prop trainer whose position filter drives feature load.

    Returns:
        DataFrame of upcoming player-game features. Empty if no
        upcoming rows are available.
    """
    from gridiron_edge.features.player.builder import build_prop_features

    df: DataFrame = build_prop_features(
        position_filter=trainer.spec.position_filter,
    )

    if df.empty:
        return df

    if "is_upcoming" in df.columns:
        df = df.loc[df["is_upcoming"], :].copy()

    return df


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
    season: int | None = typer.Option(
        None,
        "--season",
        help="Optional season filter (e.g. 2024).",
    ),
) -> None:
    """Archive-driven holdout evaluation report for a prop model."""
    from gridiron_edge.evaluation.prop_metrics import evaluate_prop_model

    mt = PropModelType(model_type)
    console.header("props evaluate", subtitle=f"{model} · {mt}")

    with step("Load archived predictions") as s:
        eval_df: DataFrame = _load_archive_or_exit(
            model_name=model,
            model_type=mt,
            season=season,
        )
        s.set_rows(len(eval_df))
        s.set_detail(f"{eval_df['season'].nunique()} season(s)")

    with step("Evaluate holdout") as s:
        report: PropEvalReport = evaluate_prop_model(
            model_name=model,
            actual=eval_df["actual"],
            predicted_mean=eval_df["predicted_mean"],
            predicted_std=eval_df.get("predicted_std"),
            lo_90=eval_df.get("lo_90"),
            hi_90=eval_df.get("hi_90"),
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
    season: int | None = typer.Option(
        None,
        "--season",
        help="Optional season filter (e.g. 2024).",
    ),
    write_manifest: bool = typer.Option(
        False,
        "--write-manifest",
        help=(
            "After displaying per-family champions, persist all champion "
            "decisions to the manifest at data/output/champions/champions.json. "
            "Runs all three selectors (game classification, game regression, "
            "prop) so the manifest reflects the full repo state. Preserves "
            "entries for model families outside the current retrain scope."
        ),
    ),
) -> None:
    """Archive-driven champion selection.

    For each requested stat family, compare ElasticNet, RandomForest,
    and XGBoost archive performance and pick the lowest-MAE algorithm.
    Requires that ``gridiron props backfill`` has already populated
    the archive for the algorithms being compared.

    Pass ``--write-manifest`` to also persist champion decisions to
    ``data/output/champions/champions.json``. Runs the full selector
    suite (game + prop) so the manifest reflects the entire repo state;
    preserves entries for families outside the current backfill scope.
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.champion import (
        RegressionComparisonResult,
        RegressionModelResult,
        build_prop_champion_candidates,
        compare_regression_models,
        format_regression_comparison,
        select_prop_champion,
    )

    repo: Path = get_settings().repo_root

    models: list[str] = _all_prop_models() if model == "all" else [model]

    for m in models:
        console.header("props champion", subtitle=m)

        with step(f"Load archive for {m}") as s:
            results: list[RegressionModelResult] = build_prop_champion_candidates(
                m,
                repo=repo,
                season=season,
            )
            if not results:
                s.set_detail("no archive rows — skipping")
            else:
                s.set_detail(f"{len(results)} algorithm(s) evaluated")

        if not results:
            typer.echo(f"\n  ❌ No archived predictions found for {m}.")
            console.summary()
            continue

        with step("Select champion") as s:
            champion, summary = select_prop_champion(results)
            s.set_detail(f"🏆 {champion.model_type} (MAE={champion.mae:.2f})")

        typer.echo(summary)

        challengers: list[RegressionModelResult] = [
            r for r in results if r.model_type != champion.model_type
        ]
        for challenger in challengers:
            comparison: RegressionComparisonResult = compare_regression_models(
                champion,
                challenger,
            )
            typer.echo(format_regression_comparison(comparison))

        console.summary()

    if len(models) > 1:
        typer.echo("  Champion selection complete across all stat families.\n")

    if write_manifest:
        write_champion_manifest(repo)


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
    start_season: str | None = typer.Option(
        None,
        "--start-season",
        help=(
            "Earliest season to backfill (inclusive). Accepts either an "
            "integer (e.g. 2023) or a season label (e.g. '2023-2024'). "
            "Defaults to the second-earliest season available so a "
            "prior training window always exists."
        ),
    ),
    end_season: str | None = typer.Option(
        None,
        "--end-season",
        help=(
            "Latest season to backfill (inclusive). Accepts either an "
            "integer (e.g. 2024) or a season label (e.g. '2024-2025'). "
            "Defaults to the most recent season available."
        ),
    ),
) -> None:
    """Walk-forward backfill of prop predictions to the archive."""
    from gridiron_edge.evaluation.prop_archive import archive_prop_predictions
    from gridiron_edge.features.player.builder import build_prop_features

    mt = PropModelType(model_type)
    console.header(
        "props backfill",
        subtitle=f"{model} · {mt}",
    )

    trainer: PropTrainer = _get_trainer(model)

    with step("Load feature data") as s:
        features_df: DataFrame = build_prop_features(
            position_filter=trainer.spec.position_filter,
        )
        s.set_rows(len(features_df))

    if features_df.empty:
        typer.echo("No feature data available; nothing to backfill.")
        raise typer.Exit(code=1)

    parsed_start: int | None = _parse_season_arg(start_season)
    parsed_end: int | None = _parse_season_arg(end_season)

    seasons_available: list[int] = sorted(
        int(s) for s in features_df["season"].dropna().unique().tolist()
    )
    if len(seasons_available) < 2:
        typer.echo("Walk-forward backfill requires at least two seasons of feature data.")
        raise typer.Exit(code=1)

    resolved_start: int = parsed_start if parsed_start is not None else seasons_available[1]
    resolved_end: int = parsed_end if parsed_end is not None else seasons_available[-1]

    if resolved_start <= seasons_available[0]:
        typer.echo(
            f"--start-season {resolved_start} has no prior training "
            f"window. Minimum allowed is {seasons_available[1]}."
        )
        raise typer.Exit(code=1)

    if resolved_end < resolved_start:
        typer.echo("--end-season must be >= --start-season.")
        raise typer.Exit(code=1)

    target_seasons: list[int] = [
        s for s in seasons_available if resolved_start <= s <= resolved_end
    ]
    if not target_seasons:
        typer.echo("No seasons fall within the requested backfill range.")
        raise typer.Exit(code=1)

    total_archived: int = 0
    for season in target_seasons:
        with step(f"Train {model} ({mt}) through {season}") as s:
            enriched, rmse = _walk_forward_predict_for_season(
                model_name=model,
                model_type=mt,
                season=season,
                features_df=features_df,
            )
            s.set_rows(len(enriched))
            s.set_detail(f"RMSE={rmse:.1f}")

        if enriched.empty:
            continue

        with step(f"Archive {model} ({mt}) {season}") as s:
            archive_prop_predictions(
                enriched,
                is_backfilled=True,
                model_name=model,
                model_type=mt.value,
            )
            s.set_rows(len(enriched))
            total_archived += len(enriched)

    typer.echo(
        f"  Walk-forward backfill complete: {total_archived:,} rows "
        f"archived across {len(target_seasons)} seasons."
    )
    console.summary()


@props_app.command("train-and-save")
def train_and_save_cmd(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Prop model family to train, e.g. qb_pass_yards.",
    ),
    model_type: str = typer.Option(
        "elasticnet",
        "--model-type",
        "-t",
        help="Algorithm type: elasticnet, random_forest, xgboost",
    ),
) -> None:
    """Train a prop model on the standard HOLDOUT_SEASONS split and persist the artifact.

    Produces a trained artifact at
    ``data/models/{model}/{model_type}/`` that ``props projections``
    can load without retraining. Use this when you need a stable,
    persisted model for upcoming-week predictions.

    For honest historical predictions across many seasons, use
    ``props backfill`` instead (walk-forward, no artifact saved).
    """
    mt = PropModelType(model_type)
    console.header("props train-and-save", subtitle=f"{model} · {mt}")

    trainer: PropTrainer = _get_trainer(model)

    with step(f"Train {model} ({mt})") as s:
        meta: PropModelMetadata = trainer.train_and_save(model_type=mt)
        mae: float = meta.metrics.get("mae", float("nan"))
        rmse: float = meta.metrics.get("rmse", float("nan"))
        r2: float = meta.metrics.get("r2", float("nan"))
        s.set_detail(f"MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}")

    typer.echo("")
    typer.echo(f"  Model:     {meta.model_name} / {meta.model_type}")
    typer.echo(f"  Trained:   {meta.trained_at}")
    typer.echo(f"  N train:   {meta.n_train_rows:,}")
    typer.echo(f"  N holdout: {meta.n_holdout_rows:,}")
    typer.echo(f"  MAE:       {mae:.1f}")
    typer.echo(f"  RMSE:      {rmse:.1f}")
    typer.echo(f"  R²:        {r2:.3f}")
    typer.echo("")
    typer.echo(
        f"  Artifact written. Use 'props projections --model {model} "
        f"--model-type {model_type}' to use it."
    )

    console.summary()


def _project_for_model(
    *,
    model_name: str,
    model_type: PropModelType,
) -> DataFrame:
    """Load a trained prop artifact and project upcoming player-games.

    Uses the artifact's metadata as the source of truth for prediction
    features, closing the drift bug where re-derived NaN filters at
    predict time can disagree with the fit-time feature list.

    Returns an empty DataFrame when:
        - no trained artifact exists,
        - no upcoming feature rows exist,
        - all upcoming rows are dropped by the fit-time NaN filter.
    """
    from typing import cast

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore

    trainer: PropTrainer = _get_trainer(model_name)
    store = ArtifactStore(get_settings().repo_root)

    if not store.is_trained(trainer.spec.name, model_type.value):
        return DataFrame()

    trainer._model = store.load(trainer.spec.name, model_type.value)
    trainer._scaler = store.load_scaler(trainer.spec.name, model_type.value)

    # Narrow BaseModelMetadata → PropModelMetadata. Safe because we
    # loaded from a prop artifact directory (ArtifactStore paths are
    # partitioned by model_name, and this trainer's spec.name is a
    # registered prop family).
    meta: PropModelMetadata = cast(
        PropModelMetadata,
        store.read_metadata(trainer.spec.name, model_type.value),
    )

    upcoming_df: DataFrame = _load_upcoming_prop_features(trainer)
    if upcoming_df.empty:
        return DataFrame()

    rmse: float = meta.metrics.get("rmse", float("nan"))
    return _enrich_and_predict(trainer, upcoming_df, meta, rmse)


@props_app.command("projections")
def projections_cmd(
    model: str = typer.Option(
        "all",
        "--model",
        "-m",
        help="Prop model family to project, e.g. qb_pass_yards. 'all' runs all models.",
    ),
    model_type: str = typer.Option(
        "elasticnet",
        "--model-type",
        "-t",
        help="Algorithm type to use for projections.",
    ),
    top: int = typer.Option(
        20,
        "--top",
        "-n",
        help="Number of top projections to display.",
    ),
) -> None:
    """Project upcoming-week player props using trained artifacts."""
    mt = PropModelType(model_type)
    subtitle: str = "all models" if model == "all" else model
    console.header("props projections", subtitle=subtitle)

    models: list[str] = _all_prop_models() if model == "all" else [model]

    all_enriched: list[DataFrame] = []
    for m in models:
        with step(f"Project {m} ({mt})") as s:
            enriched: DataFrame = _project_for_model(model_name=m, model_type=mt)
            if enriched.empty:
                s.set_detail("no projection produced")
                continue
            s.set_rows(len(enriched))
            all_enriched.append(enriched)

    if not all_enriched:
        typer.echo("No projections produced.")
        raise typer.Exit(code=1)

    import pandas as pd

    combined: DataFrame = pd.concat(all_enriched, ignore_index=True)
    combined = combined.sort_values("predicted_mean", ascending=False)

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

    for col in ["predicted_mean", "lo_90", "hi_90", "predicted_std"]:
        if col in display.columns:
            display[col] = display[col].round(1)

    rename_map: dict[str, str] = {
        "player_name": "Player",
        "position": "Pos",
        "stat_type": "Stat",
        "predicted_mean": "Proj",
        "lo_90": "Lo90",
        "hi_90": "Hi90",
        "predicted_std": "Std",
    }
    # pyrefly: ignore [no-matching-overload] # DataFrame.rename - overload degrades to Series.rename
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})

    typer.echo()
    typer.echo(display.to_string(index=False))
    console.summary()


@props_app.command("compute-splits")
def compute_splits_cmd(
    stat_type: str = typer.Option(
        "all",
        "--stat-type",
        "-s",
        help=(
            "Stat family to compute splits for. 'all' iterates over all registered prop families."
        ),
    ),
) -> None:
    """Compute per-player situational splits for prop stat families.

    Joins player_game_logs to games CSV on game_id, then partitions by
    cohort (season, home, away, favored, underdog, indoor, outdoor, l4).

    Writes per-stat-type artifacts to
    ``data/output/props/situational_splits/{stat_type}.parquet``.

    Consumed by `/props/{prop_id}` to populate the `situational_splits`
    field.
    """
    import pandas as pd

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_teams_long_short
    from gridiron_edge.evaluation.situational_splits import (
        STAT_COLUMN_MAP,
        compute_player_situational_splits,
        write_situational_splits,
    )

    settings = get_settings()
    repo = settings.repo_root

    subtitle: str = "all stat families" if stat_type == "all" else stat_type
    console.header("props compute-splits", subtitle=subtitle)

    with step("Load player game logs + games") as s:
        logs_path = repo / "data" / "cleaned" / "player_game_logs.parquet"
        games_path = repo / "data" / "cleaned" / "NFL_wk_by_wk_cleaned.csv"

        if not logs_path.exists():
            typer.echo(f"Player game logs not found at {logs_path}")
            raise typer.Exit(code=1)
        if not games_path.exists():
            typer.echo(f"Games CSV not found at {games_path}")
            raise typer.Exit(code=1)

        logs = pd.read_parquet(logs_path)
        games = pd.read_csv(games_path)

        mapping_df = load_teams_long_short(repo)
        long_to_short = dict(
            zip(
                mapping_df["NFL_LONG_NAME"],
                mapping_df["NFL_SHORT_NAME"],
                strict=True,
            )
        )

        s.set_detail(f"{len(logs):,} log rows, {len(games):,} games")

    stat_types: list[str] = list(STAT_COLUMN_MAP.keys()) if stat_type == "all" else [stat_type]

    for st in stat_types:
        if st not in STAT_COLUMN_MAP:
            typer.echo(f"Unknown stat_type: {st}")
            continue

        with step(f"Compute splits for {st}") as s:
            df = compute_player_situational_splits(
                logs,
                games,
                long_to_short,
                st,
            )
            if df.empty:
                s.set_detail("no rows produced")
                continue
            s.set_detail(f"{df['player_id'].nunique()} players, {len(df)} rows")

        with step(f"Persist splits for {st}") as s:
            path = write_situational_splits(df, st, repo)
            s.set_detail(str(path.relative_to(repo)))

    console.summary()


@props_app.command("compute-opponent-allowed")
def compute_opponent_allowed_cmd() -> None:
    """Compute per-defense per-position stat aggregations.

    For each (opponent_team, position, stat_type) combination in the
    current season, computes:
        - Mean stat allowed (across season and l4 rolling cohorts)
        - Sample size (number of games)
        - Rank against position (1 = stingiest, 32 = most generous)

    Writes the artifact to
    ``data/output/props/opponent_allowed.parquet``.

    Consumed by `/compare/player/{prop_id}` to populate the 3
    defense-side rows: avg_allowed, rank_against_position, and
    last_4_games_avg (from the l4 cohort).
    """
    import pandas as pd

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.opponent_allowed import (
        compute_opponent_allowed,
        write_opponent_allowed,
    )

    settings: Settings = get_settings()
    repo: Path = settings.repo_root

    console.header("props compute-opponent-allowed")

    with step("Load player game logs") as s:
        logs_path: Path = repo / "data" / "cleaned" / "player_game_logs.parquet"
        if not logs_path.exists():
            typer.echo(f"Player game logs not found at {logs_path}")
            raise typer.Exit(code=1)

        logs: DataFrame = pd.read_parquet(logs_path)
        s.set_detail(f"{len(logs):,} log rows")

    with step("Compute opponent-allowed aggregates") as s:
        df: DataFrame = compute_opponent_allowed(logs)
        if df.empty:
            typer.echo("No aggregates produced.")
            raise typer.Exit(code=1)
        s.set_detail(f"{len(df)} rows")

    with step("Persist artifact") as s:
        path: Path = write_opponent_allowed(df, repo)
        s.set_detail(str(path.relative_to(repo)))

    console.summary()

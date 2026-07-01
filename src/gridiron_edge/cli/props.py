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
    enriched = _enrich_predictions_for_holdout(
        trainer, holdout_df, usable, meta.metrics.get("rmse", float("nan"))
    )
    return enriched, meta.metrics.get("rmse", float("nan"))


def _walk_forward_predict_for_season(
    *,
    model_name: str,
    model_type: PropModelType,
    season: int,
    features_df: DataFrame,
) -> tuple[DataFrame, float]:
    """Train through ``season`` and predict that season's player-games.

    Used by ``backfill_cmd`` to walk forward across the historical
    range. Predictions are enriched with the standard post-process
    columns so the result is archive-ready.

    Args:
        model_name: Prop family name (e.g. ``"qb_pass_yards"``).
        model_type: Algorithm to use.
        season: Integer season label. Becomes the cutoff and the
            single prediction window.
        features_df: Pre-built features DataFrame containing all
            seasons. Sliced by ``season`` for the prediction step
            to avoid rebuilding features for each iteration.

    Returns:
        Tuple of (enriched predictions DataFrame, model RMSE).
    """
    trainer: PropTrainer = _get_trainer(model_name)

    meta: PropModelMetadata = trainer.train_through(
        cutoff_season=season,
        model_type=model_type,
    )

    season_df: DataFrame = features_df.loc[features_df["season"] == season, :].copy()

    feature_cols: list[str] = trainer._feature_columns()
    available: list[str] = [c for c in feature_cols if c in season_df.columns]
    nan_rates: Series = season_df[available].isna().mean()
    usable: list[str] = [c for c in available if nan_rates[c] < 0.5]

    target: str = trainer.spec.target_col
    season_df = season_df.dropna(subset=[*usable, target])

    if season_df.empty:
        return DataFrame(), meta.metrics.get("rmse", float("nan"))

    enriched: DataFrame = _enrich_predictions_for_holdout(
        trainer,
        season_df,
        usable,
        meta.metrics.get("rmse", float("nan")),
    )

    return enriched, meta.metrics.get("rmse", float("nan"))


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

        results: list[RegressionModelResult] = []

        for mt in model_types:
            with step(f"Load archive for {m} ({mt})") as s:
                from gridiron_edge.evaluation.prop_archive import (
                    build_prop_evaluation_df,
                )

                eval_df: DataFrame = build_prop_evaluation_df(
                    model_name=m,
                    model_type=mt.value,
                    season=season,
                )
                if eval_df.empty:
                    s.set_detail("no archive rows - skipping")
                    continue

                s.set_rows(len(eval_df))

            with step(f"Evaluate {m} ({mt})") as s:
                report: PropEvalReport = evaluate_prop_model(
                    model_name=m,
                    actual=eval_df["actual"],
                    predicted_mean=eval_df["predicted_mean"],
                    predicted_std=eval_df.get("predicted_std"),
                    lo_90=eval_df.get("lo_90"),
                    hi_90=eval_df.get("hi_90"),
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

                s.set_detail(f"MAE={result.mae:.1f}  RMSE={result.rmse:.1f}  R²={result.r2:.3f}")

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
        from gridiron_edge.cli._composites import write_champion_manifest
        from gridiron_edge.core.settings import get_settings

        write_champion_manifest(get_settings().repo_root)


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

    Returns an empty DataFrame when:
        - no trained artifact exists,
        - no upcoming feature rows exist,
        - all upcoming rows are dropped by NaN filtering.

    Callers should accumulate non-empty results and present them as a
    single projection table.
    """
    import numpy as np

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.prop_prediction.post_process import (
        TARGET_STD_MAP,
        enrich_prop_predictions,
    )

    trainer: PropTrainer = _get_trainer(model_name)
    store = ArtifactStore(get_settings().repo_root)

    if not store.is_trained(trainer.spec.name, model_type.value):
        return DataFrame()

    artifact = store.load(trainer.spec.name, model_type.value)
    scaler = store.load_scaler(trainer.spec.name, model_type.value)

    upcoming_df: DataFrame = _load_upcoming_prop_features(trainer)
    if upcoming_df.empty:
        return DataFrame()

    feature_cols: list[str] = trainer._feature_columns()
    available: list[str] = [c for c in feature_cols if c in upcoming_df.columns]
    nan_rates: Series = upcoming_df[available].isna().mean()
    usable: list[str] = [c for c in available if nan_rates[c] < 0.5]

    upcoming_clean: DataFrame = upcoming_df.dropna(subset=usable).copy()
    if upcoming_clean.empty:
        return DataFrame()

    trainer._model = artifact
    trainer._scaler = scaler

    preds: ndarray = trainer._predict(upcoming_clean.loc[:, usable])

    enriched_input: DataFrame = upcoming_clean.copy()
    enriched_input["predicted_mean"] = preds
    enriched_input["stat_type"] = trainer.spec.name

    target: str = trainer.spec.target_col
    std_col: str = TARGET_STD_MAP.get(trainer.spec.name, f"{target}_L3_std")
    if std_col not in enriched_input.columns:
        enriched_input[std_col] = np.nan

    enriched: DataFrame = enrich_prop_predictions(
        df=enriched_input,
        model_rmse=float("nan"),
        target_std_col=std_col,
    )

    return enriched


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

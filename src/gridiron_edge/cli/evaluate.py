# src/gridiron_edge/cli/evaluate.py
"""CLI commands for model evaluation.

Composite model identity:
    Model identity uses the composite ``model_key`` - ``f"{model_name}_{model_type}"``
    matching the ``ModelRegistry`` keys (e.g. ``"win_prob_random_forest"``,
    ``"total_xgboost"``). The ``backfill`` command takes ``--model-name`` +
    ``--model-type`` as separate options since it calls into ``backfill_model``
    directly; all other commands accept a single ``--model-key`` for display
    and filtering purposes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, LiteralString

from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._composites import write_champion_manifest
from gridiron_edge.evaluation.select import (
    collect_latest_forecast_run_metrics as _collect_latest_forecast_run_metrics,
)
from gridiron_edge.evaluation.select import (
    compute_report_data as _compute_report_data,
)
from gridiron_edge.evaluation.select import (
    latest_backfilled_run_id as _latest_backfilled_run_id,
)
from gridiron_edge.evaluation.select import (
    rank_models as _rank_models,
)

if TYPE_CHECKING:
    from gridiron_edge.evaluation.backfill import BackfillResult


evaluate_app = typer.Typer(
    help="Evaluate model predictions against outcomes.", no_args_is_help=True
)


def _registered_game_model_keys() -> list:
    """Return registered composite keys owned by game-model evaluation."""
    from gridiron_edge.models.game_prediction.model import (
        get_known_model_names,
    )
    from gridiron_edge.models.registry import (
        ModelRegistry,
    )

    prefixes = tuple(f"{model_name}_" for model_name in get_known_model_names())
    return [key for key in ModelRegistry.names() if key.startswith(prefixes)]


def _split_composite_key(key: str) -> tuple[str | None, str | None]:
    """Split a composite model_key into (model_name, model_type) for filtering.

    Matches against the known model_name prefixes returned by
    :func:`gridiron_edge.models.game_prediction.model.get_known_model_names`.
    Returns ``(None, None)`` when the key is ``"all"`` so callers can skip
    the filter entirely.

    Args:
        key: Composite registry key (e.g. ``"win_prob_random_forest"``)
            or ``"all"``.

    Returns:
        Tuple of ``(model_name, model_type)``. Both ``None`` if ``key == "all"``.
        Raises if ``key`` doesn't match any known prefix.
    """
    from gridiron_edge.models.game_prediction.model import get_known_model_names

    if key == "all":
        return None, None
    known_names = get_known_model_names()
    for model_name in known_names:
        prefix = f"{model_name}_"
        if key.startswith(prefix):
            return model_name, key[len(prefix) :]
    raise typer.BadParameter(
        f"Model key {key!r} is not a valid composite key. "
        f"Expected format: '{{model_name}}_{{model_type}}'. "
        f"Known model_names: {sorted(known_names)}."
    )


def _resolved_run_id_or_exit(target_key: str, *, repo: Path) -> str:
    """Resolve one composite key's latest backfilled run_id or exit with a message."""
    target_model_name, target_model_type = _split_composite_key(target_key)
    run_id = (
        _latest_backfilled_run_id(target_model_name, target_model_type, repo=repo)
        if target_model_name is not None and target_model_type is not None
        else None
    )
    if run_id is None:
        typer.echo(f"No backfilled run for {target_key!r}.")
        raise typer.Exit(1)
    return run_id


def _latest_run_evaluation_df(
    model_name: str,
    model_type: str,
    *,
    repo: Path,
) -> DataFrame:
    """Evaluate one family's most recent immutable backfill run, if any.

    Returns an empty DataFrame if the family has never been backfilled -
    the caller decides how to report that.
    """
    from gridiron_edge.evaluation.metrics import build_forecast_run_evaluation_df

    run_id = _latest_backfilled_run_id(model_name, model_type, repo=repo)
    if run_id is None:
        return DataFrame()
    return build_forecast_run_evaluation_df(
        run_id=run_id,
        model_name=model_name,
        model_type=model_type,
        repo=repo,
    )


def _latest_run_evaluation_df_for_keys(
    keys: list[str],
    *,
    repo: Path,
) -> DataFrame:
    """Concatenate each key's latest-run evaluation frame; skips families with no run."""
    import pandas as pd

    frames: list[DataFrame] = []
    for key in keys:
        name_filter, type_filter = _split_composite_key(key)
        if name_filter is None or type_filter is None:
            continue
        frame = _latest_run_evaluation_df(name_filter, type_filter, repo=repo)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return DataFrame()
    return pd.concat(frames, ignore_index=True)


@evaluate_app.command("summary")
def evaluate_summary(
    *,
    model_key: str = typer.Option(
        "all",
        help=(
            "Composite model key to evaluate (e.g. 'win_prob_random_forest'), "
            "or 'all' to compare all models."
        ),
    ),
    season: str | None = typer.Option(None, help="Filter to a specific season e.g. '2025-2026'."),
    group_by: str = typer.Option(
        "season",
        help="Group results by: season, week, model_name, or model_type.",
    ),
) -> None:
    r"""Print prediction accuracy summary for each family's latest backfill run.

    \b
    Examples:
      gridiron evaluate summary
      gridiron evaluate summary --season 2025-2026
      gridiron evaluate summary --group-by week
      gridiron evaluate summary --model-key win_prob_random_forest
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.metrics import summarise

    repo: Path = get_settings().repo_root
    subtitle: str = f"model={model_key}"
    if season:
        subtitle += f"  season={season}"
    subtitle += f"  group={group_by}"
    console.header("evaluate summary", subtitle=subtitle)

    with step("Evaluate each family's latest backfill run") as s:
        if model_key == "all":
            df_eval: DataFrame = _latest_run_evaluation_df_for_keys(
                _registered_game_model_keys(), repo=repo
            )
        else:
            df_eval = _latest_run_evaluation_df_for_keys([model_key], repo=repo)
        if season is not None and not df_eval.empty:
            df_eval = df_eval.loc[df_eval["season"] == season].reset_index(drop=True)
        if df_eval.empty:
            s.set_detail("no backfilled runs - run 'evaluate backfill' first")
        else:
            s.set_detail(f"{len(df_eval)} games")

    if not df_eval.empty:
        with step("Compute metrics") as s:
            df_summary: DataFrame = summarise(df_eval, group_by=group_by)
            s.set_detail(f"{len(df_summary)} row(s)")

        typer.echo(df_summary.to_string(index=False))

    console.summary()


@evaluate_app.command("calibration")
def evaluate_calibration(
    *,
    model_key: str = typer.Option(
        "win_prob_elo",
        help=(
            "Composite model key to evaluate (e.g. 'win_prob_random_forest'), "
            "or 'all' for all models combined."
        ),
    ),
    season: str | None = typer.Option(None, help="Filter to a specific season e.g. '2025-2026'."),
    buckets: int = typer.Option(10, help="Number of probability buckets (default 10)."),
) -> None:
    r"""Print calibration table showing predicted probability vs actual win rate.

    \b
    Examples:
      gridiron evaluate calibration
      gridiron evaluate calibration --season 2025-2026 --buckets 20
      gridiron evaluate calibration --model-key win_prob_random_forest
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.metrics import calibration_table

    repo: Path = get_settings().repo_root
    subtitle: str = f"model={model_key}"
    if season:
        subtitle += f"  season={season}"
    console.header("evaluate calibration", subtitle=subtitle)

    with step("Evaluate the family's latest backfill run") as s:
        if model_key == "all":
            df_eval: DataFrame = _latest_run_evaluation_df_for_keys(
                _registered_game_model_keys(), repo=repo
            )
        else:
            df_eval = _latest_run_evaluation_df_for_keys([model_key], repo=repo)
        if season is not None and not df_eval.empty:
            df_eval = df_eval.loc[df_eval["season"] == season].reset_index(drop=True)
        if df_eval.empty:
            s.set_detail("no backfilled runs - run 'evaluate backfill' first")
        else:
            s.set_detail(f"{len(df_eval)} games")

    if not df_eval.empty:
        with step("Build calibration table") as s:
            df_cal: DataFrame = calibration_table(df_eval, n_buckets=buckets)
            s.set_detail(f"{len(df_cal)} buckets")

        typer.echo(df_cal.to_string(index=False))

    console.summary()


def _render_backfill_result(result: BackfillResult) -> None:
    """Render structured historical reconstruction accounting."""
    import typer

    typer.echo("")
    typer.echo(f"Mode: {result.mode.value}")
    typer.echo(f"Run ID: {result.run_id or 'none'}")
    typer.echo(f"Generated events: {result.generated_count:,}")
    typer.echo(f"Inserted events: {result.inserted_count:,}")
    typer.echo(f"Existing events: {result.existing_count:,}")
    predicted = ", ".join(result.predicted_seasons) or "none"
    typer.echo(f"Predicted seasons: {predicted}")
    skipped = [item for item in result.seasons if item.reason is not None]
    if skipped:
        typer.echo("Skipped seasons:")
        for item in skipped:
            typer.echo(f"  {item.season}: {item.reason}")


@evaluate_app.command("backfill")
def evaluate_backfill(
    *,
    # Note: defaults are (win_prob, elo) as a historical convenience —
    # Elo is the cheapest and always-available model. This is not a
    # champion pick; users typically pass --model-name / --model-type
    # explicitly. To backfill the current champion, first read it via
    # `gridiron evaluate select-model` and pass the values explicitly.
    model_name: str = typer.Option(
        "win_prob",
        help="Model purpose (e.g. 'win_prob', 'total').",
    ),
    model_type: str = typer.Option(
        "elo",
        help="Model algorithm (e.g. 'random_forest', 'xgboost', 'logistic', 'elo').",
    ),
    mode: str = typer.Option(
        "auto",
        help=(
            "Backfill mode: 'walk-forward' (retrain per season - for ML models), "
            "'current-model' (use existing artifact - for analytic models like elo), "
            "or 'auto' (default per model)."
        ),
    ),
    start_season: str | None = typer.Option(
        None,
        help="First season to predict (walk-forward only), e.g. '2000-2001'.",
    ),
    end_season: str | None = typer.Option(
        None,
        help="Last season to predict (walk-forward only), e.g. '2024-2025'.",
    ),
) -> None:
    r"""Generate an immutable historical forecast run.

    By default, ML models (win_prob_logistic, win_prob_random_forest,
    win_prob_xgboost, total_random_forest, total_xgboost) use walk-forward
    backfill: for each season N, the model is retrained on data through
    N-1, then used to predict season N. Intermediate models are discarded.

    Analytic models (win_prob_elo) default to current-model backfill since
    Elo state is built chronologically and the current artifact produces
    honest historical predictions.

    \b
    Examples:
      gridiron evaluate backfill --model-name win_prob --model-type random_forest
      gridiron evaluate backfill --model-name win_prob --model-type elo
      gridiron evaluate backfill --model-name win_prob --model-type random_forest \
        --start-season 2010-2011
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.backfill import BackfillMode, backfill_model

    if mode not in {"auto", *(item.value for item in BackfillMode)}:
        raise typer.BadParameter(
            "Mode must be one of: auto, walk-forward, current-model.",
            param_hint="--mode",
        )
    resolved_mode = None if mode == "auto" else BackfillMode(mode)
    subtitle_parts: list[str] = [f"model={model_name}_{model_type}"]
    if resolved_mode is not None:
        subtitle_parts.append(f"mode={resolved_mode}")
    console.header("evaluate backfill", subtitle="  ".join(subtitle_parts))

    try:
        with step("Generate historical forecast events") as s:
            result = backfill_model(
                model_name=model_name,
                model_type=model_type,
                mode=resolved_mode,
                start_season=start_season,
                end_season=end_season,
            )
            s.set_detail(f"{result.inserted_count:,} forecast events written")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _render_backfill_result(result)
    console.summary()


@evaluate_app.command("tune")
def evaluate_tune(
    *,
    zone_k: bool = typer.Option(
        False,
        "--zone-k/--no-zone-k",
        help="Run zone-K search (one K-factor per week zone) instead of flat-K search.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply/--no-apply",
        help=(
            "After the search, generate a historical forecast run using the selected parameters."
        ),
    ),
    top: int = typer.Option(10, help="Number of top results to display."),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help=(
            "Save full results to data/output/tune/ as Parquet. "
            "Recommended for long runs - protects against terminal loss."
        ),
    ),
) -> None:
    r"""Grid search Elo parameters (K, divisor, regression) against a holdout set.

    Runs every combination of K-factor, win-probability divisor, and
    offseason regression fraction. Scores each on training seasons and
    held-out seasons (last 3 seasons) separately. Best holdout Brier
    score wins.

    Use --apply to immediately generate a historical forecast run.

    \b
    Examples:
      gridiron evaluate tune
      gridiron evaluate tune --top 20
      gridiron evaluate tune --apply
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.tune import (
        DIVISOR_VALUES,
        DIVISOR_VALUES_ZONE_K,
        HOLDOUT_SEASONS,
        K_EARLY_VALUES,
        K_MID_VALUES,
        K_POST_VALUES,
        K_VALUES,
        K_WEEK18_VALUES,
        REGRESS_VALUES,
        REGRESS_VALUES_ZONE_K,
        best_params,
        best_params_zone_k,
        run_grid_search,
        run_grid_search_zone_k,
    )

    if zone_k:
        n_combos: int = (
            len(K_EARLY_VALUES)
            * len(K_MID_VALUES)
            * len(K_WEEK18_VALUES)
            * len(K_POST_VALUES)
            * len(DIVISOR_VALUES_ZONE_K)
            * len(REGRESS_VALUES_ZONE_K)
        )
        subtitle: str = f"zone-K  {n_combos} combinations  holdout={sorted(HOLDOUT_SEASONS)}"
    else:
        n_combos = len(K_VALUES) * len(DIVISOR_VALUES) * len(REGRESS_VALUES)
        subtitle = f"flat-K  {n_combos} combinations  holdout={sorted(HOLDOUT_SEASONS)}"
    console.header("evaluate tune", subtitle=subtitle)

    with step(f"Run grid search ({n_combos} combinations)") as s:
        from gridiron_edge.core.settings import get_settings

        _out_dir: Path = get_settings().data_output / "tune"
        _fname: Literal["flat_k_tune_results.parquet", "zone_k_tune_results.parquet"] = (
            "zone_k_tune_results.parquet" if zone_k else "flat_k_tune_results.parquet"
        )
        _save: Path | None = _out_dir / _fname if save else None
        results: DataFrame = (
            run_grid_search_zone_k(save_path=_save) if zone_k else run_grid_search(save_path=_save)
        )
        best_holdout = results.iloc[0]["holdout_brier"]
        s.set_detail(f"best holdout Brier: {best_holdout:.5f}")

    with step(f"Top {top} results") as s:
        s.set_detail(f"{len(results)} total combinations scored")

    typer.echo(results.head(top).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if apply:
        if zone_k:
            params: dict[str, float] = best_params_zone_k(results)
            typer.echo(
                f"Applying best zone-K params: "
                f"k_early={params['k_early']:.0f}  k_mid={params['k_mid']:.0f}  "
                f"k_week18={params['k_week18']:.0f}  k_post={params['k_post']:.0f}  "
                f"divisor={params['divisor']:.0f}  regress={params['regress_frac']:.2f}"
            )
            with step("Backfill predictions") as s:
                from gridiron_edge.evaluation.backfill import backfill_model

                # Intentional: `evaluate tune` searches Elo hyperparameters
                # (K, divisor, regression_frac). After the search, the newly
                # tuned Elo model is backfilled to the archive. Not a
                # champion decision — the model just got tuned.
                result = backfill_model(
                    model_name="win_prob",
                    model_type="elo",
                )
                s.set_detail(
                    f"{result.inserted_count:,} backfilled forecast events written as win_prob/elo"
                )
        else:
            params = best_params(results)
            k_val: float = params["k"]
            div_val: float = params["divisor"]
            reg_val: float = params["regress_frac"]
            typer.echo(
                f"Applying best flat-K params: "
                f"k={k_val:.0f}  divisor={div_val:.0f}  regress={reg_val:.2f}"
            )
            with step("Backfill predictions") as s:
                from gridiron_edge.evaluation.backfill import backfill_model

                # Intentional: `evaluate tune` searches Elo hyperparameters
                # (K, divisor, regression_frac). After the search, the newly
                # tuned Elo model is backfilled to the archive. Not a
                # champion decision — the model just got tuned.
                result = backfill_model(
                    model_name="win_prob",
                    model_type="elo",
                )
                s.set_detail(
                    f"{result.inserted_count:,} backfilled forecast events written as win_prob/elo"
                )

    console.summary()


@evaluate_app.command("diagnostics")
def evaluate_diagnostics(
    *,
    model_key: str = typer.Option(
        "all",
        help=(
            "Composite model key to diagnose (e.g. 'win_prob_random_forest'), "
            "or 'all' for multi-model comparison."
        ),
    ),
    compare: bool = typer.Option(
        False,
        "--compare/--no-compare",
        help="Generate multi-model comparison plots for all registered models.",
    ),
) -> None:
    r"""Generate diagnostic plots for one or all models.

    Single-model mode (--model-key X):
        Saves calibration curve, confidence distribution, ROC curve,
        Brier decomposition, performance by context, and feature
        importance to data/output/evaluation/{model_key}/.

    Comparison mode (--compare):
        Overlays all models on shared calibration, ROC, and metric
        bar charts. Saves to data/output/evaluation/.

    \b
    Examples:
      gridiron evaluate diagnostics --model-key win_prob_random_forest
      gridiron evaluate diagnostics --compare
      gridiron evaluate diagnostics --model-key win_prob_random_forest --compare
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.diagnostics import (
        plot_model_comparison,
        plot_single_model,
    )

    repo: Path = get_settings().repo_root
    subtitle_parts: list[str] = []
    if model_key != "all":
        subtitle_parts.append(f"model={model_key}")
    if compare:
        subtitle_parts.append("compare")
    console.header("evaluate diagnostics", subtitle="  ".join(subtitle_parts) or "all")

    # Single-model diagnostics
    if model_key != "all":
        with step(f"Evaluate latest backfill run - {model_key}") as s:
            df_eval: DataFrame = _latest_run_evaluation_df_for_keys([model_key], repo=repo)
            if df_eval.empty:
                s.set_detail("no data - run evaluate backfill first")
                raise typer.Exit(1)
            s.set_detail(f"{len(df_eval):,} games")

        with step(f"Generate single-model plots - {model_key}") as s:
            paths: list[Path] = plot_single_model(df_eval, repo=repo)
            s.set_detail(f"{len(paths)} plots")
            for p in paths:
                typer.echo(f"  {p.relative_to(repo)}")

    # Multi-model comparison
    if compare:
        import gridiron_edge.models.elo.model
        import gridiron_edge.models.game_prediction.model  # noqa: F401

        all_keys: list[str] = _registered_game_model_keys()

        with step("Evaluate latest backfill runs - all models") as s:
            eval_dfs: dict = {}
            for key in all_keys:
                df: DataFrame = _latest_run_evaluation_df_for_keys([key], repo=repo)
                if not df.empty:
                    eval_dfs[key] = df
            s.set_detail(f"{len(eval_dfs)} models with data")

        if len(eval_dfs) < 2:
            typer.echo("Need at least 2 models with archived predictions for comparison.")
            raise typer.Exit(1)

        with step("Generate comparison plots") as s:
            paths = plot_model_comparison(eval_dfs, repo=repo)
            s.set_detail(f"{len(paths)} plots")
            for p in paths:
                typer.echo(f"  {p.relative_to(repo)}")

    console.summary()


@evaluate_app.command("select-model")
def evaluate_select_model(
    *,
    top: int = typer.Option(
        None,
        help="Show only the top N models. Defaults to all.",
    ),
    criteria: str = typer.Option(
        "brier,ece,auc",
        help=(
            "Comma-separated ordered criteria for ranking. "
            "Options: brier, ece, auc, accuracy, log_loss. "
            "First criterion is primary tiebreaker."
        ),
    ),
    write_manifest: bool = typer.Option(
        False,
        "--write-manifest",
        help=(
            "After ranking, persist champion decisions to the manifest at "
            "data/output/champions/champions.json. Runs all three selectors "
            "(game classification, game regression, prop) so the manifest "
            "reflects the full repo state. Preserves entries for model "
            "families outside the current retrain scope."
        ),
    ),
) -> None:
    r"""Rank all registered models and recommend the best for production use.

    Evaluates every model with archived predictions against four criteria:
        Brier score   (lower is better)  - overall prediction accuracy
        ECE           (lower is better)  - calibration quality
        ROC-AUC       (higher is better) - ranking quality
        Accuracy      (higher is better) - fraction of winners correctly called

    Each model is ranked on each criterion and the ranks are summed to
    produce a composite score. The model with the lowest composite rank
    wins. Ties are broken by the first listed criterion.

    \b
    Examples:
        gridiron evaluate select-model
        gridiron evaluate select-model --criteria brier,ece,auc
        gridiron evaluate select-model --top 3
        gridiron evaluate select-model --write-manifest

    """
    import pandas as pd

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    import gridiron_edge.models.elo.model
    import gridiron_edge.models.game_prediction.model  # noqa: F401

    repo: Path = get_settings().repo_root
    console.header("evaluate select-model")

    # Parse criteria
    valid_criteria: set[str] = {"brier", "ece", "auc", "accuracy", "log_loss"}
    criteria_list: list[str] = [c.strip().lower() for c in criteria.split(",")]
    invalid: list[str] = [c for c in criteria_list if c not in valid_criteria]
    if invalid:
        typer.echo(f"Unknown criteria: {invalid}. Valid: {sorted(valid_criteria)}")
        raise typer.Exit(1)

    # Compute metrics for all models with a backfilled run
    with step("Compute metrics for all models") as s:
        rows: list[dict] = _collect_latest_forecast_run_metrics(
            _registered_game_model_keys(),
            repo=repo,
        )
        s.set_detail(f"{len(rows)} models evaluated")

    if not rows:
        typer.echo("No models with a backfilled run found. Run evaluate backfill first.")
        raise typer.Exit(1)

    df = pd.DataFrame(rows)

    # Rank each criterion (lower rank = better)
    # For lower-is-better metrics: rank ascending
    # For higher-is-better metrics: rank descending
    lower_is_better: set[str] = {"brier", "ece", "log_loss"}

    rank_cols: list[str] = []
    for criterion in criteria_list:
        rank_col: str = f"rank_{criterion}"
        ascending: bool = criterion in lower_is_better
        df[rank_col] = df[criterion].rank(ascending=ascending, method="min").astype(int)
        rank_cols.append(rank_col)

    # pyrefly: ignore [bad-argument-type]  # DataFrame.sum(axis=1) - overload degrades to Series.sum
    df["composite_rank"] = df[rank_cols].sum(axis=1)
    df: DataFrame = df.sort_values(
        ["composite_rank", criteria_list[0]],
        ascending=[True, criteria_list[0] in lower_is_better],
    ).reset_index(drop=True)

    if top is not None:
        df = df.head(top)

    # Display results
    display_cols: list[str] = [
        "model_key",
        "n_games",
        "brier",
        "ece",
        "auc",
        "accuracy",
        *rank_cols,
        "composite_rank",
    ]
    typer.echo("")
    typer.echo(df[display_cols].to_string(index=False))

    # Recommendation
    best: Series = df.iloc[0]
    typer.echo("")
    typer.echo(f"Recommendation: {best['model_key']}")

    reasons: list[str] = []
    for criterion in criteria_list:
        rank_col = f"rank_{criterion}"
        if best[rank_col] == 1:
            direction: Literal["highest", "lowest"] = (
                "lowest" if criterion in lower_is_better else "highest"
            )
            reasons.append(f"{criterion.upper()}={best[criterion]:.5f} ({direction})")
    if reasons:
        typer.echo(f"Reason: Best {', '.join(reasons)}")

    # Flag if recommendation differs from current production model
    typer.echo("")
    typer.echo(
        "Note: Recommendation is based on full historical evaluation. "
        "For production use, also review calibration plots "
        "('gridiron evaluate diagnostics --compare') before switching models."
    )

    if write_manifest:
        write_champion_manifest(repo)

    console.summary()


@evaluate_app.command("prune-champions")
def evaluate_prune_champions() -> None:
    """Remove champion-manifest entries with no corresponding persisted artifact.

    Reads data/output/champions/champions.json and drops any entry whose
    (model_name, model_type) has no trained artifact under data/models/.
    Champions backed by a real artifact are preserved unchanged, including
    their own promoted_at, source_run_id, and metrics.

    Unlike `select-model --write-manifest`, this does not re-rank or
    re-promote any model - it only removes entries that reference a model
    that was never actually trained and persisted.
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.champion_resolver import prune_orphaned_champions

    repo: Path = get_settings().repo_root
    console.header("evaluate prune-champions")

    with step("Prune manifest") as s:
        result = prune_orphaned_champions(repo=repo)
        s.set_detail(f"{len(result.removed)} removed")

    typer.echo("")
    if result.removed:
        for model_name, entry in sorted(result.removed.items()):
            typer.echo(f"  removed: {model_name} ({entry['model_type']}) - no trained artifact")
    else:
        typer.echo("  nothing to prune - every champion has a trained artifact")

    console.summary()


@evaluate_app.command("model-report")
def evaluate_model_report(
    *,
    win_elo_run: str = typer.Option(..., help="Exact backfill run_id for win_prob/elo."),
    win_logistic_run: str = typer.Option(..., help="Exact backfill run_id for win_prob/logistic."),
    win_random_forest_run: str = typer.Option(
        ..., help="Exact backfill run_id for win_prob/random_forest."
    ),
    win_xgboost_run: str = typer.Option(..., help="Exact backfill run_id for win_prob/xgboost."),
    total_random_forest_run: str = typer.Option(
        ..., help="Exact backfill run_id for total/random_forest."
    ),
    total_xgboost_run: str = typer.Option(..., help="Exact backfill run_id for total/xgboost."),
    select: bool = typer.Option(
        True,
        "--select/--no-select",
        help="Select this report as current after building it.",
    ),
) -> None:
    r"""Build one immutable evaluation report across all six game-model families.

    Requires an exact backfill run_id for each of the six families (Win:
    elo, logistic, random_forest, xgboost; Total: random_forest, xgboost) so
    every family is evaluated on a caller-verified common game set. Fails
    closed if the six families' game sets disagree - use `gridiron evaluate
    backfill` to produce aligned runs first.

    \b
    Examples:
      gridiron evaluate model-report \\
        --win-elo-run <run-id> --win-logistic-run <run-id> \\
        --win-random-forest-run <run-id> --win-xgboost-run <run-id> \\
        --total-random-forest-run <run-id> --total-xgboost-run <run-id>
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.game_model_evaluation_report_builder import (
        build_and_write_game_model_evaluation_report,
    )
    from gridiron_edge.evaluation.game_model_evaluation_report_selection import (
        select_current_game_model_evaluation_report,
    )

    repo: Path = get_settings().repo_root
    console.header("evaluate model-report")

    win_run_ids: dict[str, str] = {
        "elo": win_elo_run,
        "logistic": win_logistic_run,
        "random_forest": win_random_forest_run,
        "xgboost": win_xgboost_run,
    }
    total_run_ids: dict[str, str] = {
        "random_forest": total_random_forest_run,
        "xgboost": total_xgboost_run,
    }
    generated_at = datetime.now(UTC)

    try:
        with step("Evaluate all six families on a common game set") as s:
            result = build_and_write_game_model_evaluation_report(
                win_run_ids=win_run_ids,
                total_run_ids=total_run_ids,
                generated_at=generated_at,
                repo=repo,
            )
            s.set_detail(f"{result.report.game_count} games")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo("")
    typer.echo(f"Report ID: {result.report.report_id}")
    typer.echo(f"Games: {result.report.game_count}")
    typer.echo("")
    typer.echo("Win families:")
    for metric in result.report.win_metrics:
        typer.echo(
            f"  {metric.model_type}: brier={metric.brier} ece={metric.ece} "
            f"auc={metric.auc} calibration_slope={metric.calibration_slope} "
            f"calibration_intercept={metric.calibration_intercept} "
            f"sharpness={metric.sharpness} season_stability={metric.season_stability}"
        )
    typer.echo("Total families:")
    for metric in result.report.total_metrics:
        typer.echo(
            f"  {metric.model_type}: mae={metric.mae} "
            f"median_absolute_error={metric.median_absolute_error} rmse={metric.rmse} "
            f"actual_coverage={metric.actual_coverage} (nominal={metric.nominal_coverage})"
        )

    if select:
        select_current_game_model_evaluation_report(
            result.report.report_id,
            selected_at=generated_at,
            repo=repo,
        )
        typer.echo("")
        typer.echo(f"Selected as current: {result.report.report_id}")

    console.summary()


@evaluate_app.command("explain-logistic")
def evaluate_explain_logistic(
    *,
    run_id: str = typer.Option(
        ..., "--run-id", help="Exact run_id whose win_prob/logistic evidence to explain."
    ),
) -> None:
    r"""Persist one immutable Logistic explanation evidence batch for one run.

    Reconstructs exact scaled-feature-by-coefficient log-odds contributions
    from the run's already-persisted win_prob/logistic prediction-input
    evidence, reloading the model from its exact immutable binary snapshot
    (never the live, mutable model store). Fails closed if the run has no
    win_prob/logistic evidence or if any event's reconstructed probability
    does not reconcile with its persisted estimator output within tolerance.

    \b
    Example:
      gridiron evaluate explain-logistic --run-id <run-id>
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.logistic_explanation_evidence_builder import (
        build_and_write_logistic_explanation_batch,
    )

    repo: Path = get_settings().repo_root
    console.header("evaluate explain-logistic")

    try:
        with step("Reconstruct log-odds contributions from persisted evidence") as s:
            result = build_and_write_logistic_explanation_batch(
                run_id=run_id,
                generated_at=datetime.now(UTC),
                repo=repo,
            )
            s.set_detail(f"{result.event_count} events")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo("")
    typer.echo(f"Batch ID: {result.batch.batch_id}")
    typer.echo(f"Evidence ID: {result.batch.evidence_id}")
    typer.echo(f"Events: {result.event_count}")
    typer.echo(f"Max reconciliation error: {result.max_reconciliation_error:.3e}")

    console.summary()


@evaluate_app.command("build-comparable-corpus")
def evaluate_build_comparable_corpus() -> None:
    r"""Build and persist the win_prob/logistic comparable-games corpus.

    Replays the exact feature-construction pipeline live prediction and
    walk-forward evaluation already use (the currently deployed champion's
    own epa_window and feature set) over every historical game, scales the
    result with the champion's own fitted scaler, and derives a distance
    threshold empirically from leave-one-out nearest-neighbor distances.
    Idempotent: rebuilding against unchanged model/scaler bytes and
    modeling data reproduces the same corpus_id.

    \b
    Example:
      gridiron evaluate build-comparable-corpus
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.comparable_games_corpus_builder import (
        build_and_write_comparable_games_corpus,
    )

    repo: Path = get_settings().repo_root
    console.header("evaluate build-comparable-corpus")

    try:
        with step("Replay feature construction and derive empirical threshold") as s:
            result = build_and_write_comparable_games_corpus(
                generated_at=datetime.now(UTC),
                repo=repo,
            )
            s.set_detail(f"{result.row_count} games")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo("")
    corpus = result.corpus
    typer.echo(f"Corpus ID: {corpus.corpus_id}")
    typer.echo(f"Games: {result.row_count}")
    pctl = corpus.leave_one_out_percentile
    typer.echo(f"Distance threshold (p{pctl:.0f}): {corpus.distance_threshold:.4f}")
    typer.echo(f"Leave-one-out median distance: {corpus.leave_one_out_median_distance:.4f}")

    console.summary()


@evaluate_app.command("find-comparables")
def evaluate_find_comparables(
    *,
    run_id: str = typer.Option(
        ..., "--run-id", help="Exact run_id whose win_prob/logistic evidence to match."
    ),
    k: int = typer.Option(20, "--k", help="Maximum comparables requested per event."),
) -> None:
    r"""Persist comparable-games retrieval evidence for one run's Win events.

    Matches each win_prob/logistic event already persisted in the run's
    prediction-input evidence against the comparable-games corpus by
    Euclidean distance in the champion's own standardized feature space.
    Fails closed if the corpus is stale relative to this run's evidence
    (a different model or scaler snapshot).

    \b
    Example:
      gridiron evaluate find-comparables --run-id <run-id>
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.comparable_games_evidence_builder import (
        build_and_write_comparable_games_batches,
    )

    repo: Path = get_settings().repo_root
    console.header("evaluate find-comparables")

    try:
        with step("Match run events against the comparable-games corpus") as s:
            result = build_and_write_comparable_games_batches(
                run_id=run_id,
                generated_at=datetime.now(UTC),
                repo=repo,
                k=k,
            )
            s.set_detail(f"{result.event_count} events")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo("")
    typer.echo(f"Events matched: {result.event_count}")
    for batch in result.batches:
        n, thr = batch.sample_size, batch.distance_threshold
        typer.echo(f"  {batch.game_id}: {n} comparables (threshold {thr:.4f})")

    console.summary()


def _print_ranking_section(
    ranked_df: DataFrame,
    display_cols: list[str],
    *,
    auto_select: bool,
    recommended_key: str,
    target_key: str,
    criteria_list: list[str],
    lower_is_better: set[str],
    divider: LiteralString,
) -> None:
    """Print section 1: model ranking table and recommendation."""
    typer.echo(f"\n{divider}")
    typer.echo("[1] Model Ranking")
    typer.echo(divider)
    typer.echo(ranked_df[display_cols].to_string(index=False))
    typer.echo("")

    if auto_select:
        typer.echo(f"  → Auto-selected: {recommended_key}")
    else:
        typer.echo(f"  → Ranked #1: {recommended_key}  ·  Analysing: {target_key}")

    best_in_ranked: Series = ranked_df.loc[ranked_df["model_key"] == recommended_key].iloc[0]
    reasons: list[str] = []
    for criterion in criteria_list:
        if best_in_ranked[f"rank_{criterion}"] == 1:
            direction: Literal["highest", "lowest"] = (
                "lowest" if criterion in lower_is_better else "highest"
            )
            reasons.append(f"{criterion.upper()}={best_in_ranked[criterion]:.5f} ({direction})")
    if reasons:
        typer.echo(f"  → Reason: Best {', '.join(reasons)}")


def _print_confidence_section(
    df_tiers: DataFrame,
    *,
    target_key: str,
    season: str | None,
    divider: LiteralString,
) -> None:
    """Print section 2: confidence-stratified Brier with overconfidence flag."""
    from gridiron_edge.evaluation.report import find_high_confidence_warning

    typer.echo(f"\n{divider}")
    typer.echo(f"[2] Confidence-Stratified Brier - {target_key}")
    if season:
        typer.echo(f"    season filter: {season}")
    typer.echo(divider)
    typer.echo(df_tiers.to_string(index=False))

    flag = find_high_confidence_warning(df_tiers)
    if flag is None:
        typer.echo("\n  ✓  High-confidence tiers well-calibrated.")
        return

    typer.echo(
        f"\n  ⚠  High-confidence tier '{flag.confidence_tier}': "
        f"model predicts {flag.predicted_avg:.0%} avg, "
        f"teams win {flag.actual_win_rate:.0%} - "
        f"gap {flag.calibration_gap:+.3f} ({flag.direction})"
    )


def _print_stability_section(
    df_seasons: DataFrame,
    *,
    target_key: str,
    divider: LiteralString,
) -> None:
    """Print section 3: season-over-season Brier with drift flag."""
    from gridiron_edge.evaluation.report import find_season_drift_warning

    typer.echo(f"\n{divider}")
    typer.echo(f"[3] Season-over-Season Brier - {target_key}")
    typer.echo(divider)
    typer.echo(df_seasons.to_string(index=False))

    flag = find_season_drift_warning(df_seasons)
    if flag is None:
        typer.echo("\n  ✓  Performance stable across all seasons.")
        return

    typer.echo(
        f"\n  ⚠  Possible drift: '{flag.season}' is "
        f"{flag.delta_vs_mean:+.5f} vs mean - monitor next season."
    )


def _print_misses_section(
    df_misses: DataFrame,
    *,
    target_key: str,
    top_misses: int,
    season: str | None,
    divider: LiteralString,
) -> None:
    """Print section 4: top-N misses with heuristic pattern summary."""
    from gridiron_edge.evaluation.report import (
        find_early_season_miss_pattern,
        find_overconfidence_miss_pattern,
    )

    typer.echo(f"\n{divider}")
    typer.echo(f"[4] Top {top_misses} Misses - {target_key}")
    if season:
        typer.echo(f"    season filter: {season}")
    typer.echo(divider)
    typer.echo(df_misses.to_string(index=False))

    early_flag = find_early_season_miss_pattern(df_misses, top_misses)
    if early_flag is not None:
        typer.echo(
            f"\n  ⚠  {early_flag.n_early}/{early_flag.top_misses} worst "
            f"misses in weeks 1-3 "
            f"(early-season EPA instability is a likely contributor)."
        )

    overconf_flag = find_overconfidence_miss_pattern(df_misses, top_misses)
    if overconf_flag is not None:
        typer.echo(
            f"\n  ⚠  {overconf_flag.n_losses}/{overconf_flag.top_misses} "
            f"worst misses were losses for the predicted favorite - "
            f"overconfidence pattern."
        )

    typer.echo("")


@evaluate_app.command("report")
def evaluate_report(
    *,
    model_key: str = typer.Option(
        "auto",
        help=(
            "Composite model key to report on. Use 'auto' (default) to "
            "auto-select the best model via composite ranking, or pass a "
            "specific key (e.g. 'win_prob_random_forest')."
        ),
    ),
    top_misses: int = typer.Option(
        10,
        help="Number of worst individual predictions to surface (default 10).",
    ),
    season: str | None = typer.Option(
        None,
        help="Restrict all analysis to a specific season e.g. '2025-2026'.",
    ),
    criteria: str = typer.Option(
        "brier,ece,auc",
        help=(
            "Comma-separated ranking criteria used when auto-selecting the model. "
            "Options: brier, ece, auc, accuracy, log_loss."
        ),
    ),
) -> None:
    r"""Full evaluation report: model selection + depth characterisation.

    Answers three questions that aggregate metrics alone cannot:

    \b
      1. Which model should I use?     (composite ranking, same as select-model)
      2. Does it break at high confidence?  (confidence-stratified Brier)
      3. Is performance drifting over time? (season-over-season Brier)
      4. What are its worst individual calls?  (top-N misses with context)

    With --model-key auto (default), the best model is chosen automatically.
    Pass a specific key to force analysis of that model regardless of ranking.

    \b
    Examples:
      gridiron evaluate report
      gridiron evaluate report --model-key win_prob_random_forest
      gridiron evaluate report --top-misses 20
      gridiron evaluate report --season 2025-2026
      gridiron evaluate report --model-key win_prob_xgboost --season 2025-2026
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    import gridiron_edge.models.elo.model
    import gridiron_edge.models.game_prediction.model  # noqa: F401

    repo: Path = get_settings().repo_root

    # ── Parse criteria ──────────────────────────────────────────────────────
    valid_criteria: set[str] = {"brier", "ece", "auc", "accuracy", "log_loss"}
    lower_is_better: set[str] = {"brier", "ece", "log_loss"}
    criteria_list: list[str] = [c.strip().lower() for c in criteria.split(",")]
    invalid: list[str] = [c for c in criteria_list if c not in valid_criteria]
    if invalid:
        typer.echo(f"Unknown criteria: {invalid}. Valid: {sorted(valid_criteria)}")
        raise typer.Exit(1)

    auto_select: bool = model_key == "auto"
    subtitle: str = f"model={'auto-select' if auto_select else model_key}"
    if season:
        subtitle += f"  season={season}"
    console.header("evaluate report", subtitle=subtitle)

    # ── Rank all models ─────────────────────────────────────────────────────
    with step("Compute metrics - all models") as s:
        all_rows: list[dict] = _collect_latest_forecast_run_metrics(
            _registered_game_model_keys(), repo=repo
        )
        s.set_detail(f"{len(all_rows)} models with a backfilled run")

    if not all_rows:
        typer.echo("No models with a backfilled run. Run 'gridiron evaluate backfill' first.")
        raise typer.Exit(1)

    ranked_df: DataFrame = _rank_models(
        all_rows, criteria_list=criteria_list, lower_is_better=lower_is_better
    )
    rank_cols: list[str] = [f"rank_{c}" for c in criteria_list]
    display_cols: list[str] = [
        "model_key",
        "n_games",
        "brier",
        "ece",
        "auc",
        "accuracy",
        *rank_cols,
        "composite_rank",
    ]
    recommended_key: str = str(ranked_df.iloc[0]["model_key"])
    target_key: str = recommended_key if auto_select else model_key

    if target_key not in {str(r["model_key"]) for r in all_rows}:
        typer.echo(
            f"No backfilled run for {target_key!r}. "
            f"Run 'gridiron evaluate backfill --model-name <name> --model-type <type>' first."
        )
        raise typer.Exit(1)

    # ── Compute depth metrics ────────────────────────────────────────────────
    run_id = _resolved_run_id_or_exit(target_key, repo=repo)
    try:
        _df_eval, df_tiers, df_seasons, df_misses = _compute_report_data(
            target_key=target_key, run_id=run_id, season=season, top_misses=top_misses, repo=repo
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    # ── Print report ────────────────────────────────────────────────────────
    _divider: LiteralString = "─" * 60
    _print_ranking_section(
        ranked_df,
        display_cols,
        auto_select=auto_select,
        recommended_key=recommended_key,
        target_key=target_key,
        criteria_list=criteria_list,
        lower_is_better=lower_is_better,
        divider=_divider,
    )
    _print_confidence_section(df_tiers, target_key=target_key, season=season, divider=_divider)
    _print_stability_section(df_seasons, target_key=target_key, divider=_divider)
    _print_misses_section(
        df_misses, target_key=target_key, top_misses=top_misses, season=season, divider=_divider
    )

    console.summary()


@evaluate_app.command("feature-sweep")
def evaluate_feature_sweep(
    *,
    checkpoint_dir: Path = typer.Option(  # noqa: B008
        ...,
        help="Directory for per-family sweep checkpoints (created if missing). Rerun with the "
        "same directory to resume; keep it outside data/.",
    ),
    report_dir: Path = typer.Option(  # noqa: B008
        Path("docs/reports"),
        help="Directory for the Markdown report and per-game prediction JSON.",
    ),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Skip training; build the report from existing checkpoints.",
    ),
) -> None:
    r"""Run the consolidated incremental feature sweep and write its report.

    Evaluates the baseline feature set and each wired feature batch
    cumulatively for Win (Logistic, Random Forest, XGBoost) and Total (Random
    Forest, XGBoost) on identical matched holdout games, then writes one
    Markdown report and a per-game prediction JSON. Slow and training-heavy
    (hours). Read-only with respect to data/, models, champions, and
    predictions; nothing it writes is an input to any pipeline.

    \b
    Examples:
      gridiron evaluate feature-sweep --checkpoint-dir /tmp/sweep
      gridiron evaluate feature-sweep --checkpoint-dir /tmp/sweep --report-only
    """
    from datetime import UTC, datetime
    import json

    import pandas as pd

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_modeling_file
    from gridiron_edge.evaluation.feature_sweep_columns import run_column_level
    from gridiron_edge.evaluation.feature_sweep_driver import (
        ALL_FAMILIES,
        TARGET_COLUMNS,
        load_runs,
        run_sweep,
    )
    from gridiron_edge.evaluation.feature_sweep_report import build_report, predictions_payload
    from gridiron_edge.evaluation.feature_sweep_rows import build_matched_rows

    repo: Path = get_settings().repo_root
    console.header("evaluate feature-sweep")
    frame: pd.DataFrame = load_modeling_file(repo)

    if report_only:
        with step("Load sweep checkpoints") as s:
            runs = load_runs(checkpoint_dir)
            if not runs:
                raise typer.BadParameter(f"No checkpoints found in {checkpoint_dir}")
            matched = {
                target: build_matched_rows(frame, TARGET_COLUMNS[target])
                for target in {run.spec.target for run in runs.values()}
            }
            s.set_detail(f"{len(runs)} families")
    else:
        with step("Run feature sweep (resumes from checkpoints)") as s:
            runs, matched = run_sweep(
                frame,
                repo,
                checkpoint_dir,
                families=ALL_FAMILIES,
                on_step=lambda family, name, secs: typer.echo(
                    f"  done {family} {name} ({secs / 60:.1f} min)"
                ),
            )
            s.set_detail(f"{len(runs)} families")

    with step("Column-level ablations and diagnostics") as s:
        column_level = run_column_level(runs, matched, repo)
        s.set_detail(f"{len(column_level.ablations)} families")

    generated_on = datetime.now(UTC).date()
    with step("Write report") as s:
        text, decisions = build_report(
            runs, matched, frame, generated_on=generated_on, column_level=column_level
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "phase-a-feature-sweep.md"
        predictions_path = report_dir / "phase-a-feature-sweep-predictions.json"
        report_path.write_text(text, encoding="utf-8")
        predictions_path.write_text(
            json.dumps(predictions_payload(runs, matched), separators=(",", ":")),
            encoding="utf-8",
        )
        s.set_detail(str(report_path))

    typer.echo("")
    for name, decision in decisions.items():
        typer.echo(f"{name}: {decision.status} - {decision.reason}")
    console.summary()


@evaluate_app.command("feature-selection")
def evaluate_feature_selection(
    *,
    checkpoint_dir: Path = typer.Option(  # noqa: B008
        ...,
        help="Sweep checkpoint directory from `evaluate feature-sweep` (stage 2 results are "
        "stored in its stage2/ subdirectory; rerun with the same directory to resume).",
    ),
    report_dir: Path = typer.Option(  # noqa: B008
        Path("docs/reports"),
        help="Directory for the Markdown report and the selected-subsets JSON.",
    ),
    threads: int | None = typer.Option(
        None,
        help="Estimator threads per fit during selection (default: all cores). Lower it to "
        "keep the machine usable.",
    ),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Skip selection; build the report from existing stage 2 results.",
    ),
) -> None:
    r"""Run model-driven feature selection on the sweep's full column pool.

    Requires a completed `evaluate feature-sweep`. Each family selects columns
    on training rows only (with a placebo-column control), validates the
    procedure walk-forward on 2013 onward, and confirms the deployable subset
    once on the holdout. Slow and training-heavy. Read-only with respect to
    data/, models, champions, and predictions.

    \b
    Examples:
      gridiron evaluate feature-selection --checkpoint-dir /tmp/sweep --threads 4
      gridiron evaluate feature-selection --checkpoint-dir /tmp/sweep --report-only
    """
    from datetime import UTC, datetime
    import json

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_modeling_file
    from gridiron_edge.evaluation.feature_sweep_driver import (
        TARGET_COLUMNS,
        load_runs,
    )
    from gridiron_edge.evaluation.feature_sweep_rows import build_matched_rows
    from gridiron_edge.evaluation.feature_sweep_stage2 import (
        build_stage2_report,
        load_stage2,
        run_stage2,
    )

    repo: Path = get_settings().repo_root
    console.header("evaluate feature-selection")

    runs = load_runs(checkpoint_dir)
    if not runs:
        raise typer.BadParameter(f"No sweep checkpoints found in {checkpoint_dir}")

    if report_only:
        with step("Load stage 2 results") as s:
            results = load_stage2(checkpoint_dir)
            if not results:
                raise typer.BadParameter(f"No stage 2 results found in {checkpoint_dir}")
            s.set_detail(f"{len(results)} families")
    else:
        frame = load_modeling_file(repo)
        matched = {
            target: build_matched_rows(frame, TARGET_COLUMNS[target])
            for target in {run.spec.target for run in runs.values()}
        }
        with step("Select features (resumes from checkpoints)") as s:
            results = run_stage2(
                runs,
                matched,
                repo,
                checkpoint_dir,
                threads=threads,
                on_family=lambda family, secs: typer.echo(f"  done {family} ({secs / 60:.1f} min)"),
            )
            s.set_detail(f"{len(results)} families")

    with step("Write report") as s:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "phase-a-feature-selection.md"
        subsets_path = report_dir / "phase-a-feature-selection-subsets.json"
        report_path.write_text(
            build_stage2_report(results, generated_on=datetime.now(UTC).date()),
            encoding="utf-8",
        )
        subsets_path.write_text(
            json.dumps(
                {
                    label: {
                        "early_selected": list(r.early.selected),
                        "final_selected": list(r.final.selected),
                        "final_stability": r.final.stability,
                    }
                    for label, r in results.items()
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        s.set_detail(str(report_path))

    typer.echo("")
    for label, r in results.items():
        typer.echo(
            f"{label}: final kept {len(r.final.selected)}/{r.final.n_pool}, "
            f"placebo kept {r.final.placebo_selected}/{r.final.placebo_candidates}"
        )
    console.summary()


@evaluate_app.command("feature-attribution")
def evaluate_feature_attribution(
    *,
    checkpoint_dir: Path = typer.Option(  # noqa: B008
        ...,
        help="Sweep checkpoint directory from `evaluate feature-sweep` (stage 3 results are "
        "stored in its stage3/ subdirectory; rerun with the same directory to resume).",
    ),
    report_dir: Path = typer.Option(  # noqa: B008
        Path("docs/reports"),
        help="Directory for the Markdown report.",
    ),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Skip refits; build the report from existing stage 3 results.",
    ),
) -> None:
    r"""Attribute the wired feature batches one at a time on the wide window.

    Requires a completed `evaluate feature-sweep`. Scores each batch against the
    step before it walk-forward over 2013-2025 with each step's frozen sweep
    hyperparameters (refits only), applies the pre-declared wide-window
    criterion, and writes one Markdown report. Read-only with respect to data/,
    models, champions, and predictions. Limit CPU externally (for example with
    `taskset -c 0-3 nice -n 19`).

    \b
    Examples:
      gridiron evaluate feature-attribution --checkpoint-dir /tmp/sweep
      gridiron evaluate feature-attribution --checkpoint-dir /tmp/sweep --report-only
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_modeling_file
    from gridiron_edge.evaluation.feature_sweep_driver import TARGET_COLUMNS, load_runs
    from gridiron_edge.evaluation.feature_sweep_report import batch_live_complete
    from gridiron_edge.evaluation.feature_sweep_rows import build_matched_rows
    from gridiron_edge.evaluation.feature_sweep_stage3 import (
        build_stage3_report,
        decide_stage3,
        load_stage3,
        run_stage3,
    )

    repo: Path = get_settings().repo_root
    console.header("evaluate feature-attribution")
    runs = load_runs(checkpoint_dir)
    if not runs:
        raise typer.BadParameter(f"No sweep checkpoints found in {checkpoint_dir}")
    frame = load_modeling_file(repo)

    if report_only:
        results = load_stage3(checkpoint_dir)
        if not results:
            raise typer.BadParameter(f"No stage 3 results found in {checkpoint_dir}")
    else:
        matched = {
            target: build_matched_rows(frame, TARGET_COLUMNS[target])
            for target in {run.spec.target for run in runs.values()}
        }
        with step("Attribute batches on the wide window (resumes from checkpoints)") as s:
            results = run_stage3(
                runs,
                matched,
                repo,
                checkpoint_dir,
                on_family=lambda family, secs: typer.echo(f"  done {family} ({secs / 60:.1f} min)"),
            )
            s.set_detail(f"{len(results)} families")

    live = {name: v["rows_with_nan"] for name, v in batch_live_complete(frame).items()}
    decisions, adjusted = decide_stage3(results, live)
    with step("Write report") as s:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "phase-a-feature-attribution.md"
        path.write_text(
            build_stage3_report(
                results, decisions, adjusted, live, generated_on=datetime.now(UTC).date()
            ),
            encoding="utf-8",
        )
        s.set_detail(str(path))

    typer.echo("")
    for name, decision in decisions.items():
        typer.echo(f"{name}: {decision.status} - {decision.reason}")
    console.summary()


@evaluate_app.command("feature-confirmation")
def evaluate_feature_confirmation(
    *,
    checkpoint_dir: Path = typer.Option(  # noqa: B008
        ...,
        help="Sweep checkpoint directory from `evaluate feature-sweep` (supplies the frozen "
        "hyperparameters).",
    ),
    report_dir: Path = typer.Option(  # noqa: B008
        Path("docs/reports"),
        help="Directory for the Markdown report.",
    ),
) -> None:
    r"""Confirm the final feature composition against baseline on the wide window.

    Requires the sweep checkpoints that supply frozen hyperparameters. Scores
    baseline versus baseline plus the kept QB-and-coach batch (refits only) on
    the wide window, the holdout, and Week 1 games, applies the criterion
    recorded before it ran, and writes one Markdown report. Read-only with
    respect to data/, models, champions, and predictions. Limit CPU externally
    (for example `taskset -c 0-3 nice -n 19`).

    \b
    Examples:
      gridiron evaluate feature-confirmation --checkpoint-dir /tmp/sweep
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_modeling_file
    from gridiron_edge.evaluation.feature_sweep_confirm import (
        build_confirmation_report,
        judge_confirmation,
        run_confirmation,
    )
    from gridiron_edge.evaluation.feature_sweep_driver import TARGET_COLUMNS, load_runs
    from gridiron_edge.evaluation.feature_sweep_rows import build_matched_rows

    repo: Path = get_settings().repo_root
    console.header("evaluate feature-confirmation")
    runs = load_runs(checkpoint_dir)
    if not runs:
        raise typer.BadParameter(f"No sweep checkpoints found in {checkpoint_dir}")
    frame = load_modeling_file(repo)
    matched = {
        target: build_matched_rows(frame, TARGET_COLUMNS[target])
        for target in {run.spec.target for run in runs.values()}
    }
    with step("Score final composition against baseline") as s:
        results = run_confirmation(runs, matched, repo)
        s.set_detail(f"{len(results)} families")
    verdict = judge_confirmation(results)
    with step("Write report") as s:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "phase-a-final-confirmation.md"
        path.write_text(
            build_confirmation_report(results, verdict, generated_on=datetime.now(UTC).date()),
            encoding="utf-8",
        )
        s.set_detail(str(path))
    typer.echo("")
    typer.echo(f"{'CONFIRMED' if verdict.confirmed else 'NOT CONFIRMED'}: {verdict.reason}")
    console.summary()


@evaluate_app.command("quote-movement")
def evaluate_quote_movement(
    *,
    report_dir: Path = typer.Option(  # noqa: B008
        Path("docs/reports"),
        help="Directory for the Markdown report.",
    ),
) -> None:
    r"""Measure quote drift by age and derive the market age limit.

    Compares how far the sportsbooks' consensus win probability drifts as quotes age with
    how far single books disagree at one moment, and reports the age at which drift
    exceeds the disagreement (`DECISIONS.md` D80), beside the configured limit. Reads
    the quote ledger; writes only the Markdown report.

    \b
    Examples:
      gridiron evaluate quote-movement
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.quote_movement import build_report, movement_tables
    from gridiron_edge.ingest.odds.store import load_odds_ledger
    from gridiron_edge.market.market_freshness import MAX_MARKET_AGE

    repo: Path = get_settings().repo_root
    console.header("evaluate quote-movement")
    with step("Measure quote drift by age") as s:
        ledger = load_odds_ledger(repo=repo)
        if ledger.empty:
            raise typer.BadParameter("The quote ledger is empty; collect or pull quotes first.")
        try:
            movement = movement_tables(ledger)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        s.set_detail(f"{movement.games} games")
    with step("Write report") as s:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "quote-movement.md"
        path.write_text(
            build_report(
                movement,
                configured_hours=MAX_MARKET_AGE.total_seconds() / 3600,
                generated_on=datetime.now(UTC).date(),
            ),
            encoding="utf-8",
        )
        s.set_detail(str(path))
    derived = "none" if movement.max_age_hours is None else f"{movement.max_age_hours:g} h"
    typer.echo("")
    typer.echo(
        f"derived age limit {derived}; configured {MAX_MARKET_AGE.total_seconds() / 3600:g} h"
    )
    typer.echo(movement.limit_reason)


@evaluate_app.command("market-agreement")
def evaluate_market_agreement(
    *,
    report_dir: Path = typer.Option(  # noqa: B008
        Path("docs/reports"),
        help="Directory for the Markdown report.",
    ),
) -> None:
    r"""Test whether the models beat the closing market where they disagree with it.

    For each backfilled Win and Total family, compares the model with the closing line
    (Brier or MAE), fits the calibration slope of the outcome on the model-minus-market
    gap with its 95% interval, and tabulates realized outcomes by gap bin
    (`DECISIONS.md` D85). Raw model EV is trustworthy only if the slope's interval excludes
    0 and includes 1. Reads the backfilled forecast events and the cleaned games file;
    writes only the Markdown report.

    \b
    Examples:
      gridiron evaluate market-agreement
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_games
    from gridiron_edge.evaluation.forecast_contracts import ForecastRole
    from gridiron_edge.evaluation.forecast_store import load_forecast_events
    from gridiron_edge.evaluation.market_agreement import (
        agreement_tables,
        build_report,
        prediction_frame,
    )
    from gridiron_edge.evaluation.select import latest_backfilled_run_id

    repo: Path = get_settings().repo_root
    console.header("evaluate market-agreement")
    specs = (
        ("Win - Elo", "win", "win_prob", "elo", "home_win_prob"),
        ("Win - Logistic", "win", "win_prob", "logistic", "home_win_prob"),
        ("Win - Random Forest", "win", "win_prob", "random_forest", "home_win_prob"),
        ("Win - XGBoost", "win", "win_prob", "xgboost", "home_win_prob"),
        ("Total - Random Forest", "total", "total", "random_forest", "model_total"),
        ("Total - XGBoost", "total", "total", "xgboost", "model_total"),
    )
    with step("Load backfilled forecasts") as s:
        families = {}
        skipped = []
        for name, kind, model_name, model_type, column in specs:
            run_id = latest_backfilled_run_id(model_name, model_type, repo=repo)
            if run_id is None:
                skipped.append(name)
                continue
            events = load_forecast_events(
                model_name=model_name,
                model_type=model_type,
                role=ForecastRole.BACKFILLED,
                run_id=run_id,
                repo=repo,
            )
            families[name] = (kind, prediction_frame(events, column))
        if not families:
            raise typer.BadParameter(
                "No backfilled forecasts exist; run `evaluate backfill` first."
            )
        s.set_detail(
            f"{len(families)} families" + (f"; none for {', '.join(skipped)}" if skipped else "")
        )
    with step("Compare with the closing market") as s:
        try:
            results = agreement_tables(load_games(repo), families)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        s.set_detail(f"{sum(item.games for item in results)} family-games")
    with step("Write report") as s:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "market-agreement.md"
        path.write_text(
            build_report(results, generated_on=datetime.now(UTC).date()), encoding="utf-8"
        )
        s.set_detail(str(path))
    typer.echo("")
    for item in results:
        typer.echo(
            f"{item.name}: slope {item.slope:.3f} ({item.slope_low:.3f} to {item.slope_high:.3f}), "
            f"{item.score_label} model {item.model_score:.4f} vs market {item.market_score:.4f}, "
            f"validated: {'yes' if item.validated else 'no'}"
        )


@evaluate_app.command("starting-qb-projection")
def evaluate_starting_qb_projection(
    *,
    report_dir: Path = typer.Option(  # noqa: B008
        Path("docs/reports"),
        help="Directory for the Markdown report.",
    ),
) -> None:
    r"""Backtest a projected starting quarterback against carry-forward.

    Scores a rules-based resolver (depth chart plus weekly availability status)
    and the previous-starter carry-forward against the quarterback who actually
    started, 2009 through 2024, then checks the 2025 timestamped depth chart
    snapshots at fixed lead times. Reads the cached nflverse depth charts and
    injury reports (`gridiron ingest depth-charts`, `gridiron ingest injuries`)
    and the raw games file. Writes only the Markdown report; no model, schema or
    data artifact changes.

    \b
    Examples:
      gridiron evaluate starting-qb-projection
    """
    from datetime import UTC, datetime

    import pandas as pd

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.registry import dataset_path
    from gridiron_edge.evaluation.starting_qb_projection import (
        bar_verdict,
        build_team_games,
        score_rules,
        snapshot_timing,
    )
    from gridiron_edge.evaluation.starting_qb_projection_report import build_report
    from gridiron_edge.ingest.nflverse.weekly_availability import (
        load_depth_charts,
        load_injuries,
    )

    repo: Path = get_settings().repo_root
    console.header("evaluate starting-qb-projection")
    games = pd.read_parquet(dataset_path(repo, "games_raw_nflverse"))
    try:
        depth_charts = load_depth_charts(repo=repo)
        injuries = load_injuries(repo=repo)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error
    with step("Score the resolver against carry-forward") as s:
        scored = score_rules(build_team_games(games, depth_charts, injuries))
        if scored.empty:
            raise typer.BadParameter(
                "No scorable team-games: the cached depth charts and injury reports must "
                "cover 2009-2024 (run `gridiron ingest depth-charts --all-years` and "
                "`gridiron ingest injuries --all-years`)."
            )
        verdict = bar_verdict(scored)
        s.set_detail(f"{len(scored):,} team-games")
    timing: dict[str, pd.DataFrame] = {}
    if 2025 in depth_charts and 2025 in injuries:
        with step("Score 2025 snapshots at fixed lead times") as s:
            readings = {
                "All weeks": None,
                "Week 1": [1],
                "Weeks after week 1": list(range(2, 19)),
            }
            for label, weeks in readings.items():
                timing[label] = snapshot_timing(
                    games,
                    depth_charts[2025],
                    injuries[2025],
                    season=2025,
                    lead_hours=[24, 48, 96],
                    weeks=weeks,
                )
            s.set_detail(f"{len(timing)} readings")
    with step("Write report") as s:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "starting-qb-projection.md"
        path.write_text(
            build_report(scored, verdict, timing, generated_on=datetime.now(UTC).date()),
            encoding="utf-8",
        )
        s.set_detail(str(path))
    typer.echo("")
    typer.echo(f"{'BAR CLEARED' if verdict.passes else 'BAR NOT CLEARED'}")
    console.summary()

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
from typing import Literal, LiteralString

from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._composites import write_champion_manifest
from gridiron_edge.evaluation.select import (
    collect_model_metrics as _collect_model_metrics,
)
from gridiron_edge.evaluation.select import (
    compute_report_data as _compute_report_data,
)
from gridiron_edge.evaluation.select import (
    rank_models as _rank_models,
)

evaluate_app = typer.Typer(
    help="Evaluate model predictions against outcomes.", no_args_is_help=True
)


def _split_composite_key(key: str) -> tuple[str | None, str | None]:
    """Split a composite model_key into (model_name, model_type) for filtering.

    Matches against the known model_name prefixes returned by
    :func:`gridiron_edge.models.game_prediction.predictor.get_known_model_names`.
    Returns ``(None, None)`` when the key is ``"all"`` so callers can skip
    the filter entirely.

    Args:
        key: Composite registry key (e.g. ``"win_prob_random_forest"``)
            or ``"all"``.

    Returns:
        Tuple of ``(model_name, model_type)``. Both ``None`` if ``key == "all"``.
        Raises if ``key`` doesn't match any known prefix.
    """
    from gridiron_edge.models.game_prediction.predictor import get_known_model_names

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
    r"""Print prediction accuracy summary from the archive.

    \b
    Examples:
      gridiron evaluate summary
      gridiron evaluate summary --season 2025-2026
      gridiron evaluate summary --group-by week
      gridiron evaluate summary --model-key win_prob_random_forest
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.metrics import build_evaluation_df, summarise

    name_filter, type_filter = _split_composite_key(model_key)
    subtitle: str = f"model={model_key}"
    if season:
        subtitle += f"  season={season}"
    subtitle += f"  group={group_by}"
    console.header("evaluate summary", subtitle=subtitle)

    with step("Join predictions to outcomes") as s:
        df_eval: DataFrame = build_evaluation_df(
            model_name=name_filter,
            model_type=type_filter,
            season=season,
        )
        if df_eval.empty:
            s.set_detail("no evaluated games - run 'output predictions' first")
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
    from gridiron_edge.evaluation.metrics import build_evaluation_df, calibration_table

    name_filter, type_filter = _split_composite_key(model_key)
    subtitle: str = f"model={model_key}"
    if season:
        subtitle += f"  season={season}"
    console.header("evaluate calibration", subtitle=subtitle)

    with step("Join predictions to outcomes") as s:
        df_eval: DataFrame = build_evaluation_df(
            model_name=name_filter,
            model_type=type_filter,
            season=season,
        )
        if df_eval.empty:
            s.set_detail("no evaluated games - run 'output predictions' first")
        else:
            s.set_detail(f"{len(df_eval)} games")

    if not df_eval.empty:
        with step("Build calibration table") as s:
            df_cal: DataFrame = calibration_table(df_eval, n_buckets=buckets)
            s.set_detail(f"{len(df_cal)} buckets")

        typer.echo(df_cal.to_string(index=False))

    console.summary()


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
    from gridiron_edge.evaluation.backfill import backfill_model

    resolved_mode: str | None = None if mode == "auto" else mode
    subtitle_parts: list[str] = [f"model={model_name}_{model_type}"]
    if resolved_mode is not None:
        subtitle_parts.append(f"mode={resolved_mode}")
    console.header("evaluate backfill", subtitle="  ".join(subtitle_parts))

    with step("Generate historical forecast events") as s:
        n: int = backfill_model(
            model_name=model_name,
            model_type=model_type,
            mode=resolved_mode,  # type: ignore[arg-type]
            start_season=start_season,
            end_season=end_season,
        )
        s.set_detail(f"{n:,} forecast events written")

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
                n: int = backfill_model(
                    model_name="win_prob",
                    model_type="elo",
                )
                s.set_detail(f"{n:,} backfilled forecast events written as win_prob/elo")
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
                n = backfill_model(
                    model_name="win_prob",
                    model_type="elo",
                )
                s.set_detail(f"{n:,} backfilled forecast events written as win_prob/elo")

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
    from gridiron_edge.evaluation.metrics import build_evaluation_df

    repo: Path = get_settings().repo_root
    subtitle_parts: list[str] = []
    if model_key != "all":
        subtitle_parts.append(f"model={model_key}")
    if compare:
        subtitle_parts.append("compare")
    console.header("evaluate diagnostics", subtitle="  ".join(subtitle_parts) or "all")

    # Single-model diagnostics
    if model_key != "all":
        name_filter, type_filter = _split_composite_key(model_key)
        with step(f"Load predictions - {model_key}") as s:
            df_eval: DataFrame = build_evaluation_df(
                model_name=name_filter,
                model_type=type_filter,
            )
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
        import gridiron_edge.models.elo.predictor
        import gridiron_edge.models.game_prediction.predictor  # noqa: F401
        from gridiron_edge.models.registry import ModelRegistry

        all_keys: list[str] = ModelRegistry.names()

        with step("Load predictions - all models") as s:
            eval_dfs: dict = {}
            for key in all_keys:
                nm, ty = _split_composite_key(key)
                df: DataFrame = build_evaluation_df(model_name=nm, model_type=ty)
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
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import ModelRegistry

    repo: Path = get_settings().repo_root
    console.header("evaluate select-model")

    # Parse criteria
    valid_criteria: set[str] = {"brier", "ece", "auc", "accuracy", "log_loss"}
    criteria_list: list[str] = [c.strip().lower() for c in criteria.split(",")]
    invalid: list[str] = [c for c in criteria_list if c not in valid_criteria]
    if invalid:
        typer.echo(f"Unknown criteria: {invalid}. Valid: {sorted(valid_criteria)}")
        raise typer.Exit(1)

    # Compute metrics for all models with archived predictions
    with step("Compute metrics for all models") as s:
        rows: list[dict] = _collect_model_metrics(ModelRegistry.names(), repo=repo)
        s.set_detail(f"{len(rows)} models evaluated")

    if not rows:
        typer.echo("No models with archived predictions found. Run evaluate backfill first.")
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
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import ModelRegistry

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
        all_rows: list[dict] = _collect_model_metrics(ModelRegistry.names(), repo=repo)
        s.set_detail(f"{len(all_rows)} models with archived predictions")

    if not all_rows:
        typer.echo("No models with archived predictions. Run 'gridiron evaluate backfill' first.")
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
            f"No archived predictions for {target_key!r}. "
            f"Run 'gridiron evaluate backfill --model-name <name> --model-type <type>' first."
        )
        raise typer.Exit(1)

    # ── Compute depth metrics ────────────────────────────────────────────────
    try:
        _df_eval, df_tiers, df_seasons, df_misses = _compute_report_data(
            target_key=target_key, season=season, top_misses=top_misses, repo=repo
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

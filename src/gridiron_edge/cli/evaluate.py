# src/gridiron_edge/cli/evaluate.py
"""CLI commands for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, LiteralString

from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
import typer

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


@evaluate_app.command("summary")
def evaluate_summary(
    *,
    model_version: str = typer.Option(
        "all",
        help="Model version to evaluate, or 'all' to compare all versions.",
    ),
    season: str | None = typer.Option(None, help="Filter to a specific season e.g. '2025-2026'."),
    group_by: str = typer.Option(
        "season",
        help="Group results by: season, week, or model_version.",
    ),
) -> None:
    r"""Print prediction accuracy summary from the archive.

    \b
    Examples:
      gridiron evaluate summary
      gridiron evaluate summary --season 2025-2026
      gridiron evaluate summary --group-by week
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.metrics import build_evaluation_df, summarise

    mv_filter: str | None = None if model_version == "all" else model_version
    subtitle: str = f"model={model_version}"
    if season:
        subtitle += f"  season={season}"
    subtitle += f"  group={group_by}"
    console.header("evaluate summary", subtitle=subtitle)

    with step("Join predictions to outcomes") as s:
        df_eval: DataFrame = build_evaluation_df(model_version=mv_filter, season=season)
        if df_eval.empty:
            s.set_detail("no evaluated games — run 'output predictions' first")
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
    model_version: str = typer.Option(
        "elo_v1",
        help="Model version to evaluate, or 'all' for all versions combined.",
    ),
    season: str | None = typer.Option(None, help="Filter to a specific season e.g. '2025-2026'."),
    buckets: int = typer.Option(10, help="Number of probability buckets (default 10)."),
) -> None:
    r"""Print calibration table showing predicted probability vs actual win rate.

    \b
    Examples:
      gridiron evaluate calibration
      gridiron evaluate calibration --season 2025-2026 --buckets 20
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.metrics import build_evaluation_df, calibration_table

    mv_filter_cal: str | None = None if model_version == "all" else model_version
    subtitle: str = f"model={model_version}"
    if season:
        subtitle += f"  season={season}"
    console.header("evaluate calibration", subtitle=subtitle)

    with step("Join predictions to outcomes") as s:
        df_eval: DataFrame = build_evaluation_df(model_version=mv_filter_cal, season=season)
        if df_eval.empty:
            s.set_detail("no evaluated games — run 'output predictions' first")
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
    model_version: str = typer.Option("elo_v1", help="Model version to backfill."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Re-archive all games even if already present.",
    ),
) -> None:
    r"""Archive Elo predictions for all historical games in one pass.

    Loads games and Elo state once and generates predictions for every
    game in the dataset. Use this to populate the archive before running
    evaluate summary or calibration.

    \b
    Examples:
      gridiron evaluate backfill
      gridiron evaluate backfill --overwrite
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.backfill import backfill_model

    console.header("evaluate backfill", subtitle=f"model={model_version}")

    with step("Generate + archive historical predictions") as s:
        n: int = backfill_model(model_version, overwrite=overwrite)
        s.set_detail(f"{n:,} predictions archived")

    console.summary()


@evaluate_app.command("tune")
def evaluate_tune(
    *,
    v3: bool = typer.Option(
        False,
        "--v3/--no-v3",
        help="Run zone-based K search (elo_v3) instead of flat-K search (elo_v2).",
    ),
    apply: bool = typer.Option(
        False,
        "--apply/--no-apply",
        help=(
            "After the search, backfill the best parameters as elo_v2 or elo_v3 "
            "into the prediction archive."
        ),
    ),
    top: int = typer.Option(10, help="Number of top results to display."),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help=(
            "Save full results to data/output/tune/ as Parquet. "
            "Recommended for long runs — protects against terminal loss."
        ),
    ),
) -> None:
    r"""Grid search Elo parameters (K, divisor, regression) against a holdout set.

    Runs every combination of K-factor, win-probability divisor, and
    offseason regression fraction. Scores each on training seasons and
    held-out seasons (last 3 seasons) separately. Best holdout Brier
    score wins.

    Use --apply to immediately backfill the best parameters into the
    prediction archive as model version 'elo_v2'.

    \b
    Examples:
      gridiron evaluate tune
      gridiron evaluate tune --top 20
      gridiron evaluate tune --apply
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.tune import (
        DIVISOR_VALUES,
        DIVISOR_VALUES_V3,
        HOLDOUT_SEASONS,
        K_EARLY_VALUES,
        K_MID_VALUES,
        K_POST_VALUES,
        K_VALUES,
        K_WEEK18_VALUES,
        REGRESS_VALUES,
        REGRESS_VALUES_V3,
        best_params,
        best_params_v3,
        run_grid_search,
        run_grid_search_v3,
    )

    if v3:
        n_combos: int = (
            len(K_EARLY_VALUES)
            * len(K_MID_VALUES)
            * len(K_WEEK18_VALUES)
            * len(K_POST_VALUES)
            * len(DIVISOR_VALUES_V3)
            * len(REGRESS_VALUES_V3)
        )
        subtitle: str = f"elo_v3  {n_combos} combinations  holdout={sorted(HOLDOUT_SEASONS)}"
    else:
        n_combos = len(K_VALUES) * len(DIVISOR_VALUES) * len(REGRESS_VALUES)
        subtitle = f"elo_v2  {n_combos} combinations  holdout={sorted(HOLDOUT_SEASONS)}"
    console.header("evaluate tune", subtitle=subtitle)

    with step(f"Run grid search ({n_combos} combinations)") as s:
        from gridiron_edge.core.settings import get_settings

        _out_dir: Path = get_settings().data_output / "tune"
        _fname: Literal["elo_v2_tune_results.parquet", "elo_v3_tune_results.parquet"] = (
            "elo_v3_tune_results.parquet" if v3 else "elo_v2_tune_results.parquet"
        )
        _save: Path | None = _out_dir / _fname if save else None
        results: DataFrame = (
            run_grid_search_v3(save_path=_save) if v3 else run_grid_search(save_path=_save)
        )
        best_holdout = results.iloc[0]["holdout_brier"]
        s.set_detail(f"best holdout Brier: {best_holdout:.5f}")

    with step(f"Top {top} results") as s:
        s.set_detail(f"{len(results)} total combinations scored")

    typer.echo(results.head(top).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if apply:
        if v3:
            params: dict[str, float] = best_params_v3(results)
            typer.echo(
                f"Applying best elo_v3 params: "
                f"k_early={params['k_early']:.0f}  k_mid={params['k_mid']:.0f}  "
                f"k_week18={params['k_week18']:.0f}  k_post={params['k_post']:.0f}  "
                f"divisor={params['divisor']:.0f}  regress={params['regress_frac']:.2f}"
            )
            with step("Backfill elo_v3 predictions") as s:
                from gridiron_edge.evaluation.backfill import backfill_model

                n: int = backfill_model("elo_v3", overwrite=True)
                s.set_detail(f"{n:,} predictions archived as elo_v3")
        else:
            params = best_params(results)
            k_val: float = params["k"]
            div_val: float = params["divisor"]
            reg_val: float = params["regress_frac"]
            typer.echo(
                f"Applying best elo_v2 params: "
                f"k={k_val:.0f}  divisor={div_val:.0f}  regress={reg_val:.2f}"
            )
            with step("Backfill elo_v2 predictions") as s:
                from gridiron_edge.evaluation.backfill import backfill_model

                n = backfill_model("elo_v2", overwrite=True)
                s.set_detail(f"{n:,} predictions archived as elo_v2")

    console.summary()


@evaluate_app.command("diagnostics")
def evaluate_diagnostics(
    *,
    model_version: str = typer.Option(
        "all",
        help="Model version to diagnose, or 'all' for multi-model comparison.",
    ),
    compare: bool = typer.Option(
        False,
        "--compare/--no-compare",
        help="Generate multi-model comparison plots for all registered models.",
    ),
) -> None:
    r"""Generate diagnostic plots for one or all models.

    Single-model mode (--model-version X):
        Saves calibration curve, confidence distribution, ROC curve,
        Brier decomposition, performance by context, and feature
        importance to data/output/evaluation/{model_version}/.

    Comparison mode (--compare):
        Overlays all models on shared calibration, ROC, and metric
        bar charts. Saves to data/output/evaluation/.

    \b
    Examples:
      gridiron evaluate diagnostics --model-version elo_v1
      gridiron evaluate diagnostics --model-version logistic
      gridiron evaluate diagnostics --compare
      gridiron evaluate diagnostics --model-version elo_v1 --compare
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
    if model_version != "all":
        subtitle_parts.append(f"model={model_version}")
    if compare:
        subtitle_parts.append("compare")
    console.header("evaluate diagnostics", subtitle="  ".join(subtitle_parts) or "all")

    # Single-model diagnostics
    if model_version != "all":
        with step(f"Load predictions — {model_version}") as s:
            df_eval: DataFrame = build_evaluation_df(model_version=model_version)
            if df_eval.empty:
                s.set_detail("no data — run evaluate backfill first")
                raise typer.Exit(1)
            s.set_detail(f"{len(df_eval):,} games")

        with step(f"Generate single-model plots — {model_version}") as s:
            paths: list[Path] = plot_single_model(df_eval, repo=repo)
            s.set_detail(f"{len(paths)} plots")
            for p in paths:
                typer.echo(f"  {p.relative_to(repo)}")

    # Multi-model comparison
    if compare:
        import gridiron_edge.models.elo.predictor
        import gridiron_edge.models.game_prediction.predictor  # noqa: F401
        from gridiron_edge.models.registry import PredictorRegistry

        all_models: list[str] = PredictorRegistry.names()

        with step("Load predictions — all models") as s:
            eval_dfs: dict = {}
            for mv in all_models:
                df: DataFrame = build_evaluation_df(model_version=mv)
                if not df.empty:
                    eval_dfs[mv] = df
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
) -> None:
    r"""Rank all registered models and recommend the best for production use.

    Evaluates every model with archived predictions against four criteria:
        Brier score   (lower is better)  — overall prediction accuracy
        ECE           (lower is better)  — calibration quality
        ROC-AUC       (higher is better) — ranking quality
        Accuracy      (higher is better) — fraction of winners correctly called

    Each model is ranked on each criterion and the ranks are summed to
    produce a composite score. The model with the lowest composite rank
    wins. Ties are broken by the first listed criterion.

    \b
    Examples:
      gridiron evaluate select-model
      gridiron evaluate select-model --criteria brier,ece,auc
      gridiron evaluate select-model --top 3
    """
    import pandas as pd

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import PredictorRegistry

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
        rows: list[dict] = _collect_model_metrics(PredictorRegistry.names(), repo=repo)
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

    # pyrefly: ignore [bad-argument-type]
    df["composite_rank"] = df[rank_cols].sum(axis=1)
    df: DataFrame = df.sort_values(
        ["composite_rank", criteria_list[0]],
        ascending=[True, criteria_list[0] in lower_is_better],
    ).reset_index(drop=True)

    if top is not None:
        df = df.head(top)

    # Display results
    display_cols: list[str] = [
        "model_version",
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
    typer.echo(f"Recommendation: {best['model_version']}")

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

    console.summary()


def _print_ranking_section(
    ranked_df: DataFrame,
    display_cols: list[str],
    *,
    auto_select: bool,
    recommended_mv: str,
    target_mv: str,
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
        typer.echo(f"  → Auto-selected: {recommended_mv}")
    else:
        typer.echo(f"  → Ranked #1: {recommended_mv}  ·  Analysing: {target_mv}")

    best_in_ranked: Series = ranked_df.loc[ranked_df["model_version"] == recommended_mv].iloc[0]
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
    target_mv: str,
    season: str | None,
    divider: LiteralString,
) -> None:
    """Print section 2: confidence-stratified Brier with overconfidence flag."""
    typer.echo(f"\n{divider}")
    typer.echo(f"[2] Confidence-Stratified Brier — {target_mv}")
    if season:
        typer.echo(f"    season filter: {season}")
    typer.echo(divider)
    typer.echo(df_tiers.to_string(index=False))

    high_conf: DataFrame = df_tiers.loc[df_tiers["predicted_avg"] >= 0.75, :]
    if high_conf.empty:
        typer.echo("\n  —  No high-confidence predictions (≥75%) in this dataset.")
        return

    worst_gap: float = float(high_conf["calibration_gap"].abs().max())
    if worst_gap >= 0.03:
        worst_row: DataFrame | Series = high_conf.loc[
            high_conf["calibration_gap"].abs().idxmax(), :
        ]
        direction_word: str = (
            "overconfident" if worst_row["calibration_gap"] > 0 else "underconfident"
        )
        typer.echo(
            f"\n  ⚠  High-confidence tier '{worst_row['confidence_tier']}': "
            f"model predicts {worst_row['predicted_avg']:.0%} avg, "
            f"teams win {worst_row['actual_win_rate']:.0%} — "
            f"gap {worst_row['calibration_gap']:+.3f} ({direction_word})"
        )
    else:
        typer.echo("\n  ✓  High-confidence tiers well-calibrated.")


def _print_stability_section(
    df_seasons: DataFrame,
    *,
    target_mv: str,
    divider: LiteralString,
) -> None:
    """Print section 3: season-over-season Brier with drift flag."""
    typer.echo(f"\n{divider}")
    typer.echo(f"[3] Season-over-Season Brier — {target_mv}")
    typer.echo(divider)
    typer.echo(df_seasons.to_string(index=False))

    warn_seasons: DataFrame = df_seasons.loc[df_seasons["trend"] == "⚠", :]
    if warn_seasons.empty:
        typer.echo("\n  ✓  Performance stable across all seasons.")
        return

    # pyrefly: ignore [no-matching-overload]
    worst_s: Series = warn_seasons.sort_values("delta_vs_mean", ascending=False).iloc[0]
    typer.echo(
        f"\n  ⚠  Possible drift: '{worst_s['season']}' is "
        f"{worst_s['delta_vs_mean']:+.5f} vs mean — monitor next season."
    )


def _print_misses_section(
    df_misses: DataFrame,
    *,
    target_mv: str,
    top_misses: int,
    season: str | None,
    divider: LiteralString,
) -> None:
    """Print section 4: top-N misses with heuristic pattern summary."""
    typer.echo(f"\n{divider}")
    typer.echo(f"[4] Top {top_misses} Misses — {target_mv}")
    if season:
        typer.echo(f"    season filter: {season}")
    typer.echo(divider)
    typer.echo(df_misses.to_string(index=False))

    early_mask: Series[bool] = df_misses["week"] <= 3
    n_early: int = early_mask.sum()
    if n_early >= 3:
        typer.echo(
            f"\n  ⚠  {n_early}/{top_misses} worst misses in weeks 1-3 "
            f"(early-season EPA instability is a likely contributor)."
        )

    loss_misses: DataFrame = df_misses.loc[df_misses["actual_result"] == "LOSS", :]
    n_losses: int = len(loss_misses)
    if n_losses >= top_misses // 2:
        typer.echo(
            f"\n  ⚠  {n_losses}/{top_misses} worst misses were losses for the "
            f"predicted favorite — overconfidence pattern."
        )

    typer.echo("")


@evaluate_app.command("report")
def evaluate_report(
    *,
    model_version: str = typer.Option(
        "auto",
        help=(
            "Model to report on. Use 'auto' (default) to auto-select the best "
            "model via composite ranking, or pass a specific version string."
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

    With --model-version auto (default), the best model is chosen automatically.
    Pass a specific version to force analysis of that model regardless of ranking.

    \b
    Examples:
      gridiron evaluate report
      gridiron evaluate report --model-version logistic
      gridiron evaluate report --top-misses 20
      gridiron evaluate report --season 2025-2026
      gridiron evaluate report --model-version elo_v1 --season 2025-2026
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import PredictorRegistry

    repo: Path = get_settings().repo_root

    # ── Parse criteria ──────────────────────────────────────────────────────
    valid_criteria: set[str] = {"brier", "ece", "auc", "accuracy", "log_loss"}
    lower_is_better: set[str] = {"brier", "ece", "log_loss"}
    criteria_list: list[str] = [c.strip().lower() for c in criteria.split(",")]
    invalid: list[str] = [c for c in criteria_list if c not in valid_criteria]
    if invalid:
        typer.echo(f"Unknown criteria: {invalid}. Valid: {sorted(valid_criteria)}")
        raise typer.Exit(1)

    auto_select: bool = model_version == "auto"
    subtitle: str = f"model={'auto-select' if auto_select else model_version}"
    if season:
        subtitle += f"  season={season}"
    console.header("evaluate report", subtitle=subtitle)

    # ── Rank all models ─────────────────────────────────────────────────────
    with step("Compute metrics — all models") as s:
        all_rows: list[dict] = _collect_model_metrics(PredictorRegistry.names(), repo=repo)
        s.set_detail(f"{len(all_rows)} models with archived predictions")

    if not all_rows:
        typer.echo("No models with archived predictions. Run 'gridiron evaluate backfill' first.")
        raise typer.Exit(1)

    ranked_df: DataFrame = _rank_models(
        all_rows, criteria_list=criteria_list, lower_is_better=lower_is_better
    )
    rank_cols: list[str] = [f"rank_{c}" for c in criteria_list]
    display_cols: list[str] = [
        "model_version",
        "n_games",
        "brier",
        "ece",
        "auc",
        "accuracy",
        *rank_cols,
        "composite_rank",
    ]
    recommended_mv: str = str(ranked_df.iloc[0]["model_version"])
    target_mv: str = recommended_mv if auto_select else model_version

    if target_mv not in {str(r["model_version"]) for r in all_rows}:
        typer.echo(
            f"No archived predictions for '{target_mv}'. "
            f"Run 'gridiron evaluate backfill --model-version {target_mv}' first."
        )
        raise typer.Exit(1)

    # ── Compute depth metrics ────────────────────────────────────────────────
    try:
        _df_eval, df_tiers, df_seasons, df_misses = _compute_report_data(
            target_mv=target_mv, season=season, top_misses=top_misses, repo=repo
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    console.summary()

    # ── Print report ────────────────────────────────────────────────────────
    _divider: LiteralString = "─" * 60
    _print_ranking_section(
        ranked_df,
        display_cols,
        auto_select=auto_select,
        recommended_mv=recommended_mv,
        target_mv=target_mv,
        criteria_list=criteria_list,
        lower_is_better=lower_is_better,
        divider=_divider,
    )
    _print_confidence_section(df_tiers, target_mv=target_mv, season=season, divider=_divider)
    _print_stability_section(df_seasons, target_mv=target_mv, divider=_divider)
    _print_misses_section(
        df_misses, target_mv=target_mv, top_misses=top_misses, season=season, divider=_divider
    )

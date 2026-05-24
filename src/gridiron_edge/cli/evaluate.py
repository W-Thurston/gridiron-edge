# src/gridiron_edge/cli/evaluate.py
"""CLI commands for model evaluation."""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import typer

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
    subtitle = f"model={model_version}"
    if season:
        subtitle += f"  season={season}"
    subtitle += f"  group={group_by}"
    console.header("evaluate summary", subtitle=subtitle)

    with step("Join predictions to outcomes") as s:
        df_eval = build_evaluation_df(model_version=mv_filter, season=season)
        if df_eval.empty:
            s.set_detail("no evaluated games — run 'output predictions' first")
        else:
            s.set_detail(f"{len(df_eval)} games")

    if not df_eval.empty:
        with step("Compute metrics") as s:
            df_summary = summarise(df_eval, group_by=group_by)
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
    subtitle = f"model={model_version}"
    if season:
        subtitle += f"  season={season}"
    console.header("evaluate calibration", subtitle=subtitle)

    with step("Join predictions to outcomes") as s:
        df_eval = build_evaluation_df(model_version=mv_filter_cal, season=season)
        if df_eval.empty:
            s.set_detail("no evaluated games — run 'output predictions' first")
        else:
            s.set_detail(f"{len(df_eval)} games")

    if not df_eval.empty:
        with step("Build calibration table") as s:
            df_cal = calibration_table(df_eval, n_buckets=buckets)
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
        n = backfill_model(model_version, overwrite=overwrite)
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
        n_combos = (
            len(K_EARLY_VALUES)
            * len(K_MID_VALUES)
            * len(K_WEEK18_VALUES)
            * len(K_POST_VALUES)
            * len(DIVISOR_VALUES_V3)
            * len(REGRESS_VALUES_V3)
        )
        subtitle = f"elo_v3  {n_combos} combinations  holdout={sorted(HOLDOUT_SEASONS)}"
    else:
        n_combos = len(K_VALUES) * len(DIVISOR_VALUES) * len(REGRESS_VALUES)
        subtitle = f"elo_v2  {n_combos} combinations  holdout={sorted(HOLDOUT_SEASONS)}"
    console.header("evaluate tune", subtitle=subtitle)

    with step(f"Run grid search ({n_combos} combinations)") as s:
        from gridiron_edge.core.settings import get_settings

        _out_dir = get_settings().data_output / "tune"
        _fname = "elo_v3_tune_results.parquet" if v3 else "elo_v2_tune_results.parquet"
        _save = _out_dir / _fname if save else None
        results = run_grid_search_v3(save_path=_save) if v3 else run_grid_search(save_path=_save)
        best_holdout = results.iloc[0]["holdout_brier"]
        s.set_detail(f"best holdout Brier: {best_holdout:.5f}")

    with step(f"Top {top} results") as s:
        s.set_detail(f"{len(results)} total combinations scored")

    typer.echo(results.head(top).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    if apply:
        if v3:
            params = best_params_v3(results)
            typer.echo(
                f"Applying best elo_v3 params: "
                f"k_early={params['k_early']:.0f}  k_mid={params['k_mid']:.0f}  "
                f"k_week18={params['k_week18']:.0f}  k_post={params['k_post']:.0f}  "
                f"divisor={params['divisor']:.0f}  regress={params['regress_frac']:.2f}"
            )
            with step("Backfill elo_v3 predictions") as s:
                from gridiron_edge.evaluation.backfill import backfill_model

                n = backfill_model("elo_v3", overwrite=True)
                s.set_detail(f"{n:,} predictions archived as elo_v3")
        else:
            params = best_params(results)
            k_val = params["k"]
            div_val = params["divisor"]
            reg_val = params["regress_frac"]
            typer.echo(
                f"Applying best elo_v2 params: "
                f"k={k_val:.0f}  divisor={div_val:.0f}  regress={reg_val:.2f}"
            )
            with step("Backfill elo_v2 predictions") as s:
                from gridiron_edge.evaluation.backfill import backfill_model

                n = backfill_model("elo_v2", overwrite=True)
                s.set_detail(f"{n:,} predictions archived as elo_v2")

    console.summary()

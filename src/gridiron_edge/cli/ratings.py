# src/gridiron_edge/cli/ratings.py
"""CLI commands for ratings systems."""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import typer

ratings_app = typer.Typer(help="Ratings systems (Elo, etc.)", no_args_is_help=True)
elo_app = typer.Typer(help="Elo rating system", no_args_is_help=True)
ratings_app.add_typer(elo_app, name="elo")


@elo_app.command("fit")
def elo_fit(
    *,
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Rebuild full Elo history vs incremental update.",
    ),
) -> None:
    """Build/update Elo state table."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ratings.elo import fit_elo

    mode = "full rebuild" if all_years else "incremental"
    console.header("ratings elo fit", subtitle=mode)

    with step(f"Fit Elo ({mode})"):
        fit_elo(all_years=all_years)

    console.summary()


@elo_app.command("predict")
def elo_predict(
    *,
    year: str = typer.Option(..., help="NFL season label like '2025-2026'."),
    week: int = typer.Option(..., help="Week number to predict."),
) -> None:
    """Write Elo win probabilities for upcoming games to Excel."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ratings.elo.predict import predict_elo_only

    console.header("ratings elo predict", subtitle=f"{year}  week {week}")

    with step(f"Predict Elo (week {week})"):
        predict_elo_only(year=year, week=week)

    console.summary()


@elo_app.command("evaluate")
def elo_evaluate() -> None:
    r"""Print Elo prediction accuracy by year and by week.

    Deprecated: use 'gridiron evaluate summary' for full metrics including
    Brier score, log loss, calibration, and multi-model comparison.

    \b
    Equivalent to:
      gridiron evaluate summary --model-name win_prob --model-type elo --group-by season
      gridiron evaluate summary --model-name win_prob --model-type elo --group-by week
    """
    typer.echo("Note: superseded by 'gridiron evaluate summary'.\n")

    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.metrics import build_evaluation_df, summarise

    console.header("ratings elo evaluate", subtitle="win_prob/elo")

    with step("Join predictions to outcomes") as s:
        # Intentional: `ratings elo evaluate` is Elo-specific by design.
        # The command's purpose is Elo prediction accuracy; using the
        # current champion here would defeat the point.
        df_eval = build_evaluation_df(model_name="win_prob", model_type="elo")

        if df_eval.empty:
            s.set_detail("no data - run 'gridiron evaluate backfill' first")
        else:
            s.set_detail(f"{len(df_eval)} games")

    if not df_eval.empty:
        with step("Accuracy by season") as s:
            df_year = summarise(df_eval, group_by="season")
            s.set_detail(f"{len(df_year)} seasons")
        typer.echo(df_year.to_string(index=False))

        typer.echo("")

        with step("Accuracy by week") as s:
            df_week = summarise(df_eval, group_by="week")
            s.set_detail(f"{len(df_week)} weeks")
        typer.echo(df_week.to_string(index=False))

    console.summary()

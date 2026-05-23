# src/gridiron_edge/cli/ratings.py
"""CLI commands for ratings systems."""

from __future__ import annotations

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
    """Print Elo prediction accuracy by year and by week."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.evaluation.elo import evaluate_elo

    console.header("ratings elo evaluate")

    with step("Evaluate by year"):
        evaluate_elo(time_period="YEAR")

    with step("Evaluate by week"):
        evaluate_elo(time_period="WEEK")

    console.summary()

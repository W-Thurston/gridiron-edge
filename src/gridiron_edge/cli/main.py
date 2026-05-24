# src/gridiron_edge/cli/main.py
"""Gridiron Edge CLI — main entrypoint.

Assembles the top-level Typer app from sub-module apps and registers
the run-data-pipeline command. All sub-apps are lazy-loaded via their
own modules so that --help renders quickly and imports only happen when
a command is actually invoked.
"""

from __future__ import annotations

from typing import Annotated

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli.evaluate import evaluate_app
from gridiron_edge.cli.features import features_app
from gridiron_edge.cli.ingest import ingest_app
from gridiron_edge.cli.models import models_app
from gridiron_edge.cli.output import output_app
from gridiron_edge.cli.ratings import ratings_app
from gridiron_edge.cli.sim import sim_app
from gridiron_edge.cli.transform import transform_app
from gridiron_edge.core.logging import setup_logging
from gridiron_edge.core.settings import ensure_data_dirs

_verbose_state: dict[str, bool] = {"verbose": False}


def _cli_startup(
    _ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose/--no-verbose",
            "-v",
            help="Verbose output: step details, row counts, file paths, and debug logs.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Initialise data directories, logging, and console verbosity."""
    from gridiron_edge.core.console import console

    _verbose_state["verbose"] = verbose
    console.set_verbose(verbose)
    ensure_data_dirs()
    setup_logging(verbose=verbose)


app = typer.Typer(
    name="gridiron",
    help="Gridiron Edge CLI: ingest, transform, features, ratings, output.",
    no_args_is_help=True,
    callback=_cli_startup,
)

app.add_typer(ingest_app, name="ingest")
app.add_typer(transform_app, name="transform")
app.add_typer(features_app, name="features")
app.add_typer(ratings_app, name="ratings")
app.add_typer(output_app, name="output")
app.add_typer(sim_app, name="sim")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(models_app, name="models")


# ===========================================================================
# FULL PIPELINE
# ===========================================================================


@app.command("run-data-pipeline")
def run_data_pipeline(  # noqa: PLR0912, PLR0915
    *,
    season: int | None = typer.Option(
        None,
        help=(
            "Season year (e.g. 2025). Defaults to the current season. Omit when using --all-years."
        ),
    ),
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Fetch/build full history instead of current season only.",
    ),
    fetch_games_flag: bool = typer.Option(
        True,
        "--fetch-games/--no-fetch-games",
    ),
    clean_games_flag: bool = typer.Option(
        True,
        "--clean-games/--no-clean-games",
    ),
    fetch_upcoming_flag: bool = typer.Option(
        True,
        "--fetch-upcoming/--no-fetch-upcoming",
    ),
    clean_upcoming_flag: bool = typer.Option(
        True,
        "--clean-upcoming/--no-clean-upcoming",
    ),
    fetch_weather_flag: bool = typer.Option(
        False,
        "--fetch-weather/--no-fetch-weather",
    ),
    season_year: str | None = typer.Option(
        None,
        help="Required if --fetch-weather (e.g. '2025-2026').",
    ),
    owm_api_key: str | None = typer.Option(
        None,
        help="OpenWeather API key or env var OWM_API_KEY.",
    ),
    fetch_dk_odds_flag: bool = typer.Option(
        False,
        "--fetch-odds/--no-fetch-odds",
    ),
    build_elo_flag: bool = typer.Option(
        False,
        "--build-elo/--no-build-elo",
    ),
    fit_elo_all_years: bool = typer.Option(
        False,
        "--fit-elo-all-years/--no-fit-elo-all-years",
        help="When --build-elo, rebuild full Elo history.",
    ),
    build_features_flag: bool = typer.Option(
        True,
        "--build-features/--no-build-features",
    ),
    upcoming_season: int | None = typer.Option(
        None,
        help=(
            "Season to fetch upcoming schedule for. Defaults to --season or current "
            "season. Use when fetching upcoming games from a different season than "
            "completed games (e.g. --all-years --upcoming-season 2026)."
        ),
    ),
) -> None:
    r"""Run a full end-to-end data pipeline with toggles for each stage.

    \b
    Scenario 1 — weekly refresh (most common, no flags needed):
      gridiron run-data-pipeline

    \b
    Scenario 2 — specific season:
      gridiron run-data-pipeline --season 2025

    \b
    Scenario 3 — full history rebuild:
      gridiron run-data-pipeline --all-years --upcoming-season 2026 --build-elo --fit-elo-all-years
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ingest.nflverse import (
        fetch_nflverse_games,
        fetch_nflverse_games_refresh,
        fetch_nflverse_upcoming,
    )
    from gridiron_edge.ingest.nflverse.games import _current_nfl_season
    from gridiron_edge.transform.clean import clean_nflverse_games, clean_nflverse_upcoming

    resolved_season = season or _current_nfl_season()
    upcoming_target = upcoming_season or resolved_season

    if all_years and upcoming_season:
        mode = f"full history  ·  upcoming season {upcoming_season}"
    elif all_years:
        mode = "full history rebuild"
    elif season:
        mode = f"season {resolved_season}"
    else:
        mode = f"weekly refresh  ·  season {resolved_season}"

    console.header("run-data-pipeline", subtitle=mode)

    with step("Fetch nflverse games", skip=not fetch_games_flag) as s:
        if fetch_games_flag:
            if all_years:
                path = fetch_nflverse_games()
            elif season:
                path = fetch_nflverse_games(seasons=[resolved_season])
            else:
                path = fetch_nflverse_games_refresh()
            s.set_detail(path.name)

    with step("Clean games", skip=not clean_games_flag) as s:
        if clean_games_flag:
            path = clean_nflverse_games()
            s.set_detail(path.name)

    with step("Fetch upcoming schedule", skip=not fetch_upcoming_flag) as s:
        if fetch_upcoming_flag:
            path = fetch_nflverse_upcoming(season=upcoming_target)
            s.set_detail(path.name)

    with step("Clean upcoming schedule", skip=not clean_upcoming_flag) as s:
        if clean_upcoming_flag:
            path = clean_nflverse_upcoming()
            s.set_detail(path.name)

    with step("Fetch weather", skip=not fetch_weather_flag) as s:
        if fetch_weather_flag:
            from gridiron_edge.ingest.weather import fetch_weather

            if not season_year:
                raise typer.BadParameter(
                    "When --fetch-weather is set, provide --season-year (e.g. '2025-2026').",
                )
            from gridiron_edge.cli._shared import get_owm_api_key

            key = get_owm_api_key(owm_api_key)
            fetch_weather(season_year=season_year, owm_api_key=key)
            s.set_detail(season_year)

    with step("Fetch DraftKings odds", skip=not fetch_dk_odds_flag):
        if fetch_dk_odds_flag:
            from gridiron_edge.ingest.odds import fetch_dk_odds

            fetch_dk_odds()

    with step("Fit Elo", skip=not build_elo_flag) as s:
        if build_elo_flag:
            from gridiron_edge.ratings.elo import fit_elo

            fit_elo(all_years=fit_elo_all_years)
            s.set_detail("full rebuild" if fit_elo_all_years else "incremental")

    with step("Build model inputs", skip=not build_features_flag):
        if build_features_flag:
            from gridiron_edge.features.pipeline import build_model_inputs

            build_model_inputs(all_years=all_years)

    console.summary()


# ===========================================================================
# ENTRYPOINT
# ===========================================================================


def main() -> None:
    """Entry point for the Gridiron Edge CLI."""
    app()


if __name__ == "__main__":
    main()

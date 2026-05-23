# src/gridiron_edge/cli.py
"""CLI entrypoint for Gridiron Edge.

uv run gridiron --help
"""

from __future__ import annotations

import os
from typing import Annotated

import typer

# Only lightweight imports at module level.
from gridiron_edge.core.logging import setup_logging
from gridiron_edge.core.settings import ensure_data_dirs

_verbose_state: dict[str, bool] = {"verbose": False}


def _cli_startup(
    ctx: typer.Context,
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

ingest_app = typer.Typer(help="Ingest raw data from external sources.", no_args_is_help=True)
transform_app = typer.Typer(help="Clean/curate data into canonical datasets.", no_args_is_help=True)
features_app = typer.Typer(help="Build feature tables and modeling matrices.", no_args_is_help=True)
ratings_app = typer.Typer(help="Ratings systems (Elo, etc.)", no_args_is_help=True)
elo_app = typer.Typer(help="Elo rating system", no_args_is_help=True)
output_app = typer.Typer(help="Write reports and Excel outputs.", no_args_is_help=True)
sim_app = typer.Typer(help="Monte Carlo season + playoff simulation.", no_args_is_help=True)

app.add_typer(ingest_app, name="ingest")
app.add_typer(transform_app, name="transform")
app.add_typer(features_app, name="features")
app.add_typer(ratings_app, name="ratings")
ratings_app.add_typer(elo_app, name="elo")
app.add_typer(output_app, name="output")
app.add_typer(sim_app, name="sim")


def _get_owm_api_key(owm_api_key: str | None) -> str:
    """Resolve OpenWeather API key from flag or environment variable."""
    key: str | None = owm_api_key or os.environ.get("OWM_API_KEY")
    if not key:
        raise typer.BadParameter(
            "Missing OpenWeather API key. Provide --owm-api-key or set env var OWM_API_KEY.",
        )
    return key


# ===========================================================================
# INGEST — nflverse
# ===========================================================================


@ingest_app.command("nflverse-games")
def ingest_nflverse_games(
    *,
    season: list[int] = typer.Option(  # noqa: B008
        [],
        help=(
            "Specific season year(s) to fetch (e.g. --season 2025). "
            "Repeatable: --season 2024 --season 2025. "
            "If omitted, refreshes the current season."
        ),
    ),
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Fetch full history from 1999 to present.",
    ),
    start_season: int = typer.Option(1999, help="First season when --all-years is set."),
) -> None:
    r"""Fetch NFL game results from nflverse.

    \b
    Examples:
      gridiron ingest nflverse-games                          # current season
      gridiron ingest nflverse-games --season 2025            # specific season
      gridiron ingest nflverse-games --season 2024 --season 2025
      gridiron ingest nflverse-games --all-years              # full history
      gridiron ingest nflverse-games --all-years --start-season 2015
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ingest.nflverse import fetch_nflverse_games, fetch_nflverse_games_refresh
    from gridiron_edge.ingest.nflverse.games import _current_nfl_season

    if all_years:
        label = f"Fetch nflverse games ({start_season}-present)"
    elif season:
        label = f"Fetch nflverse games (season(s): {', '.join(str(s) for s in season)})"
    else:
        label = f"Fetch nflverse games (season {_current_nfl_season()})"

    console.header("ingest nflverse-games", subtitle=label)

    with step(label) as s:
        if all_years:
            path = fetch_nflverse_games(start_season=start_season)
        elif season:
            path = fetch_nflverse_games(seasons=list(season))
        else:
            path = fetch_nflverse_games_refresh()
        s.set_detail(str(path))

    console.summary()


@ingest_app.command("nflverse-upcoming")
def ingest_nflverse_upcoming(
    *,
    season: int | None = typer.Option(
        None,
        help="Season year (e.g. 2025). Defaults to current season.",
    ),
) -> None:
    """Fetch upcoming (unplayed) games for the season from nflverse."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ingest.nflverse import fetch_nflverse_upcoming
    from gridiron_edge.ingest.nflverse.games import _current_nfl_season

    target = season or _current_nfl_season()
    console.header("ingest nflverse-upcoming", subtitle=f"Season {target}")

    with step(f"Fetch upcoming schedule (season {target})") as s:
        path = fetch_nflverse_upcoming(season=target)
        s.set_detail(str(path))

    console.summary()


@ingest_app.command("weather")
def ingest_weather(
    season_year: str = typer.Option(..., help="NFL season label like '2025-2026'."),
    owm_api_key: str | None = typer.Option(
        None,
        help="OpenWeather API key. If omitted, uses env var OWM_API_KEY.",
    ),
) -> None:
    """Pull historical weather for the most recent week in the given season."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ingest.weather import fetch_weather

    key = _get_owm_api_key(owm_api_key)
    console.header("ingest weather", subtitle=f"Season {season_year}")

    with step(f"Fetch weather ({season_year})"):
        fetch_weather(season_year=season_year, owm_api_key=key)

    console.summary()


@ingest_app.command("dk-odds")
def ingest_dk_odds() -> None:
    """Pull DraftKings odds for the current NFL week."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ingest.odds import fetch_dk_odds

    console.header("ingest dk-odds")

    with step("Fetch DraftKings odds"):
        fetch_dk_odds()

    console.summary()


# ===========================================================================
# TRANSFORM
# ===========================================================================


@transform_app.command("clean-games")
def clean_games() -> None:
    """Clean nflverse raw games into canonical games CSV."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.transform.clean import clean_nflverse_games

    console.header("transform clean-games")

    with step("Clean nflverse games") as s:
        path = clean_nflverse_games()
        s.set_detail(str(path))

    console.summary()


@transform_app.command("clean-upcoming")
def clean_upcoming() -> None:
    """Clean nflverse raw upcoming schedule into canonical schedule CSV."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.transform.clean import clean_nflverse_upcoming

    console.header("transform clean-upcoming")

    with step("Clean upcoming schedule") as s:
        path = clean_nflverse_upcoming()
        s.set_detail(str(path))

    console.summary()


# ===========================================================================
# RATINGS
# ===========================================================================


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


# ===========================================================================
# FEATURES
# ===========================================================================


@features_app.command("model-inputs")
def builder_model_inputs(
    *,
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Rebuild all modeling rows vs append new weeks.",
    ),
) -> None:
    """Build modeling base and full feature files."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.features.pipeline import build_model_inputs

    mode = "full rebuild" if all_years else "incremental"
    console.header("features model-inputs", subtitle=mode)

    with step(f"Build model inputs ({mode})"):
        build_model_inputs(all_years=all_years)

    console.summary()


# ===========================================================================
# OUTPUT
# ===========================================================================


@output_app.command("ranks")
def output_ranks(
    *,
    year: str = typer.Option(..., help="NFL season label like '2025-2026'."),
    week: int = typer.Option(..., help="Week number for rank comparison."),
) -> None:
    """Write Elo ranking changes to Excel."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.viz.excel import write_elo_rankings_csv

    console.header("output ranks", subtitle=f"{year}  week {week}")

    with step(f"Write rank changes (week {week})"):
        write_elo_rankings_csv(year=year, week=week)

    console.summary()


# ===========================================================================
# SIMULATION
# ===========================================================================


@sim_app.command("run")
def sim_run(
    *,
    n_sims: int = typer.Option(10_000, help="Number of Monte Carlo simulations."),
    k_factor: float = typer.Option(20.0, help="Elo K-factor."),
    p_tie: float = typer.Option(0.01, help="Probability of a tie game."),
    seed: int = typer.Option(1337, help="Base random seed."),
    render: bool = typer.Option(
        True,
        "--render/--no-render",
        help="Render playoff probability table PNG after simulation.",
    ),
) -> None:
    """Run Monte Carlo season + playoff simulation, write CSVs, and optionally render viz."""
    from gridiron_edge.core.console import console, step
    from gridiron_edge.sim import SimPaths, SimulationConfig, run_full_simulation

    config = SimulationConfig(
        n_sims=n_sims,
        k_factor=k_factor,
        p_tie=p_tie,
        base_seed=seed,
    )
    subtitle = f"{n_sims:,} simulations  ·  seed {seed}"
    if render:
        subtitle += "  ·  +render"
    console.header("sim run", subtitle=subtitle)

    with step(f"Simulate season + playoffs ({n_sims:,} sims)"):
        paths = SimPaths.from_settings()
        run_full_simulation(paths=paths, config=config, render=render)

    console.summary()


# ===========================================================================
# FULL PIPELINE
# ===========================================================================


@app.command("run-data-pipeline")
def run_data_pipeline(
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
      gridiron run-data-pipeline --all-years --build-elo --fit-elo-all-years
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

    if all_years:
        mode = "full history rebuild"
    elif season:
        mode = f"season {resolved_season}"
    else:
        mode = f"weekly refresh  ·  season {resolved_season}"

    console.header("run-data-pipeline", subtitle=mode)

    with step("Fetch nflverse games", skip=not fetch_games_flag) as s:
        if all_years:
            path = fetch_nflverse_games()
        elif season:
            path = fetch_nflverse_games(seasons=[resolved_season])
        else:
            path = fetch_nflverse_games_refresh()
        s.set_detail(path.name)

    with step("Clean games", skip=not clean_games_flag) as s:
        path = clean_nflverse_games()
        s.set_detail(path.name)

    with step("Fetch upcoming schedule", skip=not fetch_upcoming_flag) as s:
        path = fetch_nflverse_upcoming(season=resolved_season)
        s.set_detail(path.name)

    with step("Clean upcoming schedule", skip=not clean_upcoming_flag) as s:
        path = clean_nflverse_upcoming()
        s.set_detail(path.name)

    with step("Fetch weather", skip=not fetch_weather_flag) as s:
        from gridiron_edge.ingest.weather import fetch_weather

        if not season_year:
            raise typer.BadParameter(
                "When --fetch-weather is set, provide --season-year (e.g. '2025-2026').",
            )
        key = _get_owm_api_key(owm_api_key)
        fetch_weather(season_year=season_year, owm_api_key=key)
        s.set_detail(season_year)

    with step("Fetch DraftKings odds", skip=not fetch_dk_odds_flag):
        from gridiron_edge.ingest.odds import fetch_dk_odds

        fetch_dk_odds()

    with step("Fit Elo", skip=not build_elo_flag) as s:
        from gridiron_edge.ratings.elo import fit_elo

        fit_elo(all_years=fit_elo_all_years)
        s.set_detail("full rebuild" if fit_elo_all_years else "incremental")

    with step("Build model inputs", skip=not build_features_flag):
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

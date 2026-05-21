# src/gridiron_edge/cli.py
"""CLI entrypoint for Gridiron Edge.

poetry run gridiron --help
"""

import os

import typer

from gridiron_edge.core.logging import setup_logging
from gridiron_edge.core.settings import ensure_data_dirs
from gridiron_edge.evaluation.elo import evaluate_elo
from gridiron_edge.features.pipeline import build_model_inputs
from gridiron_edge.ingest.odds import fetch_dk_odds
from gridiron_edge.ingest.pfr import fetch_historical, fetch_upcoming
from gridiron_edge.ingest.weather import fetch_weather
from gridiron_edge.ratings.elo import fit_elo, predict_elo_only
from gridiron_edge.sim import SimPaths, SimulationConfig, run_full_simulation
from gridiron_edge.sim.season import (
    add_game_id_to_schedule,
    build_conf_div_arrays_from_csv,
    build_schedule_arrays,
    build_team_index_from_results,
    load_long_to_short_mapping,
)
from gridiron_edge.transform.clean import (
    clean_historical_games,
    clean_upcoming_schedule,
)
from gridiron_edge.viz.excel import write_elo_rank_changes


def _cli_startup() -> None:
    ensure_data_dirs()
    setup_logging()


app = typer.Typer(
    name="gridiron",
    help="Gridiron Edge CLI: ingest, transform, features, ratings, output.",
    no_args_is_help=True,
    callback=_cli_startup,
)

ingest_app = typer.Typer(
    help="Ingest raw data from external sources.",
    no_args_is_help=True,
)
transform_app = typer.Typer(
    help="Clean/curate data into canonical datasets.",
    no_args_is_help=True,
)
features_app = typer.Typer(
    help="Build feature tables and modeling matrices.",
    no_args_is_help=True,
)
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
    """Resolve OpenWeather API key from flag or env var."""
    key: str | None = owm_api_key or os.environ.get("OWM_API_KEY")
    if not key:
        raise typer.BadParameter(
            "Missing OpenWeather API key. Provide --owm-api-key or set env var OWM_API_KEY.",
        )
    return key


@ingest_app.command("pfr-historical")
def ingest_pfr_historical(
    *,
    year: str = typer.Option("2023", help="Season year to append (e.g. '2023')."),
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Scrape full history vs append only.",
    ),
) -> None:
    """Scrape historical week-by-week data from PFR via Scrapy."""
    fetch_historical(all_years=all_years, year=year)


@ingest_app.command("pfr-upcoming")
def ingest_pfr_upcoming() -> None:
    """Scrape upcoming schedule from PFR via Scrapy."""
    fetch_upcoming()


@ingest_app.command("weather")
def ingest_weather(
    season_year: str = typer.Option(..., help="NFL season label like '2023-2024'."),
    owm_api_key: str | None = typer.Option(
        None,
        help="OpenWeather API key. If omitted, uses env var OWM_API_KEY.",
    ),
) -> None:
    """Pull historical weather for the most recent week in the given season."""
    key: str = _get_owm_api_key(owm_api_key)
    fetch_weather(season_year=season_year, owm_api_key=key)


@ingest_app.command("dk-odds")
def ingest_dk_odds() -> None:
    """Pull DraftKings odds for the current NFL week and write into the Excel output."""
    fetch_dk_odds()


@transform_app.command("clean-historical")
def clean_historical() -> None:
    """Clean historical week-by-week raw scrape into cleaned file."""
    clean_historical_games()


@transform_app.command("clean-upcoming")
def clean_upcoming() -> None:
    """Clean upcoming schedule raw scrape into cleaned file."""
    clean_upcoming_schedule()


@elo_app.command("fit")
def elo_fit(
    *,
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Rebuild full Elo history vs append only.",
    ),
) -> None:
    """Build/update Elo state (writes elo_state dataset)."""
    fit_elo(all_years=all_years)


@elo_app.command("predict")
def elo_predict(
    *,
    year: str = typer.Option(..., help="NFL season label like '2025-2026'."),
    week: int = typer.Option(..., help="Week number to predict."),
) -> None:
    """Write Elo win probabilities for upcoming games to Excel."""
    predict_elo_only(year=year, week=week)


@elo_app.command("evaluate")
def elo_evaluate() -> None:
    """Print Elo prediction accuracy by year and by week."""
    print("> By Year:")
    evaluate_elo(time_period="YEAR")
    print("> By Week:")
    evaluate_elo(time_period="WEEK")


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
    build_model_inputs(all_years=all_years)


@output_app.command("ranks")
def output_ranks(
    *,
    year: str = typer.Option(..., help="NFL season label like '2025-2026'."),
    week: int = typer.Option(..., help="Week number for rank comparison."),
) -> None:
    """Write Elo ranking changes (week vs week+1) to Excel."""
    write_elo_rank_changes(year=year, week=week)


@app.command("run-data-pipeline")
def run_data_pipeline(
    *,
    year: str = typer.Option("2023", help="Year for append scrape (e.g., '2023')."),
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Scrape/build full history for ingest and features.",
    ),
    fetch_historical_flag: bool = typer.Option(
        True,
        "--fetch-historical/--no-fetch-historical",
    ),
    clean_historical_flag: bool = typer.Option(
        True,
        "--clean-historical/--no-clean-historical",
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
        help="Required if --fetch-weather. Example: '2023-2024'",
    ),
    owm_api_key: str | None = typer.Option(
        None,
        help="OpenWeather API key or env var OWM_API_KEY.",
    ),
    fetch_dk_odds_flag: bool = typer.Option(False, "--fetch-odds/--no-fetch-odds"),
    build_elo_flag: bool = typer.Option(False, "--build-elo/--no-build-elo"),
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
    """Run a full end-to-end data pipeline with toggles for each stage."""
    if fetch_historical_flag:
        fetch_historical(all_years=all_years, year=year)
    elif clean_historical_flag:
        print(
            "> Skipping historical scrape (--no-fetch-historical); "
            "clean will use existing raw CSV.",
            flush=True,
        )

    if clean_historical_flag:
        print("> Transform: clean historical games...", flush=True)
        clean_historical_games()

    if fetch_upcoming_flag:
        fetch_upcoming()
    elif clean_upcoming_flag:
        print(
            "> Skipping upcoming scrape (--no-fetch-upcoming); clean will use existing raw CSV.",
            flush=True,
        )

    if clean_upcoming_flag:
        print("> Transform: clean upcoming schedule...", flush=True)
        clean_upcoming_schedule()

    if fetch_weather_flag:
        if not season_year:
            raise typer.BadParameter(
                "When --fetch-weather is set, you must provide --season-year (e.g. '2023-2024').",
            )
        key: str = _get_owm_api_key(owm_api_key)
        fetch_weather(season_year=season_year, owm_api_key=key)

    if fetch_dk_odds_flag:
        fetch_dk_odds()

    if build_elo_flag:
        print("> Ratings: fit Elo...", flush=True)
        fit_elo(all_years=fit_elo_all_years)

    if build_features_flag:
        print("> Features: build model inputs...", flush=True)
        build_model_inputs(all_years=all_years)

    print("> run-data-pipeline complete.", flush=True)


@sim_app.command("run")
def sim_run(
    *,
    n_sims: int = typer.Option(10_000, help="Number of Monte Carlo simulations."),
    k_factor: float = typer.Option(20.0, help="Elo K-factor."),
    p_tie: float = typer.Option(0.01, help="Probability of tie game."),
    seed: int = typer.Option(1337, help="Base random seed."),
    render: bool = typer.Option(
        True,
        "--render/--no-render",
        help="Render playoff probability table image.",
    ),
) -> None:
    """Run Monte Carlo season + playoff simulation and write output CSVs."""
    import pandas as pd

    config = SimulationConfig(
        n_sims=n_sims,
        k_factor=k_factor,
        p_tie=p_tie,
        base_seed=seed,
    )
    paths = SimPaths.from_settings()

    _df_projections, _df_season_grid = run_full_simulation(paths=paths, config=config)

    if render:
        # Build viz table and render image — needs elo + full schedule data
        df_schedule = pd.read_csv(paths.schedule_file)
        df_wk_by_wk = pd.read_csv(paths.wk_by_wk_file)

        season_year = str(df_schedule["YEAR"].iloc[0])
        long_to_short = load_long_to_short_mapping(paths.mapping_file)
        team_index = build_team_index_from_results(df_wk_by_wk, long_to_short, season_year)

        df_schedule_gid = add_game_id_to_schedule(df_schedule, long_to_short)
        schedule, final_actual_week = build_schedule_arrays(
            df_schedule_gid,
            df_wk_by_wk,
            team_index,
            season_year,
        )
        conf_id, div_id = build_conf_div_arrays_from_csv(team_index, paths.conf_div_file)

        from gridiron_edge.sim.season import (
            apply_actuals_to_matrices,
            precompute_game_counts,
        )

        _gp_total, _gp_conf, _gp_div, _opp_mask = precompute_game_counts(
            schedule,
            conf_id,
            div_id,
        )
        (_, _, _, _, _, _, _, _) = apply_actuals_to_matrices(
            schedule.home,
            schedule.away,
            schedule.week_offsets,
            schedule.result,
            final_actual_week,
            conf_id,
            div_id,
        )

        # Reload sim results from saved CSVs to get pts_total_by_sim
        # (cheapest path: re-run sim kernel or load from temp CSVs)
        # For now, render is a best-effort after the simulation is complete.
        # Full integration (passing arrays through) is a follow-up once
        # run_full_simulation returns SimulationResults instead of DataFrames.
        print(
            "> Note: --render requires re-reading saved CSVs. "
            "Full array passthrough is a planned improvement.",
            flush=True,
        )

        print("> Skipping render for now — integrate build_viz_table_df() here.")

    print("> sim run complete.", flush=True)


def main() -> None:
    """Entry point for the Gridiron Edge CLI."""
    app()


if __name__ == "__main__":
    main()

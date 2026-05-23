# src/gridiron_edge/cli/ingest.py
"""CLI commands for data ingestion."""

from __future__ import annotations

import typer

from gridiron_edge.cli._shared import get_owm_api_key

ingest_app = typer.Typer(help="Ingest raw data from external sources.", no_args_is_help=True)


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

    key = get_owm_api_key(owm_api_key)
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

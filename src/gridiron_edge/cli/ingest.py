# src/gridiron_edge/cli/ingest.py
"""CLI commands for data ingestion."""

from __future__ import annotations

from pathlib import Path

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._shared import get_owm_api_key

ingest_app = typer.Typer(help="Ingest raw data from external sources.", no_args_is_help=True)


@ingest_app.command("nflverse-games")
def ingest_nflverse_games(
    *,
    season: list[int] = typer.Option(  # noqa: B008
        [],
        help=(
            "Specific season year(s) to refresh while preserving "
            "all other raw history (e.g. --season 2025). "
            "Repeatable: --season 2024 --season 2025. "
            "If omitted, refreshes the current season."
        ),
    ),
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help=(
            "Replace the raw artifact with full history from "
            "the selected start season through the present."
        ),
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
    from gridiron_edge.core.settings import current_nfl_season
    from gridiron_edge.ingest.nflverse.games import (
        fetch_nflverse_games,
        fetch_nflverse_games_refresh,
        refresh_nflverse_game_seasons,
    )

    if all_years:
        label: str = f"Fetch nflverse games ({start_season}-present)"
    elif season:
        label = f"Fetch nflverse games (season(s): {', '.join(str(s) for s in season)})"
    else:
        label = f"Fetch nflverse games (season {current_nfl_season()})"

    console.header("ingest nflverse-games", subtitle=label)

    with step(label) as s:
        if all_years:
            path: Path = fetch_nflverse_games(start_season=start_season)
        elif season:
            path = refresh_nflverse_game_seasons(seasons=list(season))
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
    from gridiron_edge.core.settings import current_nfl_season
    from gridiron_edge.ingest.nflverse import fetch_nflverse_upcoming

    target: int = season or current_nfl_season()
    console.header("ingest nflverse-upcoming", subtitle=f"Season {target}")

    with step(f"Fetch upcoming schedule (season {target})") as s:
        path: Path = fetch_nflverse_upcoming(season=target)
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

    key: str = get_owm_api_key(owm_api_key)
    console.header("ingest weather", subtitle=f"Season {season_year}")

    with step(f"Fetch weather ({season_year})"):
        fetch_weather(season_year=season_year, owm_api_key=key)

    console.summary()


@ingest_app.command("dk-odds")
def ingest_dk_odds() -> None:
    """Run the legacy best-effort DraftKings adapter explicitly.

    The supported current-market workflow uses nflverse schedule data. This
    historical adapter may be unavailable and is not required by normal data
    refresh or weekly prediction workflows.
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ingest.odds import (
        DraftKingsUnavailableError,
        fetch_dk_odds,
    )

    console.header(
        "ingest dk-odds",
        subtitle="Legacy best-effort adapter",
    )

    try:
        with step("Run legacy DraftKings adapter") as current_step:
            result = fetch_dk_odds()

            if result is None:
                detail = "No current rows returned; no files written"
            else:
                ledger_path, snapshot_path = result
                detail = f"ledger={ledger_path}; snapshot={snapshot_path}"

            current_step.set_detail(detail)
    except DraftKingsUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(detail)
    console.summary()


@ingest_app.command("pbp")
def ingest_pbp(
    *,
    season: list[int] = typer.Option(  # noqa: B008
        [],
        help="Specific season year(s) to fetch. Repeatable: --season 2024 --season 2025.",
    ),
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Fetch full history from 1999 to present (~540MB).",
    ),
    start_season: int = typer.Option(1999, help="First season when --all-years is set."),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Re-fetch and overwrite existing files.",
    ),
) -> None:
    r"""Fetch NFL play-by-play data from nflverse (stored in data/raw/pbp/).

    Downloads per-season Parquet files (~20MB each). Complete seasons are
    cached permanently and never re-fetched unless --force is set.

    \b
    Examples:
      gridiron ingest pbp                    # refresh current season
      gridiron ingest pbp --season 2025      # specific season
      gridiron ingest pbp --all-years        # full history (~540MB)
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import current_nfl_season
    from gridiron_edge.ingest.nflverse.pbp import (
        fetch_pbp,
        fetch_pbp_refresh,
    )

    if all_years:
        label: str = f"Fetch PBP ({start_season}-present)"
    elif season:
        label = f"Fetch PBP (season(s): {', '.join(str(s) for s in season)})"
    else:
        label = f"Fetch PBP (season {current_nfl_season()}, refresh)"

    console.header("ingest pbp", subtitle=label)

    with step(label) as s:
        if all_years:
            paths: list[Path] = fetch_pbp(start_season=start_season, force=force)
        elif season:
            paths = fetch_pbp(seasons=list(season), force=force)
        else:
            paths = fetch_pbp_refresh()
        s.set_detail(f"{len(paths)} file(s) written")

    console.summary()

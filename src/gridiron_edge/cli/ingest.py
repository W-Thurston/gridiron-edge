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
        label: str = f"Fetch nflverse games ({start_season}-present)"
    elif season:
        label = f"Fetch nflverse games (season(s): {', '.join(str(s) for s in season)})"
    else:
        label = f"Fetch nflverse games (season {_current_nfl_season()})"

    console.header("ingest nflverse-games", subtitle=label)

    with step(label) as s:
        if all_years:
            path: Path = fetch_nflverse_games(start_season=start_season)
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

    target: int = season or _current_nfl_season()
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


@ingest_app.command("weather-backfill")
def ingest_weather_backfill(
    *,
    season_year: str | None = typer.Option(
        None,
        help=(
            "Backfill a specific season only (e.g. '2024-2025'). "
            "If omitted alongside --all-years, all seasons are processed."
        ),
    ),
    all_years: bool = typer.Option(
        False,
        "--all-years/--no-all-years",
        help="Backfill all historical seasons.",
    ),
    owm_api_key: str | None = typer.Option(
        None,
        help="OpenWeather API key. If omitted, uses env var OWM_API_KEY.",
    ),
    max_calls: int | None = typer.Option(
        None,
        help=(
            "Maximum API calls to make in this run. "
            "Use to stay within the OWM daily limit (1,000 on the base "
            "subscription). Stops cleanly after N calls; run again tomorrow "
            "to continue. Defaults to no limit."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help=(
            "Show what would be fetched without making any API calls. "
            "Prints a season-by-season breakdown of pending games."
        ),
    ),
) -> None:
    r"""Backfill historical weather for all games not yet in the archive.

    Fetches weather at kickoff time from the OWM One Call API 3.0
    timemachine endpoint for every completed game not already present
    in weather_enriched.csv.  Already-fetched games are skipped
    automatically — only genuinely missing data is requested.

    Progress is flushed to disk every 50 games so the run can be
    interrupted and resumed without losing work.  Failed games are
    written to data/cleaned/weather_backfill_failed.csv for inspection.

    Requires an OWM One Call API 3.0 subscription (paid tier).

    \b
    Examples:
      gridiron ingest weather-backfill --all-years --dry-run
      gridiron ingest weather-backfill --all-years --max-calls 900
      gridiron ingest weather-backfill --season-year 2024-2025
      gridiron ingest weather-backfill --season-year 2024-2025 --dry-run
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.ingest.weather.backfill import backfill_weather

    key: str = get_owm_api_key(owm_api_key)

    target: str = season_year if (season_year and not all_years) else "all seasons"
    cap_label: str = f"  max={max_calls}" if max_calls is not None else ""
    subtitle: str = f"{target}{cap_label}{' [DRY RUN]' if dry_run else ''}"
    console.header("ingest weather-backfill", subtitle=subtitle)

    with step("Fetch historical weather") as s:
        n_fetched, n_failed = backfill_weather(
            season_year=season_year if not all_years else None,
            owm_api_key=key,
            dry_run=dry_run,
            max_calls=max_calls,
        )
        if dry_run:
            s.set_detail("dry run — no data written")
        else:
            s.set_detail(f"{n_fetched:,} fetched  ·  {n_failed:,} failed")

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
    from gridiron_edge.ingest.nflverse.pbp import (
        _current_nfl_season,
        fetch_pbp,
        fetch_pbp_refresh,
    )

    if all_years:
        label: str = f"Fetch PBP ({start_season}-present)"
    elif season:
        label = f"Fetch PBP (season(s): {', '.join(str(s) for s in season)})"
    else:
        label = f"Fetch PBP (season {_current_nfl_season()}, refresh)"

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

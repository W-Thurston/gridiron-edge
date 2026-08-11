# src/gridiron_edge/cli/ingest.py
"""CLI commands for data ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gridiron_edge.market.collection_plan import WeeklyQuoteCollectionPlan

from pathlib import Path

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._shared import get_odds_api_key, get_owm_api_key

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


@ingest_app.command("odds")
def ingest_odds(
    *,
    season_year: str = typer.Option(
        ...,
        "--season",
        help="NFL season label like '2026-2027'.",
    ),
    week: int = typer.Option(
        ...,
        min=1,
        max=22,
        help="NFL week number from 1 through 22.",
    ),
    odds_api_key: str | None = typer.Option(
        None,
        help="The Odds API key. If omitted, uses env var ODDS_API_KEY.",
    ),
    timeout: float = typer.Option(
        15.0,
        min=0.1,
        help="Provider request timeout in seconds.",
    ),
) -> None:
    r"""Fetch and persist current NFL featured-market quotes.

    The command reads the canonical rich schedule for matching, appends the
    observation ledger, and atomically replaces the current snapshot only after
    a successful request and nonempty matched parse.

    
    Example:
      gridiron ingest odds --season 2026-2027 --week 1
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_schedule_upcoming_rich
    from gridiron_edge.ingest.odds.the_odds_api import ingest_the_odds_api_current

    key = get_odds_api_key(odds_api_key)
    settings = get_settings()
    schedule = load_schedule_upcoming_rich(settings.repo_root)
    label = f"Fetch current odds ({season_year} week {week})"
    console.header("ingest odds", subtitle=label)

    with step(label) as current_step:
        result = ingest_the_odds_api_current(
            api_key=key,
            schedule=schedule,
            season=season_year,
            week=week,
            repo=settings.repo_root,
            timeout=timeout,
        )
        summary = (
            f"{result.quote_count} quotes, {result.game_count} games, "
            f"{result.sportsbook_count} sportsbooks"
        )
        current_step.set_detail(summary)

    typer.echo(summary)
    usage = result.usage
    if usage.requests_remaining is not None:
        typer.echo(f"Requests remaining: {usage.requests_remaining}")
    if usage.requests_used is not None:
        typer.echo(f"Requests used: {usage.requests_used}")
    if usage.request_cost is not None:
        typer.echo(f"Request cost: {usage.request_cost}")
    typer.echo(f"Ledger: {result.ledger_path}")
    typer.echo(f"Snapshot: {result.snapshot_path}")
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


@ingest_app.command("plan-odds")
def plan_odds(
    *,
    season_year: str = typer.Option(..., "--season", help="NFL season label like '2026-2027'."),
    week: int = typer.Option(..., min=1, max=22, help="NFL week number from 1 through 22."),
    plan_start: str = typer.Option(
        ..., help="Explicit UTC ISO timestamp at which planning begins."
    ),
    created_at: str = typer.Option(..., help="Explicit UTC ISO timestamp recorded on the plan."),
    poll_limit: int = typer.Option(34, min=1, help="Maximum planned provider polls for the week."),
    credit_cost_per_poll: int = typer.Option(3, min=1, help="Provider credits consumed per poll."),
) -> None:
    """Generate and persist a reviewable weekly odds-collection plan."""
    from datetime import datetime

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_schedule_upcoming_rich
    from gridiron_edge.market.collection_plan import (
        QuoteCollectionPolicy,
        build_weekly_quote_collection_plan,
    )
    from gridiron_edge.market.collection_plan_store import write_collection_plan

    settings = get_settings()
    schedule = load_schedule_upcoming_rich(settings.repo_root)
    policy = QuoteCollectionPolicy(
        weekly_poll_limit=poll_limit,
        credit_cost_per_poll=credit_cost_per_poll,
    )
    plan = build_weekly_quote_collection_plan(
        schedule,
        season=season_year,
        week=week,
        plan_start=datetime.fromisoformat(plan_start.replace("Z", "+00:00")),
        created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
        policy=policy,
    )
    path = write_collection_plan(plan, repo=settings.repo_root)
    typer.echo(f"Plan status: {plan.status.value}")
    typer.echo(f"Planned polls: {plan.planned_poll_count}/{plan.policy.weekly_poll_limit}")
    typer.echo(f"Projected credits: {plan.planned_credit_cost}")
    typer.echo(f"Omitted candidates: {plan.omitted_candidate_count}")
    typer.echo(f"Plan: {path}")


def _execute_loaded_odds_plan(
    plan: WeeklyQuoteCollectionPlan,
    *,
    evaluated_at: str,
    grace_minutes: int,
    minimum_credit_reserve: int,
    odds_api_key: str | None,
    timeout: float,
) -> None:
    """Execute one resolved plan through the shared CLI boundary."""
    from datetime import datetime, timedelta

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_schedule_upcoming_rich
    from gridiron_edge.market.collection_execution import execute_due_collection

    settings = get_settings()
    outcome = execute_due_collection(
        plan,
        schedule=load_schedule_upcoming_rich(settings.repo_root),
        api_key=get_odds_api_key(odds_api_key),
        evaluated_at=datetime.fromisoformat(evaluated_at.replace("Z", "+00:00")),
        repo=settings.repo_root,
        grace_period=timedelta(minutes=grace_minutes),
        minimum_credit_reserve=minimum_credit_reserve,
        timeout=timeout,
    )
    typer.echo(f"Execution status: {outcome.status.value}")


@ingest_app.command("select-odds-plan")
def select_odds_plan(
    *,
    season_year: str = typer.Option(..., "--season", help="NFL season label like '2026-2027'."),
    week: int = typer.Option(..., min=1, max=22, help="NFL week number from 1 through 22."),
    selected_at: str = typer.Option(..., help="Explicit UTC ISO selection timestamp."),
) -> None:
    """Explicitly select one existing validated odds-collection plan."""
    from datetime import datetime

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.market.collection_plan_store import (
        collection_plan_path,
        current_collection_plan_path,
        select_current_collection_plan,
    )

    settings = get_settings()
    selection = select_current_collection_plan(
        season=season_year,
        week=week,
        selected_at=datetime.fromisoformat(selected_at.replace("Z", "+00:00")),
        repo=settings.repo_root,
    )
    typer.echo(f"Selected season: {selection.season}")
    typer.echo(f"Selected week: {selection.week}")
    typer.echo(f"Selected at: {selection.selected_at.isoformat()}")

    plan_path: Path = collection_plan_path(
        season=selection.season,
        week=selection.week,
        repo=settings.repo_root,
    )
    selection_path: Path = current_collection_plan_path(repo=settings.repo_root)

    typer.echo(f"Plan: {plan_path}")
    typer.echo(f"Current selection: {selection_path}")


@ingest_app.command("execute-odds-plan")
def execute_odds_plan(
    *,
    season_year: str = typer.Option(..., "--season", help="NFL season label like '2026-2027'."),
    week: int = typer.Option(..., min=1, max=22, help="NFL week number from 1 through 22."),
    evaluated_at: str = typer.Option(..., help="Explicit UTC ISO evaluation timestamp."),
    grace_minutes: int = typer.Option(15, min=0, help="Inclusive due-time grace period."),
    minimum_credit_reserve: int = typer.Option(
        30,
        min=0,
        help="Credits protected from automated collection.",
    ),
    odds_api_key: str | None = typer.Option(None, help="The Odds API key or ODDS_API_KEY."),
    timeout: float = typer.Option(15.0, min=0.1, help="Provider request timeout in seconds."),
) -> None:
    """Evaluate and execute at most one due poll from a validated plan."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.market.collection_plan_store import read_collection_plan

    settings = get_settings()
    plan = read_collection_plan(season=season_year, week=week, repo=settings.repo_root)
    _execute_loaded_odds_plan(
        plan,
        evaluated_at=evaluated_at,
        grace_minutes=grace_minutes,
        minimum_credit_reserve=minimum_credit_reserve,
        odds_api_key=odds_api_key,
        timeout=timeout,
    )


@ingest_app.command("execute-selected-odds-plan")
def execute_selected_odds_plan(
    *,
    evaluated_at: str = typer.Option(..., help="Explicit UTC ISO evaluation timestamp."),
    grace_minutes: int = typer.Option(15, min=0, help="Inclusive due-time grace period."),
    minimum_credit_reserve: int = typer.Option(
        30,
        min=0,
        help="Credits protected from automated collection.",
    ),
    odds_api_key: str | None = typer.Option(None, help="The Odds API key or ODDS_API_KEY."),
    timeout: float = typer.Option(15.0, min=0.1, help="Provider request timeout in seconds."),
) -> None:
    """Execute at most one due poll from the explicitly selected plan."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.market.collection_plan_store import load_current_collection_plan

    settings = get_settings()
    plan = load_current_collection_plan(repo=settings.repo_root)
    _execute_loaded_odds_plan(
        plan,
        evaluated_at=evaluated_at,
        grace_minutes=grace_minutes,
        minimum_credit_reserve=minimum_credit_reserve,
        odds_api_key=odds_api_key,
        timeout=timeout,
    )

# src/gridiron_edge/cli/transform.py
"""CLI commands for data transformation."""

from __future__ import annotations

from pathlib import Path

# pyrefly: ignore [missing-import]
import typer

transform_app = typer.Typer(help="Clean/curate data into canonical datasets.", no_args_is_help=True)


@transform_app.command("clean-games")
def clean_games() -> None:
    """Clean nflverse raw games into canonical games CSV."""
    from gridiron_edge.core.console import console, step

    # pyrefly: ignore [missing-module-attribute]
    from gridiron_edge.transform.clean import clean_nflverse_games

    console.header("transform clean-games")

    with step("Clean nflverse games") as s:
        path = clean_nflverse_games()
        s.set_detail(str(path))

    console.summary()


@transform_app.command("clean-upcoming")
def clean_upcoming() -> None:
    """Clean nflverse raw upcoming games into the rich schedule Parquet."""
    from gridiron_edge.core.console import console, step

    # pyrefly: ignore [missing-module-attribute]
    from gridiron_edge.transform.clean import clean_nflverse_upcoming

    console.header("transform clean-upcoming")

    with step("Clean upcoming schedule") as s:
        path = clean_nflverse_upcoming()
        s.set_detail(str(path))

    console.summary()


@transform_app.command("clean-qb-coach-sidecar")
def clean_qb_coach_sidecar() -> None:
    """Build the QB-and-coach identity sidecar from cached raw nflverse schedules."""
    from gridiron_edge.core.console import console, step

    # pyrefly: ignore [missing-module-attribute]
    from gridiron_edge.transform.clean import build_qb_coach_sidecar

    console.header("transform clean-qb-coach-sidecar")

    with step("Build QB/coach sidecar") as s:
        path = build_qb_coach_sidecar()
        s.set_detail(str(path))

    console.summary()


@transform_app.command("clean-player-stats")
def clean_player_stats_command() -> None:
    """Build the cleaned player game logs from cached raw player stats.

    Reads every cached season in data/raw/player_stats/ and writes
    data/cleaned/player_game_logs.parquet. Attaching game identities looks up
    nflverse schedules, so this step needs network access.
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.transform.clean.player_stats import clean_player_stats

    console.header("transform clean-player-stats")

    with step("Clean player stats") as s:
        path = clean_player_stats()
        s.set_detail(str(path))

    console.summary()


@transform_app.command("aggregate-epa")
def transform_aggregate_epa(
    *,
    season: list[int] = typer.Option(  # noqa: B008
        [],
        help="Specific season(s) to aggregate. Defaults to all cached seasons.",
    ),
) -> None:
    r"""Aggregate play-by-play data to game-level EPA stats.

    Reads from cached data/raw/pbp/ files and writes data/cleaned/epa_by_game.parquet.
    Run after 'gridiron ingest pbp'. Incremental by default -- pass --season
    to update only specific seasons.

    \b
    Examples:
      gridiron transform aggregate-epa             # all cached seasons
      gridiron transform aggregate-epa --season 2025   # current season only
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.transform.clean.epa import aggregate_epa

    seasons_arg: list[int] | None = list(season) if season else None
    subtitle: str = f"seasons {seasons_arg}" if seasons_arg else "all cached seasons"
    console.header("transform aggregate-epa", subtitle=subtitle)

    with step("Aggregate PBP to game-level EPA") as s:
        path: Path = aggregate_epa(seasons=seasons_arg)
        s.set_detail(str(path))

    console.summary()

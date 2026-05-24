# src/gridiron_edge/cli/transform.py
"""CLI commands for data transformation."""

from __future__ import annotations

import typer

transform_app = typer.Typer(help="Clean/curate data into canonical datasets.", no_args_is_help=True)


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

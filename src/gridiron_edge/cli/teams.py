# src/gridiron_edge/cli/teams.py

"""CLI commands for team-level analytics."""

from __future__ import annotations

import logging
from logging import Logger

import typer

logger: Logger = logging.getLogger(__name__)

teams_app = typer.Typer(help="Team-level analytics.", no_args_is_help=True)


@teams_app.command("compute-cohort-splits")
def compute_cohort_splits_cmd() -> None:
    """Compute per-team cohort splits from EPA data.

    For each team and each cohort (season, l4, home, away), computes
    8 metrics (off/def EPA, third-down pct, redzone TD pct, etc.)
    and ranks each team within the cohort.

    Writes the artifact to
    ``data/output/rankings/team_cohort_splits.parquet``.

    Consumed by:
    - `/compare/teams` to populate the cohort_splits field.
    - `/teams/{abbr}` to populate the cohort_splits field.
    - `/games/{game_id}` to populate the team_comparison field.
    """
    import pandas as pd

    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_teams_long_short
    from gridiron_edge.evaluation.team_cohort_splits import (
        compute_team_cohort_splits,
        write_team_cohort_splits,
    )

    settings = get_settings()
    repo = settings.repo_root

    console.header("teams compute-cohort-splits")

    with step("Load EPA data") as s:
        epa_path = repo / "data" / "cleaned" / "epa_by_game.parquet"
        if not epa_path.exists():
            typer.echo(f"EPA data not found at {epa_path}")
            raise typer.Exit(code=1)

        epa = pd.read_parquet(epa_path)
        s.set_detail(f"{len(epa):,} EPA rows")

    with step("Load team name map") as s:
        mapping_df = load_teams_long_short(repo)
        long_to_short = dict(
            zip(
                mapping_df["NFL_LONG_NAME"],
                mapping_df["NFL_SHORT_NAME"],
                strict=True,
            )
        )
        s.set_detail(f"{len(long_to_short)} team mappings")

    with step("Compute team cohort splits") as s:
        df = compute_team_cohort_splits(epa, long_to_short)
        if df.empty:
            typer.echo("No splits produced.")
            raise typer.Exit(code=1)
        s.set_detail(f"{df['team_abbr'].nunique()} teams, {len(df)} rows")

    with step("Persist artifact") as s:
        path = write_team_cohort_splits(df, repo)
        s.set_detail(str(path.relative_to(repo)))

    console.summary()

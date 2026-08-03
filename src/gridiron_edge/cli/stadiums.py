"""CLI for reviewed stadium-reference synchronization."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.loaders import (
    load_schedule_upcoming_rich,
    load_stadiums,
)
from gridiron_edge.metadata.stadium_sync import (
    apply_approved_stadium_updates,
    audit_stadium_coverage,
    load_stadium_aliases,
    prepare_stadium_updates,
)

stadiums_app = typer.Typer(
    help="Audit, prepare, and apply reviewed stadium metadata updates.",
    no_args_is_help=True,
)


def _review_path(repo: Path, season: str) -> Path:
    return repo / "data" / "output" / "stadium_sync" / f"stadium_updates_{season}.csv"


def _stadium_path(repo: Path) -> Path:
    return repo / "data" / "cleaned" / "NFL_stadium_reference.csv"


def _alias_path(repo: Path) -> Path:
    return repo / "data" / "reference" / "stadium_aliases.csv"


def _render_counts(frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty:
        typer.echo("  none")
        return
    counts = frame.groupby(columns, dropna=False).size()
    for identity, count in counts.items():
        values = identity if isinstance(identity, tuple) else (identity,)
        label = " / ".join(str(value) for value in values)
        typer.echo(f"  {label}: {int(count)}")


@stadiums_app.command("audit")
def audit_stadiums(
    season: Annotated[
        str,
        typer.Option(
            "--season",
            help="Season label, for example 2026-2027.",
        ),
    ],
) -> None:
    """Audit current franchise-origin and scheduled-site coverage."""
    repo = get_settings().repo_root
    audit = audit_stadium_coverage(
        load_stadiums(repo),
        load_schedule_upcoming_rich(repo),
        season=season,
    )
    typer.echo(f"stadiums audit  {season}")
    _render_counts(audit, ["ISSUE"])
    if not audit.empty:
        raise typer.Exit(code=1)


@stadiums_app.command("prepare")
def prepare_stadiums(
    season: Annotated[
        str,
        typer.Option(
            "--season",
            help="Season label, for example 2026-2027.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=("Review CSV path. Defaults under data/output/stadium_sync."),
        ),
    ] = None,
) -> None:
    """Write deterministic proposals without changing canonical metadata."""
    repo = get_settings().repo_root
    updates = prepare_stadium_updates(
        load_stadiums(repo),
        load_schedule_upcoming_rich(repo),
        season=season,
        aliases=load_stadium_aliases(_alias_path(repo)),
    )
    out_path = output or _review_path(repo, season)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    updates.to_csv(out_path, index=False)
    typer.echo(f"stadiums prepare  {season}")
    _render_counts(updates, ["ACTION", "REVIEW_STATUS"])
    typer.echo(f"review: {out_path}")


@stadiums_app.command("apply")
def apply_stadiums(
    updates: Annotated[
        Path,
        typer.Option(
            "--updates",
            help="Reviewed stadium update CSV.",
        ),
    ],
    season: Annotated[
        str,
        typer.Option(
            "--season",
            help="Season label to audit after apply.",
        ),
    ],
) -> None:
    """Atomically append approved rows and report remaining coverage."""
    repo = get_settings().repo_root

    if not updates.is_file():
        raise typer.BadParameter(f"Update file does not exist: {updates}")

    review = pd.read_csv(updates)
    stadiums = load_stadiums(repo)

    result = apply_approved_stadium_updates(
        stadiums,
        review,
        path=_stadium_path(repo),
    )

    applied_count = len(result) - len(stadiums)

    remaining = audit_stadium_coverage(
        result,
        load_schedule_upcoming_rich(repo),
        season=season,
    )

    typer.echo(f"stadiums apply  {season}")
    typer.echo(f"approved rows applied: {applied_count}")
    typer.echo("remaining coverage:")
    _render_counts(
        remaining,
        ["ISSUE"],
    )

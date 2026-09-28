# src/gridiron_edge/cli/ops.py

"""Operational commands: local environment and cross-machine data upkeep."""

from __future__ import annotations

import os

# pyrefly: ignore [missing-import]
import typer

ops_app = typer.Typer(
    help="Operational commands for local environment and deployment upkeep.",
    no_args_is_help=True,
)

_DEFAULT_COLLECTOR_USER = "thursty"
_DEFAULT_COLLECTOR_REPOSITORY = "/home/thursty/apps/gridiron-edge"


@ops_app.command("pull-collector-evidence")
def pull_collector_evidence_cmd(
    host: str | None = typer.Option(
        None,
        "--host",
        help="Collector host or IP. Defaults to env var GRIDIRON_COLLECTOR_HOST.",
    ),
    user: str | None = typer.Option(
        None,
        "--user",
        help=f"SSH user on the collector host. Env var GRIDIRON_COLLECTOR_USER, "
        f"then {_DEFAULT_COLLECTOR_USER!r}.",
    ),
    remote_repository: str | None = typer.Option(
        None,
        "--remote-repository",
        help="Collector's repository path. Env var GRIDIRON_COLLECTOR_REPOSITORY, "
        f"then {_DEFAULT_COLLECTOR_REPOSITORY!r}.",
    ),
    identity_file: str | None = typer.Option(
        None,
        "--identity",
        "-i",
        help="SSH private key path. Defaults to env var GRIDIRON_COLLECTOR_SSH_KEY, "
        "or ssh's own default identity resolution if unset.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Show what would be pulled without copying anything.",
    ),
) -> None:
    """Pull collector-owned quote history, plans, and run receipts from the worker.

    One-way and additive only: mirrors the deployed quote-collection worker's
    ``data/odds`` tree into this repository's own ``data/odds``. Never
    touches ``data/output`` and never deletes a local file that is missing
    on the remote side.
    """
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.deployment.collector_sync import (
        CollectorSyncError,
        CollectorSyncTarget,
        pull_collector_evidence,
    )

    resolved_host = host or os.environ.get("GRIDIRON_COLLECTOR_HOST")
    if not resolved_host:
        raise typer.BadParameter(
            "Missing collector host. Provide --host or set env var GRIDIRON_COLLECTOR_HOST."
        )
    resolved_user = user or os.environ.get("GRIDIRON_COLLECTOR_USER") or _DEFAULT_COLLECTOR_USER
    resolved_repository = (
        remote_repository
        or os.environ.get("GRIDIRON_COLLECTOR_REPOSITORY")
        or _DEFAULT_COLLECTOR_REPOSITORY
    )
    resolved_identity = identity_file or os.environ.get("GRIDIRON_COLLECTOR_SSH_KEY")

    target = CollectorSyncTarget(
        host=resolved_host,
        user=resolved_user,
        remote_repository=resolved_repository,
        identity_file=resolved_identity,
    )
    settings = get_settings()
    typer.echo(
        f"pull-collector-evidence  {resolved_user}@{resolved_host}:{resolved_repository}/data/odds/"
    )
    try:
        result = pull_collector_evidence(
            target,
            local_repo=settings.repo_root,
            dry_run=dry_run,
        )
    except CollectorSyncError as exc:
        raise typer.BadParameter(str(exc)) from exc

    transferred = [line for line in result.stdout.splitlines() if line.strip()]
    for line in transferred:
        typer.echo(f"  {line}")
    if dry_run:
        typer.echo(f"would pull {len(transferred)} file(s)")
    elif transferred:
        typer.echo(f"pulled {len(transferred)} file(s)")
    else:
        typer.echo("up to date -- no files needed pulling")

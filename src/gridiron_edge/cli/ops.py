# src/gridiron_edge/cli/ops.py

"""Operational commands: local environment and cross-machine data upkeep."""

from __future__ import annotations

import os

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.deployment.collector_sync import CollectorSyncTarget

ops_app = typer.Typer(
    help="Operational commands for local environment and deployment upkeep.",
    no_args_is_help=True,
)

_DEFAULT_COLLECTOR_USER = "thursty"
_DEFAULT_COLLECTOR_REPOSITORY = "/home/thursty/apps/gridiron-edge"

_HOST_HELP = "Collector host or IP. Defaults to env var GRIDIRON_COLLECTOR_HOST."
_USER_HELP = (
    f"SSH user on the collector host. Env var GRIDIRON_COLLECTOR_USER, "
    f"then {_DEFAULT_COLLECTOR_USER!r}."
)
_REPOSITORY_HELP = (
    f"Collector's repository path. Env var GRIDIRON_COLLECTOR_REPOSITORY, "
    f"then {_DEFAULT_COLLECTOR_REPOSITORY!r}."
)
_IDENTITY_HELP = (
    "SSH private key path. Defaults to env var GRIDIRON_COLLECTOR_SSH_KEY, "
    "or ssh's own default identity resolution if unset."
)


def _resolve_collector_target(
    *,
    host: str | None,
    user: str | None,
    remote_repository: str | None,
    identity_file: str | None,
) -> CollectorSyncTarget:
    resolved_host = host or os.environ.get("GRIDIRON_COLLECTOR_HOST")
    if not resolved_host:
        raise typer.BadParameter(
            "Missing collector host. Provide --host or set env var GRIDIRON_COLLECTOR_HOST."
        )
    return CollectorSyncTarget(
        host=resolved_host,
        user=user or os.environ.get("GRIDIRON_COLLECTOR_USER") or _DEFAULT_COLLECTOR_USER,
        remote_repository=(
            remote_repository
            or os.environ.get("GRIDIRON_COLLECTOR_REPOSITORY")
            or _DEFAULT_COLLECTOR_REPOSITORY
        ),
        identity_file=identity_file or os.environ.get("GRIDIRON_COLLECTOR_SSH_KEY"),
    )


@ops_app.command("pull-collector-evidence")
def pull_collector_evidence_cmd(
    host: str | None = typer.Option(None, "--host", help=_HOST_HELP),
    user: str | None = typer.Option(None, "--user", help=_USER_HELP),
    remote_repository: str | None = typer.Option(
        None, "--remote-repository", help=_REPOSITORY_HELP
    ),
    identity_file: str | None = typer.Option(None, "--identity", "-i", help=_IDENTITY_HELP),
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
        pull_collector_evidence,
    )

    target = _resolve_collector_target(
        host=host, user=user, remote_repository=remote_repository, identity_file=identity_file
    )
    settings = get_settings()
    typer.echo(
        f"pull-collector-evidence  "
        f"{target.user}@{target.host}:{target.remote_repository}/data/odds/"
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


@ops_app.command("rollover-collector-week")
def rollover_collector_week_cmd(
    *,
    season_year: str = typer.Option(..., "--season", help="NFL season label like '2026-2027'."),
    week: int = typer.Option(..., min=1, max=22, help="NFL week number from 1 through 22."),
    poll_limit: int = typer.Option(34, min=1, help="Maximum planned provider polls for the week."),
    credit_cost_per_poll: int = typer.Option(3, min=1, help="Provider credits consumed per poll."),
    host: str | None = typer.Option(None, "--host", help=_HOST_HELP),
    user: str | None = typer.Option(None, "--user", help=_USER_HELP),
    remote_repository: str | None = typer.Option(
        None, "--remote-repository", help=_REPOSITORY_HELP
    ),
    identity_file: str | None = typer.Option(None, "--identity", "-i", help=_IDENTITY_HELP),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt before pushing to the worker."
    ),
) -> None:
    """Refresh, build, select, and push one week's collector plan.

    Refreshes the upcoming schedule from nflverse, builds and persists a new
    weekly collection plan, prints a summary for review, then -- after
    confirmation -- selects it as the local current plan and pushes exactly
    three artifacts to the deployed worker: the rich upcoming schedule, the
    scoped week's plan, and the current-selection pointer. Requires explicit
    ``--season``/``--week``: this never infers which week is current, the
    same as every other weekly command in this CLI.
    """
    from datetime import UTC, datetime

    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.deployment.collector_sync import (
        CollectorSyncError,
        push_collector_plan,
    )
    from gridiron_edge.ingest.nflverse.schedule import fetch_nflverse_upcoming
    from gridiron_edge.market.collection_plan import (
        QuoteCollectionPolicy,
        build_weekly_quote_collection_plan,
    )
    from gridiron_edge.market.collection_plan_store import (
        select_current_collection_plan,
        write_collection_plan,
    )
    from gridiron_edge.transform.clean.schedule_nflverse import clean_nflverse_upcoming

    target = _resolve_collector_target(
        host=host, user=user, remote_repository=remote_repository, identity_file=identity_file
    )
    settings = get_settings()
    now = datetime.now(UTC)

    typer.echo(f"refreshing upcoming schedule for {season_year} (nflverse)...")
    fetch_nflverse_upcoming(season=int(season_year.split("-")[0]), repo=settings.repo_root)
    schedule_path = clean_nflverse_upcoming(repo=settings.repo_root, ingested_at=now)
    typer.echo(f"  {schedule_path}")

    from gridiron_edge.datasets.loaders import load_schedule_upcoming_rich

    schedule = load_schedule_upcoming_rich(settings.repo_root)
    policy = QuoteCollectionPolicy(
        weekly_poll_limit=poll_limit,
        credit_cost_per_poll=credit_cost_per_poll,
    )
    plan = build_weekly_quote_collection_plan(
        schedule,
        season=season_year,
        week=week,
        plan_start=now,
        created_at=now,
        policy=policy,
    )
    plan_path = write_collection_plan(plan, repo=settings.repo_root)

    game_count = sum(len(group.game_ids) for group in plan.kickoff_groups)
    typer.echo("")
    typer.echo(f"built plan: season={plan.season} week={plan.week}")
    typer.echo(f"  status              {plan.status.value}")
    typer.echo(
        f"  games               {game_count} across {len(plan.kickoff_groups)} kickoff group(s)"
    )
    typer.echo(f"  planned polls       {plan.planned_poll_count}/{policy.weekly_poll_limit}")
    typer.echo(f"  projected credits   {plan.planned_credit_cost}")
    typer.echo(f"  omitted candidates  {plan.omitted_candidate_count}")
    if plan.polls:
        scheduled = sorted(poll.scheduled_at for poll in plan.polls)
        typer.echo(f"  first poll          {scheduled[0].isoformat()}")
        typer.echo(f"  last poll           {scheduled[-1].isoformat()}")
    typer.echo(f"  plan                {plan_path}")
    typer.echo("")

    if not yes and not typer.confirm(
        f"Select {plan.season} week {plan.week} as current and push it to "
        f"{target.user}@{target.host}?"
    ):
        typer.echo("Aborted -- plan was persisted but not selected or pushed.")
        raise typer.Exit(code=1)

    selection = select_current_collection_plan(
        season=season_year,
        week=week,
        selected_at=now,
        repo=settings.repo_root,
    )
    typer.echo(f"selected current: {selection.season} week {selection.week}")

    try:
        push_collector_plan(
            target,
            local_repo=settings.repo_root,
            season=season_year,
            week=week,
        )
    except CollectorSyncError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"pushed schedule, plan, and current selection to "
        f"{target.user}@{target.host}:{target.remote_repository}"
    )
    typer.echo("")
    typer.echo("Verify on the worker:")
    typer.echo(
        f"  ssh {target.user}@{target.host} 'cd {target.remote_repository} && "
        f"sudo .venv/bin/python -B deploy/bin/verify_quote_collection_worker.py "
        f"--repository {target.remote_repository} --user {target.user} "
        f"--group {target.user} --uv-path ~/.local/bin/uv "
        f"--environment-file /etc/gridiron-edge-collector.env "
        f"--wrapper-path /usr/local/libexec/gridiron-edge-collector "
        f"--service-path /etc/systemd/system/gridiron-edge-collector.service "
        f"--timer-path /etc/systemd/system/gridiron-edge-collector.timer'"
    )

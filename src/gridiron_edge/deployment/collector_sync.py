# src/gridiron_edge/deployment/collector_sync.py

"""One-way pull of collector-owned evidence from the deployed quote worker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class CollectorSyncError(RuntimeError):
    """The collector-evidence pull failed."""


@dataclass(frozen=True, slots=True)
class CollectorSyncTarget:
    """Explicit connection details for one deployed quote-collection worker."""

    host: str
    user: str
    remote_repository: str
    identity_file: str | None = None


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        check=False,
        capture_output=True,
        text=True,
    )


def build_rsync_command(
    target: CollectorSyncTarget,
    *,
    local_repo: Path,
    dry_run: bool = False,
) -> tuple[str, ...]:
    """Build the exact rsync invocation for one collector-evidence pull.

    Mirrors only the collector-owned ``data/odds`` tree -- current and
    historical quotes, collection plans, and collection-run claim/result
    receipts. Never touches ``data/output`` or any other locally generated
    evidence, and never deletes a local file that is missing on the remote
    side: the worker is the source of truth for its own append-only data,
    this machine's analysis artifacts are untouched by this command.
    """
    ssh_command = "ssh" if target.identity_file is None else f"ssh -i {target.identity_file}"
    source = f"{target.user}@{target.host}:{target.remote_repository.rstrip('/')}/data/odds/"
    destination = f"{local_repo / 'data' / 'odds'}/"
    command = ["rsync", "-az", "--out-format=%n", "-e", ssh_command]
    if dry_run:
        command.append("--dry-run")
    command.extend((source, destination))
    return tuple(command)


def pull_collector_evidence(
    target: CollectorSyncTarget,
    *,
    local_repo: Path,
    dry_run: bool = False,
    runner: CommandRunner = _run,
) -> subprocess.CompletedProcess[str]:
    """Pull collector-owned evidence from the deployed worker into ``local_repo``.

    Raises:
        CollectorSyncError: rsync exited nonzero.
    """
    command = build_rsync_command(target, local_repo=local_repo, dry_run=dry_run)
    result = runner(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise CollectorSyncError(f"rsync exited {result.returncode}: {detail}")
    return result


def weekly_plan_relative_path(season: str, week: int) -> str:
    """Return the repo-relative path of one week's persisted collection plan."""
    return f"data/odds/collection_plans/season={season}/week={week:02d}.json"


SCHEDULE_RELATIVE_PATH = "data/cleaned/NFL_upcoming_schedule_rich.parquet"
CURRENT_SELECTION_RELATIVE_PATH = "data/odds/collection_plans/current.json"

# Order matters: the current-selection pointer is pushed last, after the
# schedule and the scoped week's plan it will point at. If a push fails
# partway (a bad passphrase attempt mid-transfer, a dropped connection), the
# worker's `current.json` -- untouched until this final step -- keeps
# pointing at whatever it was correctly using before the rollover, rather
# than referencing a plan file that has not arrived yet.
ROLLOVER_RELATIVE_PATHS: tuple[str, ...] = (SCHEDULE_RELATIVE_PATH, CURRENT_SELECTION_RELATIVE_PATH)


def build_push_commands(
    target: CollectorSyncTarget,
    *,
    local_repo: Path,
    season: str,
    week: int,
    dry_run: bool = False,
) -> tuple[tuple[str, ...], ...]:
    """Build the exact rsync invocations for one weekly-rollover push.

    Pushes exactly the three artifacts a weekly rollover requires: the rich
    upcoming schedule, the one scoped week's collection plan, and the global
    current-selection pointer -- in that order, so the pointer is never
    pushed before the plan it selects. Each file is transferred individually
    -- never a directory mirror -- so this can never touch anything on the
    always-on worker beyond those three exact paths. ``--mkpath`` creates a
    missing remote parent directory (a new season, for example) without
    otherwise altering remote state.
    """
    ssh_command = "ssh" if target.identity_file is None else f"ssh -i {target.identity_file}"
    relative_paths = (
        SCHEDULE_RELATIVE_PATH,
        weekly_plan_relative_path(season, week),
        CURRENT_SELECTION_RELATIVE_PATH,
    )
    commands = []
    for relative_path in relative_paths:
        source = str(local_repo / relative_path)
        destination = (
            f"{target.user}@{target.host}:{target.remote_repository.rstrip('/')}/{relative_path}"
        )
        command = ["rsync", "-az", "--mkpath", "--out-format=%n", "-e", ssh_command]
        if dry_run:
            command.append("--dry-run")
        command.extend((source, destination))
        commands.append(tuple(command))
    return tuple(commands)


def push_collector_plan(
    target: CollectorSyncTarget,
    *,
    local_repo: Path,
    season: str,
    week: int,
    dry_run: bool = False,
    runner: CommandRunner = _run,
) -> tuple[subprocess.CompletedProcess[str], ...]:
    """Push one week's rollover artifacts to the deployed worker.

    Raises:
        CollectorSyncError: any rsync invocation exited nonzero. Earlier
            successful transfers in the same call are not rolled back --
            each of the three files is independently idempotent to retry.
    """
    commands = build_push_commands(
        target, local_repo=local_repo, season=season, week=week, dry_run=dry_run
    )
    results = []
    for command in commands:
        result = runner(command)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
            raise CollectorSyncError(
                f"rsync exited {result.returncode} pushing {command[-2]}: {detail}"
            )
        results.append(result)
    return tuple(results)

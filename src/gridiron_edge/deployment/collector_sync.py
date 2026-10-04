# src/gridiron_edge/deployment/collector_sync.py

"""One-way, merge-safe pull of collector-owned evidence from the deployed quote worker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

from gridiron_edge.deployment.collector_merge import MergeReport, merge_collector_tree
from gridiron_edge.ingest.odds.store import odds_write_lock

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

SYNC_STATUS_FILENAME = "sync_status.json"
DEFAULT_RESTRICTED_COMMAND = "rrsync"


class CollectorSyncError(RuntimeError):
    """The collector-evidence pull failed."""


@dataclass(frozen=True, slots=True)
class CollectorSyncTarget:
    """Explicit connection details for one deployed quote-collection worker.

    ``unattended`` means the identity is a dedicated read-only key whose server-side
    forced command (``authorized_key_line``) is rooted at the worker's ``data/odds``
    directory: the pull then never prompts and asks for the root, not an absolute path.
    """

    host: str
    user: str
    remote_repository: str
    identity_file: str | None = None
    unattended: bool = False

    def __post_init__(self) -> None:
        """Require an explicit identity for an unattended pull."""
        if self.unattended and not self.identity_file:
            raise ValueError("An unattended pull requires an explicit identity file.")


@dataclass(frozen=True, slots=True)
class PullResult:
    """The outcome of one pull: what moved and what the merge did."""

    transferred: tuple[str, ...]
    report: MergeReport
    dry_run: bool


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        check=False,
        capture_output=True,
        text=True,
    )


def _ssh_command(target: CollectorSyncTarget) -> str:
    if target.identity_file is None:
        return "ssh"
    command = f"ssh -i {target.identity_file}"
    if target.unattended:
        command += " -o IdentitiesOnly=yes -o BatchMode=yes"
    return command


def build_rsync_command(target: CollectorSyncTarget, *, destination: Path) -> tuple[str, ...]:
    """Build the exact rsync invocation that stages one copy of the worker's quote tree.

    The copy lands in ``destination`` (a staging directory), never in the live
    ``data/odds``: ``merge_collector_tree`` decides what reaches the real tree. Nothing
    is ever deleted, and ``data/output`` is never touched.
    """
    remote_root = "." if target.unattended else f"{target.remote_repository.rstrip('/')}/data/odds"
    source = f"{target.user}@{target.host}:{remote_root}/"
    return (
        "rsync",
        "-az",
        "--out-format=%n",
        "-e",
        _ssh_command(target),
        source,
        f"{destination}/",
    )


def authorized_key_line(
    public_key: str,
    *,
    remote_repository: str,
    restricted_command: str = DEFAULT_RESTRICTED_COMMAND,
) -> str:
    """Return the ``authorized_keys`` line that limits a key to reading ``data/odds``.

    ``restrict`` disables a pty, forwarding and every other capability; the forced
    ``rrsync -ro`` command allows only read-only rsync rooted at the worker's quote tree.
    """
    key = public_key.strip()
    if "\n" in key or not key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-")):
        raise ValueError("Expected exactly one SSH public key line.")
    root = f"{remote_repository.rstrip('/')}/data/odds"
    return f'restrict,command="{restricted_command} -ro {root}" {key}'


def _write_status(local_repo: Path, **fields: object) -> None:
    path = local_repo / "data" / "odds" / SYNC_STATUS_FILENAME
    previous: dict[str, object] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text())
        except ValueError:
            previous = {}
    status = {**previous, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True))
    temporary.replace(path)


def pull_collector_evidence(
    target: CollectorSyncTarget,
    *,
    local_repo: Path,
    dry_run: bool = False,
    runner: CommandRunner = _run,
) -> PullResult:
    """Stage the worker's quote tree, then merge it into ``local_repo`` without losing evidence.

    A dry run stages and plans the merge but writes nothing to ``data/odds``. Every
    attempt records its outcome in ``data/odds/sync_status.json`` (not for a dry run).

    Raises:
        CollectorSyncError: rsync failed, or the merge found a conflicting immutable file
            (everything else is still merged first).
    """
    attempted_at = datetime.now(UTC).isoformat()
    staging = local_repo / "data" / f".collector_pull_{uuid4().hex}"
    local_odds = local_repo / "data" / "odds"
    try:
        staging.mkdir(parents=True)
        result = runner(build_rsync_command(target, destination=staging))
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
            raise CollectorSyncError(f"rsync exited {result.returncode}: {detail}")
        transferred = tuple(line for line in result.stdout.splitlines() if line.strip())
        with odds_write_lock(repo=local_repo):
            report = merge_collector_tree(staging, local_odds, write=not dry_run)
        if report.conflicts:
            raise CollectorSyncError(
                "Worker files differ from immutable local files and were not replaced: "
                + ", ".join(report.conflicts)
            )
    except (CollectorSyncError, OSError, ValueError, TimeoutError) as exc:
        if not dry_run:
            _write_status(local_repo, last_attempt_at=attempted_at, error=str(exc))
        if isinstance(exc, CollectorSyncError):
            raise
        raise CollectorSyncError(str(exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not dry_run:
        _write_status(
            local_repo,
            last_attempt_at=attempted_at,
            last_success_at=attempted_at,
            error=None,
            rows_added=report.rows_added,
            files_copied=len(report.copied),
        )
    return PullResult(transferred=transferred, report=report, dry_run=dry_run)


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

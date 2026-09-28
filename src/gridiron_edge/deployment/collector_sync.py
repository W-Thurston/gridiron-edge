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

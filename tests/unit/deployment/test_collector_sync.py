# tests/unit/deployment/test_collector_sync.py
"""Tests for the one-way collector-evidence pull."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from gridiron_edge.deployment.collector_sync import (
    CollectorSyncError,
    CollectorSyncTarget,
    build_rsync_command,
    pull_collector_evidence,
)


def _complete(
    command: tuple[str, ...],
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def _target(**overrides: object) -> CollectorSyncTarget:
    defaults: dict[str, object] = {
        "host": "10.0.0.49",
        "user": "thursty",
        "remote_repository": "/home/thursty/apps/gridiron-edge",
        "identity_file": None,
    }
    defaults.update(overrides)
    return CollectorSyncTarget(**defaults)  # type: ignore[arg-type]


def test_build_rsync_command_uses_default_ssh_without_identity(tmp_path: Path) -> None:
    command = build_rsync_command(_target(), local_repo=tmp_path)

    assert command == (
        "rsync",
        "-az",
        "--out-format=%n",
        "-e",
        "ssh",
        "thursty@10.0.0.49:/home/thursty/apps/gridiron-edge/data/odds/",
        f"{tmp_path / 'data' / 'odds'}/",
    )


def test_build_rsync_command_passes_explicit_identity(tmp_path: Path) -> None:
    command = build_rsync_command(
        _target(identity_file="/home/thursty/.ssh/id_ed25519"),
        local_repo=tmp_path,
    )

    assert "-e" in command
    assert command[command.index("-e") + 1] == "ssh -i /home/thursty/.ssh/id_ed25519"


def test_build_rsync_command_strips_trailing_slash_from_remote_repository(
    tmp_path: Path,
) -> None:
    command = build_rsync_command(
        _target(remote_repository="/home/thursty/apps/gridiron-edge/"),
        local_repo=tmp_path,
    )

    assert command[-2] == "thursty@10.0.0.49:/home/thursty/apps/gridiron-edge/data/odds/"


def test_build_rsync_command_never_touches_data_output(tmp_path: Path) -> None:
    command = build_rsync_command(_target(), local_repo=tmp_path)

    assert "data/odds/" in command[-2]
    assert command[-1].endswith("data/odds/")
    assert "output" not in command[-1]


def test_build_rsync_command_dry_run_adds_flag(tmp_path: Path) -> None:
    command = build_rsync_command(_target(), local_repo=tmp_path, dry_run=True)

    assert "--dry-run" in command


def test_pull_collector_evidence_reports_transferred_files(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        return _complete(
            command,
            stdout="week=03/observations.parquet\ncollection_plans/current.json\n",
        )

    result = pull_collector_evidence(_target(), local_repo=tmp_path, runner=runner)

    assert result.returncode == 0
    assert observed["command"] == build_rsync_command(_target(), local_repo=tmp_path)


def test_pull_collector_evidence_raises_on_nonzero_exit(tmp_path: Path) -> None:
    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return _complete(
            command,
            stderr="ssh: connect to host 10.0.0.49 port 22: timed out",
            returncode=255,
        )

    with pytest.raises(CollectorSyncError, match="rsync exited 255"):
        pull_collector_evidence(_target(), local_repo=tmp_path, runner=runner)


def test_pull_collector_evidence_dry_run_does_not_change_command_destructively(
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        return _complete(command)

    pull_collector_evidence(_target(), local_repo=tmp_path, dry_run=True, runner=runner)

    assert "--dry-run" in observed["command"]

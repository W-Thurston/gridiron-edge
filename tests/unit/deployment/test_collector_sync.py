# tests/unit/deployment/test_collector_sync.py
"""Tests for the one-way collector-evidence pull."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from gridiron_edge.deployment.collector_sync import (
    ROLLOVER_RELATIVE_PATHS,
    CollectorSyncError,
    CollectorSyncTarget,
    authorized_key_line,
    build_push_commands,
    build_rsync_command,
    pull_collector_evidence,
    push_collector_plan,
    weekly_plan_relative_path,
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


def test_build_rsync_command_stages_the_remote_tree(tmp_path: Path) -> None:
    command = build_rsync_command(_target(), destination=tmp_path / "stage")

    assert command == (
        "rsync",
        "-az",
        "--out-format=%n",
        "-e",
        "ssh",
        "thursty@10.0.0.49:/home/thursty/apps/gridiron-edge/data/odds/",
        f"{tmp_path / 'stage'}/",
    )


def test_build_rsync_command_passes_explicit_identity(tmp_path: Path) -> None:
    command = build_rsync_command(
        _target(identity_file="/home/thursty/.ssh/id_ed25519"), destination=tmp_path
    )

    assert command[command.index("-e") + 1] == "ssh -i /home/thursty/.ssh/id_ed25519"


def test_build_rsync_command_strips_trailing_slash_from_remote_repository(
    tmp_path: Path,
) -> None:
    command = build_rsync_command(
        _target(remote_repository="/home/thursty/apps/gridiron-edge/"), destination=tmp_path
    )

    assert command[-2] == "thursty@10.0.0.49:/home/thursty/apps/gridiron-edge/data/odds/"


def test_unattended_command_never_prompts_and_asks_for_the_restricted_root(
    tmp_path: Path,
) -> None:
    command = build_rsync_command(
        _target(identity_file="/k/ro", unattended=True), destination=tmp_path
    )

    assert command[command.index("-e") + 1] == (
        "ssh -i /k/ro -o IdentitiesOnly=yes -o BatchMode=yes"
    )
    assert command[-2] == "thursty@10.0.0.49:./"


def test_unattended_target_requires_an_identity() -> None:
    with pytest.raises(ValueError, match="explicit identity"):
        _target(unattended=True)


def test_authorized_key_line_restricts_the_key_to_read_only_rsync() -> None:
    line = authorized_key_line(
        "ssh-ed25519 AAAAC3Nza comment\n",
        remote_repository="/home/thursty/apps/gridiron-edge/",
    )

    assert line == (
        'restrict,command="rrsync -ro /home/thursty/apps/gridiron-edge/data/odds" '
        "ssh-ed25519 AAAAC3Nza comment"
    )


@pytest.mark.parametrize(
    "text",
    ["", "-----BEGIN OPENSSH PRIVATE KEY-----", "ssh-ed25519 a\nssh-ed25519 b"],
)
def test_authorized_key_line_rejects_anything_but_one_public_key(text: str) -> None:
    with pytest.raises(ValueError, match="one SSH public key"):
        authorized_key_line(text, remote_repository="/r")


def _staging_runner(populate):
    """A runner that fills the staging directory the way rsync would."""

    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        populate(Path(command[-1]))
        return _complete(command, stdout="a\nb\n")

    return runner


def test_pull_merges_into_the_local_tree_and_cleans_up_staging(tmp_path: Path) -> None:
    def populate(stage: Path) -> None:
        receipt = stage / "collection_runs" / "season=2026-2027" / "week=04" / "r" / "result.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text("{}")

    result = pull_collector_evidence(
        _target(), local_repo=tmp_path, runner=_staging_runner(populate)
    )

    assert result.transferred == ("a", "b")
    assert result.report.copied == ["collection_runs/season=2026-2027/week=04/r/result.json"]
    assert (tmp_path / "data/odds/collection_runs/season=2026-2027/week=04/r/result.json").exists()
    assert not [
        path for path in (tmp_path / "data").iterdir() if path.name.startswith(".collector")
    ]
    status = json.loads((tmp_path / "data/odds/sync_status.json").read_text())
    assert status["error"] is None
    assert status["files_copied"] == 1
    assert status["last_success_at"] == status["last_attempt_at"]


def test_pull_dry_run_writes_nothing_and_no_status(tmp_path: Path) -> None:
    def populate(stage: Path) -> None:
        (stage / "collection_runs").mkdir()
        (stage / "collection_runs" / "r.json").write_text("{}")

    result = pull_collector_evidence(
        _target(), local_repo=tmp_path, dry_run=True, runner=_staging_runner(populate)
    )

    assert result.report.copied == ["collection_runs/r.json"]
    assert not (tmp_path / "data/odds/collection_runs/r.json").exists()
    assert not (tmp_path / "data/odds/sync_status.json").exists()


def test_pull_raises_on_nonzero_exit_and_records_the_failure(tmp_path: Path) -> None:
    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return _complete(
            command,
            stderr="ssh: connect to host 10.0.0.49 port 22: timed out",
            returncode=255,
        )

    with pytest.raises(CollectorSyncError, match="rsync exited 255"):
        pull_collector_evidence(_target(), local_repo=tmp_path, runner=runner)

    status = json.loads((tmp_path / "data/odds/sync_status.json").read_text())
    assert "timed out" in status["error"]
    assert "last_success_at" not in status


def test_failed_pull_keeps_the_previous_success_time(tmp_path: Path) -> None:
    pull_collector_evidence(_target(), local_repo=tmp_path, runner=_staging_runner(lambda s: None))
    first = json.loads((tmp_path / "data/odds/sync_status.json").read_text())

    def failing(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return _complete(command, stderr="boom", returncode=1)

    with pytest.raises(CollectorSyncError):
        pull_collector_evidence(_target(), local_repo=tmp_path, runner=failing)

    after = json.loads((tmp_path / "data/odds/sync_status.json").read_text())
    assert after["last_success_at"] == first["last_success_at"]
    assert after["error"] == "rsync exited 1: boom"


def test_pull_reports_a_conflicting_immutable_file_after_merging_the_rest(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "data/odds/collection_runs/r.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("local")

    def populate(stage: Path) -> None:
        (stage / "collection_runs").mkdir()
        (stage / "collection_runs" / "r.json").write_text("worker")
        (stage / "collection_runs" / "new.json").write_text("{}")

    with pytest.raises(CollectorSyncError, match=r"collection_runs/r\.json"):
        pull_collector_evidence(_target(), local_repo=tmp_path, runner=_staging_runner(populate))

    assert existing.read_text() == "local"
    assert (tmp_path / "data/odds/collection_runs/new.json").exists()


def test_weekly_plan_relative_path_zero_pads_week() -> None:
    assert (
        weekly_plan_relative_path("2026-2027", 3)
        == "data/odds/collection_plans/season=2026-2027/week=03.json"
    )


def test_build_push_commands_targets_exactly_three_files(tmp_path: Path) -> None:
    commands = build_push_commands(_target(), local_repo=tmp_path, season="2026-2027", week=3)

    assert len(commands) == 3
    destinations = {command[-1] for command in commands}
    assert destinations == {
        "thursty@10.0.0.49:/home/thursty/apps/gridiron-edge/"
        "data/cleaned/NFL_upcoming_schedule_rich.parquet",
        "thursty@10.0.0.49:/home/thursty/apps/gridiron-edge/"
        "data/odds/collection_plans/current.json",
        "thursty@10.0.0.49:/home/thursty/apps/gridiron-edge/"
        "data/odds/collection_plans/season=2026-2027/week=03.json",
    }


def test_build_push_commands_pushes_current_selection_last(tmp_path: Path) -> None:
    """The pointer must never be pushed before the plan it selects: a partial
    failure (a bad passphrase attempt, a dropped connection) must leave the
    worker pointing at whatever it was correctly using before the rollover,
    never at a plan file that has not arrived yet."""
    commands = build_push_commands(_target(), local_repo=tmp_path, season="2026-2027", week=3)

    destinations = [command[-1] for command in commands]
    assert destinations[-1].endswith("data/odds/collection_plans/current.json")
    assert destinations[0].endswith("data/cleaned/NFL_upcoming_schedule_rich.parquet")
    assert destinations[1].endswith("data/odds/collection_plans/season=2026-2027/week=03.json")


def test_build_push_commands_sources_come_from_local_repo(tmp_path: Path) -> None:
    commands = build_push_commands(_target(), local_repo=tmp_path, season="2026-2027", week=3)

    sources = {command[-2] for command in commands}
    expected_relative = (
        *ROLLOVER_RELATIVE_PATHS,
        weekly_plan_relative_path("2026-2027", 3),
    )
    assert sources == {str(tmp_path / path) for path in expected_relative}


def test_build_push_commands_uses_mkpath_and_never_deletes(tmp_path: Path) -> None:
    commands = build_push_commands(_target(), local_repo=tmp_path, season="2026-2027", week=3)

    for command in commands:
        assert "--mkpath" in command
        assert "--delete" not in command


def test_push_collector_plan_runs_all_three_transfers(tmp_path: Path) -> None:
    observed: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        return _complete(command)

    results = push_collector_plan(
        _target(), local_repo=tmp_path, season="2026-2027", week=3, runner=runner
    )

    assert len(results) == 3
    assert len(observed) == 3


def test_push_collector_plan_raises_on_first_failure_and_stops(tmp_path: Path) -> None:
    observed: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        if len(observed) == 1:
            return _complete(command, stderr="no such file or directory", returncode=23)
        return _complete(command)

    with pytest.raises(CollectorSyncError, match="rsync exited 23"):
        push_collector_plan(
            _target(), local_repo=tmp_path, season="2026-2027", week=3, runner=runner
        )
    assert len(observed) == 1


def test_sync_service_template_pulls_unattended_with_the_dedicated_key() -> None:
    template = Path("deploy/systemd/gridiron-edge-collector-sync.service").read_text()

    assert "pull-collector-evidence --unattended --identity" in template
    assert "gridiron_collector_ro" in template

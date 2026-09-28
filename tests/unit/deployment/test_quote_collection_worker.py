"""Tests for repository-owned quote worker deployment."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import subprocess

from pandas import DataFrame
import pytest

import gridiron_edge.deployment.quote_collection_worker as worker
from gridiron_edge.deployment.quote_collection_worker import (
    QuoteCollectionWorkerActivationError,
    QuoteCollectionWorkerConfig,
    QuoteCollectionWorkerInstallationError,
    WorkerCheckStatus,
    WorkerVerificationStatus,
    install_quote_collection_worker,
    render_service,
    render_timer,
    render_wrapper,
    verify_quote_collection_worker,
)
from gridiron_edge.market.collection_plan import build_weekly_quote_collection_plan
from gridiron_edge.market.collection_plan_store import (
    select_current_collection_plan,
    write_collection_plan,
)

SERVICE = """[Unit]
ConditionPathExists=@REPOSITORY@/data/odds/collection_plans/current.json
ConditionPathExists=@REPOSITORY@/data/cleaned/NFL_upcoming_schedule_rich.parquet

[Service]
Type=oneshot
User=@USER@
Group=@GROUP@
WorkingDirectory=@REPOSITORY@
EnvironmentFile=@ENVIRONMENT_FILE@
ExecStart=@WRAPPER@
Restart=no
"""

TIMER = """[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
AccuracySec=15s
Persistent=true
Unit=gridiron-edge-collector.service
"""


def _complete(
    command: tuple[str, ...],
    stdout: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=stdout,
        stderr="",
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    schedule = repo / "data/cleaned/NFL_upcoming_schedule_rich.parquet"
    schedule.parent.mkdir(parents=True)
    schedule.write_bytes(b"schedule")
    frame = DataFrame(
        [
            {
                "season": "2026-2027",
                "week": 1,
                "game_id": "g",
                "game_date": "2026-09-10",
                "game_time": "20:20:00",
            }
        ]
    )
    plan = build_weekly_quote_collection_plan(
        frame,
        season="2026-2027",
        week=1,
        plan_start=datetime(2026, 9, 8, 12, tzinfo=UTC),
        created_at=datetime(2026, 8, 13, 18, tzinfo=UTC),
    )
    write_collection_plan(plan, repo=repo)
    select_current_collection_plan(
        season=plan.season,
        week=plan.week,
        selected_at=datetime(2026, 8, 13, 19, tzinfo=UTC),
        repo=repo,
    )
    return repo


def _config(tmp_path: Path, repo: Path) -> QuoteCollectionWorkerConfig:
    uv_path = tmp_path / "uv"
    uv_path.write_text("#!/bin/sh\n", encoding="utf-8")
    uv_path.chmod(0o755)
    environment = tmp_path / "collector.env"
    environment.write_text("ODDS_API_KEY=secret-value\n", encoding="utf-8")
    environment.chmod(0o600)
    (tmp_path / "installed").mkdir(exist_ok=True)
    return QuoteCollectionWorkerConfig(
        repository=repo,
        deployment_user="root",
        deployment_group="root",
        uv_path=uv_path,
        environment_file=environment,
        wrapper_path=tmp_path / "installed/collector",
        service_path=tmp_path / "installed/collector.service",
        timer_path=tmp_path / "installed/collector.timer",
    )


def _permit_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker,
        "_validate_environment_file_metadata",
        lambda _path: None,
    )
    monkeypatch.setattr(
        worker,
        "_validate_environment_file_assignment",
        lambda _path: None,
    )


def _healthy_runner(command) -> subprocess.CompletedProcess[str]:
    resolved = tuple(command)
    responses = {
        ("systemctl", "is-enabled", "gridiron-edge-collector.timer"): "enabled\n",
        ("systemctl", "is-active", "gridiron-edge-collector.timer"): "active\n",
        ("timedatectl", "show", "-p", "NTPSynchronized", "--value"): "yes\n",
        ("vcgencmd", "get_throttled"): "throttled=0x0\n",
        (
            "systemctl",
            "show",
            "gridiron-edge-collector.service",
            "-p",
            "Result",
            "--value",
        ): "success\n",
        ("dmesg",): "clean\n",
    }
    return _complete(resolved, responses.get(resolved, "ok\n"))


def test_rendering_preserves_dynamic_selected_plan_boundary(tmp_path: Path) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    wrapper = render_wrapper(config)
    service = render_service(SERVICE, config)
    timer = render_timer(TIMER)

    rendered = wrapper + service + timer
    assert "execute-selected-odds-plan" in wrapper
    assert "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" in wrapper
    assert "--season" not in rendered
    assert "--week" not in rendered
    assert "secret-value" not in rendered
    assert "ODDS_API_KEY=" not in rendered
    assert "Type=oneshot" in service
    assert "Restart=no" in service
    assert "OnUnitActiveSec=5min" in timer
    assert not worker.re.findall(r"@[A-Z_]+@", service)


def test_render_service_rejects_unknown_placeholder(tmp_path: Path) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    with pytest.raises(ValueError, match="Unresolved"):
        render_service(SERVICE + "Extra=@UNKNOWN@\n", config)


def test_install_is_idempotent_and_activation_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    config = _config(tmp_path, repo)
    service_template = tmp_path / "service"
    timer_template = tmp_path / "timer"
    service_template.write_text(SERVICE, encoding="utf-8")
    timer_template.write_text(TIMER, encoding="utf-8")
    calls: list[tuple[str, ...]] = []
    _permit_test_environment(monkeypatch)

    def runner(command) -> subprocess.CompletedProcess[str]:
        resolved = tuple(command)
        calls.append(resolved)
        return _complete(resolved)

    install_quote_collection_worker(
        config,
        service_template=service_template,
        timer_template=timer_template,
        runner=runner,
    )
    deployed = (config.wrapper_path, config.service_path, config.timer_path)
    first = tuple(path.read_bytes() for path in deployed)
    install_quote_collection_worker(
        config,
        service_template=service_template,
        timer_template=timer_template,
        runner=runner,
    )
    second = tuple(path.read_bytes() for path in deployed)

    assert first == second
    assert not any(command[:3] == ("systemctl", "enable", "--now") for command in calls)
    assert _mode(config.wrapper_path) == 0o755
    assert _mode(config.service_path) == 0o644
    assert _mode(config.timer_path) == 0o644


def test_install_can_activate_timer_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    config = _config(tmp_path, repo)
    service_template = tmp_path / "service"
    timer_template = tmp_path / "timer"
    service_template.write_text(SERVICE, encoding="utf-8")
    timer_template.write_text(TIMER, encoding="utf-8")
    calls: list[tuple[str, ...]] = []
    _permit_test_environment(monkeypatch)

    def runner(command) -> subprocess.CompletedProcess[str]:
        resolved = tuple(command)
        calls.append(resolved)
        return _complete(resolved)

    install_quote_collection_worker(
        config,
        service_template=service_template,
        timer_template=timer_template,
        enable_timer=True,
        runner=runner,
    )

    assert (
        "systemctl",
        "enable",
        "--now",
        "gridiron-edge-collector.timer",
    ) in calls


def test_verifier_reports_ready_without_exposing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    config = _config(tmp_path, repo)
    config.wrapper_path.write_text("wrapper\n", encoding="utf-8")
    config.service_path.write_text("service\n", encoding="utf-8")
    config.timer_path.write_text("timer\n", encoding="utf-8")
    _permit_test_environment(monkeypatch)

    report = verify_quote_collection_worker(config, runner=_healthy_runner)
    rendered = "\n".join(check.detail for check in report.checks)

    assert report.status is WorkerVerificationStatus.READY
    assert "secret-value" not in rendered
    assert all(check.status is WorkerCheckStatus.PASSED for check in report.checks)


def test_verifier_reports_degraded_for_unresolved_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    config = _config(tmp_path, repo)
    run = repo / "data/odds/collection_runs/season=2026-2027/week=01/scheduled_at=x"
    run.mkdir(parents=True)
    (run / "claim.json").write_text("{}", encoding="utf-8")
    for path in (config.wrapper_path, config.service_path, config.timer_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("safe\n", encoding="utf-8")
    _permit_test_environment(monkeypatch)

    report = verify_quote_collection_worker(config, runner=_healthy_runner)
    unresolved = next(check for check in report.checks if check.name == "unresolved_claims")

    assert report.status is WorkerVerificationStatus.DEGRADED
    assert unresolved.status is WorkerCheckStatus.WARNING


def _runner_with_dmesg(dmesg_output: str):
    def runner(command) -> subprocess.CompletedProcess[str]:
        resolved = tuple(command)
        if resolved == ("dmesg",):
            return _complete(resolved, dmesg_output)
        return _healthy_runner(resolved)

    return runner


def test_verifier_ignores_benign_zram_and_loop_capacity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real incident, 2026-09-27: zram and loop devices report a capacity
    change once at boot -- this has no bearing on physical storage health
    and must never mark the worker degraded, no matter how long ago it
    happened or how long dmesg's ring buffer retains it."""
    repo = _repo(tmp_path)
    config = _config(tmp_path, repo)
    for path in (config.wrapper_path, config.service_path, config.timer_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("safe\n", encoding="utf-8")
    _permit_test_environment(monkeypatch)
    dmesg_output = (
        "[Wed Aug 19 02:39:44 2026] loop0: detected capacity change from 0 to 4194304\n"
        "[Wed Aug 19 02:39:44 2026] zram0: detected capacity change from 0 to 4194304\n"
    )

    report = verify_quote_collection_worker(config, runner=_runner_with_dmesg(dmesg_output))
    storage = next(check for check in report.checks if check.name == "storage_health")

    assert storage.status is WorkerCheckStatus.PASSED
    assert report.status is WorkerVerificationStatus.READY


def test_verifier_flags_real_capacity_change_on_physical_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    config = _config(tmp_path, repo)
    for path in (config.wrapper_path, config.service_path, config.timer_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("safe\n", encoding="utf-8")
    _permit_test_environment(monkeypatch)
    dmesg_output = "[   12.345678] sda: detected capacity change from 3907029168 to 0\n"

    report = verify_quote_collection_worker(config, runner=_runner_with_dmesg(dmesg_output))
    storage = next(check for check in report.checks if check.name == "storage_health")

    assert storage.status is WorkerCheckStatus.WARNING
    assert report.status is WorkerVerificationStatus.DEGRADED


def test_verifier_still_flags_usb_disconnect_on_a_zram_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benign-device exception is scoped to `capacity change` only --
    it must not blanket-exempt zram/loop lines from every marker."""
    repo = _repo(tmp_path)
    config = _config(tmp_path, repo)
    for path in (config.wrapper_path, config.service_path, config.timer_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("safe\n", encoding="utf-8")
    _permit_test_environment(monkeypatch)
    dmesg_output = "[   12.345678] usb 1-1: USB disconnect, device number 3\n"

    report = verify_quote_collection_worker(config, runner=_runner_with_dmesg(dmesg_output))
    storage = next(check for check in report.checks if check.name == "storage_health")

    assert storage.status is WorkerCheckStatus.WARNING


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _templates(tmp_path: Path) -> tuple[Path, Path]:
    service_template = tmp_path / "service"
    timer_template = tmp_path / "timer"
    service_template.write_text(SERVICE, encoding="utf-8")
    timer_template.write_text(TIMER, encoding="utf-8")
    return service_template, timer_template


def _write_old_deployment(config: QuoteCollectionWorkerConfig) -> None:
    for path, content, mode in (
        (config.wrapper_path, b"old-wrapper\n", 0o700),
        (config.service_path, b"old-service\n", 0o600),
        (config.timer_path, b"old-timer\n", 0o640),
    ):
        path.write_bytes(content)
        path.chmod(mode)


def test_staged_verification_failure_preserves_live_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    service_template, timer_template = _templates(tmp_path)
    _write_old_deployment(config)
    calls: list[tuple[str, ...]] = []
    _permit_test_environment(monkeypatch)

    def runner(command) -> subprocess.CompletedProcess[str]:
        resolved = tuple(command)
        calls.append(resolved)
        return _complete(resolved, returncode=1)

    with pytest.raises(RuntimeError, match="staged systemd-analyze verify"):
        install_quote_collection_worker(
            config,
            service_template=service_template,
            timer_template=timer_template,
            runner=runner,
        )

    assert config.wrapper_path.read_bytes() == b"old-wrapper\n"
    assert config.service_path.read_bytes() == b"old-service\n"
    assert config.timer_path.read_bytes() == b"old-timer\n"
    assert not any(call[:2] == ("systemctl", "daemon-reload") for call in calls)


def test_missing_destination_parent_fails_before_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    config = replace(
        config,
        wrapper_path=tmp_path / "missing" / "collector",
    )
    service_template, timer_template = _templates(tmp_path)
    calls: list[tuple[str, ...]] = []
    _permit_test_environment(monkeypatch)

    def runner(command) -> subprocess.CompletedProcess[str]:
        resolved = tuple(command)
        calls.append(resolved)
        return _complete(resolved)

    with pytest.raises(ValueError, match="parent does not exist"):
        install_quote_collection_worker(
            config,
            service_template=service_template,
            timer_template=timer_template,
            runner=runner,
        )

    assert calls == []


def test_daemon_reload_failure_restores_previous_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    service_template, timer_template = _templates(tmp_path)
    _write_old_deployment(config)
    calls: list[tuple[str, ...]] = []
    reload_count = 0
    _permit_test_environment(monkeypatch)

    def runner(command) -> subprocess.CompletedProcess[str]:
        nonlocal reload_count
        resolved = tuple(command)
        calls.append(resolved)
        if resolved == ("systemctl", "daemon-reload"):
            reload_count += 1
            return _complete(resolved, returncode=1 if reload_count == 1 else 0)
        return _complete(resolved)

    with pytest.raises(QuoteCollectionWorkerInstallationError, match="restored"):
        install_quote_collection_worker(
            config,
            service_template=service_template,
            timer_template=timer_template,
            enable_timer=True,
            runner=runner,
        )

    assert config.wrapper_path.read_bytes() == b"old-wrapper\n"
    assert config.service_path.read_bytes() == b"old-service\n"
    assert config.timer_path.read_bytes() == b"old-timer\n"
    assert _mode(config.wrapper_path) == 0o700
    assert _mode(config.service_path) == 0o600
    assert _mode(config.timer_path) == 0o640
    assert reload_count == 2
    assert not any(call[:3] == ("systemctl", "enable", "--now") for call in calls)


def test_daemon_reload_failure_removes_new_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    service_template, timer_template = _templates(tmp_path)
    reload_count = 0
    _permit_test_environment(monkeypatch)

    def runner(command) -> subprocess.CompletedProcess[str]:
        nonlocal reload_count
        resolved = tuple(command)
        if resolved == ("systemctl", "daemon-reload"):
            reload_count += 1
            return _complete(resolved, returncode=1 if reload_count == 1 else 0)
        return _complete(resolved)

    with pytest.raises(QuoteCollectionWorkerInstallationError):
        install_quote_collection_worker(
            config,
            service_template=service_template,
            timer_template=timer_template,
            runner=runner,
        )

    assert not config.wrapper_path.exists()
    assert not config.service_path.exists()
    assert not config.timer_path.exists()
    assert reload_count == 2


def test_restoration_reload_failure_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    service_template, timer_template = _templates(tmp_path)
    _write_old_deployment(config)
    _permit_test_environment(monkeypatch)

    def runner(command) -> subprocess.CompletedProcess[str]:
        resolved = tuple(command)
        if resolved == ("systemctl", "daemon-reload"):
            return _complete(resolved, returncode=1)
        return _complete(resolved)

    with pytest.raises(
        QuoteCollectionWorkerInstallationError,
        match="restored state also failed",
    ):
        install_quote_collection_worker(
            config,
            service_template=service_template,
            timer_template=timer_template,
            runner=runner,
        )

    assert config.wrapper_path.read_bytes() == b"old-wrapper\n"
    with pytest.raises(
        QuoteCollectionWorkerInstallationError,
        match="restored state also failed",
    ) as raised:
        install_quote_collection_worker(
            config,
            service_template=service_template,
            timer_template=timer_template,
            runner=runner,
        )

    assert config.wrapper_path.read_bytes() == b"old-wrapper\n"
    assert "secret-value" not in str(raised.value)


def test_activation_failure_keeps_installed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    service_template, timer_template = _templates(tmp_path)
    _write_old_deployment(config)
    reload_count = 0
    _permit_test_environment(monkeypatch)

    def runner(command) -> subprocess.CompletedProcess[str]:
        nonlocal reload_count
        resolved = tuple(command)
        if resolved == ("systemctl", "daemon-reload"):
            reload_count += 1
            return _complete(resolved)
        if resolved[:3] == ("systemctl", "enable", "--now"):
            return _complete(resolved, returncode=1)
        return _complete(resolved)

    with pytest.raises(QuoteCollectionWorkerActivationError, match="activation failed"):
        install_quote_collection_worker(
            config,
            service_template=service_template,
            timer_template=timer_template,
            enable_timer=True,
            runner=runner,
        )

    assert config.wrapper_path.read_text(encoding="utf-8") != "old-wrapper\n"
    assert "Type=oneshot" in config.service_path.read_text(encoding="utf-8")
    assert "OnUnitActiveSec=5min" in config.timer_path.read_text(encoding="utf-8")
    assert reload_count == 1


def test_verifier_does_not_open_environment_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _repo(tmp_path))
    for path in (config.wrapper_path, config.service_path, config.timer_path):
        path.write_text("safe\n", encoding="utf-8")
    monkeypatch.setattr(worker, "_validate_environment_file_metadata", lambda _path: None)
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path == config.environment_file:
            raise AssertionError("Verifier opened the credential file")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = verify_quote_collection_worker(config, runner=_healthy_runner)

    assert report.status is WorkerVerificationStatus.READY

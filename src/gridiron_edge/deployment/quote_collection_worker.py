"""Repository-owned installation and verification for the quote worker."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import grp
import os
from pathlib import Path
import pwd
import re
import shlex
import stat
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Final

from gridiron_edge.market.collection_plan_store import (
    load_current_collection_plan,
    read_current_collection_plan_selection,
)

SERVICE_NAME: Final[str] = "gridiron-edge-collector.service"
TIMER_NAME: Final[str] = "gridiron-edge-collector.timer"
SCHEDULE_PATH: Final[Path] = Path("data/cleaned/NFL_upcoming_schedule_rich.parquet")
SELECTION_PATH: Final[Path] = Path("data/odds/collection_plans/current.json")
_PLACEHOLDERS: Final[tuple[str, ...]] = (
    "@REPOSITORY@",
    "@USER@",
    "@GROUP@",
    "@ENVIRONMENT_FILE@",
    "@WRAPPER@",
)
_FORBIDDEN_STATIC_TOKENS: Final[tuple[str, ...]] = (
    "ODDS_API_KEY=",
    "--season",
    "--week",
)

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
DeploymentSnapshot = tuple[bytes, int] | None


class QuoteCollectionWorkerInstallationError(RuntimeError):
    """Installation failed and the previous deployment was restored."""


class QuoteCollectionWorkerActivationError(RuntimeError):
    """Files were installed, but explicit timer activation failed."""


class WorkerCheckStatus(StrEnum):
    """Outcome of one deployment verification check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class WorkerVerificationStatus(StrEnum):
    """Aggregate deployment readiness."""

    READY = "ready"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class QuoteCollectionWorkerConfig:
    """Fully resolved quote-worker deployment paths and identities."""

    repository: Path
    deployment_user: str
    deployment_group: str
    uv_path: Path
    environment_file: Path
    wrapper_path: Path
    service_path: Path
    timer_path: Path


@dataclass(frozen=True, slots=True)
class WorkerVerificationCheck:
    """One explicit deployment verification result."""

    name: str
    status: WorkerCheckStatus
    detail: str


@dataclass(frozen=True, slots=True)
class WorkerVerification:
    """Complete read-only deployment verification."""

    status: WorkerVerificationStatus
    checks: tuple[WorkerVerificationCheck, ...]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        check=False,
        capture_output=True,
        text=True,
    )


# Virtual block devices that legitimately report a capacity change once, at
# creation, with no bearing on physical storage health: zram (compressed-RAM
# swap) reports its configured size at boot, and loop devices report the
# backing file's size once mounted. Real physical-storage capacity changes
# (a failing or resizing disk) never mention these device names.
_BENIGN_CAPACITY_CHANGE_DEVICES: Final[tuple[str, ...]] = ("loop", "zram")


def _is_benign_capacity_change(marker: str, line: str) -> bool:
    return marker == "capacity change" and any(
        device in line for device in _BENIGN_CAPACITY_CHANGE_DEVICES
    )


def render_wrapper(config: QuoteCollectionWorkerConfig) -> str:
    """Render the selected-plan invocation wrapper with shell-safe paths."""
    repository = shlex.quote(str(config.repository))
    uv_path = shlex.quote(str(config.uv_path))
    content = f"""#!/bin/sh
set -eu

REPO={repository}
EVALUATED_AT=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

cd "$REPO"

exec {uv_path} run gridiron ingest execute-selected-odds-plan \\
  --evaluated-at "$EVALUATED_AT" \\
  --grace-minutes 15 \\
  --minimum-credit-reserve 30 \\
  --timeout 15
"""
    _validate_rendered(content, allow_dynamic_evaluated_at=True)
    return content


def render_service(template: str, config: QuoteCollectionWorkerConfig) -> str:
    """Render one systemd service from the repository-owned template."""
    replacements = {
        "@REPOSITORY@": str(config.repository),
        "@USER@": config.deployment_user,
        "@GROUP@": config.deployment_group,
        "@ENVIRONMENT_FILE@": str(config.environment_file),
        "@WRAPPER@": str(config.wrapper_path),
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    _validate_rendered(rendered)
    if "Type=oneshot" not in rendered or "Restart=no" not in rendered:
        raise ValueError("Collector service must remain a non-restarting oneshot.")
    if "execute-selected-odds-plan" in rendered:
        raise ValueError("Service must delegate execution through the wrapper.")
    return rendered


def render_timer(template: str) -> str:
    """Validate and return the fixed quote-worker timer template."""
    _validate_rendered(template)
    required = (
        "OnBootSec=2min",
        "OnUnitActiveSec=5min",
        "AccuracySec=15s",
        "Persistent=true",
        f"Unit={SERVICE_NAME}",
    )
    missing = [value for value in required if value not in template]
    if missing:
        raise ValueError("Collector timer is missing required settings: " + ", ".join(missing))
    return template


def validate_installation_inputs(config: QuoteCollectionWorkerConfig) -> None:
    """Validate every explicit deployment input without exposing the secret."""
    if not config.repository.is_dir():
        raise ValueError("repository must be an existing directory.")
    if not (config.repository / SCHEDULE_PATH).is_file():
        raise ValueError("The rich upcoming schedule is missing.")
    if not (config.repository / SELECTION_PATH).is_file():
        raise ValueError("The current quote-collection plan selection is missing.")
    selection = read_current_collection_plan_selection(repo=config.repository)
    plan = load_current_collection_plan(repo=config.repository)
    if (selection.season, selection.week) != (plan.season, plan.week):
        raise ValueError("The selected collection-plan scope is inconsistent.")
    if not config.uv_path.is_file() or not os.access(config.uv_path, os.X_OK):
        raise ValueError("uv_path must be an executable regular file.")
    _validate_account(config.deployment_user, config.deployment_group)
    _validate_environment_file_metadata(config.environment_file)
    _validate_environment_file_assignment(config.environment_file)
    for path in (config.wrapper_path, config.service_path, config.timer_path):
        if not path.is_absolute():
            raise ValueError("Deployed destination paths must be absolute.")
        if not path.parent.is_dir():
            raise ValueError(f"Deployment destination parent does not exist: {path.parent}")


def install_quote_collection_worker(
    config: QuoteCollectionWorkerConfig,
    *,
    service_template: Path,
    timer_template: Path,
    enable_timer: bool = False,
    runner: CommandRunner = _run,
) -> None:
    """Install one validated deployment set and optionally activate its timer."""
    validate_installation_inputs(config)
    service_source = service_template.read_text(encoding="utf-8")
    timer_source = timer_template.read_text(encoding="utf-8")
    contents = {
        config.wrapper_path: (render_wrapper(config), 0o755),
        config.service_path: (render_service(service_source, config), 0o644),
        config.timer_path: (render_timer(timer_source), 0o644),
    }

    _verify_staged_deployment(
        config,
        contents=contents,
        runner=runner,
    )
    snapshots = _snapshot_deployment_set(tuple(contents))

    try:
        _replace_deployment_set(contents)
        _successful(
            runner(("systemctl", "daemon-reload")),
            "systemctl daemon-reload",
        )
    except Exception as install_error:
        _restore_deployment_set(snapshots)
        restoration = runner(("systemctl", "daemon-reload"))
        if restoration.returncode != 0:
            raise QuoteCollectionWorkerInstallationError(
                "Installation failed; prior deployment files were restored, "
                "but systemd reload of the restored state also failed."
            ) from install_error
        raise QuoteCollectionWorkerInstallationError(
            "Installation failed and the prior deployment was restored."
        ) from install_error

    if enable_timer:
        activation = runner(("systemctl", "enable", "--now", TIMER_NAME))
        if activation.returncode != 0:
            raise QuoteCollectionWorkerActivationError(
                "Worker files were installed, but timer activation failed."
            )


def verify_quote_collection_worker(
    config: QuoteCollectionWorkerConfig,
    *,
    runner: CommandRunner = _run,
) -> WorkerVerification:
    """Return a read-only report for the installed quote collection worker."""
    checks: list[WorkerVerificationCheck] = []
    checks.extend(_artifact_checks(config))
    checks.extend(_installation_checks(config))
    checks.extend(_system_checks(config, runner))
    status = _aggregate(checks)
    return WorkerVerification(status=status, checks=tuple(checks))


def _artifact_checks(config: QuoteCollectionWorkerConfig) -> list[WorkerVerificationCheck]:
    checks: list[WorkerVerificationCheck] = []
    for name, path in (
        ("repository", config.repository),
        ("schedule", config.repository / SCHEDULE_PATH),
        ("selection", config.repository / SELECTION_PATH),
    ):
        exists = path.is_dir() if name == "repository" else path.is_file()
        checks.append(_check(name, exists, str(path)))
    try:
        selection = read_current_collection_plan_selection(repo=config.repository)
        plan = load_current_collection_plan(repo=config.repository)
        detail = f"season={selection.season} week={selection.week} polls={plan.planned_poll_count}"
        checks.append(_check("selected_plan", True, detail))
    except (FileNotFoundError, ValueError, OSError) as exc:
        checks.append(_check("selected_plan", False, type(exc).__name__))
    claims = tuple(
        (config.repository / "data/odds/collection_runs").glob(
            "season=*/week=*/scheduled_at=*/claim.json"
        )
    )
    results = {
        path.parent
        for path in (config.repository / "data/odds/collection_runs").glob(
            "season=*/week=*/scheduled_at=*/result.json"
        )
    }
    unresolved = sum(path.parent not in results for path in claims)
    checks.append(
        WorkerVerificationCheck(
            name="unresolved_claims",
            status=WorkerCheckStatus.WARNING if unresolved else WorkerCheckStatus.PASSED,
            detail=str(unresolved),
        )
    )
    return checks


def _installation_checks(config: QuoteCollectionWorkerConfig) -> list[WorkerVerificationCheck]:
    checks: list[WorkerVerificationCheck] = []
    for name, path in (
        ("wrapper", config.wrapper_path),
        ("service", config.service_path),
        ("timer", config.timer_path),
        ("environment_file", config.environment_file),
    ):
        checks.append(_check(name, path.is_file(), str(path)))
    try:
        _validate_environment_file_metadata(config.environment_file)
        checks.append(_check("environment_security", True, "root:root mode=0600"))
    except (FileNotFoundError, ValueError, OSError) as exc:
        checks.append(_check("environment_security", False, type(exc).__name__))
    for name, path in (
        ("wrapper_secret_scan", config.wrapper_path),
        ("service_secret_scan", config.service_path),
        ("timer_secret_scan", config.timer_path),
    ):
        try:
            content = path.read_text(encoding="utf-8")
            safe = "ODDS_API_KEY=" not in content
        except OSError:
            safe = False
        checks.append(_check(name, safe, "secret assignment absent" if safe else "unavailable"))
    return checks


def _system_checks(
    config: QuoteCollectionWorkerConfig,
    runner: CommandRunner,
) -> list[WorkerVerificationCheck]:
    checks: list[WorkerVerificationCheck] = []
    commands = (
        (
            "unit_syntax",
            ("systemd-analyze", "verify", str(config.service_path), str(config.timer_path)),
        ),
        ("timer_enabled", ("systemctl", "is-enabled", TIMER_NAME)),
        ("timer_active", ("systemctl", "is-active", TIMER_NAME)),
        ("clock_synchronized", ("timedatectl", "show", "-p", "NTPSynchronized", "--value")),
        ("root_filesystem", ("findmnt", "-no", "SOURCE,FSTYPE,AVAIL,TARGET", "/")),
        ("throttling", ("vcgencmd", "get_throttled")),
    )
    for name, command in commands:
        result = runner(command)
        output = (result.stdout or result.stderr).strip()
        passed = result.returncode == 0
        if name == "clock_synchronized":
            passed = passed and output.lower() == "yes"
        if name == "throttling":
            passed = passed and output == "throttled=0x0"
        checks.append(_check(name, passed, output or f"exit={result.returncode}"))
    service = runner(("systemctl", "show", SERVICE_NAME, "-p", "Result", "--value"))
    service_output = (service.stdout or service.stderr).strip()
    checks.append(
        _check(
            "latest_service_result",
            service.returncode == 0 and service_output in {"success", ""},
            service_output or "not-run",
        )
    )
    journal = runner(("journalctl", "-u", SERVICE_NAME, "--no-pager"))
    journal_text = f"{journal.stdout}\n{journal.stderr}"
    disclosed = "ODDS_API_KEY=" in journal_text
    checks.append(
        _check(
            "journal_secret_scan",
            journal.returncode == 0 and not disclosed,
            "secret assignment absent" if not disclosed else "credential disclosure detected",
        )
    )
    storage = runner(("dmesg",))
    storage_lines = f"{storage.stdout}\n{storage.stderr}".lower().splitlines()
    markers = (
        "usb disconnect",
        "i/o error",
        "capacity change",
        "synchronize cache",
    )
    found = tuple(
        marker
        for marker in markers
        if any(
            marker in line and not _is_benign_capacity_change(marker, line)
            for line in storage_lines
        )
    )
    checks.append(
        WorkerVerificationCheck(
            name="storage_health",
            status=WorkerCheckStatus.WARNING if found else WorkerCheckStatus.PASSED,
            detail=", ".join(found) if found else "no configured error markers",
        )
    )
    return checks


def _validate_rendered(content: str, *, allow_dynamic_evaluated_at: bool = False) -> None:
    unresolved = tuple(sorted(set(re.findall(r"@[A-Z_]+@", content))))
    if unresolved:
        raise ValueError("Unresolved deployment placeholders: " + ", ".join(unresolved))
    forbidden = [token for token in _FORBIDDEN_STATIC_TOKENS if token in content]
    if forbidden:
        raise ValueError("Forbidden deployment content: " + ", ".join(forbidden))
    if "--evaluated-at" in content and not allow_dynamic_evaluated_at:
        raise ValueError("Static deployment templates cannot contain evaluated-at.")


def _validate_account(user: str, group: str) -> None:
    if not user or not group:
        raise ValueError("Deployment user and group must not be empty.")
    try:
        pwd.getpwnam(user)
        grp.getgrnam(group)
    except KeyError as exc:
        raise ValueError("Deployment user or group does not exist.") from exc


def _validate_environment_file_metadata(path: Path) -> None:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("environment_file must be a regular file.")
    if metadata.st_uid != 0 or metadata.st_gid != 0:
        raise ValueError("environment_file must be owned by root:root.")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("environment_file must use mode 0600.")


def _validate_environment_file_assignment(path: Path) -> None:
    assignments = 0
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if name == "ODDS_API_KEY":
                assignments += 1
                if not separator or not value:
                    raise ValueError("ODDS_API_KEY must be nonempty.")
    if assignments != 1:
        raise ValueError("environment_file must contain exactly one ODDS_API_KEY assignment.")


def _verify_staged_deployment(
    config: QuoteCollectionWorkerConfig,
    *,
    contents: Mapping[Path, tuple[str, int]],
    runner: CommandRunner,
) -> None:
    with TemporaryDirectory(prefix="gridiron-edge-collector-") as directory:
        staging = Path(directory)
        staged_wrapper = staging / "gridiron-edge-collector"
        staged_service = staging / SERVICE_NAME
        staged_timer = staging / TIMER_NAME
        staged_config = QuoteCollectionWorkerConfig(
            repository=config.repository,
            deployment_user=config.deployment_user,
            deployment_group=config.deployment_group,
            uv_path=config.uv_path,
            environment_file=config.environment_file,
            wrapper_path=staged_wrapper,
            service_path=staged_service,
            timer_path=staged_timer,
        )
        service_source = contents[config.service_path][0]
        staged_contents = {
            staged_wrapper: (contents[config.wrapper_path][0], 0o755),
            staged_service: (
                service_source.replace(
                    str(config.wrapper_path),
                    str(staged_config.wrapper_path),
                ),
                0o644,
            ),
            staged_timer: (contents[config.timer_path][0], 0o644),
        }
        _replace_deployment_set(staged_contents)
        _successful(
            runner(
                (
                    "systemd-analyze",
                    "verify",
                    str(staged_service),
                    str(staged_timer),
                )
            ),
            "staged systemd-analyze verify",
        )


def _snapshot_deployment_set(
    destinations: Sequence[Path],
) -> dict[Path, DeploymentSnapshot]:
    snapshots: dict[Path, DeploymentSnapshot] = {}
    for destination in destinations:
        if destination.exists():
            snapshots[destination] = (
                destination.read_bytes(),
                stat.S_IMODE(destination.stat().st_mode),
            )
        else:
            snapshots[destination] = None
    return snapshots


def _restore_deployment_set(
    snapshots: Mapping[Path, DeploymentSnapshot],
) -> None:
    contents = {
        destination: snapshot for destination, snapshot in snapshots.items() if snapshot is not None
    }
    for destination, snapshot in snapshots.items():
        if snapshot is None:
            destination.unlink(missing_ok=True)
    _replace_deployment_set(contents)


def _replace_deployment_set(
    contents: Mapping[Path, tuple[str | bytes, int]],
) -> None:
    backups = _snapshot_deployment_set(tuple(contents))
    staged: dict[Path, Path] = {}
    try:
        for destination, (content, mode) in contents.items():
            if not destination.parent.is_dir():
                raise ValueError(
                    f"Deployment destination parent does not exist: {destination.parent}"
                )
            binary = isinstance(content, bytes)
            with NamedTemporaryFile(
                mode="wb" if binary else "w",
                encoding=None if binary else "utf-8",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                stream.write(content)
                temporary = Path(stream.name)
            temporary.chmod(mode)
            staged[destination] = temporary
        for destination, temporary in staged.items():
            temporary.replace(destination)
    except Exception:
        for destination, previous in backups.items():
            if previous is None:
                destination.unlink(missing_ok=True)
            else:
                previous_content, previous_mode = previous
                destination.write_bytes(previous_content)
                destination.chmod(previous_mode)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _successful(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{label} failed: {detail or result.returncode}")


def _check(name: str, passed: bool, detail: str) -> WorkerVerificationCheck:
    return WorkerVerificationCheck(
        name=name,
        status=WorkerCheckStatus.PASSED if passed else WorkerCheckStatus.FAILED,
        detail=detail,
    )


def _aggregate(checks: Sequence[WorkerVerificationCheck]) -> WorkerVerificationStatus:
    if any(check.status is WorkerCheckStatus.FAILED for check in checks):
        return WorkerVerificationStatus.BLOCKED
    if any(check.status is WorkerCheckStatus.WARNING for check in checks):
        return WorkerVerificationStatus.DEGRADED
    return WorkerVerificationStatus.READY


def _config_from_args(args: argparse.Namespace) -> QuoteCollectionWorkerConfig:
    return QuoteCollectionWorkerConfig(
        repository=args.repository.resolve(),
        deployment_user=args.user,
        deployment_group=args.group,
        uv_path=args.uv_path.resolve(),
        environment_file=args.environment_file.resolve(),
        wrapper_path=args.wrapper_path.resolve(),
        service_path=args.service_path.resolve(),
        timer_path=args.timer_path.resolve(),
    )


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--uv-path", type=Path, required=True)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--wrapper-path", type=Path, required=True)
    parser.add_argument("--service-path", type=Path, required=True)
    parser.add_argument("--timer-path", type=Path, required=True)
    return parser


def install_main() -> None:
    """CLI entry point for explicit worker installation."""
    parser = _common_parser("Install the GridIron Edge quote collection worker.")
    parser.add_argument("--service-template", type=Path, required=True)
    parser.add_argument("--timer-template", type=Path, required=True)
    parser.add_argument("--enable-timer", action="store_true")
    args = parser.parse_args()
    install_quote_collection_worker(
        _config_from_args(args),
        service_template=args.service_template,
        timer_template=args.timer_template,
        enable_timer=args.enable_timer,
    )
    print("Quote collection worker installed.")


def verify_main() -> None:
    """CLI entry point for read-only worker verification."""
    parser = _common_parser("Verify the GridIron Edge quote collection worker.")
    args = parser.parse_args()
    report = verify_quote_collection_worker(_config_from_args(args))
    print(f"Worker status: {report.status.value}")
    for check in report.checks:
        print(f"{check.status.value}: {check.name}: {check.detail}")
    if report.status is WorkerVerificationStatus.BLOCKED:
        raise SystemExit(1)

# src/gridiron_edge/cli/verify.py

"""Composite command: verify.

Python repository verification workflow: quality gates, backend tests,
an external nflverse smoke check, and model baseline comparison.

This command does not evaluate selected weekly-product readiness; use
``gridiron verify-week`` for that. Frontend build, Vitest, and ESLint checks
are separate and are not run by this command.

The default mode runs Ruff, Pyrefly, unit, integration, and e2e tests, the
fetch-games + clean-games smoke check, and baseline comparison. ``--fast``
skips e2e and smoke-pipeline. ``--strict`` converts smoke and missing-baseline
soft failures into hard failures for CI use.

Usage::

    gridiron verify
    gridiron verify --fast
    gridiron verify --strict
    gridiron verify --very-thorough
    gridiron verify --only quality-gates --only unit-tests
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._composites import (
    CompositeStage,
    StageResult,
    render_composite_summary,
    resolve_active_stages,
    run_composite,
)
from gridiron_edge.core.console import console
from gridiron_edge.core.settings import get_settings
from gridiron_edge.models.artifact import ArtifactStore, BaseModelMetadata

# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def _run_subprocess(
    cmd: list[str],
    *,
    repo_root: Path,
) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    Args:
        cmd: Command + args, passed to subprocess.run.
        repo_root: Working directory for the subprocess.

    Returns:
        Tuple of (returncode, stdout, stderr). stdout/stderr are
        captured as strings.
    """
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _summarize_pytest_output(stdout: str) -> str:
    """Extract the summary line from pytest stdout.

    Looks for the last "passed/failed" summary line in the output.
    Returns a short summary suitable for the StageResult detail field.
    """
    lines = stdout.strip().splitlines()
    # pytest's summary line is typically the last non-empty line.
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "passed" in stripped or "failed" in stripped or "error" in stripped:
            # Strip surrounding equals chars that pytest uses
            return stripped.strip("= ")
        break
    return "no summary available"


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def _stage_quality_gates(ctx: dict[str, Any]) -> StageResult:
    """Run ruff check and pyrefly check."""
    repo_root: Path = ctx["repo_root"]

    # ruff
    ruff_code, _ruff_out, ruff_err = _run_subprocess(
        ["uv", "run", "ruff", "check", "."],
        repo_root=repo_root,
    )
    if ruff_code != 0:
        return StageResult(
            success=False,
            detail=f"ruff failed (exit {ruff_code})",
            warnings=[ruff_err.strip()[:200]] if ruff_err else [],
        )

    # pyrefly
    pyrefly_code, _pyrefly_out, pyrefly_err = _run_subprocess(
        ["uvx", "pyrefly", "check"],
        repo_root=repo_root,
    )
    if pyrefly_code != 0:
        return StageResult(
            success=False,
            detail=f"pyrefly failed (exit {pyrefly_code})",
            warnings=[pyrefly_err.strip()[:200]] if pyrefly_err else [],
        )

    return StageResult(success=True, detail="ruff + pyrefly clean")


def _stage_unit_tests(ctx: dict[str, Any]) -> StageResult:
    """Run unit tests (excluding slow)."""
    repo_root: Path = ctx["repo_root"]
    code, stdout, stderr = _run_subprocess(
        ["uv", "run", "pytest", "-m", "unit and not slow", "-q"],
        repo_root=repo_root,
    )
    summary = _summarize_pytest_output(stdout)
    if code != 0:
        return StageResult(
            success=False,
            detail=f"unit tests failed ({summary})",
            warnings=[stderr.strip()[:200]] if stderr else [],
        )
    return StageResult(success=True, detail=summary)


def _stage_integration_tests(ctx: dict[str, Any]) -> StageResult:
    """Run integration tests."""
    repo_root: Path = ctx["repo_root"]
    code, stdout, stderr = _run_subprocess(
        ["uv", "run", "pytest", "-m", "integration", "-q"],
        repo_root=repo_root,
    )
    summary = _summarize_pytest_output(stdout)
    if code != 0:
        return StageResult(
            success=False,
            detail=f"integration tests failed ({summary})",
            warnings=[stderr.strip()[:200]] if stderr else [],
        )
    return StageResult(success=True, detail=summary)


def _stage_e2e_tests(ctx: dict[str, Any]) -> StageResult:
    """Run end-to-end tests."""
    repo_root: Path = ctx["repo_root"]
    code, stdout, stderr = _run_subprocess(
        ["uv", "run", "pytest", "-m", "e2e", "-q"],
        repo_root=repo_root,
    )
    summary = _summarize_pytest_output(stdout)
    if code != 0:
        return StageResult(
            success=False,
            detail=f"e2e tests failed ({summary})",
            warnings=[stderr.strip()[:200]] if stderr else [],
        )
    return StageResult(success=True, detail=summary)


def _stage_slow_tests(ctx: dict[str, Any]) -> StageResult:
    """Run slow tests (only included with --very-thorough)."""
    repo_root: Path = ctx["repo_root"]
    code, stdout, stderr = _run_subprocess(
        ["uv", "run", "pytest", "-m", "slow", "-q"],
        repo_root=repo_root,
    )
    summary = _summarize_pytest_output(stdout)
    if code != 0:
        return StageResult(
            success=False,
            detail=f"slow tests failed ({summary})",
            warnings=[stderr.strip()[:200]] if stderr else [],
        )
    return StageResult(success=True, detail=summary)


def _stage_smoke_pipeline(ctx: dict[str, Any]) -> StageResult:
    """Light run-data-pipeline check (skip almost everything).

    Runs fetch-games + clean-games against the external nflverse source.
    It does not run upcoming schedule, weather, odds, EPA, Elo, features,
    weekly readiness, or frontend checks. The goal is to verify the completed-
    game ingest and cleaning boundary, not to rebuild the complete pipeline.

    Soft-fail by default because the external source can be unavailable.
    ``--strict`` converts this to a hard failure.
    """
    from gridiron_edge.cli.main import _run_pipeline_stages

    active = {"fetch-games", "clean-games"}

    _run_pipeline_stages(
        active=active,
        all_years=False,
        resolved_season=0,
        upcoming_target=0,
        season=None,
        season_year=None,
        owm_api_key=None,
        fit_elo_all_years=False,
    )
    return StageResult(success=True, detail="fetch-games + clean-games OK")


def _parse_composite_key(pair_key: str) -> tuple[str, str] | None:
    """Parse a composite model key into (model_name, model_type).

    Composite keys use:

        {model_name}_{model_type}

    where model_type may itself contain underscores
    (e.g. random_forest).

    Returns:
        (model_name, model_type) if recognized, otherwise None.
    """
    known_model_types: tuple[str, ...] = (
        "random_forest",
        "xgboost",
        "logistic",
        "elasticnet",
        "elo",
    )

    for model_type in known_model_types:
        suffix: str = f"_{model_type}"

        if pair_key.endswith(suffix):
            model_name: str = pair_key.removesuffix(suffix)

            if model_name:
                return (model_name, model_type)

    return None


def _stage_baseline_comparison(ctx: dict[str, Any]) -> StageResult:
    """Compare current artifact metrics against the most recent full-retrain report.

    Reads `data/output/reports/full-retrain-*.md`, finds the most recent
    one, and compares each game model's current `metrics` dict against
    the values reported there.

    Soft-fails when no prior report exists (first run).
    """
    from gridiron_edge.cli.full_retrain import (
        _find_previous_baseline_report,
        _parse_baseline_report,
    )

    repo_root: Path = ctx["repo_root"]
    report_dir: Path = repo_root / "data" / "output" / "reports"

    if not report_dir.exists():
        return StageResult(
            success=False,
            detail="no full-retrain report directory found",
        )

    latest: Path | None = _find_previous_baseline_report(report_dir)

    if latest is None:
        return StageResult(
            success=False,
            detail="no full-retrain reports to compare against",
        )

    report_metrics: dict[str, dict[str, float | None]] = _parse_baseline_report(latest)

    store = ArtifactStore(repo_root)

    drifted: list[str] = []
    checked = 0

    for pair_key, baseline_metrics in report_metrics.items():
        parsed: tuple[str, str] | None = _parse_composite_key(pair_key)

        if parsed is None:
            continue

        model_name, model_type = parsed

        if not store.is_trained(model_name, model_type):
            continue

        meta: BaseModelMetadata = store.read_metadata(model_name, model_type)

        checked += 1

        for metric_name, baseline_value in baseline_metrics.items():
            if baseline_value is None:
                continue

            current_value: float | None = meta.metrics.get(metric_name)
            if current_value is None:
                continue

            tolerance = 1e-6

            if abs(current_value - baseline_value) > tolerance:
                drifted.append(f"{pair_key}:{metric_name}")
                break

    if drifted:
        return StageResult(
            success=True,
            detail=f"{len(drifted)} model(s) differ from baseline",
            warnings=drifted[:5],
            artifacts=[latest],
        )

    return StageResult(
        success=True,
        detail=f"{checked} model(s) match baseline",
        artifacts=[latest],
    )


# ---------------------------------------------------------------------------
# Stage list builders (mode-dependent)
# ---------------------------------------------------------------------------


def _build_stages(*, fast: bool, very_thorough: bool) -> list:
    """Define the stages for verify.

    Args:
        fast: If True, skip e2e tests and smoke-pipeline.
        very_thorough: If True, append slow tests.
    """
    stages: list[CompositeStage] = [
        CompositeStage(
            name="quality-gates",
            description="Run ruff + pyrefly",
            func=_stage_quality_gates,
        ),
        CompositeStage(
            name="unit-tests",
            description="Run unit tests (not slow)",
            func=_stage_unit_tests,
        ),
        CompositeStage(
            name="integration-tests",
            description="Run integration tests",
            func=_stage_integration_tests,
        ),
    ]

    if not fast:
        stages.extend(
            [
                CompositeStage(
                    name="e2e-tests",
                    description="Run end-to-end tests",
                    func=_stage_e2e_tests,
                ),
                CompositeStage(
                    name="smoke-pipeline",
                    description="Light run-data-pipeline check",
                    func=_stage_smoke_pipeline,
                    soft_fail=True,
                ),
            ]
        )

    if very_thorough:
        stages.append(
            CompositeStage(
                name="slow-tests",
                description="Run slow tests",
                func=_stage_slow_tests,
            )
        )

    stages.append(
        CompositeStage(
            name="baseline-comparison",
            description="Compare current metrics to last full-retrain",
            func=_stage_baseline_comparison,
            soft_fail=True,
        )
    )

    return stages


# Stage names for the most permissive mode, used for flag validation.
_MAX_STAGES: list[str] = [
    "quality-gates",
    "unit-tests",
    "integration-tests",
    "e2e-tests",
    "smoke-pipeline",
    "slow-tests",
    "baseline-comparison",
]
_STAGES_STR: str = ", ".join(_MAX_STAGES)
_SKIP_HELP: str = f"Stage(s) to skip. Repeatable. Valid: {_STAGES_STR}."
_ONLY_HELP: str = f"Run only these stage(s). Repeatable. Valid: {_STAGES_STR}."


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def verify_cmd(
    *,
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Skip e2e tests and smoke-pipeline for a quick check.",
    ),
    very_thorough: bool = typer.Option(
        False,
        "--very-thorough",
        help="Include slow tests (significantly extends runtime).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help=(
            "Convert soft-failures (smoke-pipeline, baseline-comparison) "
            "into hard failures. Use for CI."
        ),
    ),
    skip: list[str] = typer.Option(  # noqa: B008
        [],
        "--skip",
        help=_SKIP_HELP,
    ),
    only: list[str] = typer.Option(  # noqa: B008
        [],
        "--only",
        help=_ONLY_HELP,
    ),
) -> None:
    r"""Verify Python repository health and backend behavior.

    The default mode runs Ruff, Pyrefly, backend unit, integration, and e2e
    tests, an external nflverse fetch-games + clean-games smoke check, and a
    baseline comparison against the most recent full-retrain report. ``--fast``
    skips e2e and smoke. ``--very-thorough`` adds slow tests. ``--strict``
    converts smoke and missing-baseline soft failures into hard failures.

    This command does not run selected weekly-product readiness; use
    ``gridiron verify-week`` for that. Frontend build, Vitest, and ESLint
    remain separate checks and are not run by this command.

    \b
    Examples:
      gridiron verify
      gridiron verify --fast
      gridiron verify --strict
      gridiron verify --very-thorough
      gridiron verify --only quality-gates --only unit-tests
    """
    stages = _build_stages(fast=fast, very_thorough=very_thorough)
    active_names = [s.name for s in stages]

    # If --skip or --only filter the active set, validate against the
    # built stage list (which depends on --fast / --very-thorough).
    active = resolve_active_stages(
        all_stages=active_names,
        skip=skip,
        only=only,
    )

    subtitle_parts: list[str] = []
    if fast:
        subtitle_parts.append("fast")
    if very_thorough:
        subtitle_parts.append("very-thorough")
    if strict:
        subtitle_parts.append("strict")
    subtitle = " · ".join(subtitle_parts) or "default"

    context: dict[str, Any] = {
        "repo_root": get_settings().repo_root,
    }

    console.header("verify", subtitle=subtitle)

    summary = run_composite(
        name="verify",
        stages=stages,
        active=active,
        context=context,
        strict=strict,
    )

    render_composite_summary(summary)

    if not summary.overall_success:
        raise typer.Exit(code=1)

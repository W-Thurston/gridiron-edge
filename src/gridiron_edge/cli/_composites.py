"""Shared infrastructure for composite CLI workflows.

Composite commands compose multiple single-purpose CLI stages into a
unified workflow with consistent flag handling, staleness checking,
and summary rendering. The pattern mirrors ``run-data-pipeline`` from
``cli/main.py``.

This module provides:

- ``CompositeStage`` - declarative definition of one stage.
- ``StageResult`` - return value from a stage function.
- ``CompositeSummary`` - accounting of a full composite execution.
- ``run_composite`` - the orchestrator.
- ``render_composite_summary`` - final console summary.
- ``resolve_active_stages`` - flag handling helper.

Composite command authors define a list of ``CompositeStage`` plus
the per-stage functions, then call ``run_composite``. The orchestrator
handles stage iteration, error propagation, soft-fail semantics, and
summary rendering.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import typer

# ---------------------------------------------------------------------------
# Stage definition and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeStage:
    """Declarative definition of one stage in a composite workflow.

    Attributes:
        name: Unique identifier within the composite. Used for
            ``--skip`` / ``--only`` filters and in console output.
        description: Human-readable description shown in the
            ``step()`` context manager.
        func: Callable that executes this stage. Signature:
            ``func(context: dict[str, Any]) -> StageResult``.
        depends_on: Tuple of stage names that should have run
            successfully before this one. Used by the dependency
            validator at composite start.
        soft_fail: If True, a failure or exception does not abort the
            composite. Useful for external-service stages (e.g., odds
            fetch when the upstream service is unreliable).
    """

    name: str
    description: str
    func: Callable[[dict[str, Any]], StageResult]
    depends_on: tuple[str, ...] = ()
    soft_fail: bool = False


@dataclass
class StageResult:
    """Outcome of one stage execution.

    Attributes:
        success: True if the stage completed without error.
        detail: Optional string for the ``step()`` detail field.
        rows: Optional row count for the ``step()`` rows field.
        artifacts: Paths to files written during this stage. Surfaced
            in the final composite summary.
        warnings: Soft warnings to surface in the final summary even
            when the stage succeeded.
    """

    success: bool
    detail: str = ""
    rows: int | None = None
    artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Composite summary
# ---------------------------------------------------------------------------


@dataclass
class CompositeSummary:
    """Final accounting of a composite command's execution.

    Attributes:
        name: Composite command name for display.
        succeeded: Stage names that completed successfully.
        skipped: Stage names that were skipped via ``--skip`` /
            ``--only``.
        soft_failed: Stage names that failed but had ``soft_fail=True``.
            These do not abort the composite.
        failed: Stage names that failed without ``soft_fail=True``.
            The composite aborts at the first hard failure.
        artifacts: All paths written across stages.
        warnings: All warnings raised across stages.
    """

    name: str
    succeeded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    soft_failed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def overall_success(self) -> bool:
        """True when no hard failures occurred.

        Soft failures and skipped stages do not affect this.
        """
        return len(self.failed) == 0


# ---------------------------------------------------------------------------
# Dependency validation
# ---------------------------------------------------------------------------


def _check_dependencies(
    *,
    stages: Sequence[CompositeStage],
    active: set[str],
    assume_satisfied: set[str] | None = None,
) -> None:
    """Validate that depends_on references point at known stages.

    A stage's dependencies must all be active or come earlier in the
    stage list. Raises ``typer.BadParameter`` if a dependency is
    unmet at composite start (the caller deactivated an upstream stage).

    Args:
        stages: Ordered list of all stages in this composite.
        active: Set of stage names selected for execution.
        assume_satisfied: Stage names that completed in a prior run and
            whose outputs are on disk. Treated as pre-satisfied when
            checking dependencies. Enables resume-from-partial-run
            workflows without re-running expensive completed stages.

    Raises:
        typer.BadParameter: If any active stage has an unmet
            ``depends_on`` reference.
    """
    all_names: set[str] = {s.name for s in stages}
    seen_active: set[str] = set(assume_satisfied or ())
    unmet: list[tuple[str, str]] = []

    for stage in stages:
        if stage.name not in active:
            continue
        for dep in stage.depends_on:
            if dep not in all_names:
                unmet.append((stage.name, f"unknown stage '{dep}'"))
                continue
            if dep not in seen_active:
                unmet.append((stage.name, f"requires '{dep}'"))
        seen_active.add(stage.name)

    if unmet:
        lines = "\n  ".join(f"  {stage}: {reason}" for stage, reason in unmet)
        raise typer.BadParameter(
            f"Stage dependencies unmet:\n  {lines}\n\n"
            f"Either include the required stages or use --only with "
            f"the full chain."
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_composite(
    *,
    name: str,
    stages: Sequence[CompositeStage],
    active: set[str],
    context: dict[str, Any] | None = None,
    strict: bool = False,
    assume_satisfied: set[str] | None = None,
) -> CompositeSummary:
    """Execute a sequence of stages with consistent error handling.

    Each stage runs inside the ``step()`` context manager so console
    output matches ``run-data-pipeline``. Successful stages add their
    detail and artifacts to the summary. Failed stages either abort
    the composite (default) or are recorded as soft failures and
    skipped (when ``soft_fail=True``).

    Args:
        name: Composite command name (e.g., ``"weekly-predict"``).
        stages: Ordered sequence of ``CompositeStage`` definitions.
        active: Set of stage names to run.
        context: Optional shared context dict passed to each stage's
            ``func``. Stages can read inputs (week, season) and write
            outputs (DataFrames) here for downstream stages to consume.
        strict: If True, all soft failures are treated as hard
            failures. Used by ``verify`` to convert flaky-network
            warnings into a clean signal.
        assume_satisfied: Stage names that completed in a prior run and
            whose outputs are on disk. Forwarded to
            :func:`_check_dependencies` so dependency checks accept
            missing-but-completed stages.

    Returns:
        ``CompositeSummary`` with per-stage outcomes and consolidated
        artifact/warning lists.
    """
    from gridiron_edge.core.console import step

    _check_dependencies(
        stages=stages,
        active=active,
        assume_satisfied=assume_satisfied,
    )

    ctx: dict[str, Any] = context if context is not None else {}
    summary = CompositeSummary(name=name)

    for stage in stages:
        if stage.name not in active:
            with step(stage.description, skip=True):
                summary.skipped.append(stage.name)
            continue

        soft_fail = stage.soft_fail and not strict

        with step(stage.description) as s:
            try:
                result = stage.func(ctx)
            except Exception as exc:
                if soft_fail:
                    msg = f"{stage.name}: {type(exc).__name__}: {exc}"
                    summary.warnings.append(msg)
                    summary.soft_failed.append(stage.name)
                    s.set_detail(f"soft-failed: {type(exc).__name__}")
                    continue
                summary.failed.append(stage.name)
                s.set_detail(f"failed: {type(exc).__name__}")
                raise

            if not result.success:
                if soft_fail:
                    summary.warnings.append(f"{stage.name}: {result.detail}")
                    summary.soft_failed.append(stage.name)
                    s.set_detail(f"soft-failed: {result.detail}")
                    continue
                summary.failed.append(stage.name)
                s.set_detail(result.detail or "failed")
                return summary

            s.set_detail(result.detail)
            if result.rows is not None:
                s.set_rows(result.rows)
            summary.succeeded.append(stage.name)
            summary.artifacts.extend(result.artifacts)
            summary.warnings.extend(result.warnings)

    return summary


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def render_composite_summary(summary: CompositeSummary) -> None:
    """Render a composite execution summary to the console.

    Args:
        summary: Result of ``run_composite``.
    """
    typer.echo("")
    divider = "━" * 60
    typer.echo(divider)
    typer.echo(f"  {summary.name} summary")
    typer.echo(divider)
    typer.echo("")

    n_succeeded = len(summary.succeeded)
    n_skipped = len(summary.skipped)
    n_soft_failed = len(summary.soft_failed)
    n_failed = len(summary.failed)

    parts: list[str] = []
    parts.append(f"  ✓ {n_succeeded} stage{'s' if n_succeeded != 1 else ''} completed")
    if n_skipped:
        skipped_names = ", ".join(summary.skipped)
        parts.append(f"  ⊘ {n_skipped} skipped ({skipped_names})")
    if n_soft_failed:
        soft_names = ", ".join(summary.soft_failed)
        parts.append(f"  ⚠ {n_soft_failed} soft-failed ({soft_names})")
    if n_failed:
        fail_names = ", ".join(summary.failed)
        parts.append(f"  ✗ {n_failed} failed ({fail_names})")
    for part in parts:
        typer.echo(part)

    if summary.artifacts:
        typer.echo("")
        typer.echo("  Artifacts written:")
        for artifact in summary.artifacts:
            typer.echo(f"    {artifact}")

    if summary.warnings:
        typer.echo("")
        typer.echo("  Warnings:")
        for warning in summary.warnings:
            typer.echo(f"    {warning}")

    typer.echo("")


# ---------------------------------------------------------------------------
# Flag handling
# ---------------------------------------------------------------------------


def resolve_active_stages(
    *,
    all_stages: list[str],
    skip: list[str],
    only: list[str],
) -> set:
    """Resolve ``--skip`` and ``--only`` flags into the active stage set.

    Mirrors the pattern in ``cli/main.py::run_data_pipeline``.

    Args:
        all_stages: All stage names defined for this composite.
        skip: Stage names to skip (mutually exclusive with ``only``).
        only: Stage names to run exclusively (mutually exclusive
            with ``skip``).

    Returns:
        Set of stage names to execute.

    Raises:
        typer.BadParameter: If ``--skip`` and ``--only`` are both
            provided, or if either references an unknown stage.
    """
    if skip and only:
        raise typer.BadParameter("--skip and --only are mutually exclusive.")

    unknown = set(skip + only) - set(all_stages)
    if unknown:
        raise typer.BadParameter(
            f"Unknown stage(s): {', '.join(sorted(unknown))}. "
            f"Valid stages: {', '.join(all_stages)}."
        )

    return set(only) if only else set(all_stages) - set(skip)


def write_champion_manifest(repo: Path) -> None:
    """Run all three champion selectors and persist the manifest.

    Shared by ``evaluate select-model --write-manifest`` and
    ``props champion --write-manifest``. Uses the full catalog
    (:mod:`gridiron_edge.models.catalog`) so the manifest reflects
    the entire repo state. Preservation semantics apply — families
    outside the current backfill scope keep their prior manifest
    entries verbatim.

    Args:
        repo: Repository root.

    Side effects:
        Writes ``data/output/champions/champions.json`` and echoes a
        short summary to the console.
    """
    from gridiron_edge.core.console import step
    from gridiron_edge.evaluation.champion import promote_champions
    from gridiron_edge.models.catalog import (
        GAME_MODEL_PAIRS,
        PROP_STAT_FAMILIES,
    )

    with step("Persist champion manifest") as s:
        promote_result = promote_champions(
            game_pairs=list(GAME_MODEL_PAIRS),
            prop_families=list(PROP_STAT_FAMILIES),
            repo=repo,
        )
        fresh_summary = ", ".join(sorted(promote_result.fresh_entries.keys())) or "none"
        s.set_detail(f"fresh: {fresh_summary}; preserved: {len(promote_result.preserved_entries)}")

    typer.echo("")
    typer.echo(f"Manifest written: {promote_result.manifest_path}")
    for warning in promote_result.warnings:
        typer.echo(f"  ⚠  {warning}")


def resolve_win_prob_model_type(model_type: str) -> str:
    """Resolve a --model-type CLI value, honoring the 'auto' sentinel.

    When ``model_type == "auto"``, reads the champion manifest and
    returns the current champion's model_type for the ``win_prob``
    model_name. Any other value is returned as-is.

    Raises:
        typer.BadParameter: If ``model_type == "auto"`` but the manifest
            is missing or has no entry for ``win_prob``.
    """
    if model_type != "auto":
        return model_type

    from gridiron_edge.evaluation.champion_resolver import (
        ChampionNotFoundError,
        resolve_current_champion,
    )

    try:
        _, resolved = resolve_current_champion("win_prob")
    except ChampionNotFoundError as exc:
        raise typer.BadParameter(
            f"--model-type auto requires a champion manifest. {exc}\n\n"
            f"Run one of:\n"
            f"  gridiron full-retrain\n"
            f"  gridiron evaluate select-model --write-manifest"
        ) from exc

    return resolved

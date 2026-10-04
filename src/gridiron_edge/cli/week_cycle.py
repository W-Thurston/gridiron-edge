# src/gridiron_edge/cli/week_cycle.py

"""The weekly rhythm as one command: close the finished week, then forecast (``DECISIONS.md`` D81).

``week-cycle close`` (Tuesday) composes ``post-week`` with a completeness check and a collector
check; ``week-cycle forecast`` (Friday, after the injury report) composes ``weekly-predict`` with
worktree and injury-report checks and a summary of what the forecast assumed. Stages are the
existing composite stages, reused rather than copied.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd
from pandas import DataFrame
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

week_cycle_app = typer.Typer(
    help="Close the finished week and forecast the next one.",
    no_args_is_help=True,
)

_ROLLED_OUTCOMES = frozenset({"rolled", "selected_existing"})


# ---------------------------------------------------------------------------
# Pure checks (each testable without the network or the heavy stages)
# ---------------------------------------------------------------------------


def week_completion(games: DataFrame, *, season_int: int, week: int) -> tuple[int, int]:
    """Return ``(scheduled, played)`` regular-season games for one week of the raw games file."""
    scoped = games.loc[(games["season"].astype(int) == season_int) & (games["week"] == week)]
    if "game_type" in scoped.columns:
        scoped = scoped.loc[scoped["game_type"] == "REG"]
    return len(scoped), int(scoped["result"].notna().sum())


def injury_report_present(injuries: DataFrame | None, *, week: int) -> bool:
    """Return whether the cached injury file has regular-season reports for ``week``."""
    if injuries is None or injuries.empty:
        return False
    scoped = injuries.loc[injuries["week"].astype(int) == week]
    if "game_type" in scoped.columns:
        scoped = scoped.loc[scoped["game_type"] == "REG"]
    return not scoped.empty


def dirty_tracked_files(repo: Path) -> list[str]:
    """Return tracked files with uncommitted changes (``weekly-predict`` needs none)."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    return [line[3:] for line in result.stdout.splitlines() if line.strip()]


def collector_lines(repo: Path, *, season: str, next_week: int, now: datetime) -> list[str]:
    """Say whether the worker has rolled to ``next_week`` and how current the pull is.

    Reads the rollover receipts and sync status the pull brings to ``data/odds``; the
    worker's own plan pointer is not visible here.
    """
    odds = repo / "data" / "odds"
    target = f"{season} week {next_week}"
    rolled: str | None = None
    for path in sorted((odds / "collection_rollovers").glob("*.json")):
        try:
            receipt = json.loads(path.read_text())
        except ValueError:
            continue
        if receipt.get("outcome") in _ROLLED_OUTCOMES and receipt.get("target") == target:
            rolled = str(receipt.get("decided_at"))
    lines = [
        f"collector rolled to {target} at {rolled}"
        if rolled
        else f"no rollover receipt for {target} has been pulled yet (the worker rolls after "
        "the last kickoff of the finished week; the pull runs every 15 minutes)"
    ]
    status_path = odds / "sync_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            success = status.get("last_success_at")
            if success:
                age = now - datetime.fromisoformat(success)
                lines.append(f"collector sync last succeeded {_hours(age.total_seconds())} ago")
            if status.get("error"):
                lines.append(f"last sync attempt failed: {status['error']}")
        except (ValueError, TypeError):
            lines.append("collector sync status unreadable")
    else:
        lines.append("no collector sync status recorded")
    return lines


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{round(seconds / 60)} min"


def starter_lines(product: DataFrame) -> list[str]:
    """Summarize the recorded starting-QB assumptions of a weekly product."""
    if "starting_qb_status" not in product.columns:
        return ["starting QBs were not recorded for this product"]
    lines: list[str] = []
    counts: dict[str, int] = {}
    for side in ("away", "home"):
        for source in product[f"{side}_qb_source"].dropna():
            counts[str(source)] = counts.get(str(source), 0) + 1
    unresolved = int((product["starting_qb_status"] != "available").sum())
    lines.append(
        "starter sources: "
        + (", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "none")
        + (f"; {unresolved} game(s) not fully resolved" if unresolved else "")
    )
    for _, row in product.iterrows():
        for side in ("away", "home"):
            differs = row[f"{side}_qb_differs_from_carry_forward"]
            if not pd.isna(differs) and bool(differs):
                name = row[f"{side}_qb_name"]
                label = name if isinstance(name, str) else row[f"{side}_qb_id"]
                lines.append(
                    f"  {row['game_id']} {side}: {label} "
                    f"({row[f'{side}_qb_source']}) differs from the carried-forward starter"
                )
    return lines


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def _stage_verify_week_complete(ctx: dict[str, Any]) -> StageResult:
    """Require every scheduled game of the finished week to have a result."""
    from gridiron_edge.datasets.registry import dataset_path

    repo: Path = get_settings().repo_root
    games = pd.read_parquet(dataset_path(repo, "games_raw_nflverse"))
    scheduled, played = week_completion(games, season_int=ctx["season_int"], week=ctx["week"])
    detail = f"{played} of {scheduled} games have results"
    if scheduled == 0:
        return StageResult(success=False, detail=f"no scheduled games found for week {ctx['week']}")
    if played < scheduled:
        if ctx.get("allow_incomplete"):
            return StageResult(
                success=True,
                detail=detail,
                warnings=[f"week {ctx['week']} is incomplete: {detail}; closing it anyway"],
            )
        return StageResult(
            success=False,
            detail=f"{detail}; wait for the last game or pass --allow-incomplete",
        )
    return StageResult(success=True, detail=detail)


def _stage_check_collector(ctx: dict[str, Any]) -> StageResult:
    """Report whether the quote collector has rolled to the next week (informational)."""
    lines = collector_lines(
        get_settings().repo_root,
        season=ctx["season"],
        next_week=ctx["week"] + 1,
        now=datetime.now(UTC),
    )
    ctx.setdefault("cycle_report", []).extend(lines)
    rolled = lines[0].startswith("collector rolled")
    return StageResult(
        success=rolled,
        detail=lines[0],
    )


def _stage_next_step(ctx: dict[str, Any]) -> StageResult:
    next_week = ctx["week"] + 1
    ctx.setdefault("cycle_report", []).append(
        f"Friday, after the injury report: uv run gridiron week-cycle forecast "
        f"--season {ctx['season']} --week {next_week}"
    )
    return StageResult(success=True, detail=f"week {next_week} forecast is the next phase")


def build_close_stages() -> list[CompositeStage]:
    """Assemble ``close``: post-week stages with a completeness gate and a collector check."""
    from gridiron_edge.cli.post_week import _build_stages as post_week_stages

    by_name = {stage.name: stage for stage in post_week_stages()}
    gate = ("verify-week-complete",)
    return [
        by_name["refresh-results"],
        CompositeStage(
            name="verify-week-complete",
            description="Verify the finished week has results",
            func=_stage_verify_week_complete,
            depends_on=("refresh-results",),
        ),
        replace(by_name["refresh-next-week-state"], depends_on=gate),
        replace(by_name["refresh-season-projections"], depends_on=gate),
        replace(by_name["close-live-forecasts"], depends_on=gate),
        CompositeStage(
            name="check-collector",
            description="Check the quote collector rolled to the next week",
            func=_stage_check_collector,
            soft_fail=True,
        ),
        CompositeStage(
            name="next-step",
            description="State the next phase",
            func=_stage_next_step,
        ),
    ]


@week_cycle_app.command("close")
def close_cmd(
    *,
    week: int = typer.Option(..., help="NFL week number that just completed."),
    season: str = typer.Option(..., help="NFL season label, e.g. '2026-2027'."),
    allow_incomplete: bool = typer.Option(
        False,
        "--allow-incomplete/--require-complete",
        help="Close the week even if some scheduled games have no result yet.",
    ),
    skip: list[str] = typer.Option([], "--skip", help="Stage(s) to skip. Repeatable."),  # noqa: B008
    only: list[str] = typer.Option([], "--only", help="Run only these stage(s)."),  # noqa: B008
) -> None:
    r"""Close the finished week and set up the next one (run on Tuesday).

    Refreshes results, verifies every game of the week has one, rebuilds next-week
    state and season projections, scores the live forecasts issued before kickoff,
    and reports whether the quote collector has rolled to the next week. It does not
    forecast the next week: run ``week-cycle forecast`` on Friday, after the injury report.

    \b
    Example:
      gridiron week-cycle close --week 4 --season 2026-2027
    """
    stages = build_close_stages()
    names = [stage.name for stage in stages]
    active = resolve_active_stages(all_stages=names, skip=skip, only=only)
    context: dict[str, Any] = {
        "week": week,
        "season": season,
        "season_int": _season_int(season),
        "allow_incomplete": allow_incomplete,
    }
    console.header(
        "week-cycle close", subtitle=f"week {week} · {season} · close and set up week {week + 1}"
    )
    summary = run_composite(name="week-cycle close", stages=stages, active=active, context=context)
    render_composite_summary(summary)
    _print_report(context)
    if not summary.overall_success:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------


def _stage_check_clean_worktree(ctx: dict[str, Any]) -> StageResult:
    dirty = dirty_tracked_files(get_settings().repo_root)
    if dirty:
        shown = ", ".join(dirty[:5]) + (f" and {len(dirty) - 5} more" if len(dirty) > 5 else "")
        return StageResult(
            success=False,
            detail=f"weekly-predict needs a clean tracked worktree; uncommitted: {shown}",
        )
    return StageResult(success=True, detail="tracked worktree is clean")


def _stage_check_injury_report(ctx: dict[str, Any]) -> StageResult:
    """Refresh the injury cache and require reports for the week being forecast."""
    from gridiron_edge.ingest.nflverse.weekly_availability import fetch_injuries, load_injuries

    repo: Path = get_settings().repo_root
    season_int: int = ctx["season_int"]
    fetch_injuries(seasons=[season_int], force=True, repo=repo)
    try:
        injuries = load_injuries(seasons=[season_int], repo=repo).get(season_int)
    except FileNotFoundError:
        injuries = None
    if injury_report_present(injuries, week=ctx["week"]):
        return StageResult(success=True, detail=f"week {ctx['week']} injury reports are published")
    message = (
        f"no week {ctx['week']} injury report is published yet; the projected starters "
        "need it (the report usually lands Friday)"
    )
    if ctx.get("allow_early"):
        return StageResult(
            success=True,
            detail="proceeding early",
            warnings=[message + "; forecasting on the early picture"],
        )
    return StageResult(success=False, detail=message + "; wait, or pass --allow-early")


def _stage_report_forecast(ctx: dict[str, Any]) -> StageResult:
    """Summarize what the forecast assumed and how fresh its market is."""
    from gridiron_edge.cli.verify_week import _market_age_suffix, load_weekly_readiness
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        load_current_weekly_product,
    )

    repo: Path = get_settings().repo_root
    product = load_current_weekly_product(season=ctx["season"], week=ctx["week"], repo=repo)
    lines = starter_lines(product)
    readiness = load_weekly_readiness(season=ctx["season"], week=ctx["week"], repo=repo)
    lines.append(f"prediction ready: {'yes' if readiness.prediction_ready else 'no'}")
    lines.append(
        f"market ready: {'yes' if readiness.market_ready else 'no'}"
        + _market_age_suffix(readiness.market_fetched_at)
    )
    ctx.setdefault("cycle_report", []).extend(lines)
    return StageResult(success=True, detail="forecast summary ready")


def build_forecast_stages() -> list[CompositeStage]:
    """Assemble ``forecast``: pre-checks, the ``weekly-predict`` stages, then a summary."""
    from gridiron_edge.cli.weekly_predict import _build_stages as weekly_predict_stages

    checks = (
        CompositeStage(
            name="check-clean-worktree",
            description="Check the tracked worktree is clean",
            func=_stage_check_clean_worktree,
        ),
        CompositeStage(
            name="check-injury-report",
            description="Check the week's injury report is published",
            func=_stage_check_injury_report,
        ),
    )
    stages = list(checks)
    gate = tuple(check.name for check in checks)
    for stage in weekly_predict_stages():
        stages.append(
            replace(stage, depends_on=gate) if stage.name == "ensure-data-fresh" else stage
        )
    stages.append(
        CompositeStage(
            name="report-forecast",
            description="Summarize the forecast's assumptions and market",
            func=_stage_report_forecast,
            depends_on=("compose-weekly-product",),
            soft_fail=True,
        )
    )
    return stages


@week_cycle_app.command("forecast")
def forecast_cmd(
    *,
    week: int = typer.Option(..., help="NFL week number to forecast."),
    season: str = typer.Option(..., help="NFL season label, e.g. '2026-2027'."),
    allow_early: bool = typer.Option(
        False,
        "--allow-early/--require-injury-report",
        help="Forecast even if the week's injury report is not published yet.",
    ),
    bankroll: float | None = typer.Option(
        None, min=0.0, help="Optional bankroll for Kelly stake sizing in the edge report."
    ),
    skip: list[str] = typer.Option([], "--skip", help="Stage(s) to skip. Repeatable."),  # noqa: B008
    only: list[str] = typer.Option([], "--only", help="Run only these stage(s)."),  # noqa: B008
) -> None:
    r"""Forecast the next week once its injury report is in (run on Friday).

    Checks the tracked worktree is clean and the week's injury report is published,
    runs ``weekly-predict``, then reports the starting quarterbacks the forecast
    assumed, prediction and market readiness, and the market's age.

    \b
    Example:
      gridiron week-cycle forecast --week 5 --season 2026-2027
    """
    from gridiron_edge.cli.weekly_predict import _render_edge_preview, weekly_predict_context

    stages = build_forecast_stages()
    names = [stage.name for stage in stages]
    active = resolve_active_stages(all_stages=names, skip=skip, only=only)
    context = weekly_predict_context(season=season, week=week, bankroll=bankroll)
    context["allow_early"] = allow_early
    console.header("week-cycle forecast", subtitle=f"week {week} · {season} · forecast")
    summary = run_composite(
        name="week-cycle forecast", stages=stages, active=active, context=context
    )
    render_composite_summary(summary)
    top_edges = context.get("top_edges_preview")
    if top_edges is not None and not top_edges.empty:
        _render_edge_preview(top_edges)
    _print_report(context)
    if not summary.overall_success:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------


def _season_int(season: str) -> int:
    try:
        return int(season.split("-")[0])
    except (ValueError, IndexError) as exc:
        raise typer.BadParameter(
            f"Could not parse season {season!r}. Expected format 'YYYY-YYYY+1'."
        ) from exc


def _print_report(context: dict[str, Any]) -> None:
    lines = context.get("cycle_report") or []
    if not lines:
        return
    typer.echo("")
    for line in lines:
        typer.echo(f"  {line}")
    typer.echo("")

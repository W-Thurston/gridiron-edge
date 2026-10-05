# src/gridiron_edge/cli/week_cycle.py

"""The weekly cycle as one idempotent, state-driven command (``DECISIONS.md`` D82).

``gridiron week-cycle`` works out the target week from the schedule and does whichever step is
due: close the finished week, forecast the target week, or watch the starting quarterbacks.
The decision rules and receipts live in :mod:`gridiron_edge.weekly_cycle`; this module supplies
the real effects (the existing composite stages, the nflverse refreshes) and the commands.
Run it by hand at any time or hourly from a timer.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from dataclasses import replace
from datetime import UTC, datetime
import fcntl
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd
from pandas import DataFrame
import typer

from gridiron_edge.cli._composites import (
    CompositeStage,
    CompositeSummary,
    StageResult,
    render_composite_summary,
    resolve_active_stages,
    run_composite,
)
from gridiron_edge.core.console import console
from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.week_schedule import completed_game_ids, week_completion
from gridiron_edge.weekly_cycle.receipts import read_receipts
from gridiron_edge.weekly_cycle.runner import (
    AvailabilityRefresh,
    CyclePorts,
    StageRun,
    run_cycle,
)
from gridiron_edge.weekly_cycle.starter_watch import frame_digest, observe_starters

week_cycle_app = typer.Typer(
    help="Prepare each week by itself: close it, forecast the next, watch the starters.",
    invoke_without_command=True,
)

_ROLLED_OUTCOMES = frozenset({"rolled", "selected_existing"})


# ---------------------------------------------------------------------------
# Pure checks (each testable without the network or the heavy stages)
# ---------------------------------------------------------------------------


def coverage_gaps(expected: set[str], sources: dict[str, set[str] | None]) -> dict[str, list[str]]:
    """Return, per downstream artifact, the expected games it does not hold yet.

    A source of ``None`` means the artifact could not be read, which counts as holding none.
    """
    return {
        name: sorted(expected - (found or set()))
        for name, found in sources.items()
        if expected - (found or set())
    }


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
# Stages (the existing composite stages, reused)
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


def _read_game_ids(path: Path, column: str) -> set[str] | None:
    if not path.exists():
        return None
    return set(pd.read_parquet(path, columns=[column])[column].astype(str))


def _coverage_sources(repo: Path, season_int: int) -> dict[str, set[str] | None]:
    """Read the game ids each downstream artifact currently holds for the season."""
    from gridiron_edge.datasets.registry import dataset_path

    raw = repo / "data" / "raw"
    return {
        "player stats": _read_game_ids(
            raw / "player_stats" / f"player_stats_{season_int}.parquet", "game_id"
        ),
        "play-by-play": _read_game_ids(
            raw / "pbp" / f"play_by_play_{season_int}.parquet", "game_id"
        ),
        "EPA by game": _read_game_ids(dataset_path(repo, "epa_by_game"), "game_id"),
        "player game logs": _read_game_ids(dataset_path(repo, "player_game_logs"), "game_id"),
        "modeling file": _read_game_ids(dataset_path(repo, "modeling_full"), "GAME_ID"),
    }


def _stage_verify_week_data_current(ctx: dict[str, Any]) -> StageResult:
    """Require every artifact the next forecast learns from to hold the finished week's games.

    nflverse can publish results before player stats or play-by-play catch up, so a pull
    made too early passes the results gate but leaves the rebuilt features a week short.
    """
    from gridiron_edge.datasets.registry import dataset_path

    repo: Path = get_settings().repo_root
    games = pd.read_parquet(dataset_path(repo, "games_raw_nflverse"))
    expected = completed_game_ids(games, season_int=ctx["season_int"], week=ctx["week"])
    sources = _coverage_sources(repo, ctx["season_int"])
    gaps = coverage_gaps(expected, sources)
    report = ctx.setdefault("cycle_report", [])
    for name, found in sources.items():
        held = len(expected & (found or set()))
        report.append(f"{name}: {held} of {len(expected)} week {ctx['week']} games")
    if not gaps:
        return StageResult(success=True, detail=f"all {len(expected)} games are in every artifact")
    summary = "; ".join(f"{name} missing {len(ids)}" for name, ids in gaps.items())
    message = f"upstream data is not current for week {ctx['week']}: {summary}"
    if ctx.get("allow_incomplete"):
        return StageResult(success=True, detail=summary, warnings=[message + "; closing anyway"])
    return StageResult(success=False, detail=message + "; retrying hourly")


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


def build_close_stages() -> list[CompositeStage]:
    """Assemble ``close``: post-week stages with a completeness gate and a collector check."""
    from gridiron_edge.cli.post_week import _build_stages as post_week_stages

    by_name = {stage.name: stage for stage in post_week_stages()}
    gate = ("verify-week-complete",)
    current = ("verify-week-data-current",)
    return [
        by_name["refresh-results"],
        CompositeStage(
            name="verify-week-complete",
            description="Verify the finished week has results",
            func=_stage_verify_week_complete,
            depends_on=("refresh-results",),
        ),
        replace(by_name["refresh-next-week-state"], depends_on=gate),
        CompositeStage(
            name="verify-week-data-current",
            description="Verify the rebuilt data holds the finished week",
            func=_stage_verify_week_data_current,
            depends_on=("refresh-next-week-state",),
        ),
        replace(by_name["refresh-season-projections"], depends_on=current),
        replace(by_name["close-live-forecasts"], depends_on=current),
        CompositeStage(
            name="check-collector",
            description="Check the quote collector rolled to the next week",
            func=_stage_check_collector,
            soft_fail=True,
        ),
    ]


def _injury_report_line(repo: Path, *, season_int: int, week: int) -> str:
    """Say whether the cached injury file already holds the week's reports."""
    from gridiron_edge.ingest.nflverse.weekly_availability import load_injuries

    try:
        injuries = load_injuries(seasons=[season_int], repo=repo).get(season_int)
    except FileNotFoundError:
        injuries = None
    if injury_report_present(injuries, week=week):
        return f"week {week} injury reports were available to the starter projection"
    return (
        f"no week {week} injury report yet: starters come from the published or "
        "carried-forward quarterback until the starter watch sees one"
    )


def _stage_report_forecast(ctx: dict[str, Any]) -> StageResult:
    """Summarize what the forecast assumed and how fresh its market is."""
    from gridiron_edge.cli.verify_week import _market_age_suffix, load_weekly_readiness
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        load_current_weekly_product,
    )

    repo: Path = get_settings().repo_root
    product = load_current_weekly_product(season=ctx["season"], week=ctx["week"], repo=repo)
    lines = starter_lines(product)
    lines.append(_injury_report_line(repo, season_int=ctx["season_int"], week=ctx["week"]))
    readiness = load_weekly_readiness(season=ctx["season"], week=ctx["week"], repo=repo)
    lines.append(f"prediction ready: {'yes' if readiness.prediction_ready else 'no'}")
    lines.append(
        f"market ready: {'yes' if readiness.market_ready else 'no'}"
        + _market_age_suffix(readiness.market_fetched_at)
    )
    ctx.setdefault("cycle_report", []).extend(lines)
    return StageResult(success=True, detail="forecast summary ready")


def build_forecast_stages() -> list[CompositeStage]:
    """Assemble the forecast: the ``weekly-predict`` stages, then a summary of what they assumed."""
    from gridiron_edge.cli.weekly_predict import _build_stages as weekly_predict_stages

    stages = list(weekly_predict_stages())
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


# ---------------------------------------------------------------------------
# Real effects behind the cycle's ports
# ---------------------------------------------------------------------------


def _stage_run(summary: CompositeSummary, context: dict[str, Any], *, done: str) -> StageRun:
    """Reduce a composite summary to what a receipt records."""
    stages = dict.fromkeys(summary.succeeded, "ok")
    stages.update(dict.fromkeys(summary.soft_failed, "soft-failed"))
    stages.update(dict.fromkeys(summary.failed, "failed"))
    failed = summary.failed[0] if summary.failed else None
    detail = (
        done if failed is None else f"{failed}: {summary.failure_details.get(failed, 'failed')}"
    )
    return StageRun(
        success=summary.overall_success,
        detail=detail,
        failed_stage=failed,
        stages=stages,
        warnings=list(summary.warnings),
        report=list(context.get("cycle_report") or []),
    )


def _run_closeout(season: str, week: int, allow_incomplete: bool) -> StageRun:
    stages = build_close_stages()
    active = resolve_active_stages(all_stages=[stage.name for stage in stages], skip=[], only=[])
    context: dict[str, Any] = {
        "week": week,
        "season": season,
        "season_int": _season_int(season),
        "allow_incomplete": allow_incomplete,
    }
    console.header("week-cycle closeout", subtitle=f"week {week} · {season}")
    summary = run_composite(
        name="week-cycle closeout", stages=stages, active=active, context=context
    )
    render_composite_summary(summary)
    _print_report(context)
    return _stage_run(summary, context, done=f"week {week} closed")


def _run_forecast(season: str, week: int) -> StageRun:
    from gridiron_edge.cli.weekly_predict import weekly_predict_context

    stages = build_forecast_stages()
    active = resolve_active_stages(all_stages=[stage.name for stage in stages], skip=[], only=[])
    context = weekly_predict_context(season=season, week=week, bankroll=None)
    console.header("week-cycle forecast", subtitle=f"week {week} · {season}")
    summary = run_composite(
        name="week-cycle forecast", stages=stages, active=active, context=context
    )
    render_composite_summary(summary)
    _print_report(context)
    return _stage_run(summary, context, done=f"week {week} forecast selected")


def _run_edges(season: str, week: int) -> StageRun:
    from gridiron_edge.cli.weekly_predict import _stage_generate_edges, weekly_predict_context

    result = _stage_generate_edges(weekly_predict_context(season=season, week=week, bankroll=None))
    return StageRun(
        success=result.success,
        detail=result.detail,
        stages={"generate-edges": "ok" if result.success else "blocked"},
        warnings=result.warnings,
    )


def _refresh_availability(season_int: int) -> AvailabilityRefresh:
    from gridiron_edge.ingest.nflverse.weekly_availability import (
        fetch_depth_charts,
        fetch_injuries,
    )

    repo = get_settings().repo_root
    return AvailabilityRefresh(
        depth_charts_ok=bool(fetch_depth_charts(seasons=[season_int], force=True, repo=repo)),
        injuries_ok=bool(fetch_injuries(seasons=[season_int], force=True, repo=repo)),
    )


def _digests(season_int: int) -> dict[str, str | None]:
    from gridiron_edge.ingest.nflverse.weekly_availability import load_depth_charts, load_injuries

    repo = get_settings().repo_root
    found: dict[str, str | None] = {}
    for name, loader in (("depth_charts", load_depth_charts), ("injuries", load_injuries)):
        try:
            found[name] = frame_digest(loader(seasons=[season_int], repo=repo).get(season_int))
        except FileNotFoundError:
            found[name] = None
    return found


def _edge_inputs(season: str, week: int) -> dict[str, Any]:
    """Name the inputs an edge report derives from: the product and the newest quote."""
    from gridiron_edge.ingest.odds.store import load_current_odds
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        get_current_weekly_product_selection,
    )

    repo = get_settings().repo_root
    quotes = load_current_odds(repo=repo)
    fetched: str | None = None
    if quotes is not None:
        scoped = quotes.loc[(quotes["season"].astype(str) == season) & (quotes["week"] == week)]
        if not scoped.empty:
            fetched = scoped["fetched_at"].max().isoformat()
    selection = get_current_weekly_product_selection(season=season, week=week, repo=repo)
    return {"product_id": selection.product_id, "market_fetched_at": fetched}


def _product_id(season: str, week: int) -> str | None:
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        get_current_weekly_product_selection,
    )

    try:
        return get_current_weekly_product_selection(
            season=season, week=week, repo=get_settings().repo_root
        ).product_id
    except FileNotFoundError:
        return None


def build_ports() -> CyclePorts:
    """Return the real effects of the weekly cycle."""
    from gridiron_edge.datasets.loaders import load_schedule_upcoming_rich
    from gridiron_edge.models.game_prediction.weekly_product_store import (
        load_current_weekly_product,
    )

    repo = get_settings().repo_root
    return CyclePorts(
        clock=lambda: datetime.now(UTC),
        load_schedule=lambda: load_schedule_upcoming_rich(repo),
        product_id=_product_id,
        load_product=lambda season, week: load_current_weekly_product(
            season=season, week=week, repo=repo
        ),
        dirty_files=lambda: dirty_tracked_files(repo),
        run_closeout=_run_closeout,
        run_forecast=_run_forecast,
        run_edges=_run_edges,
        refresh_availability=_refresh_availability,
        digests=_digests,
        observe=lambda season, week, product, schedule, now: observe_starters(
            repo, season=season, week=week, product=product, schedule=schedule, now=now
        ),
        edge_inputs=_edge_inputs,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _target(season: str | None, week: int | None) -> tuple[str, int] | None:
    if (season is None) != (week is None):
        raise typer.BadParameter("Pass --season and --week together, or neither.")
    if season is None or week is None:
        return None
    _season_int(season)
    return season, week


@contextlib.contextmanager
def _single_run(repo: Path) -> Iterator[bool]:
    """Hold a non-blocking lock so a timer tick and a manual run never overlap."""
    directory = repo / "data" / "output" / "week_cycle"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@week_cycle_app.callback(invoke_without_command=True)
def week_cycle_cmd(
    ctx: typer.Context,
    *,
    season: str | None = typer.Option(None, help="Override the derived season, e.g. '2026-2027'."),
    week: int | None = typer.Option(None, help="Override the derived target week."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what is due without doing or recording anything."
    ),
    allow_incomplete: bool = typer.Option(
        False,
        "--allow-incomplete/--require-complete",
        help="Close the finished week even if some of its games have no result yet.",
    ),
) -> None:
    r"""Do whatever the weekly cycle has due (safe to run any time, and hourly).

    From the schedule it derives the target week (the earliest week with a game still
    ahead) and does the first step that is due: close the finished week (from 06:00 local,
    once every result is in), forecast the target week (when it has no current forecast),
    or compare the starting quarterbacks with those the forecast assumed and refresh the
    forecast if one changed before the week's first kickoff. Every attempt writes a receipt
    under data/output/week_cycle. Nothing due means a quiet no-op.

    \b
    Examples:
      gridiron week-cycle
      gridiron week-cycle --dry-run
      gridiron week-cycle status
    """
    if ctx.invoked_subcommand is not None:
        return
    target = _target(season, week)
    repo = get_settings().repo_root
    if dry_run:
        outcome = run_cycle(repo=repo, ports=build_ports(), target=target, dry_run=True)
    else:
        with _single_run(repo) as acquired:
            if not acquired:
                typer.echo("Another week-cycle run is in progress; nothing done.")
                return
            outcome = run_cycle(
                repo=repo,
                ports=build_ports(),
                target=target,
                allow_incomplete=allow_incomplete,
            )
    for line in outcome.lines:
        typer.echo(line)
    if outcome.failed:
        raise typer.Exit(code=1)


@week_cycle_app.command("status")
def status_cmd(
    *,
    season: str | None = typer.Option(None, help="Override the derived season."),
    week: int | None = typer.Option(None, help="Override the derived target week."),
) -> None:
    """Report what is due and the latest receipts, reading local state only."""
    repo = get_settings().repo_root
    outcome = run_cycle(repo=repo, ports=build_ports(), target=_target(season, week), dry_run=True)
    for line in outcome.lines:
        typer.echo(line)
    decision = outcome.decision
    if decision is None or decision.season is None:
        return
    weeks = sorted({w for w in (decision.week, decision.target_week) if w is not None})
    receipts = [
        receipt
        for number in weeks
        for receipt in read_receipts(repo, season=decision.season, week=number)
    ]
    typer.echo("")
    if not receipts:
        typer.echo("No receipts yet.")
        return
    typer.echo("Latest receipts:")
    for receipt in sorted(receipts, key=lambda item: item.decided)[-8:]:
        typer.echo(
            f"  {receipt.decided_at}  week {receipt.week}  {receipt.attempted}: "
            f"{receipt.outcome} - {receipt.detail}"
        )


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

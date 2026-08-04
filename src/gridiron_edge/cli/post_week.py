# src/gridiron_edge/cli/post_week.py
"""Composite command for completed-week live forecast closeout."""

from __future__ import annotations

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


def _run_pipeline(ctx: dict[str, Any], *, active: set[str]) -> None:
    """Run one explicit subset of the data pipeline."""
    from gridiron_edge.cli.main import _run_pipeline_stages

    season_int = int(ctx["season_int"])
    _run_pipeline_stages(
        active=active,
        all_years=False,
        resolved_season=season_int,
        upcoming_target=season_int,
        season=season_int,
        season_year=str(ctx["season"]),
        owm_api_key=None,
        fit_elo_all_years=False,
    )


def _stage_refresh_results(ctx: dict[str, Any]) -> StageResult:
    """Refresh completed games and canonical cleaned outcomes."""
    _run_pipeline(ctx, active={"fetch-games", "clean-games"})
    return StageResult(success=True, detail="completed results refreshed")


def _stage_refresh_next_week_state(ctx: dict[str, Any]) -> StageResult:
    """Refresh upcoming schedule, EPA, Elo, and model inputs."""
    _run_pipeline(
        ctx,
        active={
            "fetch-upcoming",
            "clean-upcoming",
            "build-epa",
            "build-elo",
            "build-features",
        },
    )
    return StageResult(success=True, detail="next-week state refreshed")


def _format_metrics(ctx: dict[str, Any]) -> str:
    """Render closeout metrics cached in the command context."""
    closeout = ctx["live_forecast_closeout"]
    parts = [
        f"{closeout.completed_outcome_count}/{closeout.scheduled_game_count} outcomes",
        (
            f"Win {closeout.matched_win_event_count}/{closeout.selected_win_count} "
            f"events, {closeout.win.evaluated_count} evaluated"
        ),
        (
            f"Total {closeout.matched_total_event_count}/{closeout.selected_total_count} "
            f"events, {closeout.total.evaluated_count} evaluated"
        ),
    ]
    if closeout.win.brier is not None:
        parts.append(
            f"Win Brier {closeout.win.brier:.4f}, "
            f"log loss {closeout.win.log_loss:.4f}, "
            f"accuracy {closeout.win.accuracy:.1%}"
        )
    if closeout.total.mae is not None:
        parts.append(
            f"Total MAE {closeout.total.mae:.2f}, "
            f"RMSE {closeout.total.rmse:.2f}, "
            f"bias {closeout.total.bias:+.2f}"
        )
    return " | ".join(parts)


def _coverage_warnings(ctx: dict[str, Any]) -> list[str]:
    """Return explicit missing-coverage messages for one closeout."""
    closeout = ctx["live_forecast_closeout"]
    fields = (
        ("missing Win components", closeout.missing_win_component_game_ids),
        ("missing Total components", closeout.missing_total_component_game_ids),
        ("missing live Win events", closeout.missing_win_event_game_ids),
        ("missing live Total events", closeout.missing_total_event_game_ids),
        ("missing outcomes", closeout.missing_outcome_game_ids),
    )
    return [f"{label}: {', '.join(values)}" for label, values in fields if values]


def _stage_close_live_forecasts(ctx: dict[str, Any]) -> StageResult:
    """Evaluate the exact live forecasts referenced by the selected product."""
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.evaluation.live_forecast_closeout import (
        load_live_forecast_closeout,
    )

    closeout = load_live_forecast_closeout(
        repo=get_settings().repo_root,
        season=str(ctx["season"]),
        week=int(ctx["week"]),
    )
    ctx["live_forecast_closeout"] = closeout
    warnings = _coverage_warnings(ctx)
    detail = _format_metrics(ctx)
    if not closeout.complete:
        detail = f"incomplete closeout: {detail}"
    return StageResult(
        success=closeout.complete,
        detail=detail,
        rows=closeout.scheduled_game_count,
        warnings=warnings,
    )


def _build_stages() -> list[CompositeStage]:
    """Define independent completed-week operational stages."""
    return [
        CompositeStage(
            name="refresh-results",
            description="Refresh completed results",
            func=_stage_refresh_results,
        ),
        CompositeStage(
            name="refresh-next-week-state",
            description="Refresh next-week state",
            func=_stage_refresh_next_week_state,
        ),
        CompositeStage(
            name="close-live-forecasts",
            description="Close selected live forecasts",
            func=_stage_close_live_forecasts,
        ),
    ]


_ALL_STAGES = [stage.name for stage in _build_stages()]
_STAGES_STR = ", ".join(_ALL_STAGES)
_SKIP_HELP = f"Stage(s) to skip. Repeatable. Valid: {_STAGES_STR}."
_ONLY_HELP = f"Run only these stage(s). Repeatable. Valid: {_STAGES_STR}."


def post_week_cmd(
    *,
    week: int = typer.Option(..., help="NFL week number that just completed."),
    season: str = typer.Option(..., help="NFL season label, e.g. '2025-2026'."),
    skip: list[str] = typer.Option([], "--skip", help=_SKIP_HELP),  # noqa: B008
    only: list[str] = typer.Option([], "--only", help=_ONLY_HELP),  # noqa: B008
) -> None:
    r"""Refresh results and evaluate the exact live forecasts issued before kickoff.

    
    Examples:
      gridiron post-week --week 1 --season 2025-2026
      gridiron post-week --only refresh-results --week 1 --season 2025-2026
      gridiron post-week --only refresh-next-week-state --week 1 --season 2025-2026
      gridiron post-week --only close-live-forecasts --week 1 --season 2025-2026
    """
    stages = _build_stages()
    active = resolve_active_stages(all_stages=_ALL_STAGES, skip=skip, only=only)
    try:
        season_int = int(season.split("-")[0])
    except (ValueError, IndexError) as exc:
        raise typer.BadParameter(
            f"Could not parse season {season!r}. Expected format 'YYYY-YYYY+1'."
        ) from exc

    context: dict[str, Any] = {
        "week": week,
        "season": season,
        "season_int": season_int,
    }
    console.header("post-week", subtitle=f"week {week} · {season} · selected live forecasts")
    summary = run_composite(
        name="post-week",
        stages=stages,
        active=active,
        context=context,
    )
    render_composite_summary(summary)
    if not summary.overall_success:
        raise typer.Exit(code=1)

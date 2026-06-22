# src/gridiron_edge/cli/post_week.py

"""Composite command: post-week.

After games complete, archive the week's predictions to the prediction
log and surface a quick performance summary. Mirrors HANDOFF.md
operational checklist steps 7-10.

Usage::

    gridiron post-week --week 1 --season 2025-2026
    gridiron post-week --week 1 --season 2025-2026 --model-type random_forest
    gridiron post-week --only backfill-predictions --week 1 --season 2025-2026
"""

from __future__ import annotations

from typing import Any, Literal

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
from gridiron_edge.evaluation.backfill import backfill_model
from gridiron_edge.evaluation.metrics import build_evaluation_df, summarise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Brier-score deviation from the current season average that triggers
# a weekly drift warning in the post-week evaluation snapshot.
#
# This is intentionally a constant rather than a CLI option. There is
# currently only one use case and one consumer; if additional drift
# reporting surfaces emerge in the future this can be promoted into a
# shared evaluation constant or configuration setting.
_BRIER_DRIFT_WARNING_THRESHOLD: float = 0.02

# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def _stage_refresh_data(ctx: dict[str, Any]) -> StageResult:
    """Refresh underlying data via the weekly run-data-pipeline path.

    Skips fetch-weather and fetch-odds (external services). Includes
    build-elo, which incrementally updates Elo state with the
    completed week's results.
    """
    from gridiron_edge.cli.main import ALL_STAGES, _run_pipeline_stages

    active = set(ALL_STAGES) - {"fetch-weather", "fetch-odds"}

    _run_pipeline_stages(
        active=active,
        all_years=False,
        resolved_season=ctx.get("season_int", 0) or 0,
        upcoming_target=ctx.get("season_int", 0) or 0,
        season=ctx.get("season_int"),
        season_year=ctx.get("season"),
        owm_api_key=None,
        fit_elo_all_years=False,
    )
    return StageResult(success=True, detail="data + Elo refreshed")


def _stage_backfill_predictions(ctx: dict[str, Any]) -> StageResult:
    """Archive predictions for the completed week.

    Uses backfill_model with start/end season set to the requested
    season so only the most-recent season is touched.
    """
    model_name: str = ctx["model_name"]
    model_type: str = ctx["model_type"]
    season: str = ctx["season"]

    n = backfill_model(
        model_name=model_name,
        model_type=model_type,
        mode=None,  # auto-resolve per model
        overwrite=True,
        start_season=season,
        end_season=season,
    )

    if n == 0:
        return StageResult(
            success=True,
            detail="no new predictions written (already archived)",
        )

    return StageResult(
        success=True,
        detail=f"{n:,} predictions archived for {season}",
        rows=n,
    )


def _stage_evaluate_summary(ctx: dict[str, Any]) -> StageResult:
    """Print a quick evaluation snapshot for the completed season.

    Loads the archive, joins to outcomes, and computes per-week
    Brier/accuracy for the season. Surfaces drift signals if the
    current week's Brier deviates from the season mean.
    """
    model_name: str = ctx["model_name"]
    model_type: str = ctx["model_type"]
    season: str = ctx["season"]
    week: int = ctx["week"]

    df_eval = build_evaluation_df(
        model_name=model_name,
        model_type=model_type,
        season=season,
    )

    if df_eval.empty:
        return StageResult(
            success=True,
            detail="no evaluated games yet for this season",
        )

    df_summary = summarise(df_eval, group_by="week")

    warnings: list[str] = []
    detail_parts: list[str] = []

    # Look for the requested week's row.
    week_row = df_summary.loc[df_summary["week"] == week, :]
    if not week_row.empty:
        row = week_row.iloc[0]
        season_mean_brier: float = df_summary["brier"].mean()
        week_brier = float(row["brier"])
        delta: float = week_brier - season_mean_brier

        detail_parts.append(f"week {week}: Brier {week_brier:.4f}, accuracy {row['accuracy']:.1%}")

        if abs(delta) > _BRIER_DRIFT_WARNING_THRESHOLD:
            direction: Literal["better", "worse"] = "worse" if delta > 0 else "better"
            warnings.append(
                f"Week {week} Brier ({week_brier:.4f}) is "
                f"{abs(delta):.4f} {direction} than season mean "
                f"({season_mean_brier:.4f})"
            )
    else:
        detail_parts.append(f"week {week}: no data")

    # Always include a brief season-wide line.
    season_brier: float = df_summary["brier"].mean()
    season_accuracy: float = df_summary["accuracy"].mean()
    detail_parts.append(f"season-to-date: Brier {season_brier:.4f}, accuracy {season_accuracy:.1%}")

    return StageResult(
        success=True,
        detail=" | ".join(detail_parts),
        rows=len(df_eval),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Stage list
# ---------------------------------------------------------------------------


def _build_stages() -> list:
    """Define the stages for post-week.

    refresh-data must run before backfill-predictions so that completed
    games are in the modeling file. evaluate-summary uses the archive
    that backfill-predictions wrote.
    """
    return [
        CompositeStage(
            name="refresh-data",
            description="Refresh data (run-data-pipeline)",
            func=_stage_refresh_data,
        ),
        CompositeStage(
            name="backfill-predictions",
            description="Archive predictions for completed week",
            func=_stage_backfill_predictions,
            depends_on=("refresh-data",),
        ),
        CompositeStage(
            name="evaluate-summary",
            description="Compute and surface week + season metrics",
            func=_stage_evaluate_summary,
            depends_on=("backfill-predictions",),
        ),
    ]


_ALL_STAGES: list[str] = [s.name for s in _build_stages()]
_STAGES_STR: str = ", ".join(_ALL_STAGES)
_SKIP_HELP: str = f"Stage(s) to skip. Repeatable. Valid: {_STAGES_STR}."
_ONLY_HELP: str = f"Run only these stage(s). Repeatable. Valid: {_STAGES_STR}."


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def post_week_cmd(
    *,
    week: int = typer.Option(..., help="NFL week number that just completed."),
    season: str = typer.Option(..., help="NFL season label, e.g. '2025-2026'."),
    model_name: str = typer.Option("win_prob", help="Model purpose. Default: win_prob."),
    model_type: str = typer.Option(
        "random_forest",
        help=("Model algorithm. One of: random_forest, xgboost, logistic, elo."),
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
    r"""Archive the completed week and surface metric snapshot.

    Composes three stages: data refresh (includes Elo update),
    prediction archive for the completed week, and a brief evaluation
    summary that compares the week's Brier against the season mean.

    \b
    Examples:
      gridiron post-week --week 1 --season 2025-2026
      gridiron post-week --week 1 --season 2025-2026 --model-type xgboost
      gridiron post-week --only backfill-predictions --week 1 --season 2025-2026
    """
    stages = _build_stages()
    active = resolve_active_stages(
        all_stages=_ALL_STAGES,
        skip=skip,
        only=only,
    )

    try:
        season_int = int(season.split("-")[0])
    except (ValueError, IndexError) as exc:
        raise typer.BadParameter(
            f"Could not parse season '{season}'. Expected format: 'YYYY-YYYY+1' (e.g. '2025-2026')."
        ) from exc

    context: dict[str, Any] = {
        "week": week,
        "season": season,
        "season_int": season_int,
        "model_name": model_name,
        "model_type": model_type,
    }

    console.header(
        "post-week",
        subtitle=(f"week {week} · {season} · model={model_name}/{model_type}"),
    )

    summary = run_composite(
        name="post-week",
        stages=stages,
        active=active,
        context=context,
    )

    render_composite_summary(summary)

    if not summary.overall_success:
        raise typer.Exit(code=1)

# src/gridiron_edge/cli/weekly_predict.py

"""Composite command: weekly-predict.

Generates predictions for the upcoming week by composing data refresh,
prediction generation, output rendering, and an edge report from an existing
source-neutral market snapshot. Mirrors the operational checklist in
HANDOFF.md steps 1-4.

Usage::

    gridiron weekly-predict --week 1 --season 2026-2027
    gridiron weekly-predict --only predict-week --week 1 --season 2026-2027
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas import DataFrame

# pyrefly: ignore [missing-import]
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
from gridiron_edge.evaluation.forecast_contracts import (
    ForecastRole,
    new_forecast_run_id,
)
from gridiron_edge.evaluation.forecast_events import build_forecast_events
from gridiron_edge.evaluation.forecast_store import write_forecast_events
from gridiron_edge.models.game_prediction.prediction_policy import PredictionPolicy

if TYPE_CHECKING:
    from gridiron_edge.market.recommendations import EdgeResult

# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def _stage_ensure_data_fresh(ctx: dict[str, Any]) -> StageResult:
    """Refresh underlying data via the weekly run-data-pipeline path.

    Skips fetch-weather because weather refresh remains out-of-band for
    this workflow. Market data is consumed from an existing source-neutral
    snapshot and is not fetched by weekly prediction orchestration.
    """
    from gridiron_edge.cli.main import ALL_STAGES, _run_pipeline_stages

    active = set(ALL_STAGES) - {"fetch-weather"}

    _run_pipeline_stages(
        active=active,
        all_years=False,
        resolved_season=ctx.get("resolved_season_int", 0) or 0,
        upcoming_target=ctx.get("upcoming_target", 0) or 0,
        season=ctx.get("season_int"),
        season_year=ctx.get("season"),
        owm_api_key=None,
        fit_elo_all_years=False,
    )
    return StageResult(success=True, detail="data refreshed")


def _canonicalize_live_elo_predictions(
    predictions: DataFrame,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Map display-oriented Elo output to canonical prediction rows."""
    return DataFrame(
        {
            "season": [season] * len(predictions),
            "week": [week] * len(predictions),
            "game_id": predictions["GAME_ID"],
            "game_date": predictions["GAME_DATE"],
            "away_team": predictions["AWAY_TEAM"],
            "home_team": predictions["HOME_TEAM"],
            "away_elo": predictions["AWAY_TEAM_ELO"],
            "home_elo": predictions["HOME_TEAM_ELO"],
            "away_win_prob": predictions["AWAY_WIN_PROB"],
            "home_win_prob": predictions["HOME_WIN_PROB"],
        }
    ).reset_index(drop=True)


def _stage_predict_week(ctx: dict[str, Any]) -> StageResult:
    """Generate and store live Elo forecasts for the upcoming week.

    The original display-oriented prediction frame remains available to
    downstream rendering. A separate canonical frame is composed into
    immutable live forecast events.
    """
    from gridiron_edge.viz.predictions import build_predictions_df

    year: str = ctx["season"]
    week: int = ctx["week"]
    repo: Path = get_settings().repo_root

    predictions: DataFrame = build_predictions_df(
        year=year,
        week=week,
        repo=repo,
    )
    if predictions.empty:
        return StageResult(
            success=False,
            detail="no predictions produced (check schedule + Elo state)",
        )

    canonical = _canonicalize_live_elo_predictions(
        predictions,
        season=year,
        week=week,
    )

    run_id = new_forecast_run_id()
    generated_at = datetime.now(UTC)

    events = build_forecast_events(
        canonical,
        model_name="win_prob",
        model_type="elo",
        run_id=run_id,
        role=ForecastRole.LIVE,
        generated_at=generated_at,
    )

    event_path = write_forecast_events(
        events,
        repo=repo,
    )

    # Downstream product composition selects this exact immutable run.
    ctx["predictions_df"] = predictions
    ctx["forecast_run_id"] = run_id
    ctx["forecast_generated_at"] = generated_at

    return StageResult(
        success=True,
        detail=f"{len(events)} live forecast events written",
        rows=len(events),
        artifacts=[event_path],
    )


def _stage_compose_weekly_product(ctx: dict[str, Any]) -> StageResult:
    """Compose, persist, and explicitly select the live Elo weekly product."""
    from gridiron_edge.datasets.loaders import load_schedule_upcoming_rich
    from gridiron_edge.datasets.writers import (
        select_current_weekly_product,
        write_weekly_product,
    )
    from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
    from gridiron_edge.evaluation.forecast_selection import (
        ForecastCandidateIdentity,
        resolve_forecast_candidates,
        select_forecast_run,
    )
    from gridiron_edge.evaluation.forecast_store import load_forecast_events
    from gridiron_edge.models.game_prediction.availability import (
        inspect_prediction_availability,
    )
    from gridiron_edge.models.game_prediction.prediction_policy import (
        resolve_prediction_policy,
    )
    from gridiron_edge.models.game_prediction.weekly_game_product import (
        build_weekly_game_product,
    )
    from gridiron_edge.models.game_prediction.weekly_spread_product import (
        load_and_attach_derived_spreads,
    )
    from gridiron_edge.models.game_prediction.weekly_total_product import (
        load_and_attach_selected_totals,
    )
    from gridiron_edge.models.game_prediction.weekly_win_product import (
        build_weekly_win_product,
    )

    season: str = ctx["season"]
    week: int = ctx["week"]
    repo: Path = get_settings().repo_root
    run_id_value = ctx.get("forecast_run_id")
    generated_at_value = ctx.get("forecast_generated_at")
    if not isinstance(run_id_value, str) or not run_id_value.strip():
        return StageResult(success=False, detail="forecast run identity is unavailable")
    if not isinstance(generated_at_value, datetime):
        return StageResult(success=False, detail="forecast generation time is unavailable")

    schedule = load_schedule_upcoming_rich(repo)
    scoped_schedule = schedule.loc[
        (schedule["season"].astype(str) == season) & (schedule["week"] == week),
        :,
    ].copy()
    if scoped_schedule.empty:
        return StageResult(success=False, detail="rich weekly schedule is empty")

    events = load_forecast_events(
        season=season,
        week=week,
        run_id=run_id_value,
        repo=repo,
    )
    selected_run = select_forecast_run(events, run_id=run_id_value)
    if not selected_run.found:
        return StageResult(success=False, detail="forecast run is not persisted")

    availability = inspect_prediction_availability(
        schedule,
        season=season,
        week=week,
        repo=repo,
    )
    policy: PredictionPolicy = resolve_prediction_policy(
        availability,
        win_champion=None,
        total_champion=None,
        win_override="elo",
    )
    win_resolutions = resolve_forecast_candidates(
        selected_run.events,
        [
            ForecastCandidateIdentity(
                game_id=str(game_id),
                model_name="win_prob",
                model_type="elo",
            )
            for game_id in scoped_schedule["game_id"]
        ],
    )
    win_product = build_weekly_win_product(
        scoped_schedule,
        selected_run.events,
        win_resolutions,
        policy=policy,
        season=season,
        week=week,
    )
    spread_product = load_and_attach_derived_spreads(
        win_product,
        repo=repo,
    )
    total_product = load_and_attach_selected_totals(
        spread_product,
        selected_run.events,
        (),
        policy=policy,
        season=season,
        week=week,
        repo=repo,
    )
    product = build_weekly_game_product(total_product)

    product_id = f"weekly_{season.replace('-', '_')}_wk{week:02d}_{run_id_value}"
    identity = WeeklyProductIdentity(
        product_id=product_id,
        run_id=run_id_value,
        season=season,
        week=week,
        generated_at=generated_at_value,
    )
    artifact = write_weekly_product(repo, product, identity=identity)
    select_current_weekly_product(
        repo,
        product_id,
        season=season,
        week=week,
        selected_at=datetime.now(UTC),
    )
    ctx["weekly_product_id"] = product_id
    ctx["weekly_product_path"] = artifact
    return StageResult(
        success=True,
        detail=f"{len(product)} weekly product rows selected",
        rows=len(product),
        artifacts=[artifact],
    )


def _stage_render_outputs(ctx: dict[str, Any]) -> StageResult:
    """Render predictions to PNG + HTML."""
    from gridiron_edge.viz.predictions import (
        build_predictions_df,
        render_predictions_html,
        render_predictions_image,
    )

    year: str = ctx["season"]
    week: int = ctx["week"]
    repo: Path = get_settings().repo_root

    # Prefer the context-cached DataFrame from predict-week, else rebuild.
    df = ctx.get("predictions_df")
    if df is None:
        df = build_predictions_df(year=year, week=week, repo=repo)
        if df.empty:
            return StageResult(
                success=False,
                detail="no predictions to render",
            )

    artifacts: list[Path] = []

    png_path = render_predictions_image(df, year=year, week=week, repo=repo)
    artifacts.append(png_path)

    html_path = render_predictions_html(df, year=year, week=week, repo=repo)
    artifacts.append(html_path)

    return StageResult(
        success=True,
        detail=f"rendered {len(artifacts)} outputs",
        artifacts=artifacts,
    )


def _edge_stage_detail(result: EdgeResult, *, min_ev: float) -> str:
    """Return a deterministic composite-stage detail from diagnostics."""
    from gridiron_edge.market.edge_diagnostics import EdgeResultState

    if result.diagnostics.blockers:
        blockers = ", ".join(blocker.value for blocker in result.diagnostics.blockers)
        return f"edge calculation blocked: {blockers}"
    if result.diagnostics.state is EdgeResultState.NO_CALCULABLE_EDGES:
        return "no calculable edges from available inputs"
    if result.diagnostics.state is EdgeResultState.NO_POSITIVE_EDGES:
        return "calculated markets contain no positive-EV edges"
    if result.rows.empty:
        return f"positive edges exist below min_ev={min_ev:.1%}"
    return f"{len(result.rows)} edges (top EV {result.rows['ev'].max():.1%})"


def _edge_stage_detail(result: EdgeResult, *, min_ev: float) -> str:
    """Return a deterministic composite-stage detail from diagnostics."""
    from gridiron_edge.market.edge_diagnostics import EdgeResultState

    if result.diagnostics.blockers:
        blockers = ", ".join(blocker.value for blocker in result.diagnostics.blockers)
        return f"edge calculation blocked: {blockers}"
    if result.diagnostics.state is EdgeResultState.NO_CALCULABLE_EDGES:
        return "no calculable edges from available inputs"
    if result.diagnostics.state is EdgeResultState.NO_POSITIVE_EDGES:
        return "calculated markets contain no positive-EV edges"
    if result.rows.empty:
        return f"positive edges exist below min_ev={min_ev:.1%}"
    return f"{len(result.rows)} edges (top EV {result.rows['ev'].max():.1%})"


def _stage_generate_edges(ctx: dict[str, Any]) -> StageResult:
    """Generate persisted weekly edges through the unified domain service."""
    from gridiron_edge.market.weekly_edge_service import (
        build_weekly_edge_result,
    )

    season: str = ctx["season"]
    week: int = ctx["week"]
    repo: Path = get_settings().repo_root
    bankroll = ctx.get("bankroll")
    if bankroll is not None and not isinstance(bankroll, int | float):
        return StageResult(success=False, detail="bankroll must be numeric")

    kelly_multiplier = 0.25
    min_ev = 0.0
    result = build_weekly_edge_result(
        season=season,
        week=week,
        bankroll=None if bankroll is None else float(bankroll),
        kelly_multiplier=kelly_multiplier,
        min_ev=min_ev,
        repo=repo,
    )
    detail = _edge_stage_detail(result, min_ev=min_ev)
    if result.diagnostics.blockers:
        return StageResult(success=False, detail=detail)
    if result.rows.empty:
        return StageResult(success=True, detail=detail)

    ranked = result.rows
    ctx["top_edges_preview"] = ranked.head(5).reset_index(drop=True).copy()
    out_dir = repo / "data" / "output" / "edges"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"edges_{season}_wk{week:02d}.csv"
    ranked.to_csv(out_path, index=False)
    return StageResult(
        success=True,
        detail=detail,
        rows=len(ranked),
        artifacts=[out_path],
    )


# ---------------------------------------------------------------------------
# Stage list
# ---------------------------------------------------------------------------


def _render_edge_preview(ranked: DataFrame) -> None:
    """Render a small console preview of the highest-EV edges."""
    if ranked.empty:
        return

    typer.echo("")
    typer.echo("Top edges")
    typer.echo("-" * 60)

    for _, row in ranked.iterrows():
        matchup: str = f"{row['away_team']} @ {row['home_team']}"
        market: str = f"{row['market_type']}:{row['side']}"

        stake = row.get("kelly_stake")
        # pyrefly: ignore [bad-argument-type]
        stake_text = "—" if pd.isna(stake) else f"${float(stake):.2f}"
        typer.echo(f"{matchup:20s}  {market:16s}  EV {row['ev']:+.1%}  Stake {stake_text}")


def _build_stages() -> list[CompositeStage]:
    """Define the stages for weekly-predict.

    Order matters: each stage's depends_on points to stages earlier in
    the list. Edge generation depends on predictions and consumes an existing
    source-neutral market snapshot without fetching external data.
    """
    return [
        CompositeStage(
            name="ensure-data-fresh",
            description="Ensure data is fresh",
            func=_stage_ensure_data_fresh,
        ),
        CompositeStage(
            name="predict-week",
            description="Generate predictions for upcoming week",
            func=_stage_predict_week,
            depends_on=("ensure-data-fresh",),
        ),
        CompositeStage(
            name="compose-weekly-product",
            description="Compose and select the weekly game product",
            func=_stage_compose_weekly_product,
            depends_on=("predict-week",),
        ),
        CompositeStage(
            name="render-outputs",
            description="Render predictions PNG + HTML",
            func=_stage_render_outputs,
            depends_on=("predict-week",),
        ),
        CompositeStage(
            name="generate-edges",
            description="Generate edge report against current odds",
            func=_stage_generate_edges,
            depends_on=("compose-weekly-product",),
            soft_fail=True,
        ),
    ]


_ALL_STAGES: list[str] = [s.name for s in _build_stages()]
_STAGES_STR: str = ", ".join(_ALL_STAGES)
_SKIP_HELP: str = f"Stage(s) to skip. Repeatable. Valid: {_STAGES_STR}."
_ONLY_HELP: str = f"Run only these stage(s). Repeatable. Valid: {_STAGES_STR}."


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def weekly_predict_cmd(
    *,
    week: int = typer.Option(..., help="NFL week number to predict."),
    season: str = typer.Option(..., help="NFL season label, e.g. '2026-2027'."),
    bankroll: float | None = typer.Option(
        None,
        min=0.0,
        help="Optional bankroll for Kelly stake sizing in the edge report.",
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
    r"""Generate predictions and edge report for the upcoming week.

    Composes data refresh, live Elo prediction, weekly-product selection,
    and an edge report from the existing current market snapshot. Edge
    generation soft-fails when current market data is unavailable.

    \b
    Examples:
      gridiron weekly-predict --week 1 --season 2026-2027
      gridiron weekly-predict --only predict-week --week 1 --season 2026-2027
    """
    stages = _build_stages()
    active = resolve_active_stages(
        all_stages=_ALL_STAGES,
        skip=skip,
        only=only,
    )

    # Derive integer season for run-data-pipeline downstream.
    try:
        season_int = int(season.split("-")[0])
    except (ValueError, IndexError) as exc:
        raise typer.BadParameter(
            f"Could not parse season '{season}'. Expected format: 'YYYY-YYYY+1' (e.g. '2026-2027')."
        ) from exc

    context: dict[str, Any] = {
        "week": week,
        "season": season,
        "season_int": season_int,
        "resolved_season_int": season_int,
        "upcoming_target": season_int,
        "model_type": "elo",
        "bankroll": bankroll,
    }

    console.header(
        "weekly-predict",
        subtitle=f"week {week} · {season} · model=elo",
    )

    summary: CompositeSummary = run_composite(
        name="weekly-predict",
        stages=stages,
        active=active,
        context=context,
    )
    render_composite_summary(summary)

    top_edges = context.get("top_edges_preview")
    if top_edges is not None and not top_edges.empty:
        _render_edge_preview(top_edges)

    if not summary.overall_success:
        raise typer.Exit(code=1)

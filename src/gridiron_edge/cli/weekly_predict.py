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
from typing import Any

from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._composites import (
    CompositeStage,
    CompositeSummary,
    StageResult,
    render_composite_summary,
    resolve_active_stages,
    resolve_win_prob_model_type,
    run_composite,
)
from gridiron_edge.core.console import console
from gridiron_edge.core.settings import get_settings
from gridiron_edge.evaluation.archive import load_prediction_log
from gridiron_edge.evaluation.forecast_contracts import (
    ForecastRole,
    new_forecast_run_id,
)
from gridiron_edge.evaluation.forecast_events import build_forecast_events
from gridiron_edge.evaluation.forecast_store import write_forecast_events
from gridiron_edge.ingest.odds.store import load_current_odds

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

    # Rendering consumes the original display-oriented frame.
    ctx["predictions_df"] = predictions

    return StageResult(
        success=True,
        detail=f"{len(events)} live forecast events written",
        rows=len(events),
        artifacts=[event_path],
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


def _stage_generate_edges(ctx: dict[str, Any]) -> StageResult:
    """Generate an edge report from the existing current market snapshot.

    Missing market data remains a source-neutral soft failure. This stage does
    not invoke an external market adapter.
    """
    from gridiron_edge.market.recommendations import (
        build_edge_report,
        rank_edges,
    )
    from gridiron_edge.models.game_prediction.post_process import (
        get_margin_std,
        get_total_std,
    )

    year: str = ctx["season"]
    week: int = ctx["week"]
    model_type: str = ctx.get("model_type", "random_forest")
    repo: Path = get_settings().repo_root

    predictions = load_prediction_log(
        season=year,
        week=week,
        model_name="win_prob",
        model_type=model_type,
    )
    if predictions.empty:
        return StageResult(
            success=False,
            detail=f"no predictions for win_prob/{model_type} week {week}",
        )

    odds = load_current_odds()
    if odds is None or odds.empty:
        return StageResult(
            success=False,
            detail="no current market snapshot available",
        )

    margin_std = get_margin_std("win_prob", model_type)
    total_std = get_total_std("total", model_type, default=13.0)

    edge_report = build_edge_report(
        predictions,
        odds,
        margin_std=margin_std,
        total_std=total_std,
        bankroll=ctx.get("bankroll", 1000.0),
        kelly_multiplier=0.25,
    )

    if edge_report.empty:
        return StageResult(
            success=True,
            detail="no edges (predictions did not match any odds)",
        )

    ranked = rank_edges(edge_report, min_ev=0.0)
    ctx["top_edges_preview"] = ranked.head(5).reset_index(drop=True).copy()

    out_dir = repo / "data" / "output" / "edges"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"edges_{year}_wk{week:02d}.csv"
    ranked.to_csv(out_path, index=False)

    return StageResult(
        success=True,
        detail=f"{len(ranked)} edges (top EV {ranked['ev'].max():.1%})",
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

        typer.echo(
            f"{matchup:20s}  {market:16s}  EV {row['ev']:+.1%}  Stake ${row['kelly_stake']:.2f}"
        )


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
            name="render-outputs",
            description="Render predictions PNG + HTML",
            func=_stage_render_outputs,
            depends_on=("predict-week",),
        ),
        CompositeStage(
            name="generate-edges",
            description="Generate edge report against current odds",
            func=_stage_generate_edges,
            depends_on=("predict-week",),
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
    model_type: str = typer.Option(
        "auto",
        help=(
            "Win-probability model algorithm to use for edges. "
            "One of: random_forest, xgboost, logistic, elo. "
            "Defaults to 'auto', which resolves to the current champion "
            "from the manifest at data/output/champions/champions.json."
        ),
    ),
    bankroll: float = typer.Option(
        1000.0,
        help="Bankroll for Kelly stake sizing in the edge report.",
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

    Composes four stages: data refresh, prediction, output rendering,
    and an edge report from the existing current market snapshot. Edge
    generation soft-fails when current market data is unavailable.

    \b
    Examples:
      gridiron weekly-predict --week 1 --season 2026-2027
      gridiron weekly-predict --week 1 --season 2026-2027 --model-type xgboost
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

    # Resolve --model-type auto sentinel against the champion manifest.
    # Runs after Typer/user-input validation so genuine input errors
    # surface first.
    resolved_model_type = resolve_win_prob_model_type(model_type)

    context: dict[str, Any] = {
        "week": week,
        "season": season,
        "season_int": season_int,
        "resolved_season_int": season_int,
        "upcoming_target": season_int,
        "model_type": resolved_model_type,
        "bankroll": bankroll,
    }

    console.header(
        "weekly-predict",
        subtitle=f"week {week} · {season} · model={resolved_model_type}",
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

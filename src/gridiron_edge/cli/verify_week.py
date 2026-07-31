# src/gridiron_edge/cli/verify_week.py

"""Read-only weekly operational readiness verification."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import re
from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.loaders import (
    load_schedule_upcoming_rich,
)
from gridiron_edge.evaluation.champion_resolver import (
    ChampionNotFoundError,
    resolve_current_champion,
)
from gridiron_edge.evaluation.forecast_contracts import (
    SelectedForecast,
)
from gridiron_edge.evaluation.forecast_selection import (
    ForecastCandidateIdentity,
    ForecastCandidateStatus,
    resolve_forecast_candidates,
    select_forecast_events,
    select_forecast_run,
)
from gridiron_edge.evaluation.forecast_store import (
    empty_forecast_events,
    load_forecast_events,
)
from gridiron_edge.evaluation.weekly_readiness import (
    WeeklyReadiness,
    WeeklyReadinessBlocker,
    evaluate_weekly_readiness,
)
from gridiron_edge.ingest.odds.store import load_current_odds

if TYPE_CHECKING:
    from gridiron_edge.market.recommendations import EdgeResult

_SEASON_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<start>\d{4})-(?P<end>\d{4})$")

_EMPTY_RICH_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
)

_EMPTY_MARKET_COLUMNS: Final[tuple[str, ...]] = (
    "fetched_at",
    "sportsbook",
    "season",
    "week",
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "market",
    "side",
    "odds",
    "line",
)


def validate_season_label(season: str) -> str:
    """Validate and return an NFL season label."""
    match = _SEASON_PATTERN.fullmatch(season)
    if match is None:
        raise ValueError(
            f"Could not parse season {season!r}. Expected format YYYY-YYYY, for example 2026-2027."
        )

    start = int(match.group("start"))
    end = int(match.group("end"))

    if end != start + 1:
        raise ValueError(
            f"Invalid season {season!r}. "
            "The ending year must be one greater than the starting year."
        )

    return season


def _format_timestamp(
    value: datetime | None,
) -> str:
    """Format an artifact timestamp for diagnostic output."""
    if value is None:
        return "unavailable"

    return value.isoformat()


def _empty_rich_schedule() -> DataFrame:
    """Return an empty rich schedule identity frame."""
    return DataFrame(
        {
            "season": pd.Series(dtype="string"),
            "week": pd.Series(dtype="int64"),
            "game_id": pd.Series(dtype="string"),
        }
    )


def _empty_markets() -> DataFrame:
    """Return an empty canonical long-format market frame."""
    return DataFrame(columns=_EMPTY_MARKET_COLUMNS)


def _load_schedule(
    repo: Path,
) -> DataFrame:
    """Load rich schedule data without creating or refreshing it."""
    try:
        return load_schedule_upcoming_rich(repo)
    except FileNotFoundError:
        return _empty_rich_schedule()


def _schedule_for_readiness(
    rich_schedule: DataFrame,
) -> DataFrame:
    """Project rich schedule identity onto the readiness schema."""
    required: set[str] = {
        "season",
        "week",
        "game_id",
    }
    missing: list[str] = sorted(required - set(rich_schedule.columns))
    if missing:
        raise ValueError(
            "Rich upcoming schedule is missing required columns: " + ", ".join(missing)
        )

    return DataFrame(
        {
            "YEAR": rich_schedule["season"].astype("string"),
            # pyrefly: ignore [missing-attribute]
            "WEEK_NUM": pd.to_numeric(
                rich_schedule["week"],
                errors="raise",
            ).astype(int),
            "GAME_ID": rich_schedule["game_id"].astype("string"),
        }
    )


def _load_markets(
    repo: Path,
) -> DataFrame:
    """Load the current market snapshot without fetching it."""
    markets = load_current_odds(repo=repo)
    if markets is None:
        return _empty_markets()

    return markets.copy()


def _scheduled_game_ids(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
) -> list:
    """Return scheduled rich-artifact game IDs deterministically."""
    required: set[str] = {
        "season",
        "week",
        "game_id",
    }
    if not required.issubset(schedule.columns):
        return []

    scoped = schedule.loc[
        (schedule["season"].astype(str) == season) & (schedule["week"] == week),
        "game_id",
    ]

    return sorted({str(game_id) for game_id in scoped.dropna() if str(game_id).strip()})


def _select_run_predictions(
    events: DataFrame,
    *,
    run_id: str,
) -> tuple[
    DataFrame,
    tuple[WeeklyReadinessBlocker, ...],
]:
    """Select one explicit run and retain its win-probability rows."""
    result = select_forecast_run(
        events,
        run_id=run_id,
    )

    if not result.found:
        return (
            empty_forecast_events(),
            (WeeklyReadinessBlocker.MISSING_FORECAST_SELECTION,),
        )

    predictions = result.events.loc[
        result.events["model_name"] == "win_prob",
        :,
    ].copy()

    if predictions.empty:
        return (
            predictions,
            (WeeklyReadinessBlocker.MISSING_FORECAST_SELECTION,),
        )

    if predictions["game_id"].duplicated().any():
        return (
            predictions.iloc[0:0].copy(),
            (WeeklyReadinessBlocker.AMBIGUOUS_FORECAST_SELECTION,),
        )

    return predictions.reset_index(drop=True), ()


def _select_current_predictions(
    events: DataFrame,
    *,
    scheduled_game_ids: list[str],
    model_type: str,
) -> tuple[
    DataFrame,
    tuple[WeeklyReadinessBlocker, ...],
]:
    """Select uniquely eligible champion events for scheduled games."""
    identities = [
        ForecastCandidateIdentity(
            game_id=game_id,
            model_name="win_prob",
            model_type=model_type,
        )
        for game_id in scheduled_game_ids
    ]

    resolutions = resolve_forecast_candidates(
        events,
        identities,
    )

    references: list[SelectedForecast] = []
    missing = False
    ambiguous = False

    for resolution in resolutions:
        if resolution.status is ForecastCandidateStatus.SELECTED:
            if resolution.selected is not None:
                references.append(resolution.selected)
        elif resolution.status is ForecastCandidateStatus.MISSING:
            missing = True
        elif resolution.status is ForecastCandidateStatus.AMBIGUOUS:
            ambiguous = True

    blockers: list[WeeklyReadinessBlocker] = []

    if missing:
        blockers.append(WeeklyReadinessBlocker.MISSING_FORECAST_SELECTION)

    if ambiguous:
        blockers.append(WeeklyReadinessBlocker.AMBIGUOUS_FORECAST_SELECTION)

    selected = select_forecast_events(
        events,
        references,
    )

    return selected.events, tuple(blockers)


def _load_edge_result(
    *,
    season: str,
    week: int,
    repo: Path,
) -> EdgeResult:
    """Load weekly edge rows and diagnostics without modifying artifacts."""
    from gridiron_edge.market.weekly_edge_service import (
        build_weekly_edge_result,
    )

    return build_weekly_edge_result(
        season=season,
        week=week,
        bankroll=None,
        kelly_multiplier=0.25,
        min_ev=0.0,
        repo=repo,
    )


def _with_selection_blockers(
    readiness: WeeklyReadiness,
    selection_blockers: tuple[
        WeeklyReadinessBlocker,
        ...,
    ],
) -> WeeklyReadiness:
    """Append selection blockers without duplicating existing blockers."""
    combined = tuple(
        dict.fromkeys(
            (
                *readiness.blockers,
                *selection_blockers,
            )
        )
    )

    return replace(
        readiness,
        blockers=combined,
    )


def _edge_readiness_blockers(
    result: EdgeResult,
) -> tuple[WeeklyReadinessBlocker, ...]:
    """Translate unified edge blockers into readiness blockers."""
    from gridiron_edge.market.edge_diagnostics import EdgeDiagnosticBlocker

    mapping = {
        EdgeDiagnosticBlocker.NO_PREDICTIONS: (WeeklyReadinessBlocker.MISSING_WEEKLY_PRODUCT),
        EdgeDiagnosticBlocker.NO_MARKET_DATA: (WeeklyReadinessBlocker.MISSING_MARKET_DATA),
        EdgeDiagnosticBlocker.MARKET_WRONG_SCOPE: (WeeklyReadinessBlocker.MARKET_SCOPE_MISMATCH),
        EdgeDiagnosticBlocker.MARKET_STALE: (WeeklyReadinessBlocker.STALE_MARKET_DATA),
        EdgeDiagnosticBlocker.ZERO_MATCHED_GAMES: (
            WeeklyReadinessBlocker.ZERO_PREDICTION_MARKET_MATCHES
        ),
        EdgeDiagnosticBlocker.INCOMPLETE_MARKETS: (WeeklyReadinessBlocker.INCOMPLETE_MARKETS),
    }

    return tuple(dict.fromkeys(mapping[blocker] for blocker in result.diagnostics.blockers))


def _render_weekly_readiness(
    readiness: WeeklyReadiness,
) -> None:
    """Render every weekly readiness count and blocker."""
    typer.echo("Coverage")
    typer.echo(f"  Scheduled games                 {readiness.scheduled_game_count}")
    typer.echo(f"  Selected win predictions        {readiness.selected_win_prediction_count}")
    typer.echo(f"  Spread values                   {readiness.spread_value_count}")
    typer.echo(f"  Total predictions               {readiness.total_prediction_count}")
    typer.echo(f"  Projected scores                {readiness.projected_score_count}")
    typer.echo(f"  Complete provenance             {readiness.complete_provenance_count}")

    typer.echo("")
    typer.echo("Markets")
    typer.echo(f"  Games with market data          {readiness.market_game_count}")
    typer.echo(f"  Prediction-market matches       {readiness.prediction_market_match_count}")
    typer.echo(f"  Eligible markets                {readiness.eligible_market_count}")
    typer.echo(f"  Positive edges                  {readiness.positive_edge_count}")

    typer.echo("")
    typer.echo("Artifacts")
    typer.echo(
        f"  Prediction generated at         {_format_timestamp(readiness.prediction_generated_at)}"
    )
    typer.echo(
        f"  Market fetched at               {_format_timestamp(readiness.market_fetched_at)}"
    )
    typer.echo(f"  Market source                   {readiness.market_source or 'unavailable'}")

    typer.echo("")

    if readiness.ready:
        typer.echo("Ready")
        typer.echo("  No blockers")
        return

    typer.echo("Blocked")
    for blocker in readiness.blockers:
        typer.echo(f"  {blocker.value}")


def load_weekly_readiness(
    *,
    season: str,
    week: int,
    run_id: str | None = None,
    repo: Path | None = None,
) -> WeeklyReadiness:
    """Load existing weekly artifacts and derive readiness.

    This function performs reads and in-memory analysis only. It does not
    fetch, generate, enrich, persist, or render artifacts.
    """
    validate_season_label(season)

    if week < 1 or week > 22:
        raise ValueError("week must be between 1 and 22.")

    if run_id is not None and not run_id.strip():
        raise ValueError("run_id must not be empty when provided.")

    resolved_repo = repo or get_settings().repo_root

    rich_schedule = _load_schedule(resolved_repo)
    readiness_schedule = _schedule_for_readiness(rich_schedule)
    markets = _load_markets(resolved_repo)

    events = load_forecast_events(
        season=season,
        week=week,
        repo=resolved_repo,
    )

    game_ids = _scheduled_game_ids(
        rich_schedule,
        season=season,
        week=week,
    )

    if run_id is not None:
        predictions, selection_blockers = _select_run_predictions(
            events,
            run_id=run_id,
        )
    else:
        try:
            _, model_type = resolve_current_champion(
                "win_prob",
                repo=resolved_repo,
            )
        except ChampionNotFoundError:
            predictions = empty_forecast_events()
            selection_blockers = (WeeklyReadinessBlocker.MISSING_FORECAST_SELECTION,)
        else:
            predictions, selection_blockers = _select_current_predictions(
                events,
                scheduled_game_ids=game_ids,
                model_type=model_type,
            )

    edge_result = _load_edge_result(
        season=season,
        week=week,
        repo=resolved_repo,
    )

    readiness = evaluate_weekly_readiness(
        season=season,
        week=week,
        schedule=readiness_schedule,
        predictions=predictions,
        markets=markets,
        edges=edge_result.rows,
    )

    readiness = replace(
        readiness,
        market_game_count=(edge_result.diagnostics.market_game_count),
        prediction_market_match_count=(edge_result.diagnostics.matched_game_count),
        eligible_market_count=(edge_result.diagnostics.eligible_market_count),
        positive_edge_count=(edge_result.diagnostics.positive_edge_count),
    )

    return _with_selection_blockers(
        readiness,
        (
            *selection_blockers,
            *_edge_readiness_blockers(edge_result),
        ),
    )


def _render_weekly_readiness(
    readiness: WeeklyReadiness,
) -> None:
    """Render every weekly readiness count and blocker."""
    typer.echo("Coverage")
    typer.echo(f"  Scheduled games                 {readiness.scheduled_game_count}")
    typer.echo(f"  Selected win predictions        {readiness.selected_win_prediction_count}")
    typer.echo(f"  Spread values                   {readiness.spread_value_count}")
    typer.echo(f"  Total predictions               {readiness.total_prediction_count}")
    typer.echo(f"  Projected scores                {readiness.projected_score_count}")
    typer.echo(f"  Complete provenance             {readiness.complete_provenance_count}")

    typer.echo("")
    typer.echo("Markets")
    typer.echo(f"  Games with market data          {readiness.market_game_count}")
    typer.echo(f"  Prediction-market matches       {readiness.prediction_market_match_count}")
    typer.echo(f"  Eligible markets                {readiness.eligible_market_count}")
    typer.echo(f"  Positive edges                  {readiness.positive_edge_count}")

    typer.echo("")
    typer.echo("Artifacts")
    typer.echo(
        f"  Prediction generated at         {_format_timestamp(readiness.prediction_generated_at)}"
    )
    typer.echo(
        f"  Market fetched at               {_format_timestamp(readiness.market_fetched_at)}"
    )
    typer.echo(f"  Market source                   {readiness.market_source or 'unavailable'}")

    typer.echo("")

    if readiness.ready:
        typer.echo("Ready")
        typer.echo("  No blockers")
        return

    typer.echo("Blocked")
    for blocker in readiness.blockers:
        typer.echo(f"  {blocker.value}")


def verify_week_cmd(
    *,
    season: str = typer.Option(
        ...,
        "--season",
        help=("NFL season label in YYYY-YYYY format, for example 2026-2027."),
    ),
    week: int = typer.Option(
        ...,
        "--week",
        min=1,
        max=22,
        help="NFL week number from 1 through 22.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help=(
            "Exact forecast run to verify. When omitted, "
            "resolve the current win-probability champion's "
            "eligible events without using recency."
        ),
    ),
) -> None:
    """Verify weekly operational readiness without modifying data."""
    try:
        validated_season = validate_season_label(season)
        readiness = load_weekly_readiness(
            season=validated_season,
            week=week,
            run_id=run_id,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"verify-week  {validated_season} week {week}")
    typer.echo("")

    _render_weekly_readiness(readiness)

    if not readiness.ready:
        raise typer.Exit(code=1)

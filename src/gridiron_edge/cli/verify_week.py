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
    load_current_weekly_product,
    load_schedule_upcoming_rich,
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


def _empty_predictions() -> DataFrame:
    """Return the minimum prediction contract for readiness evaluation."""
    return DataFrame(
        columns=[
            "season",
            "week",
            "game_id",
            "home_win_prob",
            "model_spread",
            "model_total",
            "projected_home_score",
            "projected_away_score",
            "event_id",
            "run_id",
            "model_name",
            "model_type",
            "generated_at",
        ]
    )


def _selected_product_for_readiness(product: DataFrame) -> DataFrame:
    """Adapt one explicitly selected weekly product to readiness inputs."""
    required = {
        "season",
        "week",
        "game_id",
        "home_win_prob",
        "model_spread",
        "model_total",
        "projected_home_score",
        "projected_away_score",
        "win_event_id",
        "product_run_id",
        "win_model_name",
        "win_model_type",
        "product_generated_at",
    }
    missing = sorted(required - set(product.columns))
    if missing:
        raise ValueError(
            "Selected weekly product is missing required columns: " + ", ".join(missing)
        )
    return DataFrame(
        {
            "season": product["season"],
            "week": product["week"],
            "game_id": product["game_id"],
            "home_win_prob": product["home_win_prob"],
            "model_spread": product["model_spread"],
            "model_total": product["model_total"],
            "projected_home_score": product["projected_home_score"],
            "projected_away_score": product["projected_away_score"],
            "event_id": product["win_event_id"],
            "run_id": product["product_run_id"],
            "model_name": product["win_model_name"],
            "model_type": product["win_model_type"],
            "generated_at": product["product_generated_at"],
        }
    )


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
    repo: Path | None = None,
) -> WeeklyReadiness:
    """Load the selected weekly product and derive read-only readiness."""
    validate_season_label(season)
    if week < 1 or week > 22:
        raise ValueError("week must be between 1 and 22.")

    resolved_repo = repo or get_settings().repo_root
    rich_schedule = _load_schedule(resolved_repo)
    readiness_schedule = _schedule_for_readiness(rich_schedule)
    markets = _load_markets(resolved_repo)

    selection_blockers: tuple[WeeklyReadinessBlocker, ...] = ()
    try:
        product = load_current_weekly_product(
            resolved_repo,
            season=season,
            week=week,
        )
    except FileNotFoundError:
        predictions = _empty_predictions()
        selection_blockers = (WeeklyReadinessBlocker.MISSING_WEEKLY_PRODUCT,)
    else:
        predictions = _selected_product_for_readiness(product)

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
        market_game_count=edge_result.diagnostics.market_game_count,
        prediction_market_match_count=edge_result.diagnostics.matched_game_count,
        eligible_market_count=edge_result.diagnostics.eligible_market_count,
        positive_edge_count=edge_result.diagnostics.positive_edge_count,
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
) -> None:
    """Verify weekly operational readiness without modifying data."""
    try:
        validated_season = validate_season_label(season)
        readiness = load_weekly_readiness(
            season=validated_season,
            week=week,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"verify-week  {validated_season} week {week}")
    typer.echo("")

    _render_weekly_readiness(readiness)

    if not readiness.ready:
        raise typer.Exit(code=1)

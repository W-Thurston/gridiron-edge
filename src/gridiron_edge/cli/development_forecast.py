# src/gridiron_edge/cli/development_forecast.py

"""Command: generate-development-forecast.

Generates and persists one retrospective, development-role forecast run
(immutable forecast events plus per-family prediction-input evidence) for an
already-retained season and week, entirely from data already on disk.

Unlike ``weekly-predict``, this command never fetches an external source: the
schedule is adapted from the retained, history-preserving complete game
record rather than the fetch-derived upcoming-schedule snapshot, which only
ever holds the current season's not-yet-played games. This lets the command
regenerate a week that has already been played, honestly labeled as
development validation rather than pre-kickoff live issuance (see
``ForecastRole.DEVELOPMENT``).

This command does not compose or select a weekly product and does not verify
readiness; it only generates and publishes the underlying forecast events and
evidence. Composing and selecting the resulting run into a weekly product is
a separate, explicit step.

Usage::

    gridiron generate-development-forecast --season 2026-2027 --week 2
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame

# pyrefly: ignore [missing-import]
import typer

from gridiron_edge.cli._prediction_publication import (
    _publish_and_reload_evidence,
    _publish_binary_snapshots,
    _require_execution_provenance,
)
from gridiron_edge.core.console import console, step
from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.loaders import load_games
from gridiron_edge.evaluation.forecast_contracts import new_forecast_run_id
from gridiron_edge.evaluation.forecast_store import write_forecast_events
from gridiron_edge.evaluation.prediction_input_sources import (
    capture_prediction_source_artifacts,
    recapture_and_require_same_prediction_sources,
    resolve_clean_source_revision,
)

_RETAINED_HISTORY_RICH_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "game_day_of_week",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "neutral_site",
)


def build_retained_history_schedule(games: DataFrame) -> DataFrame:
    """Adapt retained complete game history into the rich-schedule shape.

    ``_scope_schedule`` and ``_build_elo_schedule`` require exactly the
    columns below, regardless of source. The fetch-derived rich upcoming
    schedule supplies them for not-yet-played games; this adapter supplies
    the same shape for any already-retained game, including ones already
    played, by renaming the equivalent columns already present in the
    complete, history-preserving ``games`` dataset.
    """
    required = {
        "GAME_ID",
        "WEEK_NUM",
        "YEAR",
        "GAME_DAY_OF_WEEK",
        "GAME_DATE",
        "GAMETIME",
        "AWAY_TEAM",
        "HOME_TEAM",
        "IS_NEUTRAL_SITE",
    }
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError("Retained game history is missing required columns: " + ", ".join(missing))

    schedule = DataFrame(
        {
            "season": games["YEAR"].astype(str),
            "week": pd.to_numeric(games["WEEK_NUM"], errors="raise").astype(int),
            "game_id": games["GAME_ID"].astype(str),
            "game_day_of_week": games["GAME_DAY_OF_WEEK"],
            "game_date": games["GAME_DATE"],
            "game_time": games["GAMETIME"],
            "away_team": games["AWAY_TEAM"],
            "home_team": games["HOME_TEAM"],
            "neutral_site": pd.to_numeric(games["IS_NEUTRAL_SITE"], errors="raise"),
        }
    )
    return schedule.loc[:, list(_RETAINED_HISTORY_RICH_COLUMNS)].reset_index(drop=True)


def _generate_development_forecast(
    *,
    season: str,
    week: int,
    repo: Path,
) -> tuple[str, int, int, tuple[Path, ...]]:
    """Execute and publish one evidence-authenticated development forecast run.

    Returns the run ID, event count, evidence-artifact count, and every
    published artifact path.
    """
    from gridiron_edge.models.game_prediction.weekly_execution import (
        execute_development_weekly_prediction_policy,
    )

    source_revision = resolve_clean_source_revision(repo)
    source_artifacts = capture_prediction_source_artifacts(repo)
    schedule = build_retained_history_schedule(load_games(repo))
    run_id = new_forecast_run_id()
    generated_at = datetime.now(UTC)

    execution = execute_development_weekly_prediction_policy(
        schedule,
        season=season,
        week=week,
        repo=repo,
        run_id=run_id,
        generated_at=generated_at,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
    )
    _require_execution_provenance(
        execution.input_evidence,
        source_revision=source_revision,
        source_artifacts=source_artifacts,
    )
    recapture_and_require_same_prediction_sources(
        repo,
        source_artifacts,
    )
    snapshot_paths = _publish_binary_snapshots(
        execution.input_evidence,
        repo=repo,
    )
    evidence_paths = _publish_and_reload_evidence(
        execution.input_evidence,
        forecast_events=execution.events,
        repo=repo,
    )
    write_result = write_forecast_events(execution.events, repo=repo)

    return (
        run_id,
        len(execution.events),
        len(execution.input_evidence),
        (*snapshot_paths, *evidence_paths, write_result.path),
    )


def generate_development_forecast_cmd(
    *,
    week: int = typer.Option(..., help="NFL week number to generate."),
    season: str = typer.Option(..., help="NFL season label, e.g. '2026-2027'."),
) -> None:
    r"""Generate one development-role forecast run from retained history.

    Resolves a clean tracked Git source revision, captures the bounded
    source-artifact inventory, adapts the retained complete game history into
    a rich-schedule frame for the requested season and week, and executes the
    policy-selected development-role Win and Total models through the same
    evidence-aware boundaries the live command uses. Persists immutable
    forecast events and per-family prediction-input evidence. Never fetches
    an external source. Does not compose a weekly product or change the
    current selection.

    \b
    Examples:
      gridiron generate-development-forecast --season 2026-2027 --week 2
    """
    console.header(
        "generate-development-forecast",
        subtitle=f"week {week} · {season} · retained history · development role",
    )

    repo: Path = get_settings().repo_root

    try:
        with step("Generate development forecast") as s:
            run_id, event_count, evidence_count, artifacts = _generate_development_forecast(
                season=season,
                week=week,
                repo=repo,
            )
            s.set_detail(
                f"{event_count} development forecast events written with "
                f"{evidence_count} input evidence artifacts"
            )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise typer.Exit(code=1) from exc

    typer.echo("")
    typer.echo(f"Run ID: {run_id}")
    for artifact in artifacts:
        typer.echo(f"  {artifact}")

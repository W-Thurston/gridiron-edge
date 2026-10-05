# src/gridiron_edge/evaluation/weekly_recap_build.py

"""Build, store and select a finished week's recap from persisted state (``DECISIONS.md`` D86).

Shared by ``gridiron evaluate weekly-recap`` and the closeout's ``build-weekly-recap`` stage, so a
recap built by hand and one built at closeout pass the same refusals and follow the same selection
rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd

from gridiron_edge.datasets import loaders
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.evaluation import forecast_store
from gridiron_edge.evaluation.week_schedule import scheduled_games
from gridiron_edge.evaluation.weekly_recap import (
    BuiltWeeklyRecap,
    WeeklyRecapRefusalError,
    build_weekly_recap,
    unplayed_game_ids,
)
from gridiron_edge.evaluation.weekly_recap_store import (
    RecapWriteOutcome,
    get_current_weekly_recap_selection,
    select_current_weekly_recap,
    write_weekly_recap,
)
from gridiron_edge.ratings.history import season_start


class RecapSelectionOutcome(StrEnum):
    """What storing and selecting a built recap did."""

    NEW = "new"
    REPLAYED = "replayed"
    RESELECTED = "reselected"


@dataclass(frozen=True, slots=True)
class RecapSelectionResult:
    """The recap now selected for its week, the one it replaced, and how it got there."""

    recap_id: str
    previous_recap_id: str | None
    outcome: RecapSelectionOutcome


def build_weekly_recap_from_repo(
    *,
    repo: Path,
    season: str,
    week: int,
    built_at: datetime,
) -> BuiltWeeklyRecap:
    """Build one week's recap from the raw schedule, the selected product and the cleaned games.

    Raises:
        WeeklyRecapRefusalError: No game is scheduled, a game has no result yet, no product is
            selected, or the builder refuses the persisted state (it names the reason).
    """
    raw = pd.read_parquet(dataset_path(repo, "games_raw_nflverse"))
    schedule = scheduled_games(raw, season_int=season_start(season), week=week)
    if schedule.empty:
        raise WeeklyRecapRefusalError("no regular-season games are scheduled.")
    unplayed = unplayed_game_ids(schedule)
    if unplayed:
        raise WeeklyRecapRefusalError(
            f"{len(unplayed)} of {len(schedule)} games have no result yet: {', '.join(unplayed)}"
        )
    try:
        product = loaders.load_current_weekly_product(repo, season=season, week=week)
    except FileNotFoundError as error:
        raise WeeklyRecapRefusalError("no weekly product is selected.") from error
    return build_weekly_recap(
        season=season,
        week=week,
        schedule=schedule,
        product=product,
        forecast_events=forecast_store.load_forecast_events(repo=repo, season=season, week=week),
        games=loaders.load_games(repo),
        built_at=built_at,
    )


def store_and_select_weekly_recap(
    built: BuiltWeeklyRecap,
    *,
    selected_at: datetime,
    repo: Path,
) -> RecapSelectionResult:
    """Write the recap (or accept an exact replay) and select it for its week.

    A recap that is already the week's selection is a replay and the selection is left untouched;
    a stored recap that is not selected is reselected.
    """
    recap = built.recap
    previous = get_current_weekly_recap_selection(season=recap.season, week=recap.week, repo=repo)
    previous_id = previous.recap_id if previous is not None else None
    written = write_weekly_recap(built, repo=repo)
    if previous_id == recap.recap_id:
        return RecapSelectionResult(recap.recap_id, previous_id, RecapSelectionOutcome.REPLAYED)
    select_current_weekly_recap(recap.recap_id, selected_at=selected_at, repo=repo)
    outcome = (
        RecapSelectionOutcome.NEW
        if written is RecapWriteOutcome.WRITTEN
        else RecapSelectionOutcome.RESELECTED
    )
    return RecapSelectionResult(recap.recap_id, previous_id, outcome)

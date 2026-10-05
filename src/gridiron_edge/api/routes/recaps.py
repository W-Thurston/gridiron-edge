# src/gridiron_edge/api/routes/recaps.py

"""Recap endpoints: the selected weekly and season recaps, served read-only (``DECISIONS.md`` D86).

Every route reads only through ``weekly_recap_store``, which verifies each manifest's identity
and game-frame digest on load. Nothing is scored or rebuilt at request time.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path
from pandas import DataFrame

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.schemas.recaps import (
    SeasonRecapResponse,
    WeeklyRecapList,
    WeeklyRecapResponse,
)
from gridiron_edge.api.serializers.recaps import (
    serialize_season_recap,
    serialize_weekly_recap,
    serialize_weekly_recap_list,
)
from gridiron_edge.evaluation.weekly_recap import WeeklyRecap
from gridiron_edge.evaluation.weekly_recap_store import (
    WeeklyRecapSelection,
    get_current_season_recap_selection,
    get_current_weekly_recap_selection,
    list_current_weekly_recap_selections,
    load_season_recap,
    load_weekly_recap,
)

router = APIRouter(prefix="/recaps", tags=["recaps"])

_VERIFICATION_FAILED = "Weekly recap artifacts failed verification."


@router.get("", response_model=WeeklyRecapList)
def list_weekly_recaps(settings: SettingsDep) -> WeeklyRecapList:
    """Return every selected weekly recap, ordered by season and week."""
    try:
        selections = list_current_weekly_recap_selections(repo=settings.repo_root)
        recaps = [(selection, *_load_selected(selection, settings)) for selection in selections]
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=_VERIFICATION_FAILED) from exc
    return serialize_weekly_recap_list(recaps)


# Declared before /{season}/{week} so "summary" is never parsed as a week.
@router.get("/{season}/summary", response_model=SeasonRecapResponse)
def get_season_recap(
    settings: SettingsDep,
    season: str = Path(description="Season, e.g. '2026-2027'.", min_length=1),
) -> SeasonRecapResponse:
    """Return the selected season recap: each week beside the season to date, and calibration."""
    try:
        selection = get_current_season_recap_selection(season=season, repo=settings.repo_root)
        if selection is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No season recap is selected for {season}. Build it with "
                    f"`gridiron evaluate weekly-recap --season {season} --season-only`."
                ),
            )
        recap = load_season_recap(selection.season_recap_id, repo=settings.repo_root)
        if recap.season != selection.season:
            raise ValueError("Selected season recap covers another season.")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=_VERIFICATION_FAILED) from exc
    return serialize_season_recap(selection, recap)


@router.get("/{season}/{week}", response_model=WeeklyRecapResponse)
def get_weekly_recap(
    settings: SettingsDep,
    season: str = Path(description="Season, e.g. '2026-2027'.", min_length=1),
    week: int = Path(ge=1, description="Week number, 1 or later."),
) -> WeeklyRecapResponse:
    """Return the selected recap of one week."""
    try:
        selection = get_current_weekly_recap_selection(
            season=season, week=week, repo=settings.repo_root
        )
        if selection is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No weekly recap is selected for {season} week {week}. Build it with "
                    f"`gridiron evaluate weekly-recap --season {season} --week {week}`."
                ),
            )
        recap, games = _load_selected(selection, settings)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=_VERIFICATION_FAILED) from exc
    return serialize_weekly_recap(selection, recap, games)


def _load_selected(
    selection: WeeklyRecapSelection,
    settings: SettingsDep,
) -> tuple[WeeklyRecap, DataFrame]:
    recap, games = load_weekly_recap(selection.recap_id, repo=settings.repo_root)
    if (recap.season, recap.week) != (selection.season, selection.week):
        raise ValueError("Selected weekly recap covers another season or week.")
    return recap, games

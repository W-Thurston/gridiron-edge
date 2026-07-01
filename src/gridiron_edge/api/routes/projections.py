# src/gridiron_edge/api/routes/projections.py

"""Season and playoff projections endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Query

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_projections_summary_df,
    load_team_name_map,
    resolve_current_season_week,
)
from gridiron_edge.api.schemas.projections import ProjectionsList
from gridiron_edge.api.serializers.projections import serialize_projections

router = APIRouter(prefix="/projections", tags=["projections"])


@router.get("", response_model=ProjectionsList)
def get_projections(
    settings: SettingsDep,
    season: str | None = Query(
        default=None,
        description=(
            "Season to project, e.g. '2025-2026'. Currently ignored — "
            "returns the latest available projections regardless. Reserved "
            "for future multi-season history."
        ),
    ),
) -> ProjectionsList:
    """Return Monte Carlo season and playoff projections for all teams."""
    df, computed_at = load_projections_summary_df(settings)
    long_to_short = load_team_name_map(settings)

    if season is None:
        resolved_season, _ = resolve_current_season_week(settings)
    else:
        resolved_season = season

    return serialize_projections(df, long_to_short, resolved_season, computed_at)

# src/gridiron_edge/api/routes/weeks.py

"""Current-week endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import resolve_current_week
from gridiron_edge.api.schemas.weeks import CurrentWeek
from gridiron_edge.api.serializers.weeks import serialize_current_week

router = APIRouter(prefix="/weeks", tags=["weeks"])


@router.get("/current", response_model=CurrentWeek)
def get_current_week(settings: SettingsDep) -> CurrentWeek:
    """Return the current NFL season and week."""
    season, week, source = resolve_current_week(settings)
    return serialize_current_week(season=season, week=week, source=source)

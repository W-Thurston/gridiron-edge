# src/gridiron_edge/api/routes/defense.py

"""Defense endpoints — per-team allowed aggregates for charts."""

from __future__ import annotations

from fastapi import APIRouter, Query

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import load_defense_allowed
from gridiron_edge.api.schemas.defense import DefenseAllowedResponse
from gridiron_edge.api.serializers.defense import serialize_defense_allowed

router = APIRouter(prefix="/defense", tags=["defense"])


@router.get("/{team}/allowed", response_model=DefenseAllowedResponse)
def get_defense_allowed(
    settings: SettingsDep,
    team: str,
    stat_type: str = Query(
        description="Stat family, e.g. 'rb_rush_yards', 'wr_rec_yards'.",
    ),
) -> DefenseAllowedResponse:
    """Return a team's allowed aggregates for a stat_type, all cohorts.

    Powers the Compare Player-vs-Defense bar chart's team-allowed
    average line + the "matchup, plainly" verdict card, keyed on an
    arbitrary team (independent of any specific prop/game).

    - Unknown stat_type: 200 with position "" + cohorts null.
    - Team/stat with no data: 200 with cohorts null, field_status
      marking the opponent-allowed blocker.
    """
    position, cohorts = load_defense_allowed(
        settings,
        team=team,
        stat_type=stat_type,
    )
    return serialize_defense_allowed(
        team=team,
        position=position,
        stat_type=stat_type,
        cohorts=cohorts,
    )

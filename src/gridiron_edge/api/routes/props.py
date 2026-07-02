# src/gridiron_edge/api/routes/props.py

"""Props endpoints — champion-model prop projections for the current week."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from gridiron_edge.api._prop_id import decode_prop_id
from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_prop,
    load_props_for_week,
    resolve_current_season_week,
)
from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas.props import PropDetail, PropList
from gridiron_edge.api.serializers.props import (
    serialize_prop_detail,
    serialize_props_list,
)
from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError

router = APIRouter(prefix="/props", tags=["props"])


def _resolve_scope(
    settings: SettingsDep,
    season: str | None,
    week: int | None,
) -> tuple[str, int]:
    """Return (season, week), defaulting to current when not provided.

    Lazy: only reads the games table when a default is actually needed.
    """
    if season is not None and week is not None:
        return (season, week)
    resolved_season, resolved_week = resolve_current_season_week(settings)
    return (season or resolved_season, week or resolved_week)


@router.get("", response_model=PropList)
def list_props(
    settings: SettingsDep,
    season: str | None = Query(
        default=None,
        description="Season, e.g. '2026-2027'. Defaults to current.",
    ),
    week: int | None = Query(
        default=None,
        description="Week number. Defaults to current.",
    ),
    stat_type: str | None = Query(
        default=None,
        description="Prop family filter, e.g. 'qb_pass_yards'.",
    ),
    position: str | None = Query(
        default=None,
        description="Position filter, e.g. 'QB'.",
    ),
) -> PropList:
    """Return champion-model prop predictions for (season, week).

    Iterates the registered prop stat families (or just ``stat_type``
    when provided), resolves each family's current champion, filters
    the archive to that champion's rows. Families without a resolved
    champion are silently skipped.

    - Zero families resolve champion: 200 with empty list,
      ``_meta.field_status["items"]`` marked NO_CHAMPION_MANIFEST.
    - Some families resolve, others don't: 200 with list of resolved
      families (silent skip).
    - Resolved but empty archive: 200 with empty list, no field_status.
    """
    resolved_season, resolved_week = _resolve_scope(settings, season, week)

    try:
        rows = load_props_for_week(
            settings,
            season=resolved_season,
            week=resolved_week,
            stat_type=stat_type,
            position=position,
        )
    except ChampionNotFoundError:
        meta = ResponseMeta().with_blocked(
            "items",
            *Unavailable.NO_CHAMPION_MANIFEST,
        )
        return PropList(
            season=resolved_season,
            week=resolved_week,
            stat_type=stat_type,
            position=position,
            items=[],
            total=0,
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    return serialize_props_list(
        rows,
        season=resolved_season,
        week=resolved_week,
        stat_type=stat_type,
        position=position,
    )


@router.get("/{prop_id}", response_model=PropDetail)
def get_prop(
    settings: SettingsDep,
    prop_id: str,
) -> PropDetail:
    """Return champion-model prop prediction and metadata for one prop.

    - Malformed or unknown prop_id: 404.
    - Champion for this stat_type not resolved: 200 with projection
      and line_context null, field_status blocks marked.
    - Prop not in archive: 404.
    """
    game_id, player_id, stat_type = decode_prop_id(prop_id)

    try:
        row = load_prop(
            settings,
            game_id=game_id,
            player_id=player_id,
            stat_type=stat_type,
        )
    except ChampionNotFoundError:
        # Champion for this family not resolved. Return 200 with the
        # requested identity fields populated but projection/line_context
        # null, plus field_status marking why.
        meta = ResponseMeta()
        meta = meta.with_blocked("projection", *Unavailable.NO_CHAMPION_MANIFEST)
        meta = meta.with_blocked("line_context", *Unavailable.NO_CHAMPION_MANIFEST)
        return PropDetail(
            prop_id=prop_id,
            game_id=game_id,
            player_id=player_id,
            player_name="",
            position="",
            team="",
            stat_type=stat_type,
            model_key="",
            response_meta=meta,  # pyrefly: ignore[unexpected-keyword]
        )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prop not found: {prop_id}",
        )

    return serialize_prop_detail(row)

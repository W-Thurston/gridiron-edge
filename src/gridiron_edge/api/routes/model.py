# src/gridiron_edge/api/routes/model.py

"""Model performance endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pandas import DataFrame

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_bets_df,
    load_current_model_performance_report,
    load_evaluation_df,
)
from gridiron_edge.api.schemas.model_performance import (
    HistoricalModelPerformance,
    HistoricalModelPerformanceSeries,
    ModelPerformance,
)
from gridiron_edge.api.serializers.model_performance import (
    serialize_historical_model_performance,
    serialize_historical_model_performance_series,
    serialize_model_performance,
)

router = APIRouter(prefix="/model", tags=["model"])

_VALID_GROUP_BY = ("season", "week", "model_name", "model_type")


@router.get("/performance", response_model=ModelPerformance)
def get_model_performance(
    settings: SettingsDep,
    season: str | None = Query(
        default=None,
        description="Filter to a specific season, e.g. '2025-2026'.",
    ),
    model_name: str | None = Query(
        default=None,
        description="Filter to a specific model purpose, e.g. 'win_prob'.",
    ),
    model_type: str | None = Query(
        default=None,
        description=("Filter to a specific model algorithm, e.g. 'random_forest'."),
    ),
    group_by: str = Query(
        default="season",
        description="Group results by: season, week, model_name, or model_type.",
    ),
) -> ModelPerformance:
    """Return model prediction quality and betting performance metrics."""
    from gridiron_edge.betting.performance import summary as perf_summary
    from gridiron_edge.evaluation.metrics import summarise

    if group_by not in _VALID_GROUP_BY:
        # FastAPI's Pydantic validation doesn't catch this on a plain str;
        # raise HTTPException with a clear message.
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail=(f"Invalid group_by '{group_by}'. Must be one of: {list(_VALID_GROUP_BY)}."),
        )

    # Model-quality side: build_latest_run_evaluation_df → summarise
    df_eval = load_evaluation_df(
        settings,
        model_name=model_name,
        model_type=model_type,
        season=season,
    )
    summary_df: DataFrame = (
        summarise(df_eval, group_by=group_by) if not df_eval.empty else DataFrame()
    )

    # Betting-side: filter bets to those with model context matching the
    # requested model_name/model_type, then compute summary on the slice.
    bets = load_bets_df(settings)
    if not bets.empty:
        if model_name is not None:
            bets = bets.loc[bets["model_name"] == model_name]
        if model_type is not None:
            bets = bets.loc[bets["model_type"] == model_type]

    # pyrefly: ignore [bad-argument-type]
    model_bet_summary = perf_summary(bets) if not bets.empty else {}

    filters = {
        "season": season,
        "model_name": model_name,
        "model_type": model_type,
        "group_by": group_by,
    }

    return serialize_model_performance(
        df_eval,
        summary_df,
        model_bet_summary,
        filters,
    )


@router.get(
    "/historical-performance",
    response_model=HistoricalModelPerformance,
)
def get_historical_model_performance(
    settings: SettingsDep,
) -> HistoricalModelPerformance:
    """Return the explicitly selected historical walk-forward summary."""
    try:
        current = load_current_model_performance_report(settings)
    except FileNotFoundError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail="Historical model-performance artifacts failed verification.",
        ) from exc
    return serialize_historical_model_performance(current)


@router.get(
    "/historical-performance/series",
    response_model=HistoricalModelPerformanceSeries,
)
def get_historical_model_performance_series(
    settings: SettingsDep,
) -> HistoricalModelPerformanceSeries:
    """Return verified persisted historical chart series."""
    try:
        current = load_current_model_performance_report(settings)
    except FileNotFoundError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail="Historical model-performance artifacts failed verification.",
        ) from exc
    return serialize_historical_model_performance_series(current)

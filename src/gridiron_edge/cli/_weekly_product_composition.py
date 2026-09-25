# src/gridiron_edge/cli/_weekly_product_composition.py

"""Shared, schedule-agnostic weekly-product composition and selection.

Factored out of ``weekly_predict.py``'s ``compose-weekly-product`` stage so a
non-live caller (development regeneration) can reuse the identical validated
composition path against its own schedule and forecast run, rather than the
fetch-derived upcoming schedule the live pipeline always uses. Composition
itself has no live-only assumption: it is driven entirely by the explicit
``schedule``, ``events``, and ``policy`` passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.evaluation.forecast_selection import (
    ForecastCandidateIdentity,
    resolve_forecast_candidates,
)
from gridiron_edge.models.game_prediction.prediction_policy import PredictionPolicy
from gridiron_edge.models.game_prediction.weekly_game_product import (
    build_weekly_game_product,
)
from gridiron_edge.models.game_prediction.weekly_spread_product import (
    load_and_attach_derived_spreads,
)
from gridiron_edge.models.game_prediction.weekly_total_product import (
    load_and_attach_selected_totals,
)
from gridiron_edge.models.game_prediction.weekly_win_product import (
    build_weekly_win_product,
)


@dataclass(frozen=True)
class ComposedWeeklyProduct:
    """The persisted, explicitly-selected result of one composition."""

    product_id: str
    artifact: Path
    row_count: int


def compose_and_select_weekly_product(
    *,
    schedule: DataFrame,
    events: DataFrame,
    policy: PredictionPolicy,
    run_id: str,
    generated_at: datetime,
    season: str,
    week: int,
    repo: Path | None = None,
) -> ComposedWeeklyProduct:
    """Compose, persist, and explicitly select one weekly product.

    ``schedule`` must already carry the rich-schedule shape ``_scope_schedule``
    requires (season/week/game_id/game_day_of_week/game_date/game_time/
    away_team/home_team/neutral_site), live from the fetch-derived upcoming
    schedule or development from retained history. ``events`` must be one
    run's exact selected forecast events, and ``policy`` the exact decision
    that produced them.
    """
    resolved_repo = repo or get_settings().repo_root

    scoped_schedule = schedule.loc[
        (schedule["season"].astype(str) == season) & (schedule["week"] == week),
        :,
    ].copy()
    if scoped_schedule.empty:
        raise ValueError("Rich schedule has no rows for the requested season and week.")

    win_resolutions: tuple = ()
    if policy.win.model_type is not None:
        win_resolutions = resolve_forecast_candidates(
            events,
            [
                ForecastCandidateIdentity(
                    game_id=str(game_id),
                    model_name="win_prob",
                    model_type=policy.win.model_type,
                )
                for game_id in scoped_schedule["game_id"]
            ],
        )

    total_resolutions: tuple = ()
    if policy.total.model_type is not None:
        total_resolutions = resolve_forecast_candidates(
            events,
            [
                ForecastCandidateIdentity(
                    game_id=str(game_id),
                    model_name="total",
                    model_type=policy.total.model_type,
                )
                for game_id in scoped_schedule["game_id"]
            ],
        )

    win_product = build_weekly_win_product(
        scoped_schedule,
        events,
        win_resolutions,
        policy=policy,
        season=season,
        week=week,
    )
    spread_product = load_and_attach_derived_spreads(
        win_product,
        repo=resolved_repo,
    )
    total_product = load_and_attach_selected_totals(
        spread_product,
        events,
        total_resolutions,
        policy=policy,
        season=season,
        week=week,
        repo=resolved_repo,
    )
    product = build_weekly_game_product(total_product)

    product_id = f"weekly_{season.replace('-', '_')}_wk{week:02d}_{run_id}"
    identity = WeeklyProductIdentity(
        product_id=product_id,
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
    )
    artifact = write_weekly_product(resolved_repo, product, identity=identity)
    select_current_weekly_product(
        resolved_repo,
        product_id,
        season=season,
        week=week,
        selected_at=datetime.now(UTC),
    )
    return ComposedWeeklyProduct(
        product_id=product_id,
        artifact=artifact,
        row_count=len(product),
    )

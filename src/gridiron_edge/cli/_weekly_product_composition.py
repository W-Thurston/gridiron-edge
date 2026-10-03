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
from hashlib import sha256
from pathlib import Path

from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.loaders import load_qb_coach_sidecar
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.evaluation.forecast_selection import (
    ForecastCandidateIdentity,
    resolve_forecast_candidates,
)
from gridiron_edge.evaluation.prediction_input_evidence import PredictionSourceState
from gridiron_edge.evaluation.prediction_input_evidence_store import (
    list_prediction_input_evidence_by_run,
)
from gridiron_edge.models.game_prediction.prediction_policy import PredictionPolicy
from gridiron_edge.models.game_prediction.weekly_game_product import (
    build_weekly_game_product,
)
from gridiron_edge.models.game_prediction.weekly_spread_product import (
    load_and_attach_derived_spreads,
)
from gridiron_edge.models.game_prediction.weekly_starting_qb import (
    attach_starting_qb_assumptions,
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


def _bound_sidecar(run_id: str, *, repo: Path) -> DataFrame | None:
    """Load the QB/coach sidecar the forecast run was bound to, else ``None``.

    The run's prediction-input evidence records the sidecar's SHA-256. The
    sidecar is rewritten on every build, so the file is used only if it still
    matches that digest; a mismatch means the starters recorded here would not
    be the ones that fed the forecast, and composition fails instead.
    """
    sidecar_path = dataset_path(repo, "qb_coach_sidecar")
    relative = sidecar_path.relative_to(repo).as_posix()
    digests = {
        reference.content_digest
        for evidence in list_prediction_input_evidence_by_run(run_id, repo=repo)
        for reference in evidence.source_artifacts
        if reference.relative_path == relative and reference.state is PredictionSourceState.PRESENT
    }
    if not digests or not sidecar_path.exists():
        return None
    if len(digests) != 1:
        raise ValueError("Prediction-input evidence disagrees about the QB/coach sidecar identity.")
    if sha256(sidecar_path.read_bytes()).hexdigest() not in digests:
        raise ValueError(
            "The QB/coach sidecar changed after the forecast run; the assumed starting "
            "quarterbacks cannot be recorded. Rerun the forecast."
        )
    return load_qb_coach_sidecar(repo)


def _starter_names(sidecar: DataFrame, *, repo: Path, season_label: str) -> dict[str, str]:
    """Map player ids to names from the sidecar, then the cached depth chart."""
    names: dict[str, str] = {}
    for side in ("AWAY", "HOME"):
        pairs = sidecar[[f"{side}_QB_ID", f"{side}_QB_NAME"]].dropna()
        names.update(zip(pairs.iloc[:, 0].astype(str), pairs.iloc[:, 1].astype(str), strict=True))
    try:
        from gridiron_edge.ingest.nflverse.weekly_availability import load_depth_charts

        season = int(season_label.split("-")[0])
        chart = load_depth_charts(seasons=[season], repo=repo).get(season)
    except (FileNotFoundError, ValueError):
        chart = None
    if chart is not None and {"gsis_id", "player_name"} <= set(chart.columns):
        known = chart[["gsis_id", "player_name"]].dropna().drop_duplicates("gsis_id")
        for player_id, name in zip(
            known["gsis_id"].astype(str), known["player_name"].astype(str), strict=True
        ):
            names.setdefault(player_id, name)
    return names


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
    sidecar = _bound_sidecar(run_id, repo=resolved_repo)
    names = (
        None
        if sidecar is None
        else _starter_names(sidecar, repo=resolved_repo, season_label=season)
    )
    product = build_weekly_game_product(
        attach_starting_qb_assumptions(total_product, sidecar, names)
    )

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

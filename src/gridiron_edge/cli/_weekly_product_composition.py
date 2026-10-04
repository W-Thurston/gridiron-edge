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

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.loaders import load_qb_coach_sidecar
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.datasets.writers import (
    select_current_weekly_product,
    write_weekly_product,
)
from gridiron_edge.evaluation.forecast_contracts import ForecastRole, WeeklyProductIdentity
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
from gridiron_edge.models.game_prediction.weekly_product_store import cited_run_ids
from gridiron_edge.models.game_prediction.weekly_spread_product import (
    load_and_attach_derived_spreads,
)
from gridiron_edge.models.game_prediction.weekly_starting_qb import (
    STARTING_QB_COLUMNS,
    attach_starting_qb_assumptions,
    unrecorded_starting_qb_record,
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
    carried_game_ids: tuple[str, ...] = ()
    carried_run_ids: tuple[str, ...] = ()


_STORAGE_COLUMNS = (
    "product_schema_version",
    "product_id",
    "product_run_id",
    "product_generated_at",
)


def _kickoff(row: pd.Series) -> datetime:
    from gridiron_edge.market.collection_plan import _schedule_kickoff_utc

    return _schedule_kickoff_utc(row["game_date"], row["game_time"])


def carried_game_ids(
    previous: DataFrame | None,
    scoped_schedule: DataFrame,
    *,
    generated_at: datetime,
) -> tuple[str, ...]:
    """Return the games whose previous row is kept instead of being forecast again.

    A game moves from upcoming to played, and its last pre-kickoff forecast must survive that
    move: a game that has kicked off, or whose result has landed and so left the upcoming
    schedule, keeps the row of the previous product. Games not yet started are forecast by the
    new run, and a started game with no previous row stays missing (never forecast late).
    """
    if previous is None or previous.empty:
        return ()
    scheduled = set(scoped_schedule["game_id"].astype(str))
    carried: list[str] = []
    for _, row in previous.iterrows():
        game_id = str(row["game_id"])
        if game_id not in scheduled or _kickoff(row) <= generated_at:
            carried.append(game_id)
    return tuple(sorted(carried))


def require_pregame_live_forecasts(product: DataFrame) -> None:
    """Reject a live forecast generated after its game's kickoff.

    A forecast made once the game is under way is not a forecast of it, and would be scored
    as one. Development and backfilled forecasts are exempt (they are retrospective).
    """
    late: list[str] = []
    for _, row in product.iterrows():
        kickoff = _kickoff(row)
        for family in ("win", "total"):
            generated = row[f"{family}_generated_at"]
            if str(row[f"{family}_role"]) != ForecastRole.LIVE.value or pd.isna(generated):
                continue
            if pd.Timestamp(generated).to_pydatetime() > kickoff:
                late.append(f"{row['game_id']} ({family})")
    if late:
        raise ValueError("Live forecasts were generated after kickoff: " + ", ".join(sorted(late)))


def _carry_rows(previous: DataFrame, game_ids: tuple[str, ...], columns: pd.Index) -> DataFrame:
    """Return the previous product's rows for ``game_ids`` in the new product's columns."""
    kept = previous.loc[previous["game_id"].astype(str).isin(game_ids)].drop(
        columns=list(_STORAGE_COLUMNS), errors="ignore"
    )
    kept = kept.reset_index(drop=True)
    if not set(STARTING_QB_COLUMNS) <= set(kept.columns):
        record = unrecorded_starting_qb_record(kept.index)
        kept = pd.concat([kept, record.loc[:, [c for c in record if c not in kept]]], axis=1)
    return kept.loc[:, list(columns)]


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


def _require_carried_events_exist(
    previous: DataFrame, game_ids: tuple[str, ...], season: str, week: int, repo: Path
) -> None:
    """Require every forecast event a carried row cites to still be in the forecast store."""
    from gridiron_edge.evaluation.forecast_store import load_forecast_events

    rows = previous.loc[previous["game_id"].astype(str).isin(game_ids)]
    cited = {
        str(value)
        for column in ("win_event_id", "total_event_id")
        for value in rows[column].dropna()
    }
    stored = set(load_forecast_events(season=season, week=week, repo=repo)["event_id"].astype(str))
    if cited - stored:
        raise ValueError(
            "Carried rows cite forecast events that are not in the forecast store: "
            + ", ".join(sorted(cited - stored))
        )


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
    previous_product: DataFrame | None = None,
) -> ComposedWeeklyProduct:
    """Compose, persist, and explicitly select one weekly product.

    ``schedule`` must already carry the rich-schedule shape ``_scope_schedule``
    requires (season/week/game_id/game_day_of_week/game_date/game_time/
    away_team/home_team/neutral_site), live from the fetch-derived upcoming
    schedule or development from retained history. ``events`` must be one
    run's exact selected forecast events, and ``policy`` the exact decision
    that produced them.

    ``previous_product`` is the week's current product when this is a refresh. Games that
    have kicked off, or whose results landed and so left the upcoming schedule, keep their
    previous row (their last pre-kickoff forecast); only the remaining games take ``events``.
    The product declares the earlier runs it draws on as carried runs.
    """
    resolved_repo = repo or get_settings().repo_root

    scoped_schedule = schedule.loc[
        (schedule["season"].astype(str) == season) & (schedule["week"] == week),
        :,
    ].copy()
    if scoped_schedule.empty:
        raise ValueError("Rich schedule has no rows for the requested season and week.")
    kept_ids = carried_game_ids(previous_product, scoped_schedule, generated_at=generated_at)
    scoped_schedule = scoped_schedule.loc[~scoped_schedule["game_id"].astype(str).isin(kept_ids)]
    if scoped_schedule.empty:
        raise ValueError("Every game of the week is already carried; there is nothing to forecast.")

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
    if kept_ids and previous_product is not None:
        _require_carried_events_exist(previous_product, kept_ids, season, week, resolved_repo)
        product = pd.concat(
            [product, _carry_rows(previous_product, kept_ids, product.columns)],
            ignore_index=True,
        )
        product = product.sort_values(["game_date", "game_time", "game_id"], kind="stable")
        product = product.reset_index(drop=True)
    require_pregame_live_forecasts(product)

    product_id = f"weekly_{season.replace('-', '_')}_wk{week:02d}_{run_id}"
    identity = WeeklyProductIdentity(
        product_id=product_id,
        run_id=run_id,
        season=season,
        week=week,
        generated_at=generated_at,
        carried_run_ids=tuple(sorted(cited_run_ids(product) - {run_id})),
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
        carried_game_ids=kept_ids,
        carried_run_ids=identity.carried_run_ids,
    )

# src/gridiron_edge/evaluation/live_forecast_closeout.py
"""Close out exact forecast events selected by one weekly product."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.datasets.loaders import load_current_weekly_product, load_games
from gridiron_edge.evaluation.forecast_contracts import (
    WEEKLY_PRODUCT_FORECAST_ROLES,
    ForecastRole,
)
from gridiron_edge.evaluation.forecast_store import load_forecast_events

_AVAILABLE_WIN = "available"
_AVAILABLE_SPREAD = "available"
_AVAILABLE_TOTAL = {"available", "uncertainty_unavailable"}


@dataclass(frozen=True, slots=True)
class WinCloseoutMetrics:
    """Metrics for evaluated selected Win forecasts."""

    evaluated_count: int
    brier: float | None
    log_loss: float | None
    accuracy: float | None


@dataclass(frozen=True, slots=True)
class SpreadCloseoutMetrics:
    """Metrics for evaluated selected Spread forecasts."""

    evaluated_count: int
    mae: float | None
    rmse: float | None
    bias: float | None


@dataclass(frozen=True, slots=True)
class TotalCloseoutMetrics:
    """Metrics for evaluated selected Total forecasts."""

    evaluated_count: int
    mae: float | None
    rmse: float | None
    bias: float | None


@dataclass(frozen=True, slots=True)
class LiveForecastCloseout:
    """Coverage reconciliation and metrics for one selected weekly product."""

    season: str
    week: int
    product_id: str
    product_run_id: str
    win_role: ForecastRole | None
    total_role: ForecastRole | None
    scheduled_game_count: int
    completed_outcome_count: int
    selected_win_count: int
    matched_win_event_count: int
    selected_spread_count: int
    matched_spread_event_count: int
    selected_total_count: int
    matched_total_event_count: int
    missing_win_component_game_ids: tuple[str, ...]
    missing_spread_component_game_ids: tuple[str, ...]
    missing_total_component_game_ids: tuple[str, ...]
    missing_win_event_game_ids: tuple[str, ...]
    missing_spread_event_game_ids: tuple[str, ...]
    missing_total_event_game_ids: tuple[str, ...]
    missing_outcome_game_ids: tuple[str, ...]
    win: WinCloseoutMetrics
    spread: SpreadCloseoutMetrics
    total: TotalCloseoutMetrics
    reconciliation: DataFrame

    @property
    def complete(self) -> bool:
        """Return whether every scheduled game has forecasts and an outcome."""
        return not any(
            (
                self.missing_win_component_game_ids,
                self.missing_spread_component_game_ids,
                self.missing_total_component_game_ids,
                self.missing_win_event_game_ids,
                self.missing_spread_event_game_ids,
                self.missing_total_event_game_ids,
                self.missing_outcome_game_ids,
            )
        )


_PRODUCT_COLUMNS = {
    "product_id",
    "product_run_id",
    "win_role",
    "total_role",
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "win_status",
    "win_event_id",
    "win_run_id",
    "win_model_name",
    "win_model_type",
    "spread_status",
    "spread_source_event_id",
    "spread_model_name",
    "spread_model_type",
    "total_status",
    "total_event_id",
    "total_run_id",
    "total_model_name",
    "total_model_type",
}
_GAME_COLUMNS = {
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_SCORE",
    "HOME_SCORE",
}
_EVENT_COLUMNS = {
    "event_id",
    "run_id",
    "role",
    "season",
    "week",
    "game_id",
    "model_name",
    "model_type",
    "home_win_prob",
    "model_spread",
    "model_total",
}


def _require_columns(frame: DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _single_text(frame: DataFrame, column: str, *, label: str) -> str:
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1 or not values[0].strip():
        raise ValueError(f"{label} must contain one nonempty {column} value.")
    return values[0]


def _selected_role(
    product: DataFrame,
    *,
    status_column: str,
    available_statuses: set[str],
    role_column: str,
    family: str,
) -> ForecastRole | None:
    """Return one supported role for the selected family."""
    selected = product.loc[
        product[status_column].astype(str).isin(available_statuses),
        role_column,
    ]

    if selected.empty:
        return None

    values = tuple(sorted(selected.dropna().astype(str).str.strip().unique().tolist()))
    if len(values) != 1 or not values:
        raise ValueError(f"Selected {family} components must use one nonempty forecast role.")

    try:
        role = ForecastRole(values[0])
    except ValueError as exc:
        raise ValueError(
            f"Selected {family} components contain an unsupported forecast role: {values[0]!r}."
        ) from exc

    if role not in WEEKLY_PRODUCT_FORECAST_ROLES:
        raise ValueError(
            f"Selected {family} components require a live or development forecast role."
        )

    return role


def _selected_product_roles(
    product: DataFrame,
) -> tuple[ForecastRole | None, ForecastRole | None]:
    """Resolve one coherent role across selected Win and Total components."""
    win_role = _selected_role(
        product,
        status_column="win_status",
        available_statuses={_AVAILABLE_WIN},
        role_column="win_role",
        family="Win",
    )
    total_role = _selected_role(
        product,
        status_column="total_status",
        available_statuses=_AVAILABLE_TOTAL,
        role_column="total_role",
        family="Total",
    )

    selected_roles = {
        role
        for role in (
            win_role,
            total_role,
        )
        if role is not None
    }
    if len(selected_roles) > 1:
        raise ValueError("Selected Win and Total components must use one forecast role.")

    return win_role, total_role


def _family_reference_columns(
    family: str,
) -> tuple[str, str, str, str, str]:
    """Return event, run, model, and role columns for one family."""
    if family == "spread":
        return (
            "spread_source_event_id",
            "win_run_id",
            "spread_model_name",
            "spread_model_type",
            "win_role",
        )

    prefix = "win" if family == "win" else "total"
    return (
        f"{prefix}_event_id",
        f"{prefix}_run_id",
        f"{prefix}_model_name",
        f"{prefix}_model_type",
        f"{prefix}_role",
    )


def _event_matches(
    row: Series,
    event: Series,
    *,
    family: str,
) -> bool:
    (
        event_column,
        run_column,
        model_name_column,
        model_type_column,
        role_column,
    ) = _family_reference_columns(family)

    return all(
        (
            str(event["event_id"]) == str(row[event_column]),
            str(event["game_id"]) == str(row["game_id"]),
            str(event["run_id"]) == str(row[run_column]),
            str(event["role"]) == str(row[role_column]),
            str(event["model_name"]) == str(row[model_name_column]),
            str(event["model_type"]) == str(row[model_type_column]),
            str(event["season"]) == str(row["season"]),
            int(event["week"]) == int(row["week"]),
        )
    )


def _referenced_event(
    row: Series,
    events_by_id: DataFrame,
    *,
    family: str,
) -> Series | None:
    event_column, _, _, _, _ = _family_reference_columns(family)
    event_id = row[event_column]
    if pd.isna(event_id) or str(event_id).strip() == "":
        return None
    if str(event_id) not in events_by_id.index:
        return None
    event = events_by_id.loc[str(event_id)]
    if isinstance(event, DataFrame):
        return None
    return event if _event_matches(row, event, family=family) else None


def _win_metrics(rows: DataFrame) -> WinCloseoutMetrics:
    evaluable = rows.loc[rows["win_evaluable"], :]
    count = len(evaluable)
    if count == 0:
        return WinCloseoutMetrics(0, None, None, None)
    probability = pd.to_numeric(evaluable["home_win_prob"], errors="coerce").astype(float)
    outcome = pd.to_numeric(evaluable["actual_home_win"], errors="coerce").astype(float)
    clipped = probability.clip(1e-7, 1 - 1e-7)
    brier = float(((probability - outcome) ** 2).mean())
    loss = float(-(outcome * np.log(clipped) + (1 - outcome) * np.log(1 - clipped)).mean())
    accuracy = float(((probability >= 0.5) == outcome.astype(bool)).mean())
    return WinCloseoutMetrics(count, brier, loss, accuracy)


def _spread_metrics(rows: DataFrame) -> SpreadCloseoutMetrics:
    evaluable = rows.loc[rows["spread_evaluable"], :]
    count = len(evaluable)
    if count == 0:
        return SpreadCloseoutMetrics(0, None, None, None)
    prediction = pd.to_numeric(evaluable["projected_home_margin"], errors="coerce").astype(float)
    actual = pd.to_numeric(evaluable["actual_margin"], errors="coerce").astype(float)
    error = prediction - actual
    return SpreadCloseoutMetrics(
        count,
        float(error.abs().mean()),
        float(math.sqrt(float((error**2).mean()))),
        float(error.mean()),
    )


def _total_metrics(rows: DataFrame) -> TotalCloseoutMetrics:
    evaluable = rows.loc[rows["total_evaluable"], :]
    count = len(evaluable)
    if count == 0:
        return TotalCloseoutMetrics(0, None, None, None)
    prediction = pd.to_numeric(evaluable["model_total"], errors="coerce").astype(float)
    actual = pd.to_numeric(evaluable["actual_total"], errors="coerce").astype(float)
    error = prediction - actual
    return TotalCloseoutMetrics(
        count,
        float(error.abs().mean()),
        float(math.sqrt(float((error**2).mean()))),
        float(error.mean()),
    )


def close_live_forecasts(  # noqa: PLR0915
    *,
    product: DataFrame,
    forecast_events: DataFrame,
    games: DataFrame,
) -> LiveForecastCloseout:
    """Reconcile exact selected events with completed outcomes without I/O."""
    _require_columns(product, _PRODUCT_COLUMNS, label="Weekly product")
    _require_columns(forecast_events, _EVENT_COLUMNS, label="Forecast events")
    _require_columns(games, _GAME_COLUMNS, label="Cleaned games")
    if product.empty:
        raise ValueError("Weekly product must contain at least one scheduled game.")
    if product["game_id"].astype(str).duplicated().any():
        raise ValueError("Weekly product contains duplicate game IDs.")
    if forecast_events["event_id"].astype(str).duplicated().any():
        raise ValueError("Forecast events contain duplicate event IDs.")

    season = _single_text(product, "season", label="Weekly product")
    product_id = _single_text(product, "product_id", label="Weekly product")
    product_run_id = _single_text(product, "product_run_id", label="Weekly product")
    win_role, total_role = _selected_product_roles(product)
    numeric_weeks = pd.to_numeric(product["week"], errors="coerce")
    weeks = numeric_weeks.dropna().astype(int).unique().tolist()
    if len(weeks) != 1:
        raise ValueError("Weekly product must contain one week value.")
    week = int(weeks[0])

    numeric_game_weeks = pd.to_numeric(games["WEEK_NUM"], errors="coerce")
    scoped_games = games.loc[
        games["YEAR"].astype(str).eq(season) & numeric_game_weeks.eq(week),
        ["GAME_ID", "AWAY_SCORE", "HOME_SCORE"],
    ].copy()
    if scoped_games["GAME_ID"].astype(str).duplicated().any():
        raise ValueError("Cleaned games contain duplicate Game IDs in the requested scope.")
    outcome_by_game_id: dict[str, tuple[float, float]] = {}
    for game in scoped_games.itertuples(index=False):
        if pd.notna(game.AWAY_SCORE) and pd.notna(game.HOME_SCORE):
            away_score = float(cast(float | int, game.AWAY_SCORE))
            home_score = float(cast(float | int, game.HOME_SCORE))
            outcome_by_game_id[str(game.GAME_ID)] = (
                away_score,
                home_score,
            )

    events_by_id = forecast_events.copy()
    events_by_id.index = events_by_id["event_id"].astype(str)

    records: list[dict[str, object]] = []
    for _, row in product.iterrows():
        game_id = str(row["game_id"])
        win_selected = str(row["win_status"]) == _AVAILABLE_WIN
        spread_selected = str(row["spread_status"]) == _AVAILABLE_SPREAD
        total_selected = str(row["total_status"]) in _AVAILABLE_TOTAL
        win_event = _referenced_event(row, events_by_id, family="win") if win_selected else None
        spread_event = (
            _referenced_event(row, events_by_id, family="spread") if spread_selected else None
        )
        total_event = (
            _referenced_event(row, events_by_id, family="total") if total_selected else None
        )
        scores = outcome_by_game_id.get(game_id)
        outcome_available = scores is not None
        away_score: float | None = None
        home_score: float | None = None
        if scores is not None:
            away_score, home_score = scores
        tied = away_score is not None and home_score is not None and away_score == home_score
        actual_home_win: bool | object = pd.NA
        actual_margin: float | object = pd.NA
        actual_total: float | object = pd.NA
        if away_score is not None and home_score is not None:
            actual_margin = home_score - away_score
            actual_total = home_score + away_score
            if not tied:
                actual_home_win = home_score > away_score
        home_win_prob = win_event["home_win_prob"] if win_event is not None else pd.NA
        model_spread = spread_event["model_spread"] if spread_event is not None else pd.NA
        projected_home_margin: float | object = pd.NA
        if pd.notna(model_spread):
            projected_home_margin = -float(cast(float | int, model_spread))
        model_total = total_event["model_total"] if total_event is not None else pd.NA
        records.append(
            {
                "game_id": game_id,
                "away_team": row["away_team"],
                "home_team": row["home_team"],
                "win_component_selected": win_selected,
                "win_event_id": row["win_event_id"],
                "win_event_matched": win_event is not None,
                "home_win_prob": home_win_prob,
                "spread_component_selected": spread_selected,
                "spread_event_id": row["spread_source_event_id"],
                "spread_event_matched": spread_event is not None,
                "model_spread": model_spread,
                "projected_home_margin": projected_home_margin,
                "total_component_selected": total_selected,
                "total_event_id": row["total_event_id"],
                "total_event_matched": total_event is not None,
                "model_total": model_total,
                "away_score": away_score if away_score is not None else pd.NA,
                "home_score": home_score if home_score is not None else pd.NA,
                "outcome_available": outcome_available,
                "actual_home_win": actual_home_win,
                "actual_margin": actual_margin,
                "actual_total": actual_total,
                "win_evaluable": (
                    win_event is not None
                    and outcome_available
                    and not tied
                    and pd.notna(home_win_prob)
                ),
                "spread_evaluable": (
                    spread_event is not None and outcome_available and pd.notna(model_spread)
                ),
                "total_evaluable": (
                    total_event is not None and outcome_available and pd.notna(model_total)
                ),
            }
        )

    reconciliation = DataFrame.from_records(records)

    def ids(mask: Series) -> tuple[str, ...]:
        return tuple(sorted(reconciliation.loc[mask, "game_id"].astype(str).tolist()))

    missing_win_components = ids(~reconciliation["win_component_selected"])
    missing_spread_components = ids(~reconciliation["spread_component_selected"])
    missing_total_components = ids(~reconciliation["total_component_selected"])
    missing_win_events = ids(
        reconciliation["win_component_selected"] & ~reconciliation["win_event_matched"]
    )
    missing_spread_events = ids(
        reconciliation["spread_component_selected"] & ~reconciliation["spread_event_matched"]
    )
    missing_total_events = ids(
        reconciliation["total_component_selected"] & ~reconciliation["total_event_matched"]
    )
    missing_outcomes = ids(~reconciliation["outcome_available"])

    return LiveForecastCloseout(
        season=season,
        week=week,
        product_id=product_id,
        product_run_id=product_run_id,
        win_role=win_role,
        total_role=total_role,
        scheduled_game_count=len(reconciliation),
        completed_outcome_count=int(reconciliation["outcome_available"].sum()),
        selected_win_count=int(reconciliation["win_component_selected"].sum()),
        matched_win_event_count=int(reconciliation["win_event_matched"].sum()),
        selected_spread_count=int(reconciliation["spread_component_selected"].sum()),
        matched_spread_event_count=int(reconciliation["spread_event_matched"].sum()),
        selected_total_count=int(reconciliation["total_component_selected"].sum()),
        matched_total_event_count=int(reconciliation["total_event_matched"].sum()),
        missing_win_component_game_ids=missing_win_components,
        missing_spread_component_game_ids=missing_spread_components,
        missing_total_component_game_ids=missing_total_components,
        missing_win_event_game_ids=missing_win_events,
        missing_spread_event_game_ids=missing_spread_events,
        missing_total_event_game_ids=missing_total_events,
        missing_outcome_game_ids=missing_outcomes,
        win=_win_metrics(reconciliation),
        spread=_spread_metrics(reconciliation),
        total=_total_metrics(reconciliation),
        reconciliation=reconciliation,
    )


def load_live_forecast_closeout(
    *,
    repo: Path,
    season: str,
    week: int,
) -> LiveForecastCloseout:
    """Load the selected product, exact events, and outcomes for closeout."""
    product = load_current_weekly_product(repo, season=season, week=week)
    events = load_forecast_events(repo=repo, season=season, week=week)
    games = load_games(repo)
    return close_live_forecasts(
        product=product,
        forecast_events=events,
        games=games,
    )

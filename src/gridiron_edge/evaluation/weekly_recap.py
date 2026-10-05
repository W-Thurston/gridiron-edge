# src/gridiron_edge/evaluation/weekly_recap.py

"""Recap one finished week: every scheduled game's forecast against the result and the market.

The recap is built from persisted state only (``DECISIONS.md`` D86): the week's regular-season
schedule from the raw nflverse games file, the week's *selected* weekly product, the exact
forecast events it cites, and the cleaned games file's outcomes and closing lines. It reuses the
closeout's reconciliation (:func:`close_live_forecasts`) for event matching, outcomes and ties,
so its model metrics over in-product games equal the closeout's.

Conventions: one row per game, Away/Home; win means ``HOME_WIN``; every margin is home minus
away. The closing market is ``VEGAS_LINE`` (expected home margin) and ``OVER_UNDER``, and the
market's home win probability is Phi(``VEGAS_LINE`` / ``MARGIN_STD``), as in ``DECISIONS.md``
D85. The product's own ``spread_line`` and ``total_line`` are the market at forecast time.

The starting-QB section (:mod:`gridiron_edge.evaluation.weekly_recap_starters`) adds, per game
and side, the assumed starter, nflverse's value and the carried-forward starter against who
started.

A scheduled game missing from the product is a row with status ``not_in_product``; a missing
value always carries a status. The recap's identity is the SHA-256 of its content excluding
``built_at``, so unchanged inputs replay to the same identity.

Public API
----------
build_weekly_recap      schedule, product, events, games, game logs, captures -> BuiltWeeklyRecap
recap_metrics           recap game rows -> model and market metrics, family by family
unplayed_game_ids       scoped schedule -> games without a result
weekly_recap_id         recap -> content identity excluding built_at
validate_weekly_recap   recap -> raises on an invalid manifest
weekly_recap_from_payload / weekly_recap_payload   JSON round trip
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Any, Final, cast, get_origin, get_type_hints

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.evaluation.game_model_evaluation_report import (
    EvaluationFrameReference,
    frame_content_digest,
)
from gridiron_edge.evaluation.live_forecast_closeout import close_live_forecasts
from gridiron_edge.evaluation.market_agreement import MARGIN_STD
from gridiron_edge.evaluation.weekly_recap_starters import (
    ACTUAL_STARTER_CONVENTION,
    STARTER_BOOLEAN_COLUMNS,
    STARTER_STRING_COLUMNS,
    SidecarCapture,
    StarterSummary,
    player_names,
    starter_columns,
    starter_summary,
    week_passers,
)
from gridiron_edge.market.collection_plan import _schedule_kickoff_utc
from gridiron_edge.models.game_prediction.weekly_product_store import cited_run_ids

WEEKLY_RECAP_SCHEMA_VERSION: Final[int] = 2
# Two errors closer than this are a tie.
_CLOSER_TOLERANCE: Final[float] = 1e-9
_PROBABILITY_CLIP: Final[float] = 1e-7
_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "away_team",
    "home_team",
    "gameday",
    "gametime",
    "away_score",
    "home_score",
    "result",
    "away_qb_id",
    "home_qb_id",
)
_OUTCOME_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "AWAY_SCORE",
    "HOME_SCORE",
    "VEGAS_LINE",
    "OVER_UNDER",
)


class WeeklyRecapRefusalError(ValueError):
    """The week cannot be recapped from the persisted state as it stands."""


class ProductRowStatus(StrEnum):
    """Whether the selected product holds a row for a scheduled game."""

    IN_PRODUCT = "in_product"
    NOT_IN_PRODUCT = "not_in_product"


class ForecastStatus(StrEnum):
    """Whether one forecast family can be scored for a game."""

    AVAILABLE = "available"
    FORECAST_MISSING = "forecast_missing"
    NOT_IN_PRODUCT = "not_in_product"
    # Win only: the forecast exists, but a tied game has no home-win outcome.
    OUTCOME_TIE = "outcome_tie"


class ForecastMarketStatus(StrEnum):
    """Whether the product recorded the market lines it was composed against."""

    AVAILABLE = "available"
    NOT_RECORDED = "not_recorded"
    NOT_IN_PRODUCT = "not_in_product"


class ClosingLineStatus(StrEnum):
    """Whether the cleaned games file holds a closing line."""

    AVAILABLE = "available"
    CLOSING_LINE_MISSING = "closing_line_missing"


class Closer(StrEnum):
    """Which side's forecast landed closer to the result."""

    MODEL = "model"
    MARKET = "market"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    """Win-probability scores over ``count`` decided games."""

    count: int
    brier: float | None
    log_loss: float | None
    accuracy: float | None


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    """Point-forecast errors (prediction minus result) over ``count`` games."""

    count: int
    mae: float | None
    rmse: float | None
    bias: float | None


@dataclass(frozen=True, slots=True)
class CloserCounts:
    """How often each side landed closer, over games both sides forecast."""

    model: int
    market: int
    tie: int


@dataclass(frozen=True, slots=True)
class WinComparison:
    """Model against closing-market win probability."""

    model: ProbabilityMetrics
    market: ProbabilityMetrics
    closer: CloserCounts


@dataclass(frozen=True, slots=True)
class PointComparison:
    """Model against closing-market point forecast."""

    model: ErrorMetrics
    market: ErrorMetrics
    closer: CloserCounts


@dataclass(frozen=True, slots=True)
class WeekMetrics:
    """The week's model and market metrics, family by family.

    Each side is scored over its own available games, with its count; ``closer`` covers only
    games both sides forecast. Projected-score errors are per team score.
    """

    win: WinComparison
    spread: PointComparison
    total: PointComparison
    projected_score: PointComparison


@dataclass(frozen=True, slots=True)
class RecapCounts:
    """Row counts of the recap's game frame."""

    scheduled_games: int
    in_product: int
    not_in_product: int
    carried_rows: int
    outcome_ties: int


@dataclass(frozen=True, slots=True)
class RecapConventions:
    """The conventions every value in the recap is stated in."""

    orientation: str
    closing_market: str
    market_at_forecast: str
    market_win_probability: str
    margin_std: float
    actual_starter: str


RECAP_CONVENTIONS: Final[RecapConventions] = RecapConventions(
    orientation="one row per game, Away/Home; win means HOME_WIN; margins are home minus away",
    closing_market="cleaned games VEGAS_LINE (expected home margin) and OVER_UNDER",
    market_at_forecast="weekly product spread_line (expected home margin) and total_line",
    market_win_probability="Phi(VEGAS_LINE / margin_std)",
    margin_std=MARGIN_STD,
    actual_starter=ACTUAL_STARTER_CONVENTION,
)


@dataclass(frozen=True, slots=True)
class WeeklyRecap:
    """Manifest of one immutable weekly recap."""

    schema_version: int
    recap_id: str
    built_at: datetime
    season: str
    week: int
    product_id: str
    product_run_id: str
    carried_run_ids: tuple[str, ...]
    product_schema_version: int
    forecast_role: str | None
    schedule_digest: str
    outcome_digest: str
    passer_digest: str
    starter_capture_digests: tuple[str, ...]
    conventions: RecapConventions
    counts: RecapCounts
    metrics: WeekMetrics
    starters: StarterSummary
    games: EvaluationFrameReference


@dataclass(frozen=True, slots=True)
class BuiltWeeklyRecap:
    """A recap manifest with the game frame it references."""

    recap: WeeklyRecap
    games: DataFrame


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def unplayed_game_ids(schedule: DataFrame) -> tuple[str, ...]:
    """Return the scheduled games without a result, sorted."""
    return tuple(sorted(schedule.loc[schedule["result"].isna(), "game_id"].astype(str)))


def _require_columns(frame: DataFrame, required: tuple[str, ...], *, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _single(product: DataFrame, column: str) -> str:
    values = product[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Weekly product must carry one {column} value.")
    return values[0]


def _outcomes(games: DataFrame, schedule: DataFrame) -> DataFrame:
    """Return the cleaned outcome row of every scheduled game, refusing any disagreement."""
    _require_columns(games, _OUTCOME_COLUMNS, label="Cleaned games")
    ids = set(schedule["game_id"].astype(str))
    cleaned = games.loc[games["GAME_ID"].astype(str).isin(ids), list(_OUTCOME_COLUMNS)].copy()
    cleaned["GAME_ID"] = cleaned["GAME_ID"].astype(str)
    if cleaned["GAME_ID"].duplicated().any():
        raise ValueError("Cleaned games contain duplicate Game IDs for the week.")
    by_id = cleaned.set_index("GAME_ID")
    problems: list[str] = []
    for row in schedule.itertuples(index=False):
        game_id = str(row.game_id)
        if game_id not in by_id.index:
            problems.append(f"{game_id} (not in the cleaned games file)")
            continue
        cleaned_scores = (by_id.at[game_id, "AWAY_SCORE"], by_id.at[game_id, "HOME_SCORE"])
        raw_scores = (row.away_score, row.home_score)
        if any(pd.isna(value) for value in (*cleaned_scores, *raw_scores)) or tuple(
            float(cast(float, value)) for value in cleaned_scores
        ) != tuple(float(cast(float, value)) for value in raw_scores):
            problems.append(
                f"{game_id} (raw {raw_scores[0]}-{raw_scores[1]}, "
                f"cleaned {cleaned_scores[0]}-{cleaned_scores[1]})"
            )
    if problems:
        raise WeeklyRecapRefusalError(
            "The raw and cleaned games files disagree on scores: " + "; ".join(sorted(problems))
        )
    return by_id


# ---------------------------------------------------------------------------
# Per-game values
# ---------------------------------------------------------------------------


def _number(value: object) -> float | None:
    if value is None or pd.isna(cast(Any, value)):
        return None
    return float(cast(float, value))


def _phi(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _closer(model_error: float | None, market_error: float | None) -> str | None:
    if model_error is None or market_error is None:
        return None
    model_abs, market_abs = abs(model_error), abs(market_error)
    if abs(model_abs - market_abs) <= _CLOSER_TOLERANCE:
        return Closer.TIE.value
    return (Closer.MODEL if model_abs < market_abs else Closer.MARKET).value


def _difference(prediction: float | None, actual: float) -> float | None:
    return None if prediction is None else prediction - actual


def _family_status(*, in_product: bool, selected: bool, value: float | None) -> ForecastStatus:
    if not in_product:
        return ForecastStatus.NOT_IN_PRODUCT
    if not selected or value is None:
        return ForecastStatus.FORECAST_MISSING
    return ForecastStatus.AVAILABLE


def _game_row(  # noqa: PLR0915
    game: Series,
    outcome: Series,
    product_row: Series | None,
    closeout_row: Series | None,
    *,
    season: str,
    week: int,
    product_run_id: str,
    market_recorded: bool,
) -> dict[str, object]:
    in_product = product_row is not None and closeout_row is not None
    away_score = float(cast(float, outcome["AWAY_SCORE"]))
    home_score = float(cast(float, outcome["HOME_SCORE"]))
    actual_margin = home_score - away_score
    actual_total = home_score + away_score
    tied = actual_margin == 0
    home_won = None if tied else actual_margin > 0
    outcome_value = None if home_won is None else float(home_won)

    closing_margin = _number(outcome["VEGAS_LINE"])
    closing_total = _number(outcome["OVER_UNDER"])
    closing_prob = None if closing_margin is None else _phi(closing_margin / MARGIN_STD)
    implied_home = implied_away = None
    if closing_margin is not None and closing_total is not None:
        implied_home = (closing_total + closing_margin) / 2
        implied_away = (closing_total - closing_margin) / 2

    row: dict[str, object] = {
        "season": season,
        "week": week,
        "game_id": str(game["game_id"]),
        "away_team": str(game["away_team"]),
        "home_team": str(game["home_team"]),
        "kickoff": game["_kickoff"],
        "product_row_status": (
            ProductRowStatus.IN_PRODUCT if in_product else ProductRowStatus.NOT_IN_PRODUCT
        ).value,
        "forecast_role": None,
        "forecast_run_id": None,
        "forecast_generated_at": None,
        "carried": None,
        "forecast_market_status": ForecastMarketStatus.NOT_IN_PRODUCT.value,
        "forecast_home_margin_line": None,
        "forecast_total_line": None,
        "forecast_market_ingested_at": None,
        "closing_spread_status": (
            ClosingLineStatus.CLOSING_LINE_MISSING
            if closing_margin is None
            else ClosingLineStatus.AVAILABLE
        ).value,
        "closing_home_margin": closing_margin,
        "closing_home_win_prob": closing_prob,
        "closing_total_status": (
            ClosingLineStatus.CLOSING_LINE_MISSING
            if closing_total is None
            else ClosingLineStatus.AVAILABLE
        ).value,
        "closing_total": closing_total,
        "market_implied_away_score": implied_away,
        "market_implied_home_score": implied_home,
        "away_score": away_score,
        "home_score": home_score,
        "actual_margin": actual_margin,
        "actual_total": actual_total,
        "actual_home_win": home_won,
    }

    win_prob = margin = total = proj_home = proj_away = None
    win_selected = spread_selected = total_selected = score_selected = False
    if in_product:
        assert product_row is not None and closeout_row is not None
        run_id = str(product_row["win_run_id"])
        row["forecast_role"] = str(product_row["win_role"])
        row["forecast_run_id"] = run_id
        row["forecast_generated_at"] = pd.Timestamp(product_row["win_generated_at"])
        row["carried"] = run_id != product_run_id
        forecast_margin = _number(product_row.get("spread_line")) if market_recorded else None
        forecast_total = _number(product_row.get("total_line")) if market_recorded else None
        if forecast_margin is not None and forecast_total is not None:
            row["forecast_market_status"] = ForecastMarketStatus.AVAILABLE.value
            row["forecast_home_margin_line"] = forecast_margin
            row["forecast_total_line"] = forecast_total
            row["forecast_market_ingested_at"] = pd.Timestamp(product_row["ingested_at"])
        else:
            row["forecast_market_status"] = ForecastMarketStatus.NOT_RECORDED.value
        win_selected = bool(closeout_row["win_event_matched"])
        spread_selected = bool(closeout_row["spread_event_matched"])
        total_selected = bool(closeout_row["total_event_matched"])
        score_selected = str(product_row["projected_score_status"]) == "available"
        win_prob = _number(closeout_row["home_win_prob"])
        margin = _number(closeout_row["projected_home_margin"])
        total = _number(closeout_row["model_total"])
        proj_home = _number(product_row["projected_home_score"])
        proj_away = _number(product_row["projected_away_score"])

    win_status = _family_status(in_product=in_product, selected=win_selected, value=win_prob)
    if win_status is ForecastStatus.AVAILABLE and tied:
        win_status = ForecastStatus.OUTCOME_TIE
    win_model_brier = win_market_brier = None
    if outcome_value is not None:
        if win_status is ForecastStatus.AVAILABLE and win_prob is not None:
            win_model_brier = (win_prob - outcome_value) ** 2
        if closing_prob is not None:
            win_market_brier = (closing_prob - outcome_value) ** 2
    row |= {
        "win_status": win_status.value,
        "home_win_prob": win_prob if win_status is not ForecastStatus.FORECAST_MISSING else None,
        "win_model_brier": win_model_brier,
        "win_market_brier": win_market_brier,
        "win_closer": _closer(win_model_brier, win_market_brier),
    }

    spread_status = _family_status(in_product=in_product, selected=spread_selected, value=margin)
    spread_model = _difference(
        margin if spread_status is ForecastStatus.AVAILABLE else None, actual_margin
    )
    spread_market = _difference(closing_margin, actual_margin)
    row |= {
        "spread_status": spread_status.value,
        "projected_home_margin": margin if spread_status is ForecastStatus.AVAILABLE else None,
        "spread_model_error": spread_model,
        "spread_market_error": spread_market,
        "spread_closer": _closer(spread_model, spread_market),
    }

    total_status = _family_status(in_product=in_product, selected=total_selected, value=total)
    total_model = _difference(
        total if total_status is ForecastStatus.AVAILABLE else None, actual_total
    )
    total_market = _difference(closing_total, actual_total)
    row |= {
        "total_status": total_status.value,
        "model_total": total if total_status is ForecastStatus.AVAILABLE else None,
        "total_model_error": total_model,
        "total_market_error": total_market,
        "total_closer": _closer(total_model, total_market),
    }

    score_status = _family_status(
        in_product=in_product,
        selected=score_selected and proj_away is not None,
        value=proj_home,
    )
    score_model = score_market = None
    if score_status is ForecastStatus.AVAILABLE:
        assert proj_home is not None and proj_away is not None
        score_model = (abs(proj_away - away_score) + abs(proj_home - home_score)) / 2
    if implied_home is not None and implied_away is not None:
        score_market = (abs(implied_away - away_score) + abs(implied_home - home_score)) / 2
    available_score = score_status is ForecastStatus.AVAILABLE
    row |= {
        "projected_score_status": score_status.value,
        "projected_away_score": proj_away if available_score else None,
        "projected_home_score": proj_home if available_score else None,
        "projected_score_model_abs_error": score_model,
        "projected_score_market_abs_error": score_market,
        "projected_score_closer": _closer(score_model, score_market),
    }
    return row


_STRING_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "game_id",
    "away_team",
    "home_team",
    "product_row_status",
    "forecast_role",
    "forecast_run_id",
    "forecast_market_status",
    "closing_spread_status",
    "closing_total_status",
    "win_status",
    "win_closer",
    "spread_status",
    "spread_closer",
    "total_status",
    "total_closer",
    "projected_score_status",
    "projected_score_closer",
    *STARTER_STRING_COLUMNS,
)
_TIMESTAMP_COLUMNS: Final[tuple[str, ...]] = (
    "kickoff",
    "forecast_generated_at",
    "forecast_market_ingested_at",
)
_BOOLEAN_COLUMNS: Final[tuple[str, ...]] = ("carried", "actual_home_win", *STARTER_BOOLEAN_COLUMNS)


def _typed_frame(records: list[dict[str, object]]) -> DataFrame:
    """Give every column one explicit dtype so the frame and its digest replay exactly."""
    frame = DataFrame.from_records(records)
    for column in frame.columns:
        if column == "week":
            frame[column] = frame[column].astype("int64")
        elif column in _STRING_COLUMNS:
            frame[column] = Series(
                [None if pd.isna(value) else str(value) for value in frame[column]],
                index=frame.index,
                dtype=object,
            )
        elif column in _TIMESTAMP_COLUMNS:
            frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[us, UTC]")
        elif column in _BOOLEAN_COLUMNS:
            frame[column] = frame[column].astype("boolean")
        else:
            frame[column] = pd.to_numeric(frame[column]).astype("float64")
    return frame


# ---------------------------------------------------------------------------
# Week metrics
# ---------------------------------------------------------------------------


def _probability_metrics(probability: Series, outcome: Series) -> ProbabilityMetrics:
    """Score win probabilities with the closeout's formulas."""
    count = len(probability)
    if count == 0:
        return ProbabilityMetrics(0, None, None, None)
    p = probability.astype(float)
    y = outcome.astype(float)
    clipped = p.clip(_PROBABILITY_CLIP, 1 - _PROBABILITY_CLIP)
    return ProbabilityMetrics(
        count,
        float(((p - y) ** 2).mean()),
        float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
        float(((p >= 0.5) == y.astype(bool)).mean()),
    )


def _error_metrics(error: Series) -> ErrorMetrics:
    count = len(error)
    if count == 0:
        return ErrorMetrics(0, None, None, None)
    values = error.astype(float)
    return ErrorMetrics(
        count,
        float(values.abs().mean()),
        float(math.sqrt(float((values**2).mean()))),
        float(values.mean()),
    )


def _closer_counts(closer: Series) -> CloserCounts:
    values = closer.dropna()
    return CloserCounts(
        model=int((values == Closer.MODEL.value).sum()),
        market=int((values == Closer.MARKET.value).sum()),
        tie=int((values == Closer.TIE.value).sum()),
    )


def _score_errors(games: DataFrame, prefix: str) -> Series:
    """Return per-team-score signed errors from one side's two projected scores."""
    if prefix == "projected":
        away, home = games["projected_away_score"], games["projected_home_score"]
    else:
        away, home = games["market_implied_away_score"], games["market_implied_home_score"]
    mask = away.notna() & home.notna()
    return pd.concat(
        [
            (away[mask] - games.loc[mask, "away_score"]),
            (home[mask] - games.loc[mask, "home_score"]),
        ],
        ignore_index=True,
    )


def recap_metrics(games: DataFrame) -> WeekMetrics:
    """Score any set of recap game rows: one week, weeks to date, or one role's rows.

    Each side is scored over its own available rows; a tie has no win outcome.
    """
    decided = games["actual_home_win"].notna()
    model_win = games.loc[decided & games["win_status"].eq(ForecastStatus.AVAILABLE.value)]
    market_win = games.loc[decided & games["closing_home_win_prob"].notna()]
    return WeekMetrics(
        win=WinComparison(
            model=_probability_metrics(
                model_win["home_win_prob"], model_win["actual_home_win"].astype(bool)
            ),
            market=_probability_metrics(
                market_win["closing_home_win_prob"], market_win["actual_home_win"].astype(bool)
            ),
            closer=_closer_counts(games["win_closer"]),
        ),
        spread=PointComparison(
            model=_error_metrics(games["spread_model_error"].dropna()),
            market=_error_metrics(games["spread_market_error"].dropna()),
            closer=_closer_counts(games["spread_closer"]),
        ),
        total=PointComparison(
            model=_error_metrics(games["total_model_error"].dropna()),
            market=_error_metrics(games["total_market_error"].dropna()),
            closer=_closer_counts(games["total_closer"]),
        ),
        projected_score=PointComparison(
            model=_error_metrics(_score_errors(games, "projected")),
            market=_error_metrics(_score_errors(games, "market")),
            closer=_closer_counts(games["projected_score_closer"]),
        ),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_weekly_recap(
    *,
    season: str,
    week: int,
    schedule: DataFrame,
    product: DataFrame,
    forecast_events: DataFrame,
    games: DataFrame,
    player_game_logs: DataFrame,
    sidecar_captures: Mapping[str, SidecarCapture],
    built_at: datetime,
) -> BuiltWeeklyRecap:
    """Build one week's recap from the scoped schedule and the selected product, without I/O.

    ``schedule`` is the week's regular-season rows of the raw games file; ``product`` is the
    week's selected product; ``games`` is the cleaned games file; ``player_game_logs`` names who
    passed for each team; ``sidecar_captures`` maps a run id to the verified capture of the
    sidecar that run was bound to (the starter record of rows that carry none). Refuses a week
    with an unplayed game, a product row for an unscheduled game, a selected component whose
    forecast event does not match, and a score the raw and cleaned files disagree on.
    """
    _utc(built_at, "built_at")
    _require_columns(schedule, _SCHEDULE_COLUMNS, label="Schedule")
    if schedule.empty:
        raise WeeklyRecapRefusalError(
            f"No regular-season games are scheduled for {season} week {week}."
        )
    if schedule["game_id"].astype(str).duplicated().any():
        raise ValueError("Schedule contains duplicate game IDs.")
    unplayed = unplayed_game_ids(schedule)
    if unplayed:
        raise WeeklyRecapRefusalError(
            f"{len(unplayed)} of {len(schedule)} games have no result yet: " + ", ".join(unplayed)
        )

    closeout = close_live_forecasts(product=product, forecast_events=forecast_events, games=games)
    if closeout.season != season or closeout.week != week:
        raise ValueError(
            f"Weekly product is {closeout.season} week {closeout.week}, not {season} week {week}."
        )
    scheduled_ids = set(schedule["game_id"].astype(str))
    unscheduled = sorted(set(product["game_id"].astype(str)) - scheduled_ids)
    if unscheduled:
        raise WeeklyRecapRefusalError(
            "The selected product holds games that are not scheduled: " + ", ".join(unscheduled)
        )
    unmatched = sorted(
        {
            *closeout.missing_win_event_game_ids,
            *closeout.missing_spread_event_game_ids,
            *closeout.missing_total_event_game_ids,
        }
    )
    if unmatched:
        raise WeeklyRecapRefusalError(
            "Selected components cite no matching forecast event: " + ", ".join(unmatched)
        )
    outcomes = _outcomes(games, schedule)

    product_run_id = closeout.product_run_id
    market_recorded = {"spread_line", "total_line", "ingested_at"} <= set(product.columns)
    product_rows = {str(row["game_id"]): row for _, row in product.iterrows()}
    closeout_rows = {str(row["game_id"]): row for _, row in closeout.reconciliation.iterrows()}
    ordered = schedule.assign(
        _kickoff=[
            _schedule_kickoff_utc(day, time)
            for day, time in zip(schedule["gameday"], schedule["gametime"], strict=True)
        ]
    ).sort_values(["_kickoff", "game_id"])
    season_int = int(_single(schedule, "season"))
    passers = week_passers(player_game_logs, season_int=season_int, week=week)
    starters = starter_columns(
        ordered,
        product,
        captures=sidecar_captures,
        passers=passers,
        names=player_names(player_game_logs, season_int=season_int, week=week),
    )
    records = [
        _game_row(
            game,
            cast(Series, outcomes.loc[str(game["game_id"])]),
            product_rows.get(str(game["game_id"])),
            closeout_rows.get(str(game["game_id"])),
            season=season,
            week=week,
            product_run_id=product_run_id,
            market_recorded=market_recorded,
        )
        | starters[str(game["game_id"])]
        for _, game in ordered.iterrows()
    ]
    frame = _typed_frame(records)
    frame_digest = frame_content_digest(frame)

    role = closeout.win_role or closeout.total_role
    in_product = frame["product_row_status"].eq(ProductRowStatus.IN_PRODUCT.value)
    recap = WeeklyRecap(
        schema_version=WEEKLY_RECAP_SCHEMA_VERSION,
        recap_id="",
        built_at=built_at,
        season=season,
        week=week,
        product_id=closeout.product_id,
        product_run_id=product_run_id,
        carried_run_ids=tuple(sorted(cited_run_ids(product) - {product_run_id})),
        product_schema_version=int(_single(product, "product_schema_version")),
        forecast_role=None if role is None else role.value,
        schedule_digest=frame_content_digest(
            _canonical_rows(schedule, _SCHEDULE_COLUMNS, key="game_id")
        ),
        outcome_digest=frame_content_digest(
            _canonical_rows(outcomes.reset_index(), _OUTCOME_COLUMNS, key="GAME_ID")
        ),
        passer_digest=frame_content_digest(passers.astype(object)),
        starter_capture_digests=tuple(
            sorted({capture.digest for capture in sidecar_captures.values()})
        ),
        conventions=RECAP_CONVENTIONS,
        counts=RecapCounts(
            scheduled_games=len(frame),
            in_product=int(in_product.sum()),
            not_in_product=int((~in_product).sum()),
            carried_rows=int(frame["carried"].fillna(False).sum()),
            outcome_ties=int(frame["actual_home_win"].isna().sum()),
        ),
        metrics=recap_metrics(frame),
        starters=starter_summary(frame),
        games=EvaluationFrameReference(
            artifact=f"schema={WEEKLY_RECAP_SCHEMA_VERSION}/games/{frame_digest}.parquet",
            row_count=len(frame),
            columns=tuple(str(column) for column in frame.columns),
            content_digest=frame_digest,
        ),
    )
    recap = replace(recap, recap_id=weekly_recap_id(recap))
    validate_weekly_recap(recap)
    return BuiltWeeklyRecap(recap=recap, games=frame)


def _canonical_rows(frame: DataFrame, columns: tuple[str, ...], *, key: str) -> DataFrame:
    """Return the identity-relevant columns in a stable order and dtype."""
    rows = frame.loc[:, list(columns)].sort_values(key).reset_index(drop=True)
    return rows.astype(object)


# ---------------------------------------------------------------------------
# Identity, validation and JSON
# ---------------------------------------------------------------------------


def weekly_recap_payload(recap: WeeklyRecap) -> dict[str, object]:
    """Return the recap manifest as JSON-ready values."""
    payload: dict[str, object] = dict(asdict(recap))
    payload["built_at"] = recap.built_at.isoformat()
    return payload


def weekly_recap_id(recap: WeeklyRecap) -> str:
    """Return the SHA-256 of the recap's content, excluding ``built_at`` and the id itself."""
    payload = weekly_recap_payload(recap)
    del payload["built_at"], payload["recap_id"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def validate_weekly_recap(recap: WeeklyRecap) -> None:
    """Validate the schema version, timestamps, frame reference and identity."""
    if recap.schema_version != WEEKLY_RECAP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported weekly recap schema version {recap.schema_version}; "
            f"expected {WEEKLY_RECAP_SCHEMA_VERSION}."
        )
    _utc(recap.built_at, "built_at")
    if recap.week < 1 or not recap.season.strip():
        raise ValueError("Weekly recap needs a season and a week of at least 1.")
    artifact = recap.games.artifact
    if not artifact.strip() or artifact.startswith("/") or ".." in artifact.split("/"):
        raise ValueError("Weekly recap game frame must be a safe relative path.")
    if recap.games.row_count != recap.counts.scheduled_games:
        raise ValueError("Weekly recap game frame must hold one row per scheduled game.")
    if recap.recap_id != weekly_recap_id(recap):
        raise ValueError("Weekly recap ID does not match its content.")


def weekly_recap_from_payload(payload: object) -> WeeklyRecap:
    """Parse a stored manifest, requiring exactly the current schema's fields."""
    recap = cast(WeeklyRecap, _decode(WeeklyRecap, payload, "recap"))
    validate_weekly_recap(recap)
    return recap


def _decode(kind: Any, value: object, label: str) -> object:  # noqa: ANN401
    if is_dataclass(kind):
        if not isinstance(value, dict):
            raise ValueError(f"Weekly recap {label} must be a JSON object.")
        names = {field.name for field in fields(kind)}
        if set(value) != names:
            raise ValueError(f"Weekly recap {label} fields do not match the schema.")
        hints = get_type_hints(kind)
        return kind(
            **{name: _decode(hints[name], value[name], f"{label}.{name}") for name in names}
        )
    if kind is datetime:
        if not isinstance(value, str):
            raise ValueError(f"Weekly recap {label} must be an ISO timestamp.")
        return datetime.fromisoformat(value)
    if get_origin(kind) is tuple:
        if not isinstance(value, list):
            raise ValueError(f"Weekly recap {label} must be a list.")
        return tuple(value)
    return value


def _utc(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")


__all__ = [
    "RECAP_CONVENTIONS",
    "WEEKLY_RECAP_SCHEMA_VERSION",
    "BuiltWeeklyRecap",
    "Closer",
    "ClosingLineStatus",
    "ForecastMarketStatus",
    "ForecastStatus",
    "ProductRowStatus",
    "WeekMetrics",
    "WeeklyRecap",
    "WeeklyRecapRefusalError",
    "build_weekly_recap",
    "recap_metrics",
    "unplayed_game_ids",
    "validate_weekly_recap",
    "weekly_recap_from_payload",
    "weekly_recap_id",
    "weekly_recap_payload",
]

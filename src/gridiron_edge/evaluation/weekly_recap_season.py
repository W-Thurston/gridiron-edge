# src/gridiron_edge/evaluation/weekly_recap_season.py

"""Roll a season's selected weekly recaps up into week-by-week and season-to-date aggregates.

The season recap is built from the selected weekly recaps' game frames only (``DECISIONS.md``
D86), never from products: each week's values come from the recap's own metrics function over
that week's rows, so a week reads exactly as its recap does. Season-to-date values pool every
game through the week (they are not an average of weekly values); the live-only scope keeps the
rows whose forecast role is ``live``.

Beside the model and market metrics it carries, per family, the per-game paired model-minus-market
difference with its standard error and an approximate 95% interval, and model and market
reliability over ten fixed equal-width bins of the home win probability with counts and ECE. A
finished week with no selected recap is a row with status ``not_built``.

Each week also carries its edge-evidence status and recorded-wager summary from its recap; the
season adds those wagers up and counts the ledger's bets without a ``game_id`` placed in the
season's window (March 1 of its first year to March 1 of the next, UTC) as unattributed
(``DECISIONS.md`` D89).

The identity is the SHA-256 of the content excluding ``built_at``; the content names every weekly
recap id it read, so a newly selected weekly recap gives a new season identity.

Public API
----------
build_season_recap          selected weekly recaps, finished weeks, bet ledger -> SeasonRecap
season_recap_id             season recap -> content identity excluding built_at
validate_season_recap       season recap -> raises on an invalid manifest
season_recap_from_payload / season_recap_payload   JSON round trip
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
import math
import types
from typing import Any, Final, Union, cast, get_args, get_origin, get_type_hints

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.metrics import expected_calibration_error, reliability_bins
from gridiron_edge.evaluation.weekly_recap import (
    RECAP_CONVENTIONS,
    WEEKLY_RECAP_SCHEMA_VERSION,
    ForecastStatus,
    RecapConventions,
    WeeklyRecap,
    WeeklyRecapRefusalError,
    WeekMetrics,
    recap_metrics,
)
from gridiron_edge.evaluation.weekly_recap_markets import (
    WagerSummary,
    ledger_digest,
    sum_wager_summaries,
    wager_summary,
)
from gridiron_edge.evaluation.weekly_recap_starters import StarterSummary, starter_summary

SEASON_RECAP_SCHEMA_VERSION: Final[int] = 2
# An unattributed bet belongs to the season whose window holds its placement time.
_SEASON_WINDOW_MONTH: Final[int] = 3
# Declared before any forecast probability was read: ten equal-width bins, the edges of
# ``expected_calibration_error``'s default.
CALIBRATION_BIN_COUNT: Final[int] = 10
# Normal quantile of the approximate 95% interval around a paired mean difference.
INTERVAL_Z: Final[float] = 1.96


class SeasonWeekStatus(StrEnum):
    """Whether a finished week of the season has a selected weekly recap."""

    RECAPPED = "recapped"
    NOT_BUILT = "not_built"


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """The mean per-game model-minus-market difference over games both sides forecast.

    Negative means the model's error was lower. The standard error and interval need two games.
    """

    count: int
    mean: float | None
    standard_error: float | None
    interval_low: float | None
    interval_high: float | None


@dataclass(frozen=True, slots=True)
class PairedDifferences:
    """Win Brier and absolute spread, total and per-team-score error, model minus market."""

    win: PairedDifference
    spread: PairedDifference
    total: PairedDifference
    projected_score: PairedDifference


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One fixed probability bin: its decided games, mean forecast and observed home-win rate."""

    lower: float
    upper: float
    count: int
    mean_forecast: float | None
    observed_rate: float | None


@dataclass(frozen=True, slots=True)
class SideCalibration:
    """One side's reliability over every bin, and its ECE over ``count`` decided games."""

    count: int
    ece: float | None
    bins: tuple[ReliabilityBin, ...]


@dataclass(frozen=True, slots=True)
class Calibration:
    """Model and closing-market home win probability on the same bins."""

    model: SideCalibration
    market: SideCalibration


@dataclass(frozen=True, slots=True)
class ScopeSummary:
    """The season so far over one scope (every row, or the live rows)."""

    weeks: int
    games: int
    metrics: WeekMetrics
    paired: PairedDifferences
    calibration: Calibration
    starters: StarterSummary


@dataclass(frozen=True, slots=True)
class SeasonWeek:
    """One finished week: its own values and the season to date through it.

    Every metric is ``None`` when the week has no selected recap; a live metric scores no game
    when the week has no live row.
    """

    week: int
    status: SeasonWeekStatus
    recap_id: str | None
    forecast_role: str | None
    games: int
    live_games: int
    metrics: WeekMetrics | None
    live_metrics: WeekMetrics | None
    to_date: WeekMetrics | None
    live_to_date: WeekMetrics | None
    edge_status: str | None
    wagers: WagerSummary | None


@dataclass(frozen=True, slots=True)
class SeasonWagers:
    """Recorded wagers over the recapped weeks, and the season's bets without a game."""

    recapped_weeks: WagerSummary
    unattributed: WagerSummary
    unattributed_digest: str


@dataclass(frozen=True, slots=True)
class SeasonConventions:
    """The conventions the season's aggregates are stated in."""

    recap: RecapConventions
    season_to_date: str
    live_only: str
    calibration_bins: str
    paired_difference: str
    interval: str
    interval_z: float
    unattributed_wagers: str


SEASON_CONVENTIONS: Final[SeasonConventions] = SeasonConventions(
    recap=RECAP_CONVENTIONS,
    season_to_date="every recapped game through the week, pooled",
    live_only="rows whose forecast role is live",
    calibration_bins=(
        f"{CALIBRATION_BIN_COUNT} equal-width bins of the home win probability, [lower, upper) "
        "with 1.0 in the last bin; decided games only; ECE weights each bin's "
        "|mean forecast - observed rate| by its share of games"
    ),
    paired_difference=(
        "per game both sides forecast, model minus market: win Brier, and absolute spread, total "
        "and per-team-score error; negative means the model's error was lower"
    ),
    interval=(
        "mean +/- interval_z x standard error (sample standard deviation with ddof 1 over "
        "sqrt(n)): an approximate 95% interval by the normal approximation, not a verdict"
    ),
    interval_z=INTERVAL_Z,
    unattributed_wagers=(
        "ledger bets without a game_id placed from March 1 of the season's first year to March 1 "
        "of the next (UTC)"
    ),
)


@dataclass(frozen=True, slots=True)
class SeasonRecap:
    """Manifest of one immutable season recap."""

    schema_version: int
    season_recap_id: str
    built_at: datetime
    season: str
    recap_schema_version: int
    weekly_recap_ids: tuple[str, ...]
    conventions: SeasonConventions
    weeks: tuple[SeasonWeek, ...]
    all_games: ScopeSummary
    live_games: ScopeSummary
    wagers: SeasonWagers


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def _paired(model: Series, market: Series) -> PairedDifference:
    both = model.notna() & market.notna()
    difference = (model[both] - market[both]).astype(float)
    count = len(difference)
    if count == 0:
        return PairedDifference(0, None, None, None, None)
    mean = float(difference.mean())
    if count < 2:
        return PairedDifference(count, mean, None, None, None)
    error = float(difference.std(ddof=1)) / math.sqrt(count)
    return PairedDifference(
        count, mean, error, mean - INTERVAL_Z * error, mean + INTERVAL_Z * error
    )


def _paired_differences(games: DataFrame) -> PairedDifferences:
    return PairedDifferences(
        win=_paired(games["win_model_brier"], games["win_market_brier"]),
        spread=_paired(games["spread_model_error"].abs(), games["spread_market_error"].abs()),
        total=_paired(games["total_model_error"].abs(), games["total_market_error"].abs()),
        projected_score=_paired(
            games["projected_score_model_abs_error"], games["projected_score_market_abs_error"]
        ),
    )


def _optional(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def _side_calibration(probability: Series, outcome: Series) -> SideCalibration:
    p = probability.astype(float).reset_index(drop=True)
    y = outcome.astype(float).reset_index(drop=True)
    bins = reliability_bins(p, y, n_bins=CALIBRATION_BIN_COUNT)
    return SideCalibration(
        count=len(p),
        ece=(
            float(expected_calibration_error(p, y, n_bins=CALIBRATION_BIN_COUNT))
            if len(p)
            else None
        ),
        bins=tuple(
            ReliabilityBin(
                lower=float(row["bin_lower"]),
                upper=float(row["bin_upper"]),
                count=int(row["n_games"]),
                mean_forecast=_optional(row["mean_predicted_prob"]),
                observed_rate=_optional(row["observed_rate"]),
            )
            for row in bins.to_dict("records")
        ),
    )


def _calibration(games: DataFrame) -> Calibration:
    """Score the rows ``recap_metrics`` scores: decided games with each side's probability."""
    decided = games["actual_home_win"].notna()
    model = games.loc[decided & games["win_status"].eq(ForecastStatus.AVAILABLE.value)]
    market = games.loc[decided & games["closing_home_win_prob"].notna()]
    return Calibration(
        model=_side_calibration(model["home_win_prob"], model["actual_home_win"]),
        market=_side_calibration(market["closing_home_win_prob"], market["actual_home_win"]),
    )


def _live(games: DataFrame) -> DataFrame:
    return games.loc[games["forecast_role"].eq(ForecastRole.LIVE.value)]


def _scope_summary(games: DataFrame) -> ScopeSummary:
    return ScopeSummary(
        weeks=int(games["week"].nunique()),
        games=len(games),
        metrics=recap_metrics(games),
        paired=_paired_differences(games),
        calibration=_calibration(games),
        starters=starter_summary(games),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def unattributed_bets(bets: DataFrame, *, season: str) -> DataFrame:
    """Return the ledger's bets without a ``game_id`` placed in the season's window."""
    if bets.empty:
        return bets
    start_year = int(season[:4])
    start = pd.Timestamp(datetime(start_year, _SEASON_WINDOW_MONTH, 1, tzinfo=UTC))
    end = pd.Timestamp(datetime(start_year + 1, _SEASON_WINDOW_MONTH, 1, tzinfo=UTC))
    placed = pd.to_datetime(bets["placed_at"], utc=True)
    return bets.loc[bets["game_id"].isna() & placed.ge(start) & placed.lt(end)]


def build_season_recap(
    *,
    season: str,
    recaps: Sequence[tuple[WeeklyRecap, DataFrame]],
    finished_weeks: Sequence[int],
    bets: DataFrame,
    built_at: datetime,
) -> SeasonRecap:
    """Roll the season's selected weekly recaps up, without I/O.

    ``recaps`` are the season's selected recaps with their verified game frames;
    ``finished_weeks`` are the weeks whose every scheduled game has a result; ``bets`` is the bet
    ledger, read only for the season's unattributed bets (each week's wagers come from its
    recap). A finished week without a recap is ``not_built``. Refuses a season with no recap, a
    recap of another season, two recaps of one week, and a recap whose metrics do not match its
    own game frame.
    """
    _utc(built_at, "built_at")
    if not recaps:
        raise WeeklyRecapRefusalError(f"No weekly recap is selected for {season}.")
    by_week: dict[int, tuple[WeeklyRecap, DataFrame]] = {}
    for recap, games in recaps:
        if recap.season != season:
            raise ValueError(f"Weekly recap {recap.recap_id} covers {recap.season}, not {season}.")
        if recap.week in by_week:
            raise ValueError(f"Two weekly recaps are given for {season} week {recap.week}.")
        if recap_metrics(games) != recap.metrics:
            raise ValueError(
                f"Weekly recap {recap.recap_id} metrics do not match its own game frame."
            )
        by_week[recap.week] = (recap, games)

    pooled = pd.concat(
        [by_week[week][1] for week in sorted(by_week)], ignore_index=True
    ).reset_index(drop=True)
    weeks: list[SeasonWeek] = []
    for week in sorted(set(by_week) | {int(week) for week in finished_weeks}):
        if week not in by_week:
            weeks.append(
                SeasonWeek(
                    week=week,
                    status=SeasonWeekStatus.NOT_BUILT,
                    recap_id=None,
                    forecast_role=None,
                    games=0,
                    live_games=0,
                    metrics=None,
                    live_metrics=None,
                    to_date=None,
                    live_to_date=None,
                    edge_status=None,
                    wagers=None,
                )
            )
            continue
        recap, games = by_week[week]
        to_date = pooled.loc[pooled["week"] <= week]
        weeks.append(
            SeasonWeek(
                week=week,
                status=SeasonWeekStatus.RECAPPED,
                recap_id=recap.recap_id,
                forecast_role=recap.forecast_role,
                games=len(games),
                live_games=len(_live(games)),
                metrics=recap.metrics,
                live_metrics=recap_metrics(_live(games)),
                to_date=recap_metrics(to_date),
                live_to_date=recap_metrics(_live(to_date)),
                edge_status=recap.edges.status.value,
                wagers=recap.wagers.total,
            )
        )

    unattributed = unattributed_bets(bets, season=season)

    season_recap = SeasonRecap(
        schema_version=SEASON_RECAP_SCHEMA_VERSION,
        season_recap_id="",
        built_at=built_at,
        season=season,
        recap_schema_version=WEEKLY_RECAP_SCHEMA_VERSION,
        weekly_recap_ids=tuple(by_week[week][0].recap_id for week in sorted(by_week)),
        conventions=SEASON_CONVENTIONS,
        weeks=tuple(weeks),
        all_games=_scope_summary(pooled),
        live_games=_scope_summary(_live(pooled)),
        wagers=SeasonWagers(
            recapped_weeks=sum_wager_summaries(
                by_week[week][0].wagers.total for week in sorted(by_week)
            ),
            unattributed=wager_summary(unattributed),
            unattributed_digest=ledger_digest(unattributed),
        ),
    )
    season_recap = replace(season_recap, season_recap_id=season_recap_id(season_recap))
    validate_season_recap(season_recap)
    return season_recap


# ---------------------------------------------------------------------------
# Identity, validation and JSON
# ---------------------------------------------------------------------------


def season_recap_payload(recap: SeasonRecap) -> dict[str, object]:
    """Return the season recap manifest as JSON-ready values."""
    payload: dict[str, object] = dict(asdict(recap))
    payload["built_at"] = recap.built_at.isoformat()
    return payload


def season_recap_id(recap: SeasonRecap) -> str:
    """Return the SHA-256 of the content, excluding ``built_at`` and the id itself."""
    payload = season_recap_payload(recap)
    del payload["built_at"], payload["season_recap_id"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def validate_season_recap(recap: SeasonRecap) -> None:
    """Validate the schema versions, timestamps, weeks and identity."""
    if recap.schema_version != SEASON_RECAP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported season recap schema version {recap.schema_version}; "
            f"expected {SEASON_RECAP_SCHEMA_VERSION}."
        )
    if recap.recap_schema_version != WEEKLY_RECAP_SCHEMA_VERSION:
        raise ValueError("Season recap was rolled up from another weekly recap schema.")
    _utc(recap.built_at, "built_at")
    if not recap.season.strip():
        raise ValueError("Season recap needs a season.")
    weeks = [week.week for week in recap.weeks]
    if weeks != sorted(set(weeks)) or any(week < 1 for week in weeks):
        raise ValueError("Season recap weeks must be distinct, ascending and at least 1.")
    recapped = tuple(
        week.recap_id
        for week in recap.weeks
        if week.status is SeasonWeekStatus.RECAPPED and week.recap_id is not None
    )
    if recapped != recap.weekly_recap_ids or not recapped:
        raise ValueError("Season recap weekly recap ids must match its recapped weeks.")
    if recap.season_recap_id != season_recap_id(recap):
        raise ValueError("Season recap ID does not match its content.")


def season_recap_from_payload(payload: object) -> SeasonRecap:
    """Parse a stored manifest, requiring exactly the current schema's fields."""
    recap = cast(SeasonRecap, _decode(SeasonRecap, payload, "season recap"))
    validate_season_recap(recap)
    return recap


def _decode(kind: Any, value: object, label: str) -> object:  # noqa: ANN401, PLR0911
    origin = get_origin(kind)
    if origin in (Union, types.UnionType):
        options = [option for option in get_args(kind) if option is not type(None)]
        if value is None:
            return None
        (option,) = options
        return _decode(option, value, label)
    if is_dataclass(kind):
        if not isinstance(value, dict):
            raise ValueError(f"Season recap {label} must be a JSON object.")
        names = {field.name for field in fields(kind)}
        if set(value) != names:
            raise ValueError(f"Season recap {label} fields do not match the schema.")
        hints = get_type_hints(kind)
        return kind(
            **{name: _decode(hints[name], value[name], f"{label}.{name}") for name in names}
        )
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"Season recap {label} must be a list.")
        (item, _) = get_args(kind)
        return tuple(_decode(item, element, f"{label}[]") for element in value)
    if kind is datetime:
        if not isinstance(value, str):
            raise ValueError(f"Season recap {label} must be an ISO timestamp.")
        return datetime.fromisoformat(value)
    if isinstance(kind, type) and issubclass(kind, StrEnum):
        return kind(value)
    return value


def _utc(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")


__all__ = [
    "CALIBRATION_BIN_COUNT",
    "INTERVAL_Z",
    "SEASON_CONVENTIONS",
    "SEASON_RECAP_SCHEMA_VERSION",
    "Calibration",
    "PairedDifference",
    "PairedDifferences",
    "ReliabilityBin",
    "ScopeSummary",
    "SeasonRecap",
    "SeasonWagers",
    "SeasonWeek",
    "SeasonWeekStatus",
    "SideCalibration",
    "build_season_recap",
    "season_recap_from_payload",
    "season_recap_id",
    "season_recap_payload",
    "unattributed_bets",
    "validate_season_recap",
]

# src/gridiron_edge/ingest/odds/nflverse_schedule.py

"""Adapt nflverse rich-schedule market fields to the generic odds contract."""

from __future__ import annotations

from datetime import timedelta
import math
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.ingest.odds.store import QUOTE_COLUMNS

NFLVERSE_PROVIDER: Final[str] = "nflverse"
MARKET_COLUMNS: Final[tuple[str, ...]] = QUOTE_COLUMNS

_REQUIRED_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "over_odds",
    "under_odds",
    "source",
    "ingested_at",
)

_SIDE_ORDER: Final[tuple[tuple[str, str], ...]] = (
    ("moneyline", "away"),
    ("moneyline", "home"),
    ("spread", "away"),
    ("spread", "home"),
    ("total", "over"),
    ("total", "under"),
)


def _require_columns(schedule: DataFrame) -> None:
    """Require rich-schedule identity, provenance, and market columns."""
    missing = sorted(set(_REQUIRED_SCHEDULE_COLUMNS) - set(schedule.columns))
    if missing:
        raise ValueError("Rich schedule is missing required market columns: " + ", ".join(missing))


def _validate_identity(schedule: DataFrame) -> None:
    """Validate canonical schedule identity within the requested scope."""
    for column in ("season", "game_id", "away_team", "home_team"):
        values = schedule[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Rich schedule column {column!r} must be nonempty.")

    duplicated = schedule["game_id"].duplicated(keep=False)
    if duplicated.any():
        game_ids = sorted(schedule.loc[duplicated, "game_id"].astype(str).unique().tolist())
        raise ValueError("Rich schedule contains duplicate game IDs: " + ", ".join(game_ids))


def _validate_source(schedule: DataFrame) -> None:
    """Require the rich schedule to identify nflverse as its source."""
    sources = {str(value).strip() for value in schedule["source"].dropna() if str(value).strip()}
    if sources != {"nflverse"}:
        raise ValueError("Rich schedule market adapter requires source 'nflverse'.")


def _normalize_ingested_at(schedule: DataFrame) -> pd.Timestamp:
    """Return the single timezone-aware UTC ingestion timestamp."""
    timestamps: list[pd.Timestamp] = []
    for value in schedule["ingested_at"]:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("Rich schedule ingested_at must contain valid datetimes.")
        if timestamp.tzinfo is None:
            raise ValueError("Rich schedule ingested_at must be timezone-aware UTC.")
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("Rich schedule ingested_at must use UTC.")
        timestamps.append(timestamp)

    unique = {timestamp.value for timestamp in timestamps}
    if len(unique) != 1:
        raise ValueError("Requested rich schedule scope must have one ingestion timestamp.")
    return timestamps[0]


def _optional_float(value: object) -> float | None:
    """Return a nullable finite numeric market value."""
    if value is None or value is pd.NA:
        return None

    if not isinstance(value, int | float):
        raise ValueError("Schedule market values must be numeric or null.")

    numeric = float(value)
    if math.isnan(numeric):
        return None
    if not math.isfinite(numeric):
        raise ValueError("Schedule market values must be finite.")

    return numeric


def _market_values(
    row: Series,
    market: str,
    side: str,
) -> tuple[float | None, float | None]:
    """Return odds and line values for one canonical market side."""
    if market == "moneyline":
        column = "away_moneyline" if side == "away" else "home_moneyline"
        return _optional_float(row[column]), None

    if market == "spread":
        odds_column = "away_spread_odds" if side == "away" else "home_spread_odds"
        home_line = _optional_float(row["spread_line"])
        line = None if home_line is None else -home_line if side == "away" else home_line
        return _optional_float(row[odds_column]), line

    odds_column = "over_odds" if side == "over" else "under_odds"
    return _optional_float(row[odds_column]), _optional_float(row["total_line"])


def adapt_nflverse_schedule_markets(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
) -> DataFrame:
    """Convert rich nflverse schedule markets to canonical long odds rows.

    ``spread_line`` follows nflverse's home-team orientation. The home side
    retains that line and the away side receives its additive inverse. Six
    rows are emitted per scheduled game so incomplete market sides remain
    explicit through nullable odds and line values.
    """
    _require_columns(schedule)
    if not season.strip():
        raise ValueError("season must not be empty.")
    if week < 1:
        raise ValueError("week must be at least 1.")

    scoped = schedule.loc[
        (schedule["season"].astype(str) == season) & (schedule["week"] == week),
        :,
    ].copy()
    if scoped.empty:
        return DataFrame(columns=list(MARKET_COLUMNS))

    _validate_identity(scoped)
    _validate_source(scoped)
    fetched_at = _normalize_ingested_at(scoped)

    rows: list[dict[str, object]] = []
    for _, schedule_row in scoped.iterrows():
        base: dict[str, object] = {
            "fetched_at": fetched_at,
            "provider": NFLVERSE_PROVIDER,
            "provider_event_id": None,
            "sportsbook": None,
            "sportsbook_updated_at": None,
            "commence_time": None,
            "is_live": False,
            "season": season,
            "week": week,
            "game_id": str(schedule_row["game_id"]),
            "game_date": schedule_row["game_date"],
            "away_team": str(schedule_row["away_team"]),
            "home_team": str(schedule_row["home_team"]),
        }
        for market, side in _SIDE_ORDER:
            odds, line = _market_values(schedule_row, market, side)
            rows.append(
                {
                    **base,
                    "market": market,
                    "side": side,
                    "odds": odds,
                    "line": line,
                }
            )

    result = DataFrame(rows, columns=list(MARKET_COLUMNS))
    result["fetched_at"] = pd.to_datetime(result["fetched_at"], utc=True)
    result["odds"] = pd.to_numeric(result["odds"], errors="coerce")
    result["line"] = pd.to_numeric(result["line"], errors="coerce")
    return result

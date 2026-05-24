# src/gridiron_edge/ingest/odds/store.py

"""Odds storage utilities — append-only ledger and current snapshot.

All odds are stored in long format (one row per market/side/game/pull),
which makes multi-sportsbook comparison and time-series line movement
analysis straightforward.

Storage layout:
    data/odds/dk_odds_log.parquet    — full historical ledger (all pulls)
    data/odds/dk_odds_current.parquet — latest pull only (for viz)

Schema (dk_odds_log):
    fetched_at      datetime64[ns]  UTC timestamp of the pull
    sportsbook      str             "draftkings" | "fanduel" | ...
    season          str             "2026-2027"
    week            int             NFL week number
    game_id         str             "2026_01_KC_LAC" (canonical GAME_ID)
    game_date       str             "2026-09-05"
    away_team       str             long name ("Kansas City Chiefs")
    home_team       str             long name ("Los Angeles Chargers")
    market          str             "moneyline" | "spread" | "total"
    side            str             "away" | "home" | "over" | "under"
    odds            float           American odds (e.g. -110.0, +150.0)
    line            float           spread/total value; NaN for moneyline
"""

from __future__ import annotations

import datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

# Column order for the ledger — enforced on every write.
_LEDGER_COLUMNS: list[str] = [
    "fetched_at",
    "sportsbook",
    "season",
    "week",
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "market",
    "side",
    "odds",
    "line",
]


def _odds_dir(repo: Path | None = None) -> Path:
    """Return the odds data directory, creating it if needed.

    Args:
        repo: Repository root path. Defaults to ``get_settings().repo_root``.

    Returns:
        Absolute path to ``data/odds/``.
    """
    root: Path = repo or get_settings().repo_root
    path: Path = root / "data" / "odds"
    path.mkdir(parents=True, exist_ok=True)
    return path


def wide_to_long(
    df_wide: pd.DataFrame,
    *,
    sportsbook: str,
    season: str,
    week: int,
    fetched_at: datetime.datetime | None = None,
) -> pd.DataFrame:
    """Convert a wide per-team odds DataFrame to the long ledger format.

    The wide format from ``pull_dk_sportsbook_odds_refactored()`` has one
    row per team per game with columns like ``moneyline``, ``spread_value``,
    ``spread_odds``, ``total_OU_value``, etc.  This function melts it into
    one row per market per side per game.

    Args:
        df_wide: DataFrame from ``pull_dk_sportsbook_odds_refactored()``.
            Expected columns: ``team``, ``opponent``, ``location``,
            ``start_time``, ``event_id``, ``moneyline``, ``spread_value``,
            ``spread_odds``, ``total_OU_value``, ``over_total_odds``,
            ``under_total_odds``.
        sportsbook: Sportsbook identifier (e.g. ``"draftkings"``).
        season: NFL season label (e.g. ``"2026-2027"``).
        week: NFL week number.
        fetched_at: UTC timestamp of the pull. Defaults to ``datetime.now(UTC)``.

    Returns:
        Long-format DataFrame with columns matching ``_LEDGER_COLUMNS``.
    """
    ts = fetched_at or datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None)

    rows: list[dict] = []

    # Deduplicate to one row per game (not per team)
    games = (
        df_wide.loc[df_wide["location"] == 1, :]
        .copy()
        .rename(columns={"team": "home_team_raw", "opponent": "away_team_raw"})
    )
    if games.empty:
        # Fallback: group by event_id and take first
        games: DataFrame = df_wide.groupby("event_id", sort=False).first().reset_index()

    away_rows = df_wide[df_wide["location"] == 0].set_index("event_id")
    home_rows = df_wide[df_wide["location"] == 1].set_index("event_id")

    for event_id, home_row in home_rows.iterrows():
        if event_id not in away_rows.index:
            continue
        away_row = away_rows.loc[[event_id]].iloc[0]  # type: ignore[index]

        home_team = str(home_row["team"])
        away_team = str(away_row["team"])
        start_time = home_row.get("start_time")
        game_date: str = (
            pd.Timestamp(start_time).strftime("%Y-%m-%d") if pd.notna(start_time) else ""
        )

        # Build a canonical game_id from start_time + teams
        # We don't have the full GAME_ID from DK so we construct a best-effort key.
        # Format: YYYY_WW_AWAYSHORT_HOMESHORT — will be matched to canonical IDs downstream.
        game_id: str = f"{season[:4]}_{week:02d}_{event_id}"

        base: dict[str, Any | int | str] = {
            "fetched_at": ts,
            "sportsbook": sportsbook,
            "season": season,
            "week": week,
            "game_id": game_id,
            "game_date": game_date,
            "away_team": away_team,
            "home_team": home_team,
        }

        # Moneyline
        if pd.notna(away_row.get("moneyline")):
            rows.append(
                {
                    **base,
                    "market": "moneyline",
                    "side": "away",
                    "odds": float(away_row["moneyline"]),
                    "line": float("nan"),
                }
            )
        if pd.notna(home_row.get("moneyline")):
            rows.append(
                {
                    **base,
                    "market": "moneyline",
                    "side": "home",
                    "odds": float(home_row["moneyline"]),
                    "line": float("nan"),
                }
            )

        # Spread
        if pd.notna(away_row.get("spread_value")) and pd.notna(away_row.get("spread_odds")):
            rows.append(
                {
                    **base,
                    "market": "spread",
                    "side": "away",
                    "odds": float(away_row["spread_odds"]),
                    "line": float(away_row["spread_value"]),
                }
            )
        if pd.notna(home_row.get("spread_value")) and pd.notna(home_row.get("spread_odds")):
            rows.append(
                {
                    **base,
                    "market": "spread",
                    "side": "home",
                    "odds": float(home_row["spread_odds"]),
                    "line": float(home_row["spread_value"]),
                }
            )

        # Total (only need one row per over/under, use home row)
        if pd.notna(home_row.get("total_OU_value")):
            if pd.notna(home_row.get("over_total_odds")):
                rows.append(
                    {
                        **base,
                        "market": "total",
                        "side": "over",
                        "odds": float(home_row["over_total_odds"]),
                        "line": float(home_row["total_OU_value"]),
                    }
                )
            if pd.notna(home_row.get("under_total_odds")):
                rows.append(
                    {
                        **base,
                        "market": "total",
                        "side": "under",
                        "odds": float(home_row["under_total_odds"]),
                        "line": float(home_row["total_OU_value"]),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=_LEDGER_COLUMNS)

    return pd.DataFrame(rows, columns=_LEDGER_COLUMNS)


def append_to_odds_ledger(
    df_long: pd.DataFrame,
    *,
    repo: Path | None = None,
) -> Path:
    """Append new odds rows to the historical ledger Parquet file.

    Reads the existing ledger (if any), removes any rows with the same
    ``(sportsbook, season, week, fetched_at)`` combination to avoid
    duplicates from re-runs, then appends the new rows and writes back.

    Args:
        df_long: Long-format odds DataFrame from ``wide_to_long()``.
        repo: Repository root path.

    Returns:
        Absolute path to the ledger file.
    """
    path: Path = _odds_dir(repo) / "dk_odds_log.parquet"

    if path.exists():
        existing: DataFrame = pd.read_parquet(path)
        # Drop any rows from the same pull (idempotent re-runs)
        if not df_long.empty:
            key_cols: list[str] = ["sportsbook", "season", "week", "fetched_at"]
            key_vals: DataFrame = df_long.loc[key_cols, :].drop_duplicates()
            mask: Series = pd.Series([True] * len(existing))
            for _, krow in key_vals.iterrows():
                match = (
                    (existing["sportsbook"] == krow["sportsbook"])
                    & (existing["season"] == krow["season"])
                    & (existing["week"] == krow["week"])
                    & (existing["fetched_at"] == krow["fetched_at"])
                )
                mask = mask & ~match
            existing = existing.loc[mask, :]
        df_out: DataFrame = pd.concat([existing, df_long], ignore_index=True)
    else:
        df_out = df_long.copy()

    df_out.to_parquet(path, index=False)
    logger.info("Odds ledger: %d total rows → %s", len(df_out), path)
    return path


def write_current_odds_snapshot(
    df_long: pd.DataFrame,
    *,
    repo: Path | None = None,
) -> Path:
    """Write the current odds pull as a snapshot for downstream use.

    Overwrites ``data/odds/dk_odds_current.parquet`` with the latest pull.
    Used by the predictions viz to get the current week's odds without
    reading the full historical ledger.

    Args:
        df_long: Long-format odds DataFrame from ``wide_to_long()``.
        repo: Repository root path.

    Returns:
        Absolute path to the snapshot file.
    """
    path: Path = _odds_dir(repo) / "dk_odds_current.parquet"
    df_long.to_parquet(path, index=False)
    logger.info("Odds snapshot written: %d rows → %s", len(df_long), path)
    return path


def load_current_odds(
    *,
    market: str | None = None,
    repo: Path | None = None,
) -> pd.DataFrame | None:
    """Load the current odds snapshot, optionally filtered by market.

    Args:
        market: If provided, filter to rows where ``market == market``
            (e.g. ``"moneyline"``).
        repo: Repository root path.

    Returns:
        Long-format DataFrame, or ``None`` if no snapshot exists.
    """
    path: Path = _odds_dir(repo) / "dk_odds_current.parquet"
    if not path.exists():
        return None
    df: DataFrame = pd.read_parquet(path)
    if market is not None:
        df = df.loc[df["market"] == market, :].copy()
    return df


def load_odds_ledger(
    *,
    sportsbook: str | None = None,
    season: str | None = None,
    week: int | None = None,
    market: str | None = None,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Load the historical odds ledger with optional filters.

    Uses Parquet predicate pushdown for efficient filtered reads when
    pyarrow is available.

    Args:
        sportsbook: Filter to a specific sportsbook (e.g. ``"draftkings"``).
        season: Filter to a specific season (e.g. ``"2026-2027"``).
        week: Filter to a specific week.
        market: Filter to a specific market (e.g. ``"moneyline"``).
        repo: Repository root path.

    Returns:
        Long-format DataFrame. Empty DataFrame if no ledger exists yet.
    """
    path = _odds_dir(repo) / "dk_odds_log.parquet"
    if not path.exists():
        return pd.DataFrame(columns=_LEDGER_COLUMNS)

    filters: list[tuple] = []
    if sportsbook is not None:
        filters.append(("sportsbook", "==", sportsbook))
    if season is not None:
        filters.append(("season", "==", season))
    if week is not None:
        filters.append(("week", "==", week))
    if market is not None:
        filters.append(("market", "==", market))

    return pd.read_parquet(path, filters=filters or None)

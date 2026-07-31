# src/gridiron_edge/ingest/odds/store.py

"""Source-neutral odds storage with an append ledger and current snapshot.

All markets use one long-format contract with one row per market side, game,
and ingestion snapshot. Sources may retain incomplete canonical side rows with
nullable odds or lines so missing market values remain distinguishable from
missing games.

Storage layout:
    data/odds/odds_log.parquet     - historical source-labeled snapshots
    data/odds/odds_current.parquet - explicitly written current snapshot

Schema:
    fetched_at      datetime64[ns, UTC]  UTC timestamp of ingestion
    sportsbook      str                 source identifier
    season          str                 "2026-2027"
    week            int                 NFL week number
    game_id         str                 canonical game ID
    game_date       str                 "2026-09-05"
    away_team       str                 long away-team name
    home_team       str                 long home-team name
    market          str                 "moneyline" | "spread" | "total"
    side            str                 "away" | "home" | "over" | "under"
    odds            float               American odds, nullable
    line            float               spread or total value, nullable
"""

from __future__ import annotations

import datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

# Column order for the ledger - enforced on every write.
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

_VALID_MARKET_SIDES: dict[str, frozenset[str]] = {
    "moneyline": frozenset({"away", "home"}),
    "spread": frozenset({"away", "home"}),
    "total": frozenset({"over", "under"}),
}

_REQUIRED_TEXT_COLUMNS: tuple[str, ...] = (
    "sportsbook",
    "season",
    "game_id",
    "away_team",
    "home_team",
    "market",
    "side",
)


def _normalize_odds_schema(rows: DataFrame) -> DataFrame:
    """Require the exact generic odds schema and canonical column order."""
    missing = sorted(set(_LEDGER_COLUMNS) - set(rows.columns))
    unknown = sorted(set(rows.columns) - set(_LEDGER_COLUMNS))

    if missing:
        raise ValueError("Invalid odds schema; missing columns: " + ", ".join(missing))
    if unknown:
        raise ValueError("Invalid odds schema; unknown columns: " + ", ".join(unknown))

    return rows.loc[:, _LEDGER_COLUMNS].copy()


def _validate_odds_text_columns(rows: DataFrame) -> None:
    """Require nonempty market identity and provenance values."""
    for column in _REQUIRED_TEXT_COLUMNS:
        values = rows[column]
        if values.isna().any():
            raise ValueError(f"Odds column {column!r} must contain nonempty values.")
        if values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Odds column {column!r} must contain nonempty values.")


def _normalize_odds_week(rows: DataFrame) -> None:
    """Validate and normalize NFL week values in place."""
    if rows["week"].isna().any():
        raise ValueError("Odds week must not be null.")

    rows["week"] = rows["week"].astype(int)

    if (rows["week"] < 1).any():
        raise ValueError("Odds week must be at least 1.")


def _normalize_odds_fetched_at(rows: DataFrame) -> None:
    """Validate and normalize source ingestion timestamps in place."""
    for value in rows["fetched_at"]:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            raise ValueError("Odds fetched_at values must be valid datetimes.")
        if timestamp.tzinfo is None:
            raise ValueError("Odds fetched_at values must be timezone-aware UTC.")
        if timestamp.utcoffset() != datetime.timedelta(0):
            raise ValueError("Odds fetched_at values must use UTC.")

    fetched_at = pd.to_datetime(
        rows["fetched_at"],
        utc=True,
        errors="coerce",
    )
    # pyrefly: ignore [missing-attribute]
    if fetched_at.isna().any():
        raise ValueError("Odds fetched_at values must be valid datetimes.")

    rows["fetched_at"] = fetched_at


def _validate_market_side_pairs(rows: DataFrame) -> None:
    """Require canonical sides for each market family."""
    invalid_pairs = sorted(
        {
            (
                str(row["market"]),
                str(row["side"]),
            )
            for _, row in rows.iterrows()
            if str(row["side"])
            not in _VALID_MARKET_SIDES.get(
                str(row["market"]),
                frozenset(),
            )
        }
    )

    if invalid_pairs:
        rendered = ", ".join(f"{market}/{side}" for market, side in invalid_pairs)
        raise ValueError("Odds rows contain invalid market/side pairs: " + rendered)


def validate_odds_rows(rows: DataFrame) -> DataFrame:
    """Validate and normalize source-labeled long-format market rows.

    Canonical rows may retain nullable odds and line values. Validation
    does not require complete market pairs and does not fabricate missing
    prices.
    """
    normalized = _normalize_odds_schema(rows)

    if normalized.empty:
        return normalized

    _validate_odds_text_columns(normalized)
    _normalize_odds_week(normalized)
    _normalize_odds_fetched_at(normalized)
    _validate_market_side_pairs(normalized)

    normalized["odds"] = pd.to_numeric(
        normalized["odds"],
        errors="coerce",
    )
    normalized["line"] = pd.to_numeric(
        normalized["line"],
        errors="coerce",
    )

    return normalized


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
    one row per market per side per game and resolves canonical
    ``game_id`` values via ``_game_id.resolve_dk_game_ids``.

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
        Rows whose teams can't be resolved to canonical game_ids are
        excluded.
    """
    from gridiron_edge.ingest.odds._game_id import resolve_dk_game_ids

    # Keep UTC tz on the timestamp; archive consumers expect tz-aware
    # datetimes per archive/H1.
    ts = fetched_at or datetime.datetime.now(tz=datetime.UTC)

    # Resolve canonical game_ids from team names. The resolver handles
    # both the intermediate (home_team/away_team) and wide (team/opponent/
    # location) DataFrame formats and adds a ``game_id`` column.
    season_year: int = int(season[:4])
    df_with_gid = resolve_dk_game_ids(df_wide, season_year=season_year, week=week)

    rows: list[dict] = []
    away_rows = df_with_gid[df_with_gid["location"] == 0].set_index("event_id")
    home_rows = df_with_gid[df_with_gid["location"] == 1].set_index("event_id")

    for event_id, home_row in home_rows.iterrows():
        if event_id not in away_rows.index:
            continue
        away_row = away_rows.loc[[event_id]].iloc[0]  # type: ignore[index]

        # Skip games whose teams didn't resolve to canonical short codes.
        canonical_game_id = home_row.get("game_id")
        if canonical_game_id is None or pd.isna(canonical_game_id):
            logger.warning(
                "wide_to_long: skipping event_id=%s - game_id resolution failed "
                "for home='%s' away='%s'",
                event_id,
                home_row.get("team"),
                away_row.get("team"),
            )
            continue

        home_team = str(home_row["team"])
        away_team = str(away_row["team"])
        start_time = home_row.get("start_time")
        game_date: str = (
            pd.Timestamp(start_time).strftime("%Y-%m-%d") if pd.notna(start_time) else ""
        )

        base: dict[str, Any] = {
            "fetched_at": ts,
            "sportsbook": sportsbook,
            "season": season,
            "week": week,
            "game_id": canonical_game_id,
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
    path: Path = _odds_dir(repo) / "odds_log.parquet"
    normalized = validate_odds_rows(df_long)

    if not path.exists():
        df_out = normalized
    elif normalized.empty:
        # Nothing to add; just return the existing file path unchanged.
        return path
    else:
        existing: DataFrame = pd.read_parquet(path)

        # Drop any rows from the same pull (idempotent re-runs).
        # The bug fixed here: df_long.loc[col_list, :] tried to select
        # rows whose *index* matched those strings (not column projection).
        # df_long[col_list] is the correct column-projection form.
        key_cols: list[str] = ["sportsbook", "season", "week", "fetched_at"]
        key_vals: DataFrame = normalized.loc[:, key_cols].drop_duplicates()

        # Build a boolean mask: True for rows we want to KEEP from existing
        # (i.e. rows whose key tuple does NOT appear in df_long).
        existing_keys = existing[key_cols].apply(tuple, axis=1)
        new_keys = key_vals.apply(tuple, axis=1)
        existing = existing.loc[~existing_keys.isin(new_keys), :]

        existing = validate_odds_rows(existing)
        df_out = pd.concat([existing, normalized], ignore_index=True)

    df_out.to_parquet(path, index=False)
    logger.info("Odds ledger: %d total rows → %s", len(df_out), path)
    return path


def write_current_odds_snapshot(
    df_long: pd.DataFrame,
    *,
    repo: Path | None = None,
) -> Path:
    """Write the current odds pull as a snapshot for downstream use.

    Overwrites ``data/odds/odds_current.parquet`` with the supplied snapshot.
    Used by the predictions viz to get the current week's odds without
    reading the full historical ledger.

    Args:
        df_long: Long-format odds DataFrame from ``wide_to_long()``.
        repo: Repository root path.

    Returns:
        Absolute path to the snapshot file.
    """
    path: Path = _odds_dir(repo) / "odds_current.parquet"
    normalized = validate_odds_rows(df_long)
    normalized.to_parquet(path, index=False)
    logger.info("Odds snapshot written: %d rows → %s", len(normalized), path)
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
    path: Path = _odds_dir(repo) / "odds_current.parquet"
    if not path.exists():
        return None
    df: DataFrame = validate_odds_rows(pd.read_parquet(path))
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
    path = _odds_dir(repo) / "odds_log.parquet"
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

    return validate_odds_rows(pd.read_parquet(path, filters=filters or None))

"""Provider-aware quote storage for observations and the current snapshot.

The canonical long-format contract contains one row per provider, sportsbook,
game, market, side, and local observation. Generic storage owns validation and
atomic persistence only; provider parsing and canonical game matching belong to
provider adapters.
"""

from __future__ import annotations

import datetime
import logging
from logging import Logger
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

QUOTE_COLUMNS: tuple[str, ...] = (
    "fetched_at",
    "provider",
    "provider_event_id",
    "sportsbook",
    "sportsbook_updated_at",
    "commence_time",
    "is_live",
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
)


OBSERVATION_FETCH_IDENTITY_COLUMNS: tuple[str, ...] = (
    "fetched_at",
    "provider",
    "provider_event_id",
    "sportsbook",
    "game_id",
    "market",
    "side",
)

OBSERVATION_SORT_COLUMNS: tuple[str, ...] = (
    "fetched_at",
    "provider",
    "provider_event_id",
    "sportsbook",
    "game_id",
    "market",
    "side",
    "line",
    "odds",
    "sportsbook_updated_at",
    "is_live",
)

OBSERVATION_IDENTITY_COLUMNS: tuple[str, ...] = (
    "fetched_at",
    "provider",
    "provider_event_id",
    "sportsbook",
    "game_id",
    "market",
    "side",
    "odds",
    "line",
    "sportsbook_updated_at",
    "is_live",
)

_VALID_MARKET_SIDES: dict[str, frozenset[str]] = {
    "moneyline": frozenset({"away", "home"}),
    "spread": frozenset({"away", "home"}),
    "total": frozenset({"over", "under"}),
}

_REQUIRED_TEXT_COLUMNS: tuple[str, ...] = (
    "provider",
    "season",
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "market",
    "side",
)

_NULLABLE_TEXT_COLUMNS: tuple[str, ...] = (
    "provider_event_id",
    "sportsbook",
)


def empty_quote_frame() -> DataFrame:
    """Return an empty frame with the canonical quote column order."""
    return DataFrame(columns=list(QUOTE_COLUMNS))


def _require_exact_schema(rows: DataFrame) -> DataFrame:
    """Require the exact provider-aware quote schema and column order."""
    missing = sorted(set(QUOTE_COLUMNS) - set(rows.columns))
    unknown = sorted(set(rows.columns) - set(QUOTE_COLUMNS))
    if missing:
        raise ValueError("Invalid quote schema; missing columns: " + ", ".join(missing))
    if unknown:
        raise ValueError("Invalid quote schema; unknown columns: " + ", ".join(unknown))
    return rows.loc[:, QUOTE_COLUMNS].copy()


def _validate_required_text(rows: DataFrame) -> None:
    """Require nonempty canonical and provider identity text."""
    for column in _REQUIRED_TEXT_COLUMNS:
        values = rows[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Quote column {column!r} must contain nonempty values.")


def _normalize_nullable_text(rows: DataFrame) -> None:
    """Normalize optional text to string-or-null and reject empty strings."""
    for column in _NULLABLE_TEXT_COLUMNS:
        values = rows[column]
        present = values.notna()
        if values.loc[present].astype(str).str.strip().eq("").any():
            raise ValueError(f"Quote column {column!r} must be null or nonempty.")
        rows[column] = values.where(~present, values.astype("string").str.strip())


def _normalize_week(rows: DataFrame) -> None:
    """Normalize positive integer NFL week values in place."""
    numeric: Series = Series(
        pd.to_numeric(rows["week"], errors="coerce"),
        index=rows.index,
    )
    if numeric.isna().any() or (numeric % 1 != 0).any() or (numeric < 1).any():
        raise ValueError("Quote week must contain positive integers.")
    rows["week"] = numeric.astype(int)


def _normalize_utc_timestamp(
    rows: DataFrame,
    column: str,
    *,
    required: bool,
) -> None:
    """Normalize one required or nullable timezone-aware UTC timestamp."""
    normalized: list[str] = []
    for value in rows[column]:
        if pd.isna(value):
            if required:
                raise ValueError(f"Quote {column} values must be valid datetimes.")
            normalized.append("NaT")
            continue
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError(f"Quote {column} values must be timezone-aware UTC.")
        if timestamp.utcoffset() != datetime.timedelta(0):
            raise ValueError(f"Quote {column} values must use UTC.")
        normalized.append(timestamp.isoformat())
    rows[column] = pd.to_datetime(normalized, utc=True)


def _normalize_is_live(rows: DataFrame) -> None:
    """Require explicit boolean live-state values."""
    if rows["is_live"].isna().any():
        raise ValueError("Quote is_live values must not be null.")
    invalid = [value for value in rows["is_live"] if not isinstance(value, bool)]
    if invalid:
        raise ValueError("Quote is_live values must be boolean.")
    rows["is_live"] = rows["is_live"].astype(bool)


def _validate_market_side_pairs(rows: DataFrame) -> None:
    """Require canonical sides for each supported market family."""
    invalid_pairs = sorted(
        {
            (str(row["market"]), str(row["side"]))
            for _, row in rows.iterrows()
            if str(row["side"]) not in _VALID_MARKET_SIDES.get(str(row["market"]), frozenset())
        }
    )
    if invalid_pairs:
        rendered = ", ".join(f"{market}/{side}" for market, side in invalid_pairs)
        raise ValueError("Quote rows contain invalid market/side pairs: " + rendered)


def _normalize_numeric(rows: DataFrame, column: str, *, reject_zero: bool) -> None:
    """Normalize nullable finite numeric quote values."""
    raw = rows[column]
    converted: Series = Series(
        pd.to_numeric(raw, errors="coerce"),
        index=raw.index,
    )
    invalid_parse = raw.notna() & converted.isna()
    if invalid_parse.any():
        raise ValueError(f"Quote {column} values must be numeric or null.")
    finite = converted.dropna().map(
        lambda value: bool(pd.notna(value) and abs(value) != float("inf"))
    )
    if not finite.all():
        raise ValueError(f"Quote {column} values must be finite when provided.")
    if reject_zero and converted.dropna().eq(0).any():
        raise ValueError("Quote odds values must not be zero.")
    rows[column] = converted.astype(float)


def validate_quote_rows(rows: DataFrame) -> DataFrame:
    """Validate and normalize provider-aware long-format quote rows."""
    normalized = _require_exact_schema(rows)
    if normalized.empty:
        return normalized

    _validate_required_text(normalized)
    _normalize_nullable_text(normalized)
    _normalize_week(normalized)
    _normalize_utc_timestamp(normalized, "fetched_at", required=True)
    _normalize_utc_timestamp(normalized, "sportsbook_updated_at", required=False)
    _normalize_utc_timestamp(normalized, "commence_time", required=False)
    _normalize_is_live(normalized)
    _validate_market_side_pairs(normalized)
    _normalize_numeric(normalized, "odds", reject_zero=True)
    _normalize_numeric(normalized, "line", reject_zero=False)
    return normalized


def _odds_dir(repo: Path | None = None) -> Path:
    """Return and create the canonical odds directory."""
    root = repo or get_settings().repo_root
    path = root / "data" / "odds"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _history_root(repo: Path | None = None) -> Path:
    """Return and create the canonical partitioned quote-history root."""
    path = _odds_dir(repo) / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def odds_history_partition_path(
    *,
    season: str,
    week: int,
    repo: Path | None = None,
) -> Path:
    """Return the deterministic path for one season-and-week partition."""
    normalized_season = season.strip()
    if not normalized_season or not all(
        character.isalnum() or character in {"-", "_"} for character in normalized_season
    ):
        raise ValueError("Quote season must be a safe nonempty path component.")
    if week < 1:
        raise ValueError("Quote week must be at least 1.")
    return (
        _history_root(repo)
        / f"season={normalized_season}"
        / f"week={week:02d}"
        / "observations.parquet"
    )


def _single_partition_scope(rows: DataFrame) -> tuple[str, int]:
    """Return the one season-and-week scope owned by an append."""
    if rows.empty:
        raise ValueError("Historical quote append requires at least one row.")
    scopes = rows.loc[:, ["season", "week"]].drop_duplicates()
    if len(scopes) != 1:
        raise ValueError("Historical quote append must contain exactly one season-and-week scope.")
    scope = scopes.iloc[0]
    return str(scope["season"]), int(scope["week"])


def _matching_history_paths(
    *,
    season: str | None,
    week: int | None,
    repo: Path | None,
) -> list[Path]:
    """Return deterministic matching history partitions."""
    root = _history_root(repo)
    if season is not None and week is not None:
        path = odds_history_partition_path(season=season, week=week, repo=repo)
        return [path] if path.is_file() else []

    season_pattern = f"season={season}" if season is not None else "season=*"
    week_pattern = f"week={week:02d}" if week is not None else "week=*"
    return sorted(root.glob(f"{season_pattern}/{week_pattern}/observations.parquet"))


def _atomic_write_parquet(rows: DataFrame, path: Path) -> None:
    """Write Parquet beside the destination and atomically replace it."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        rows.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_observation_conflicts(rows: DataFrame) -> None:
    """Reject conflicting market-side observations within one local fetch."""
    duplicated = rows.duplicated(
        subset=list(OBSERVATION_FETCH_IDENTITY_COLUMNS),
        keep=False,
    )
    if not duplicated.any():
        return

    conflicts = rows.loc[
        duplicated,
        list(OBSERVATION_FETCH_IDENTITY_COLUMNS),
    ].drop_duplicates()
    rendered = [
        "/".join("<null>" if pd.isna(value) else str(value) for value in identity)
        for identity in conflicts.itertuples(index=False, name=None)
    ]
    raise ValueError(
        "Quote observations conflict within one local fetch: " + ", ".join(sorted(rendered))
    )


def _canonicalize_observations(rows: DataFrame) -> DataFrame:
    """Deduplicate, validate, and deterministically order observations."""
    unique = rows.drop_duplicates(
        subset=list(OBSERVATION_IDENTITY_COLUMNS),
        keep="last",
    ).reset_index(drop=True)
    _validate_observation_conflicts(unique)
    return unique.sort_values(
        list(OBSERVATION_SORT_COLUMNS),
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)


def append_to_odds_ledger(
    quotes: DataFrame,
    *,
    repo: Path | None = None,
) -> Path:
    """Append normalized observations to one atomic weekly partition."""
    normalized = validate_quote_rows(quotes)
    season, week = _single_partition_scope(normalized)
    path = odds_history_partition_path(season=season, week=week, repo=repo)

    if path.exists():
        existing = validate_quote_rows(pd.read_parquet(path))
        existing_scope = _single_partition_scope(existing)
        if existing_scope != (season, week):
            raise ValueError("Existing quote partition contains an invalid scope.")
        combined = pd.concat([existing, normalized], ignore_index=True)
    else:
        combined = normalized

    output = _canonicalize_observations(validate_quote_rows(combined))
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(output, path)
    logger.info("Odds observation partition: %d rows -> %s", len(output), path)
    return path


def write_current_odds_snapshot(
    quotes: DataFrame,
    *,
    repo: Path | None = None,
) -> Path:
    """Atomically replace the current validated quote snapshot."""
    path = _odds_dir(repo) / "odds_current.parquet"
    normalized = validate_quote_rows(quotes)
    _atomic_write_parquet(normalized, path)
    logger.info("Current odds snapshot: %d rows -> %s", len(normalized), path)
    return path


def load_current_odds(
    *,
    market: str | None = None,
    repo: Path | None = None,
) -> DataFrame | None:
    """Load the current quote snapshot, optionally filtered by market."""
    path = _odds_dir(repo) / "odds_current.parquet"
    if not path.exists():
        return None
    rows = validate_quote_rows(pd.read_parquet(path))
    if market is not None:
        rows = rows.loc[rows["market"] == market, :].copy()
    return rows


def load_odds_ledger(
    *,
    provider: str | None = None,
    sportsbook: str | None = None,
    season: str | None = None,
    week: int | None = None,
    market: str | None = None,
    repo: Path | None = None,
) -> DataFrame:
    """Load partitioned quote history with optional provider-aware filters."""
    paths = _matching_history_paths(season=season, week=week, repo=repo)
    if not paths:
        return empty_quote_frame()

    frames = [validate_quote_rows(pd.read_parquet(path)) for path in paths]
    rows = validate_quote_rows(pd.concat(frames, ignore_index=True))
    for column, value in (
        ("provider", provider),
        ("sportsbook", sportsbook),
        ("season", season),
        ("week", week),
        ("market", market),
    ):
        if value is not None:
            rows = rows.loc[rows[column] == value, :].copy()
    if rows.empty:
        return empty_quote_frame()
    combined_history = rows
    return combined_history.sort_values(
        [
            "season",
            "week",
            *OBSERVATION_SORT_COLUMNS,
        ],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)

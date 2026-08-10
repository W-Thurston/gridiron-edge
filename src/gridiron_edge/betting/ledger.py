# src/gridiron_edge/betting/ledger.py
"""Append-only bet ledger with immutable reference-offer evidence.

Follows the same Parquet append-only pattern as ``evaluation/archive.py``.
The ledger stores every bet placed, its model context at bet time, and
settlement results including PnL.

Public API::

    log_bet(...)         Record a new bet, returns bet_id (UUID)
    settle_bet(...)      Settle a bet with result and compute PnL
    load_bets(...)       Load bets with optional filters
    compute_pnl(...)     Pure function: stake + odds + result -> PnL

Storage lives at ``data/betting/bet_ledger.parquet``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from logging import Logger
from math import isfinite
from pathlib import Path
from typing import Final, Literal
import uuid

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.market.odds_math import american_to_decimal

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

BetStatus: type[BetStatus] = Literal["open", "won", "lost", "push"]
MarketType: type[MarketType] = Literal["moneyline", "spread", "total"]

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BET_COLUMNS: Final[list[str]] = [
    "bet_id",
    "game_id",
    "placed_at",
    "market_type",
    "side",
    "line",
    "odds",
    "stake",
    "book",
    "reference_provider",
    "reference_provider_event_id",
    "reference_sportsbook",
    "reference_market_fetched_at",
    "reference_sportsbook_updated_at",
    "reference_commence_time",
    "reference_american_odds",
    "reference_line",
    "model_name",
    "model_type",
    "model_prob",
    "model_ev",
    "edge_strength",
    "confidence_tier",
    "status",
    "settled_at",
    "pnl",
    "closing_line",
    "closing_odds",
    "clv",
]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _bet_ledger_path(repo: Path | None = None) -> Path:
    """Return the path to the bet ledger Parquet file.

    Creates the parent directory if it does not exist.

    Args:
        repo: Repository root override. Defaults to ``get_settings().repo_root``.

    Returns:
        Absolute path to ``data/betting/bet_ledger.parquet``.
    """
    if repo is None:
        from gridiron_edge.core.settings import get_settings

        repo = get_settings().repo_root
    path: Path = repo / "data" / "betting" / "bet_ledger.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _empty_ledger() -> pd.DataFrame:
    """Return an empty DataFrame with the correct ledger schema."""
    return pd.DataFrame(columns=_BET_COLUMNS)


def _require_ledger_schema(
    df: DataFrame,
    *,
    label: str,
) -> None:
    """Require the exact current persisted bet-ledger schema."""
    actual: list[str] = df.columns.tolist()
    expected: list[str] = _BET_COLUMNS

    missing: list[str] = [column for column in expected if column not in actual]
    extra: list[str] = [column for column in actual if column not in expected]

    problems: list[str] = []

    if missing:
        problems.append("missing columns: " + ", ".join(missing))

    if extra:
        problems.append("extra columns: " + ", ".join(extra))

    if not missing and not extra and actual != expected:
        problems.append("columns are not in canonical order")

    if problems:
        raise ValueError(
            f"{label} does not match the current bet-ledger schema: " + "; ".join(problems)
        )


def _validate_model_identity(
    model_name: str | None,
    model_type: str | None,
) -> None:
    """Require model_name and model_type to form one optional identity.

    A manual or otherwise unattributed bet omits both values. A
    model-attributed bet supplies both nonempty values.
    """
    if model_name is None and model_type is None:
        return

    if model_name is None or model_type is None:
        raise ValueError("model_name and model_type must be provided together.")

    if not model_name.strip():
        raise ValueError("model_name must be a nonempty string when model identity is provided.")

    if not model_type.strip():
        raise ValueError("model_type must be a nonempty string when model identity is provided.")


def _require_utc_timestamp(
    value: datetime,
    *,
    label: str,
) -> None:
    """Require one reference timestamp to be timezone-aware UTC."""
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")


def _validate_optional_reference_text(
    value: str | None,
    *,
    label: str,
) -> None:
    """Require optional reference text to be null or nonempty."""
    if value is not None and not value.strip():
        raise ValueError(f"{label} must be null or a nonempty string.")


def _validate_reference_provenance(
    *,
    reference_provider: str | None,
    reference_provider_event_id: str | None,
    reference_sportsbook: str | None,
    reference_market_fetched_at: datetime | None,
    reference_sportsbook_updated_at: datetime | None,
    reference_commence_time: datetime | None,
    reference_american_odds: int | None,
    reference_line: float | None,
) -> None:
    """Validate one optional exact reference-offer evidence contract."""
    values = (
        reference_provider,
        reference_provider_event_id,
        reference_sportsbook,
        reference_market_fetched_at,
        reference_sportsbook_updated_at,
        reference_commence_time,
        reference_american_odds,
        reference_line,
    )
    if all(value is None for value in values):
        return

    if reference_provider is None or not reference_provider.strip():
        raise ValueError(
            "reference_provider must be a nonempty string when any reference "
            "offer field is provided."
        )
    if reference_market_fetched_at is None:
        raise ValueError(
            "reference_market_fetched_at is required when reference offer provenance is provided."
        )

    _validate_optional_reference_text(
        reference_provider_event_id,
        label="reference_provider_event_id",
    )
    _validate_optional_reference_text(
        reference_sportsbook,
        label="reference_sportsbook",
    )
    _require_utc_timestamp(
        reference_market_fetched_at,
        label="reference_market_fetched_at",
    )
    if reference_sportsbook_updated_at is not None:
        _require_utc_timestamp(
            reference_sportsbook_updated_at,
            label="reference_sportsbook_updated_at",
        )
    if reference_commence_time is not None:
        _require_utc_timestamp(
            reference_commence_time,
            label="reference_commence_time",
        )
    if reference_american_odds is not None and (
        reference_american_odds == 0 or not isfinite(reference_american_odds)
    ):
        raise ValueError("reference_american_odds must be finite and nonzero when provided.")
    if reference_line is not None and not isfinite(reference_line):
        raise ValueError("reference_line must be finite when provided.")


def _read_ledger(repo: Path | None = None) -> pd.DataFrame:
    """Read the bet ledger from disk.

    Returns an empty DataFrame with the correct schema if the file does not
    exist.

    Args:
        repo: Repository root override.

    Returns:
        DataFrame with all recorded bets.
    """
    path: Path = _bet_ledger_path(repo)
    if not path.exists():
        return _empty_ledger()
    df: DataFrame = pd.read_parquet(path)
    _require_ledger_schema(
        df,
        label="Existing bet ledger",
    )
    return df


def _write_ledger(df: pd.DataFrame, repo: Path | None = None) -> Path:
    """Write the bet ledger to disk.

    Args:
        df: Full ledger DataFrame.
        repo: Repository root override.

    Returns:
        Path to the written file.
    """
    _require_ledger_schema(
        df,
        label="Bet ledger write",
    )

    path: Path = _bet_ledger_path(repo)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Pure PnL calculation
# ---------------------------------------------------------------------------


def compute_pnl(
    stake: float,
    american_odds: int,
    result: BetStatus,
) -> float:
    """Compute profit/loss for a settled bet.

    Args:
        stake: Amount wagered.
        american_odds: American odds at time of bet (e.g. -110, +150).
        result: Settlement result.

    Returns:
        Profit (positive) or loss (negative). Zero for push or open bets.
    """
    if result == "won":
        decimal_odds: float = american_to_decimal(american_odds)
        return stake * (decimal_odds - 1.0)
    if result == "lost":
        return -stake
    # push or open
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_bet(
    game_id: str,
    *,
    market_type: str,
    side: str,
    odds: int,
    stake: float,
    book: str,
    line: float | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    model_prob: float | None = None,
    model_ev: float | None = None,
    edge_strength: str | None = None,
    confidence_tier: str | None = None,
    reference_provider: str | None = None,
    reference_provider_event_id: str | None = None,
    reference_sportsbook: str | None = None,
    reference_market_fetched_at: datetime | None = None,
    reference_sportsbook_updated_at: datetime | None = None,
    reference_commence_time: datetime | None = None,
    reference_american_odds: int | None = None,
    reference_line: float | None = None,
    placed_at: datetime | None = None,
    repo: Path | None = None,
) -> str:
    """Record a new bet in the ledger.

    Creates the ledger file if it does not exist. Appends a single row
    with status ``"open"`` and generates a UUID for the ``bet_id``.

    Args:
        game_id: Canonical game identifier (e.g. ``"2026_01_KC_LAC"``).
        market_type: One of ``"moneyline"``, ``"spread"``, ``"total"``.
        side: Bet side (e.g. ``"home"``, ``"away"``, ``"over"``, ``"under"``).
        odds: American odds at time of bet.
        stake: Dollar amount wagered.
        book: Sportsbook name (e.g. ``"draftkings"``).
        line: Point spread or total line. ``None`` for moneyline bets.
        model_name: Model purpose used to identify the edge
            (e.g. ``"win_prob"``, ``"qb_pass_yards"``).
        model_type: Algorithm used to compute the edge
            (e.g. ``"random_forest"``, ``"elasticnet"``).
        model_prob: Model probability at bet time.
        model_ev: Expected value at bet time.
        edge_strength: Edge classification at bet time.
        confidence_tier: Confidence tier at bet time.
        reference_provider: Provider that supplied the reference offer.
        reference_provider_event_id: Optional provider event identity.
        reference_sportsbook: Optional sportsbook for the reference offer.
        reference_market_fetched_at: UTC local observation timestamp.
        reference_sportsbook_updated_at: Optional UTC source update time.
        reference_commence_time: Optional UTC kickoff evidence.
        reference_american_odds: Optional reference-offer American odds.
        reference_line: Optional reference-offer point value.
        placed_at: Timestamp of bet placement. Defaults to ``utcnow()``.
        repo: Repository root override.

    Returns:
        The generated ``bet_id`` (UUID string).

    Raises:
        ValueError: If model identity is incomplete or contains an empty
            model_name or model_type.
    """
    _validate_model_identity(model_name, model_type)
    _validate_reference_provenance(
        reference_provider=reference_provider,
        reference_provider_event_id=reference_provider_event_id,
        reference_sportsbook=reference_sportsbook,
        reference_market_fetched_at=reference_market_fetched_at,
        reference_sportsbook_updated_at=reference_sportsbook_updated_at,
        reference_commence_time=reference_commence_time,
        reference_american_odds=reference_american_odds,
        reference_line=reference_line,
    )

    bet_id = str(uuid.uuid4())
    if placed_at is None:
        placed_at = datetime.now(UTC)

    row: dict[str, datetime | float | int | str | None] = {
        "bet_id": bet_id,
        "game_id": game_id,
        "placed_at": placed_at,
        "market_type": market_type,
        "side": side,
        "line": line,
        "odds": odds,
        "stake": stake,
        "book": book,
        "reference_provider": reference_provider,
        "reference_provider_event_id": reference_provider_event_id,
        "reference_sportsbook": reference_sportsbook,
        "reference_market_fetched_at": reference_market_fetched_at,
        "reference_sportsbook_updated_at": reference_sportsbook_updated_at,
        "reference_commence_time": reference_commence_time,
        "reference_american_odds": reference_american_odds,
        "reference_line": reference_line,
        "model_name": model_name,
        "model_type": model_type,
        "model_prob": model_prob,
        "model_ev": model_ev,
        "edge_strength": edge_strength,
        "confidence_tier": confidence_tier,
        "status": "open",
        "settled_at": None,
        "pnl": None,
        "closing_line": None,
        "closing_odds": None,
        "clv": None,
    }

    new_row = pd.DataFrame([row], columns=_BET_COLUMNS)
    existing: DataFrame = _read_ledger(repo)

    if existing.empty:
        combined: DataFrame = new_row
    else:
        combined = pd.concat(
            [existing.dropna(axis=1, how="all"), new_row.dropna(axis=1, how="all")],
            ignore_index=True,
        ).reindex(columns=_BET_COLUMNS)

    _write_ledger(combined, repo)

    logger.info("Bet logged: %s  %s %s %s @ %s", bet_id, market_type, side, game_id, odds)
    return bet_id


def settle_bet(bet_id: str, result: BetStatus, *, repo: Path | None = None) -> pd.Series:
    """Settle an open bet with the given result.

    Computes PnL from the bet's odds and stake. Closing fields remain
    null until a validated closeout policy writes them.

    Args:
        bet_id: UUID of the bet to settle.
        result: Settlement result - ``"won"``, ``"lost"``, or ``"push"``.
        repo: Repository root override.

    Returns:
        The settled bet row as a ``pd.Series``.

    Raises:
        ValueError: If ``bet_id`` is not found or the bet is already settled.
    """
    if result not in ("won", "lost", "push"):
        msg: str = f"Invalid result: {result!r}. Must be 'won', 'lost', or 'push'."
        raise ValueError(msg)

    ledger: DataFrame = _read_ledger(repo)
    mask: Series[bool] = ledger["bet_id"] == bet_id
    if not mask.any():
        msg = f"Bet not found: {bet_id}"
        raise ValueError(msg)

    idx: int | str = mask.idxmax()
    bet: DataFrame | Series = ledger.loc[idx]

    if bet["status"] != "open":
        msg = f"Bet {bet_id} is already settled (status={bet['status']!r})."
        raise ValueError(msg)

    # Compute PnL
    pnl: float = compute_pnl(bet["stake"], int(bet["odds"]), result)

    closing_line = None
    closing_odds = None
    clv = None

    # Update ledger
    ledger.loc[idx, "status"] = result
    ledger["settled_at"] = pd.to_datetime(ledger["settled_at"], utc=True)
    ledger.loc[idx, "settled_at"] = datetime.now(UTC)
    ledger.loc[idx, "pnl"] = pnl
    ledger.loc[idx, "closing_line"] = closing_line
    ledger.loc[idx, "closing_odds"] = closing_odds
    ledger.loc[idx, "clv"] = clv

    _write_ledger(ledger, repo)

    logger.info("Bet settled: %s -> %s (PnL=%.2f)", bet_id, result, pnl)
    return pd.Series(ledger.loc[idx])


def load_bets(
    *,
    status: str | None = None,
    season: str | None = None,
    week: int | None = None,
    market_type: str | None = None,
    book: str | None = None,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Load bets from the ledger with optional filters.

    Args:
        status: Filter to bets with this status.
        season: Filter to bets whose ``game_id`` starts with this season year.
        week: Filter to bets whose ``game_id`` contains this week number.
        market_type: Filter to this market type.
        book: Filter to this sportsbook.
        repo: Repository root override.

    Returns:
        Filtered DataFrame of bets. Empty with correct schema if no matches
        or ledger does not exist.
    """
    df: DataFrame = _read_ledger(repo)
    if df.empty:
        return df

    if status is not None:
        df = df.loc[df["status"] == status, :]
    if market_type is not None:
        df = df.loc[df["market_type"] == market_type, :]
    if book is not None:
        df = df.loc[df["book"] == book, :]
    if season is not None:
        # game_id format: YYYY_WW_AWAY_HOME - season year is first 4 chars
        df = df.loc[df["game_id"].str.startswith(season[:4]), :]
    if week is not None:
        # game_id format: YYYY_WW_AWAY_HOME - week is chars 5:7
        week_str: str = f"{week:02d}"
        df = df.loc[df["game_id"].str[5:7] == week_str, :]

    return df.reset_index(drop=True)

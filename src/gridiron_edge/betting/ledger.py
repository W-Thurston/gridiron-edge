# src/gridiron_edge/betting/ledger.py
"""Append-only bet ledger with settlement and CLV enrichment.

Follows the same Parquet append-only pattern as ``evaluation/archive.py``.
The ledger stores every bet placed, its model context at bet time, and
settlement results including PnL and closing line value.

Public API::

    log_bet(...)         Record a new bet, returns bet_id (UUID)
    settle_bet(...)      Settle a bet with result, compute PnL + CLV
    load_bets(...)       Load bets with optional filters
    compute_pnl(...)     Pure function: stake + odds + result -> PnL

Storage lives at ``data/betting/bet_ledger.parquet``.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal
import uuid

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.market.odds_math import american_to_decimal

if TYPE_CHECKING:
    pass

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

    # Backward compatibility for ledgers written before the
    # model_name / model_type migration (Unit 6a). Old ledgers stored
    # model identity as a single ``model_version`` column. New schema
    # adds ``model_name`` and ``model_type`` as NA, and the obsolete
    # ``model_version`` is dropped by the final column projection.
    for col in _BET_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df.loc[:, _BET_COLUMNS]


def _write_ledger(df: pd.DataFrame, repo: Path | None = None) -> Path:
    """Write the bet ledger to disk.

    Args:
        df: Full ledger DataFrame.
        repo: Repository root override.

    Returns:
        Path to the written file.
    """
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
    market_type: str,
    side: str,
    odds: int,
    stake: float,
    book: str,
    *,
    line: float | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    model_prob: float | None = None,
    model_ev: float | None = None,
    edge_strength: str | None = None,
    confidence_tier: str | None = None,
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
        placed_at: Timestamp of bet placement. Defaults to ``utcnow()``.
        repo: Repository root override.

    Returns:
        The generated ``bet_id`` (UUID string).
    """
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


def settle_bet(
    bet_id: str,
    result: BetStatus,
    *,
    repo: Path | None = None,
    odds_ledger: pd.DataFrame | None = None,
) -> pd.Series:
    """Settle an open bet with the given result.

    Computes PnL from the bet's odds and stake. If an ``odds_ledger`` is
    provided, also computes closing line value (CLV).

    Args:
        bet_id: UUID of the bet to settle.
        result: Settlement result - ``"won"``, ``"lost"``, or ``"push"``.
        repo: Repository root override.
        odds_ledger: Optional long-format odds DataFrame for CLV lookup.

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

    # Compute CLV if odds ledger is available
    closing_line = None
    closing_odds = None
    clv = None
    if odds_ledger is not None and not odds_ledger.empty:
        clv_result: tuple[float | None, int | None, float | None] | None = _compute_clv_for_bet(
            bet, odds_ledger
        )
        if clv_result is not None:
            closing_line, closing_odds, clv = clv_result

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


# ---------------------------------------------------------------------------
# CLV helper
# ---------------------------------------------------------------------------


def _compute_clv_for_bet(
    bet: pd.Series,
    odds_ledger: pd.DataFrame,
) -> tuple[float | None, int | None, float | None] | None:
    """Compute closing line value for a single settled bet.

    Returns (closing_line, closing_odds, clv) or None if closing data
    is not available.
    """
    try:
        from gridiron_edge.market.clv import extract_closing_odds
    except ImportError:
        return None

    game_id = bet["game_id"]
    market = bet["market_type"]
    side = bet["side"]

    # Filter odds ledger to this game + market
    mask = (odds_ledger["game_id"] == game_id) & (odds_ledger["market"] == market)
    game_odds: Series = odds_ledger[mask]
    if game_odds.empty:
        return None

    # Get closing odds for our side
    # pyrefly: ignore [bad-argument-type]
    closing: DataFrame = extract_closing_odds(game_odds)
    side_mask = closing["side"] == side
    if not side_mask.any():
        return None

    closing_row = closing[side_mask].iloc[0]
    closing_odds_val = int(closing_row["odds"])
    closing_line_val = closing_row.get("line", None)

    if market == "moneyline":
        clv_val: float | None = _ml_clv(int(bet["odds"]), closing_odds_val)
    else:
        clv_val = _line_clv(bet.get("line", None), closing_line_val, market, side)

    return (
        float(closing_line_val) if closing_line_val is not None else None,
        closing_odds_val,
        clv_val,
    )


def _ml_clv(bet_odds: int, closing_odds: int) -> float | None:
    """Moneyline CLV using the canonical closing_line_value helper.

    Unifies the ledger's CLV computation with the formula used in
    ``market/clv.py``. The single-sided ledger row does not carry the
    opposing odds, so the calculation still operates on raw implied
    probabilities rather than no-vig probabilities. The canonical
    helper performs the relative-change calculation in a single place.

    Returns ``None`` for unusable inputs (e.g. American odds of zero,
    which ``american_to_implied_prob`` rejects) so the ledger's
    settlement path never raises mid-write.
    """
    from gridiron_edge.market.clv import closing_line_value
    from gridiron_edge.market.odds_math import american_to_implied_prob

    try:
        bet_prob: float = american_to_implied_prob(bet_odds)
        close_prob: float = american_to_implied_prob(closing_odds)
    except ValueError:
        return None

    if bet_prob <= 0 or close_prob <= 0:
        return None

    try:
        return closing_line_value(bet_prob, close_prob)
    except ValueError:
        return None


def _line_clv(
    bet_line: float | None,
    closing_line: float | None,
    market: str,
    side: str,
) -> float | None:
    """Point-based CLV for spread or total bets."""
    if bet_line is None or closing_line is None:
        return None
    bl: float = bet_line
    cl: float = closing_line
    if market == "spread":
        return bl - cl if side == "home" else cl - bl
    # total
    return cl - bl if side == "over" else bl - cl

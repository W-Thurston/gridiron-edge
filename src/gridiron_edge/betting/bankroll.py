# src/gridiron_edge/betting/bankroll.py
"""Bankroll transaction log — tracks every dollar in and out.

Decoupled from the bet ledger. The CLI layer orchestrates calls to both
``ledger.py`` and ``bankroll.py`` so that bet placement deducts stake
and settlement credits the gross return.

Transaction types::

    deposit       Money added to the bankroll.
    withdraw      Money removed from the bankroll.
    bet_placed    Stake leaves the bankroll (placed a bet).
    bet_settled   Gross return enters the bankroll (won/push payout).

Storage lives at ``data/betting/bankroll_txn.parquet``.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Final, Literal
import uuid

import pandas as pd
from pandas import DataFrame

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TxnType: type[TxnType] = Literal["deposit", "withdraw", "bet_placed", "bet_settled"]

_INFLOWS: frozenset[str] = frozenset({"deposit", "bet_settled"})
_OUTFLOWS: frozenset[str] = frozenset({"withdraw", "bet_placed"})

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_TXN_COLUMNS: Final[list[str]] = [
    "txn_id",
    "timestamp",
    "txn_type",
    "amount",
    "reference_id",
    "note",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _txn_path(repo: Path | None = None) -> Path:
    """Return the path to the bankroll transaction log.

    Creates the parent directory if it does not exist.

    Args:
        repo: Repository root override.

    Returns:
        Absolute path to ``data/betting/bankroll_txn.parquet``.
    """
    if repo is None:
        from gridiron_edge.core.settings import get_settings

        repo = get_settings().repo_root
    path: Path = repo / "data" / "betting" / "bankroll_txn.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _empty_txn_log() -> pd.DataFrame:
    """Return an empty DataFrame with the correct transaction schema."""
    return pd.DataFrame(columns=_TXN_COLUMNS)


def _read_txn_log(repo: Path | None = None) -> pd.DataFrame:
    """Read the transaction log from disk.

    Returns an empty DataFrame with the correct schema if the file does
    not exist.
    """
    path: Path = _txn_path(repo)
    if not path.exists():
        return _empty_txn_log()
    df: DataFrame = pd.read_parquet(path)
    for col in _TXN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df.loc[:, _TXN_COLUMNS]


def _write_txn_log(df: pd.DataFrame, repo: Path | None = None) -> Path:
    """Write the transaction log to disk."""
    path: Path = _txn_path(repo)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------


def signed_amount(txn_type: str, amount: float) -> float:
    """Return the signed amount for balance calculations.

    Deposits and bet settlements are positive (money in).
    Withdrawals and bet placements are negative (money out).
    """
    if txn_type in _INFLOWS:
        return amount
    return -amount


def _signed_amount_series(
    txn_types: pd.Series,
    amounts: pd.Series,
) -> pd.Series:
    """Vectorized version of ``signed_amount`` for a Series of transactions.

    Mirrors the scalar logic: inflows return positive, outflows return
    negative. Used by ``current_balance`` and ``balance_history`` to
    avoid row-wise apply (bankroll/H1).
    """
    import numpy as np

    return pd.Series(
        np.where(txn_types.isin(_INFLOWS), amounts, -amounts),
        index=txn_types.index,
    )


# ---------------------------------------------------------------------------
# Internal append helper
# ---------------------------------------------------------------------------


def _append_txn(
    txn_type: TxnType,
    amount: float,
    *,
    reference_id: str | None = None,
    note: str | None = None,
    repo: Path | None = None,
) -> str:
    """Append a single transaction to the log. Returns the txn_id.

    Args:
        txn_type: One of ``"deposit"``, ``"withdraw"``, ``"bet_placed"``,
            ``"bet_settled"``.
        amount: Transaction amount (must be >= 0).
        reference_id: Optional reference (e.g. bet_id).
        note: Optional human-readable note.
        repo: Repository root override.

    Returns:
        The generated ``txn_id`` (UUID string).

    Raises:
        ValueError: If ``amount`` is negative.
    """
    if amount < 0:
        msg: str = f"Transaction amount must be >= 0, got {amount}"
        raise ValueError(msg)

    txn_id = str(uuid.uuid4())
    row: dict[str, datetime | float | str | None] = {
        "txn_id": txn_id,
        "timestamp": datetime.now(UTC),
        "txn_type": txn_type,
        "amount": amount,
        "reference_id": reference_id,
        "note": note,
    }

    new_row = pd.DataFrame([row], columns=_TXN_COLUMNS)
    existing: DataFrame = _read_txn_log(repo)

    if existing.empty:
        combined: DataFrame = new_row
    else:
        combined = pd.concat(
            [existing.dropna(axis=1, how="all"), new_row.dropna(axis=1, how="all")],
            ignore_index=True,
        ).reindex(columns=_TXN_COLUMNS)

    _write_txn_log(combined, repo)
    logger.info("Txn %s: %s %.2f", txn_id, txn_type, amount)
    return txn_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deposit(
    amount: float,
    *,
    note: str | None = None,
    repo: Path | None = None,
) -> str:
    """Record a deposit (money added to bankroll).

    Args:
        amount: Deposit amount (must be > 0).
        note: Optional note.
        repo: Repository root override.

    Returns:
        Transaction ID.

    Raises:
        ValueError: If ``amount`` is not positive.
    """
    if amount <= 0:
        msg: str = f"Deposit amount must be positive, got {amount}"
        raise ValueError(msg)
    return _append_txn("deposit", amount, note=note, repo=repo)


def withdraw(
    amount: float,
    *,
    note: str | None = None,
    repo: Path | None = None,
) -> str:
    """Record a withdrawal (money removed from bankroll).

    Does **not** check whether the balance would go negative — that is
    a CLI-layer concern, not a library concern.

    Args:
        amount: Withdrawal amount (must be > 0).
        note: Optional note.
        repo: Repository root override.

    Returns:
        Transaction ID.

    Raises:
        ValueError: If ``amount`` is not positive.
    """
    if amount <= 0:
        msg: str = f"Withdrawal amount must be positive, got {amount}"
        raise ValueError(msg)
    return _append_txn("withdraw", amount, note=note, repo=repo)


def record_bet_placed(
    stake: float,
    *,
    bet_id: str | None = None,
    repo: Path | None = None,
) -> str:
    """Record a bet placement (stake leaves the bankroll).

    Args:
        stake: Amount wagered (must be > 0).
        bet_id: The bet's UUID for cross-referencing.
        repo: Repository root override.

    Returns:
        Transaction ID.

    Raises:
        ValueError: If ``stake`` is not positive.
    """
    if stake <= 0:
        msg: str = f"Stake must be positive, got {stake}"
        raise ValueError(msg)
    return _append_txn(
        "bet_placed",
        stake,
        reference_id=bet_id,
        note=f"Bet placed: {bet_id}",
        repo=repo,
    )


def record_bet_settled(
    stake: float,
    pnl: float,
    *,
    bet_id: str | None = None,
    repo: Path | None = None,
) -> str:
    """Record a bet settlement (gross return enters the bankroll).

    The gross return is ``stake + pnl``:

    - **Won:** ``stake + profit`` (e.g. 100 + 150 = 250)
    - **Lost:** ``stake + (-stake) = 0`` (nothing returns)
    - **Push:** ``stake + 0 = stake`` (original stake returns)

    If the gross return is zero (a loss), the transaction is still
    recorded for audit trail purposes.

    Args:
        stake: Original stake amount.
        pnl: Profit/loss from ``compute_pnl()``.
        bet_id: The bet's UUID for cross-referencing.
        repo: Repository root override.

    Returns:
        Transaction ID.
    """
    gross_return: float = stake + pnl
    # Clamp to zero (shouldn't happen, but defensive)
    gross_return = max(gross_return, 0.0)
    return _append_txn(
        "bet_settled",
        gross_return,
        reference_id=bet_id,
        note=f"Bet settled: {bet_id} (PnL={pnl:+.2f})",
        repo=repo,
    )


def current_balance(*, repo: Path | None = None) -> float:
    """Compute the current bankroll balance.

    Returns the sum of all signed transactions. Returns ``0.0`` if
    no transactions exist.
    """
    df: DataFrame = _read_txn_log(repo)
    if df.empty:
        return 0.0
    signs = _signed_amount_series(df["txn_type"], df["amount"])
    return float(signs.sum())


def balance_history(*, repo: Path | None = None) -> pd.DataFrame:
    """Build a running balance history.

    Returns a DataFrame sorted by timestamp with columns:
    ``timestamp``, ``txn_type``, ``amount``, ``signed_amount``,
    ``running_balance``.
    """
    df: DataFrame = _read_txn_log(repo)
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp", "txn_type", "amount", "signed_amount", "running_balance"],
        )
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["signed_amount"] = _signed_amount_series(df["txn_type"], df["amount"])
    df["running_balance"] = df["signed_amount"].cumsum()
    return df.loc[:, ["timestamp", "txn_type", "amount", "signed_amount", "running_balance"]]


def load_transactions(
    *,
    txn_type: str | None = None,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Load transactions with optional filter.

    Args:
        txn_type: Filter to this transaction type.
        repo: Repository root override.

    Returns:
        Filtered DataFrame. Empty with correct schema if no matches.
    """
    df: DataFrame = _read_txn_log(repo)
    if df.empty:
        return df
    if txn_type is not None:
        df = df.loc[df["txn_type"] == txn_type, :]
    return df.reset_index(drop=True)

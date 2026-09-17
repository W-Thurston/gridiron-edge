# src/gridiron_edge/betting/bankroll.py
"""Bankroll transaction ledger for every cash inflow and outflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import logging
from math import isfinite
import os
from pathlib import Path
from typing import Final, Literal
import uuid

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)

TxnType: type[TxnType] = Literal["deposit", "withdraw", "bet_placed", "bet_settled"]
_INFLOWS: Final[frozenset[str]] = frozenset({"deposit", "bet_settled"})
_OUTFLOWS: Final[frozenset[str]] = frozenset({"withdraw", "bet_placed"})

_TXN_COLUMNS: Final[list[str]] = [
    "txn_id",
    "source_transaction_id",
    "timestamp",
    "txn_type",
    "amount",
    "balance_after",
    "reference_id",
    "note",
]
_LEGACY_TXN_COLUMNS: Final[list[str]] = [
    "txn_id",
    "timestamp",
    "txn_type",
    "amount",
    "reference_id",
    "note",
]
_BANKROLL_SOURCE_KIND: Final[str] = "bankroll_transaction_ledger"


@dataclass(frozen=True, slots=True)
class BankrollSnapshot:
    """Content-identified bankroll evidence at one UTC cutoff."""

    amount: float
    observed_at: datetime
    source_kind: str
    source_id: str


def _txn_path(repo: Path | None = None) -> Path:
    """Return the canonical bankroll transaction path."""
    if repo is None:
        from gridiron_edge.core.settings import get_settings

        repo = get_settings().repo_root
    path = repo / "data" / "betting" / "bankroll_txn.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _empty_txn_log() -> DataFrame:
    return pd.DataFrame(columns=_TXN_COLUMNS)


def _read_txn_log(repo: Path | None = None) -> DataFrame:
    """Read current data or normalize the immediately preceding schema."""
    path = _txn_path(repo)
    if not path.exists():
        return _empty_txn_log()
    df = pd.read_parquet(path)
    actual = df.columns.tolist()
    if actual == _TXN_COLUMNS:
        return df.loc[:, _TXN_COLUMNS]
    if actual != _LEGACY_TXN_COLUMNS:
        raise ValueError(
            "Existing bankroll transaction log does not match the current "
            "or supported legacy schema."
        )
    normalized = df.copy()
    normalized["source_transaction_id"] = None
    normalized["balance_after"] = None
    return normalized.loc[:, _TXN_COLUMNS]


def _write_txn_log(df: DataFrame, repo: Path | None = None) -> Path:
    """Normalize a supported frame and atomically publish the canonical schema."""
    actual = df.columns.tolist()
    if actual == _TXN_COLUMNS:
        canonical = df.loc[:, _TXN_COLUMNS]
    elif actual == _LEGACY_TXN_COLUMNS:
        canonical = df.copy()
        canonical["source_transaction_id"] = None
        canonical["balance_after"] = None
        canonical = canonical.loc[:, _TXN_COLUMNS]
    else:
        raise ValueError(
            "Bankroll transaction write does not match the canonical or supported legacy schema."
        )
    path = _txn_path(repo)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        canonical.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _require_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def signed_amount(txn_type: str, amount: float) -> float:
    """Return one amount with its bankroll sign."""
    return amount if txn_type in _INFLOWS else -amount


def _signed_amount_series(txn_types: pd.Series, amounts: pd.Series) -> pd.Series:
    import numpy as np

    return pd.Series(
        np.where(txn_types.isin(_INFLOWS), amounts, -amounts),
        index=txn_types.index,
    )


def _append_txn(
    txn_type: TxnType,
    amount: float,
    *,
    reference_id: str | None = None,
    note: str | None = None,
    timestamp: datetime | None = None,
    source_transaction_id: str | None = None,
    balance_after: float | None = None,
    repo: Path | None = None,
) -> str:
    """Append one validated transaction and return its internal UUID."""
    if not isfinite(amount) or amount < 0:
        raise ValueError(f"Transaction amount must be finite and >= 0, got {amount}")
    if balance_after is not None and not isfinite(balance_after):
        raise ValueError("balance_after must be finite when provided.")
    recorded_at = (
        datetime.now(UTC)
        if timestamp is None
        else _require_utc(
            timestamp,
            label="transaction timestamp",
        )
    )
    txn_id = str(uuid.uuid4())
    row: dict[str, object] = {
        "txn_id": txn_id,
        "source_transaction_id": source_transaction_id,
        "timestamp": recorded_at,
        "txn_type": txn_type,
        "amount": amount,
        "balance_after": balance_after,
        "reference_id": reference_id,
        "note": note,
    }
    new_row = pd.DataFrame([row], columns=_TXN_COLUMNS)
    existing = _read_txn_log(repo)
    combined = (
        new_row
        if existing.empty
        else pd.concat(
            [
                existing.dropna(axis=1, how="all"),
                new_row.dropna(axis=1, how="all"),
            ],
            ignore_index=True,
        ).reindex(columns=_TXN_COLUMNS)
    )
    _write_txn_log(combined, repo)
    logger.info("Txn %s: %s %.2f", txn_id, txn_type, amount)
    return txn_id


def deposit(
    amount: float,
    *,
    note: str | None = None,
    timestamp: datetime | None = None,
    source_transaction_id: str | None = None,
    balance_after: float | None = None,
    repo: Path | None = None,
) -> str:
    """Record money added to the bankroll."""
    if not isfinite(amount) or amount <= 0:
        raise ValueError(f"Deposit amount must be finite and positive, got {amount}")
    return _append_txn(
        "deposit",
        amount,
        note=note,
        timestamp=timestamp,
        source_transaction_id=source_transaction_id,
        balance_after=balance_after,
        repo=repo,
    )


def withdraw(
    amount: float,
    *,
    note: str | None = None,
    timestamp: datetime | None = None,
    source_transaction_id: str | None = None,
    balance_after: float | None = None,
    repo: Path | None = None,
) -> str:
    """Record money removed from the bankroll."""
    if not isfinite(amount) or amount <= 0:
        raise ValueError(f"Withdrawal amount must be finite and positive, got {amount}")
    return _append_txn(
        "withdraw",
        amount,
        note=note,
        timestamp=timestamp,
        source_transaction_id=source_transaction_id,
        balance_after=balance_after,
        repo=repo,
    )


def record_bet_placed(
    stake: float,
    *,
    bet_id: str | None = None,
    placed_at: datetime | None = None,
    source_transaction_id: str | None = None,
    balance_after: float | None = None,
    repo: Path | None = None,
) -> str:
    """Record stake leaving the available cash bankroll."""
    if not isfinite(stake) or stake <= 0:
        raise ValueError(f"Stake must be finite and positive, got {stake}")
    return _append_txn(
        "bet_placed",
        stake,
        reference_id=bet_id,
        note=f"Bet placed: {bet_id}",
        timestamp=placed_at,
        source_transaction_id=source_transaction_id,
        balance_after=balance_after,
        repo=repo,
    )


def record_bet_settled(
    stake: float,
    pnl: float,
    *,
    bet_id: str | None = None,
    settled_at: datetime | None = None,
    source_transaction_id: str | None = None,
    balance_after: float | None = None,
    gross_return: float | None = None,
    repo: Path | None = None,
) -> str:
    """Record gross settlement return entering the cash bankroll."""
    returned = max(stake + pnl, 0.0) if gross_return is None else gross_return
    if not isfinite(returned) or returned < 0:
        raise ValueError("gross_return must be finite and nonnegative.")
    return _append_txn(
        "bet_settled",
        returned,
        reference_id=bet_id,
        note=f"Bet settled: {bet_id} (PnL={pnl:+.2f})",
        timestamp=settled_at,
        source_transaction_id=source_transaction_id,
        balance_after=balance_after,
        repo=repo,
    )


def current_balance(*, repo: Path | None = None) -> float:
    """Return the current available cash bankroll."""
    df = _read_txn_log(repo)
    if df.empty:
        return 0.0
    return float(_signed_amount_series(df["txn_type"], df["amount"]).sum())


def balance_history(*, repo: Path | None = None) -> DataFrame:
    """Return transactions in time order with a running cash balance."""
    df = _read_txn_log(repo)
    columns = ["timestamp", "txn_type", "amount", "signed_amount", "running_balance"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    ordered = df.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
    ordered["signed_amount"] = _signed_amount_series(
        ordered["txn_type"],
        ordered["amount"],
    )
    ordered["running_balance"] = ordered["signed_amount"].cumsum()
    return ordered.loc[:, columns]


def load_transactions(
    *,
    txn_type: str | None = None,
    repo: Path | None = None,
) -> DataFrame:
    """Load bankroll transactions with an optional type filter."""
    df = _read_txn_log(repo)
    if df.empty:
        return df
    if txn_type is not None:
        df = df.loc[df["txn_type"] == txn_type, :]
    return df.reset_index(drop=True)


def _require_valid_evidence(transactions: DataFrame) -> None:
    if transactions.columns.tolist() != _TXN_COLUMNS:
        raise ValueError("Bankroll transaction log does not match the canonical schema.")
    if transactions["txn_id"].isna().any():
        raise ValueError("Bankroll transaction log contains a null txn_id.")
    if transactions["txn_id"].duplicated().any():
        raise ValueError("Bankroll transaction log contains duplicate txn_id values.")
    for value in transactions["timestamp"]:
        _require_utc(pd.Timestamp(value).to_pydatetime(), label="transaction timestamp")
    unknown = set(transactions["txn_type"]) - _INFLOWS - _OUTFLOWS
    if unknown:
        raise ValueError(
            f"Bankroll transaction log contains unknown txn_type values: {sorted(unknown)}"
        )
    amounts = transactions["amount"].to_numpy(dtype=float)
    import numpy as np

    if not bool(np.isfinite(amounts).all()) or bool((amounts < 0).any()):
        raise ValueError("Bankroll transaction log contains a non-finite or negative amount.")


def bankroll_snapshot_as_of(
    cutoff: datetime,
    *,
    repo: Path | None = None,
) -> BankrollSnapshot | None:
    """Derive deterministic bankroll evidence visible at a UTC cutoff."""
    cutoff_utc = _require_utc(cutoff, label="cutoff")
    transactions = _read_txn_log(repo)
    if transactions.empty:
        return None
    _require_valid_evidence(transactions)
    visible = transactions.loc[transactions["timestamp"] <= cutoff_utc, :].copy()
    if visible.empty:
        return None
    visible = visible.sort_values(["timestamp", "txn_id"], kind="stable").reset_index(drop=True)
    amount = float(_signed_amount_series(visible["txn_type"], visible["amount"]).sum())
    material_rows = [
        {
            "txn_id": str(row["txn_id"]),
            "timestamp": pd.Timestamp(row["timestamp"]).to_pydatetime().isoformat(),
            "txn_type": str(row["txn_type"]),
            "amount": float(row["amount"]),
            "reference_id": None if pd.isna(row["reference_id"]) else str(row["reference_id"]),
        }
        for _, row in visible.iterrows()
    ]
    source_id = _digest({"cutoff": cutoff_utc.isoformat(), "transactions": material_rows})
    return BankrollSnapshot(amount, cutoff_utc, _BANKROLL_SOURCE_KIND, source_id)

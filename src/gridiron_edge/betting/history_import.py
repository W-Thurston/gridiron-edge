# src/gridiron_edge/betting/history_import.py
"""Validated replacement import for normalized historical betting records.

The importer treats the normalized bet-slip CSV and cash-transaction CSV as
separate evidence sources. Validation constructs both canonical ledgers fully
in memory before publication. Dry runs never write. Replacement publication is
coordinated under the bet-ledger lock and uses compensating restoration if any
write or reload validation fails.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Final
import uuid

import pandas as pd
from pandas import DataFrame

from gridiron_edge.betting._artifact_transaction import (
    restore_artifacts,
    snapshot_artifact,
)
from gridiron_edge.betting.bankroll import (
    _TXN_COLUMNS,
    _read_txn_log,
    _txn_path,
    _write_txn_log,
    signed_amount,
)
from gridiron_edge.betting.ledger import (
    _BET_COLUMNS,
    _HISTORICAL_MARKETS,
    _LEDGER_LOCK,
    _bet_ledger_path,
    _read_ledger,
    _write_ledger,
)

_EXPECTED_BET_COLUMNS: Final[list[str]] = [
    "placed_at_local",
    "source_bet_id",
    "description",
    "market_type",
    "game_id",
    "side",
    "line",
    "american_odds",
    "stake",
    "status",
    "paid_amount",
    "potential_payout",
    "sportsbook",
    "funding_type",
    "source_transaction_id",
    "notes",
]
_EXPECTED_TRANSACTION_COLUMNS: Final[list[str]] = [
    "timestamp_local",
    "txn_type",
    "amount",
    "balance_after",
    "source_transaction_id",
    "linked_source_bet_id",
    "notes",
]
_VALID_STATUSES: Final[frozenset[str]] = frozenset({"open", "won", "lost", "push"})
_VALID_FUNDING_TYPES: Final[frozenset[str]] = frozenset({"cash", "bonus", "unresolved"})
_TRANSACTION_TYPE_MAP: Final[dict[str, str]] = {
    "opening_balance": "deposit",
    "deposit": "deposit",
    "withdraw": "withdraw",
    "bet_placed": "bet_placed",
    "bet_settled": "bet_settled",
}
_BALANCE_TOLERANCE: Final[float] = 0.005


@dataclass(frozen=True, slots=True)
class HistoricalImportResult:
    """Validated historical-import totals and publication status."""

    bet_count: int
    settled_bet_count: int
    open_bet_count: int
    won_bet_count: int
    lost_bet_count: int
    transaction_count: int
    opening_balance: float
    cash_wager_outflow: float
    settlement_inflow: float
    final_bankroll: float
    bonus_funding: float
    unresolved_funding: float
    published: bool


@dataclass(frozen=True, slots=True)
class _PreparedImport:
    bets: DataFrame
    transactions: DataFrame
    result: HistoricalImportResult


def import_history(
    *,
    bets_path: Path,
    transactions_path: Path,
    replace_existing: bool = False,
    repo: Path,
) -> HistoricalImportResult:
    """Validate normalized history and optionally replace both ledgers.

    Args:
        bets_path: Normalized historical bet-slip CSV.
        transactions_path: Normalized cash-transaction CSV.
        replace_existing: Publish both canonical ledgers when true. False is
            a non-writing dry run.
        repo: Repository root containing ``data/betting``.

    Returns:
        Validated totals and whether publication occurred.

    Raises:
        ValueError: If either source or the cross-source accounting contract is
            invalid.
        RuntimeError: If publication fails and compensating restoration is
            incomplete.
    """
    prepared = _prepare_import(
        bets_path=bets_path,
        transactions_path=transactions_path,
    )
    if not replace_existing:
        return prepared.result

    with _LEDGER_LOCK:
        ledger_snapshot = snapshot_artifact(_bet_ledger_path(repo))
        transaction_snapshot = snapshot_artifact(_txn_path(repo))
        try:
            _write_ledger(prepared.bets, repo)
            _write_txn_log(prepared.transactions, repo)
            reloaded_bets = _read_ledger(repo)
            reloaded_transactions = _read_txn_log(repo)
            _validate_published(
                expected=prepared,
                bets=reloaded_bets,
                transactions=reloaded_transactions,
            )
        except Exception:
            restore_artifacts(
                (ledger_snapshot, transaction_snapshot),
                failure_message=(
                    "Historical import failed and artifact restoration was incomplete."
                ),
            )
            raise

    return replace(prepared.result, published=True)


def _prepare_import(
    *,
    bets_path: Path,
    transactions_path: Path,
) -> _PreparedImport:
    source_bets = pd.read_csv(bets_path, keep_default_na=False)
    source_transactions = pd.read_csv(
        transactions_path,
        keep_default_na=False,
    )
    _require_columns(source_bets, _EXPECTED_BET_COLUMNS, label="Bet import")
    _require_columns(
        source_transactions,
        _EXPECTED_TRANSACTION_COLUMNS,
        label="Transaction import",
    )
    bets, source_to_internal = _build_bets(source_bets)
    transactions, totals = _build_transactions(
        source_transactions,
        source_to_internal=source_to_internal,
    )
    bonus = float(bets.loc[bets["funding_type"] == "bonus", "stake"].sum())
    unresolved = float(bets.loc[bets["funding_type"] == "unresolved", "stake"].sum())
    result = HistoricalImportResult(
        bet_count=len(bets),
        settled_bet_count=int((bets["status"] != "open").sum()),
        open_bet_count=int((bets["status"] == "open").sum()),
        won_bet_count=int((bets["status"] == "won").sum()),
        lost_bet_count=int((bets["status"] == "lost").sum()),
        transaction_count=len(transactions),
        opening_balance=totals["opening_balance"],
        cash_wager_outflow=totals["cash_wager_outflow"],
        settlement_inflow=totals["settlement_inflow"],
        final_bankroll=totals["final_bankroll"],
        bonus_funding=bonus,
        unresolved_funding=unresolved,
        published=False,
    )
    return _PreparedImport(bets, transactions, result)


def _require_columns(
    frame: DataFrame,
    expected: list[str],
    *,
    label: str,
) -> None:
    actual = frame.columns.tolist()
    if actual != expected:
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        problems: list[str] = []
        if missing:
            problems.append("missing columns: " + ", ".join(missing))
        if extra:
            problems.append("extra columns: " + ", ".join(extra))
        if not missing and not extra:
            problems.append("columns are not in canonical order")
        raise ValueError(f"{label} schema invalid: {'; '.join(problems)}")


def _build_bets(
    source: DataFrame,
) -> tuple[DataFrame, dict[str, str]]:
    if source.empty:
        raise ValueError("Bet import must contain at least one row.")
    source_ids = source["source_bet_id"].astype(str)
    if source_ids.str.strip().eq("").any():
        raise ValueError("Bet import contains an empty source_bet_id.")
    if source_ids.duplicated().any():
        raise ValueError("Bet import contains duplicate source_bet_id values.")

    rows: list[dict[str, object]] = []
    source_to_internal: dict[str, str] = {}
    for number, (_, row) in enumerate(source.iterrows(), start=2):
        source_id = str(row["source_bet_id"]).strip()
        bet_id = str(uuid.uuid4())
        source_to_internal[source_id] = bet_id
        market = str(row["market_type"]).strip()
        status = str(row["status"]).strip()
        game_id = _optional_text(row["game_id"])
        side = _optional_text(row["side"])
        description = _optional_text(row["description"])
        odds = _integer(row["american_odds"], label=f"Bet row {number} odds")
        stake = _positive_float(row["stake"], label=f"Bet row {number} stake")
        line = _optional_float(row["line"], label=f"Bet row {number} line")
        paid = _optional_float(
            row["paid_amount"],
            label=f"Bet row {number} paid_amount",
        )
        potential = _optional_float(
            row["potential_payout"],
            label=f"Bet row {number} potential_payout",
        )
        funding = str(row["funding_type"]).strip()

        if market not in _HISTORICAL_MARKETS:
            raise ValueError(f"Bet row {number} has unknown market_type: {market!r}")
        if status not in _VALID_STATUSES:
            raise ValueError(f"Bet row {number} has unknown status: {status!r}")
        if funding not in _VALID_FUNDING_TYPES:
            raise ValueError(f"Bet row {number} has unknown funding_type: {funding!r}")
        book = str(row["sportsbook"]).strip()
        if not book:
            raise ValueError(f"Bet row {number} has an empty sportsbook.")
        if market in {"moneyline", "spread", "total", "player_prop"} and not game_id:
            raise ValueError(f"Bet row {number} market {market!r} requires game_id.")
        if market == "moneyline" and side not in {"home", "away"}:
            raise ValueError(f"Bet row {number} moneyline requires home or away side.")
        if market in {"spread", "total"} and line is None:
            raise ValueError(f"Bet row {number} market {market!r} requires line.")
        if market in {"parlay", "same_game_parlay", "special"} and not description:
            raise ValueError(f"Bet row {number} market {market!r} requires description.")

        pnl = _historical_pnl(
            status=status,
            stake=stake,
            paid_amount=paid,
            label=f"Bet row {number}",
        )
        rows.append(
            {
                "bet_id": bet_id,
                "source_bet_id": source_id,
                "game_id": game_id,
                "description": description,
                "placed_at": _parse_timestamp(
                    row["placed_at_local"],
                    label=f"Bet row {number} placed_at_local",
                ),
                "market_type": market,
                "side": side,
                "line": line,
                "odds": odds,
                "stake": stake,
                "book": book,
                "funding_type": funding,
                "paid_amount": paid,
                "potential_payout": potential,
                "reference_provider": None,
                "reference_provider_event_id": None,
                "reference_sportsbook": None,
                "reference_market_fetched_at": None,
                "reference_sportsbook_updated_at": None,
                "reference_commence_time": None,
                "reference_american_odds": None,
                "reference_line": None,
                "recommended_bet_result_id": None,
                "recommendation_evaluation_id": None,
                "candidate_reference_id": None,
                "recommendation_policy_id": None,
                "model_name": None,
                "model_type": None,
                "model_prob": None,
                "model_ev": None,
                "edge_strength": None,
                "confidence_tier": None,
                "status": status,
                "settled_at": None,
                "pnl": pnl,
                "closing_line": None,
                "closing_odds": None,
                "clv": None,
            }
        )

    return pd.DataFrame(rows, columns=_BET_COLUMNS), source_to_internal


def _historical_pnl(
    *,
    status: str,
    stake: float,
    paid_amount: float | None,
    label: str,
) -> float | None:
    if status == "open":
        if paid_amount is not None:
            raise ValueError(f"{label} open bet must not provide paid_amount.")
        return None
    if status == "lost":
        if paid_amount not in {None, 0.0}:
            raise ValueError(f"{label} lost bet cannot have a positive paid_amount.")
        return -stake
    if status == "push":
        if paid_amount is not None and not isclose(
            paid_amount,
            stake,
            abs_tol=_BALANCE_TOLERANCE,
        ):
            raise ValueError(f"{label} push paid_amount must equal stake.")
        return 0.0
    if paid_amount is None or paid_amount <= 0:
        raise ValueError(f"{label} won bet requires positive paid_amount.")
    return paid_amount - stake


def _build_transactions(
    source: DataFrame,
    *,
    source_to_internal: dict[str, str],
) -> tuple[DataFrame, dict[str, float]]:
    if source.empty:
        raise ValueError("Transaction import must contain at least one row.")
    source = source.copy()
    source["_source_order"] = range(len(source))
    parsed = [
        _parse_timestamp(value, label=f"Transaction row {position + 2} timestamp_local")
        for position, value in enumerate(source["timestamp_local"])
    ]
    source["_timestamp"] = parsed
    source = source.sort_values(
        ["_timestamp", "_source_order"],
        kind="stable",
    ).reset_index(drop=True)

    source_ids = source["source_transaction_id"].astype(str).str.strip()
    nonempty_ids = source_ids.loc[source_ids.ne("")]
    if nonempty_ids.duplicated().any():
        raise ValueError("Transaction import contains duplicate source_transaction_id values.")

    rows: list[dict[str, object]] = []
    running = 0.0
    opening = 0.0
    wager_outflow = 0.0
    settlement_inflow = 0.0

    for source_order, (_, row) in enumerate(source.iterrows()):
        number = source_order + 2
        source_type = str(row["txn_type"]).strip()
        txn_type = _historical_transaction_type(
            source_type,
            source_order=source_order,
            row_number=number,
        )
        amount = _nonnegative_float(
            row["amount"],
            label=f"Transaction row {number} amount",
        )
        balance_after = _finite_float(
            row["balance_after"],
            label=f"Transaction row {number} balance_after",
        )
        if source_type == "opening_balance":
            if amount <= 0:
                raise ValueError("opening_balance amount must be positive.")
            opening = amount
        running += signed_amount(txn_type, amount)
        if not isclose(running, balance_after, abs_tol=_BALANCE_TOLERANCE):
            raise ValueError(
                f"Transaction row {number} running balance mismatch: "
                f"computed {running:.2f}, source {balance_after:.2f}."
            )
        if txn_type == "bet_placed":
            wager_outflow += amount
        elif txn_type == "bet_settled":
            settlement_inflow += amount

        linked_source_id = _optional_text(row["linked_source_bet_id"])
        if linked_source_id and linked_source_id not in source_to_internal:
            raise ValueError(
                f"Transaction row {number} references unknown source bet {linked_source_id!r}."
            )
        source_transaction_id = _optional_text(row["source_transaction_id"])
        if source_type != "opening_balance" and source_transaction_id is None:
            raise ValueError(f"Transaction row {number} requires source_transaction_id.")
        rows.append(
            {
                "txn_id": str(uuid.uuid4()),
                "source_transaction_id": source_transaction_id,
                "timestamp": row["_timestamp"],
                "txn_type": txn_type,
                "amount": amount,
                "balance_after": balance_after,
                "reference_id": (
                    source_to_internal[linked_source_id] if linked_source_id else None
                ),
                "note": _optional_text(row["notes"]),
            }
        )

    if opening <= 0:
        raise ValueError("Transaction import requires one opening_balance row.")
    transactions = pd.DataFrame(rows, columns=_TXN_COLUMNS)
    return transactions, {
        "opening_balance": opening,
        "cash_wager_outflow": wager_outflow,
        "settlement_inflow": settlement_inflow,
        "final_bankroll": running,
    }


def _historical_transaction_type(
    source_type: str,
    *,
    source_order: int,
    row_number: int,
) -> str:
    """Map one source transaction type and enforce opening-row placement."""
    if source_type not in _TRANSACTION_TYPE_MAP:
        raise ValueError(f"Transaction row {row_number} has unknown txn_type: {source_type!r}")
    if source_type == "opening_balance" and source_order != 0:
        raise ValueError("opening_balance must be the first transaction.")
    return _TRANSACTION_TYPE_MAP[source_type]


def _parse_timestamp(value: object, *, label: str) -> datetime:
    try:
        timestamp = pd.Timestamp(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid timestamp.") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must include an explicit timezone offset.")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _optional_text(value: object) -> str | None:
    text = str(value).strip()
    return text or None


def _integer(value: object, *, label: str) -> int:
    number = _finite_float(value, label=label)
    if not number.is_integer() or number == 0:
        raise ValueError(f"{label} must be a nonzero integer.")
    return int(number)


def _finite_float(value: object, *, label: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _positive_float(value: object, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if number <= 0:
        raise ValueError(f"{label} must be positive.")
    return number


def _nonnegative_float(value: object, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if number < 0:
        raise ValueError(f"{label} must be nonnegative.")
    return number


def _optional_float(value: object, *, label: str) -> float | None:
    if str(value).strip() == "":
        return None
    return _finite_float(value, label=label)


def _validate_published(
    *,
    expected: _PreparedImport,
    bets: DataFrame,
    transactions: DataFrame,
) -> None:
    if bets.columns.tolist() != _BET_COLUMNS:
        raise ValueError("Published bet ledger has an invalid schema.")
    if transactions.columns.tolist() != _TXN_COLUMNS:
        raise ValueError("Published transaction ledger has an invalid schema.")
    if bets["source_bet_id"].tolist() != expected.bets["source_bet_id"].tolist():
        raise ValueError("Published bet ledger does not match validated source identities.")
    if (
        transactions["source_transaction_id"].tolist()
        != expected.transactions["source_transaction_id"].tolist()
    ):
        raise ValueError("Published transaction ledger does not match validated source identities.")
    balance = float(
        sum(
            signed_amount(str(row["txn_type"]), float(row["amount"]))
            for _, row in transactions.iterrows()
        )
    )
    if not isclose(
        balance,
        expected.result.final_bankroll,
        abs_tol=_BALANCE_TOLERANCE,
    ):
        raise ValueError("Published transaction ledger has an invalid final balance.")

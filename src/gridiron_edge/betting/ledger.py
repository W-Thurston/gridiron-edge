# src/gridiron_edge/betting/ledger.py
"""Mutable bet ledger with immutable recorded-offer evidence.

The canonical ledger stores live single-game wagers and normalized historical
wagers. Writes publish the complete ledger atomically. Same-process mutation is
coordinated by ``_LEDGER_LOCK``; multi-process writers remain unsupported.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from math import isfinite
import os
from pathlib import Path
import threading
from typing import Final, Literal
import uuid

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.market.odds_math import american_to_decimal

logger = logging.getLogger(__name__)

BetStatus: type[BetStatus] = Literal["open", "won", "lost", "push"]
MarketType: type[MarketType] = Literal["moneyline", "spread", "total"]
FundingType: type[FundingType] = Literal["cash", "bonus", "unresolved"]

_HISTORICAL_MARKETS: Final[frozenset[str]] = frozenset(
    {
        "moneyline",
        "spread",
        "total",
        "player_prop",
        "parlay",
        "same_game_parlay",
        "special",
    }
)

_BET_COLUMNS: Final[list[str]] = [
    "bet_id",
    "source_bet_id",
    "game_id",
    "description",
    "placed_at",
    "market_type",
    "side",
    "line",
    "odds",
    "stake",
    "book",
    "funding_type",
    "paid_amount",
    "potential_payout",
    "reference_provider",
    "reference_provider_event_id",
    "reference_sportsbook",
    "reference_market_fetched_at",
    "reference_sportsbook_updated_at",
    "reference_commence_time",
    "reference_american_odds",
    "reference_line",
    "recommended_bet_result_id",
    "recommendation_evaluation_id",
    "candidate_reference_id",
    "recommendation_policy_id",
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

_LEGACY_BET_COLUMNS: Final[list[str]] = [
    column
    for column in _BET_COLUMNS
    if column
    not in {
        "source_bet_id",
        "description",
        "funding_type",
        "paid_amount",
        "potential_payout",
    }
]
_NEW_COLUMN_DEFAULTS: Final[dict[str, float | str | None]] = {
    "source_bet_id": None,
    "description": None,
    "funding_type": "cash",
    "paid_amount": None,
    "potential_payout": None,
}

_LEDGER_LOCK: Final[threading.RLock] = threading.RLock()


def _bet_ledger_path(repo: Path | None = None) -> Path:
    """Return the canonical bet-ledger path."""
    if repo is None:
        from gridiron_edge.core.settings import get_settings

        repo = get_settings().repo_root
    path = repo / "data" / "betting" / "bet_ledger.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _empty_ledger() -> DataFrame:
    """Return an empty canonical ledger."""
    return pd.DataFrame(columns=_BET_COLUMNS)


def _schema_problem(actual: list[str], expected: list[str]) -> str | None:
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    problems: list[str] = []
    if missing:
        problems.append("missing columns: " + ", ".join(missing))
    if extra:
        problems.append("extra columns: " + ", ".join(extra))
    if not missing and not extra and actual != expected:
        problems.append("columns are not in canonical order")
    return "; ".join(problems) if problems else None


def _require_ledger_schema(df: DataFrame, *, label: str) -> None:
    """Require the exact canonical persisted schema."""
    problem = _schema_problem(df.columns.tolist(), _BET_COLUMNS)
    if problem:
        raise ValueError(f"{label} does not match the current bet-ledger schema: {problem}")


def _normalize_legacy_ledger(df: DataFrame) -> DataFrame:
    """Upgrade only the immediately preceding ledger schema in memory."""
    actual = df.columns.tolist()
    if actual == _BET_COLUMNS:
        return df.loc[:, _BET_COLUMNS]
    if actual != _LEGACY_BET_COLUMNS:
        problem = _schema_problem(actual, _BET_COLUMNS)
        raise ValueError(
            "Existing bet ledger does not match the current or supported "
            f"legacy schema: {problem or 'unsupported schema'}"
        )
    normalized = df.copy()
    for column, default in _NEW_COLUMN_DEFAULTS.items():
        normalized[column] = default
    return normalized.loc[:, _BET_COLUMNS]


def _read_ledger(repo: Path | None = None) -> DataFrame:
    """Read and normalize the canonical or immediately preceding ledger."""
    path = _bet_ledger_path(repo)
    if not path.exists():
        return _empty_ledger()
    return _normalize_legacy_ledger(pd.read_parquet(path))


def _write_ledger(df: DataFrame, repo: Path | None = None) -> Path:
    """Atomically publish a complete canonical ledger."""
    _require_ledger_schema(df, label="Bet ledger write")
    path = _bet_ledger_path(repo)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _validate_model_identity(model_name: str | None, model_type: str | None) -> None:
    if model_name is None and model_type is None:
        return
    if model_name is None or model_type is None:
        raise ValueError("model_name and model_type must be provided together.")
    if not model_name.strip():
        raise ValueError("model_name must be a nonempty string when model identity is provided.")
    if not model_type.strip():
        raise ValueError("model_type must be a nonempty string when model identity is provided.")


def _validate_recommendation_provenance(
    *,
    recommended_bet_result_id: str | None,
    recommendation_evaluation_id: str | None,
    candidate_reference_id: str | None,
    recommendation_policy_id: str | None,
) -> None:
    identities = {
        "recommended_bet_result_id": recommended_bet_result_id,
        "recommendation_evaluation_id": recommendation_evaluation_id,
        "candidate_reference_id": candidate_reference_id,
        "recommendation_policy_id": recommendation_policy_id,
    }
    if all(value is None for value in identities.values()):
        return
    if any(value is None for value in identities.values()):
        raise ValueError(
            "Recommendation provenance must be entirely absent or provide "
            "result, evaluation, candidate, and policy identities."
        )
    for label, value in identities.items():
        if value is None or not value.strip():
            raise ValueError(f"{label} must be a nonempty string.")


def _require_utc_timestamp(value: datetime, *, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")


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
        raise ValueError("reference_provider must be nonempty when reference evidence is provided.")
    if reference_market_fetched_at is None:
        raise ValueError("reference_market_fetched_at is required with reference evidence.")
    for label, value in (
        ("reference_provider_event_id", reference_provider_event_id),
        ("reference_sportsbook", reference_sportsbook),
    ):
        if value is not None and not value.strip():
            raise ValueError(f"{label} must be null or nonempty.")
    _require_utc_timestamp(
        reference_market_fetched_at,
        label="reference_market_fetched_at",
    )
    for label, value in (
        ("reference_sportsbook_updated_at", reference_sportsbook_updated_at),
        ("reference_commence_time", reference_commence_time),
    ):
        if value is not None:
            _require_utc_timestamp(value, label=label)
    if reference_american_odds is not None and (
        reference_american_odds == 0 or not isfinite(reference_american_odds)
    ):
        raise ValueError("reference_american_odds must be finite and nonzero when provided.")
    if reference_line is not None and not isfinite(reference_line):
        raise ValueError("reference_line must be finite.")


def compute_pnl(stake: float, american_odds: int, result: BetStatus) -> float:
    """Return net profit or loss for a settlement result."""
    if result == "won":
        return stake * (american_to_decimal(american_odds) - 1.0)
    if result == "lost":
        return -stake
    return 0.0


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
    recommended_bet_result_id: str | None = None,
    recommendation_evaluation_id: str | None = None,
    candidate_reference_id: str | None = None,
    recommendation_policy_id: str | None = None,
    placed_at: datetime | None = None,
    repo: Path | None = None,
    source_bet_id: str | None = None,
    description: str | None = None,
    funding_type: FundingType = "cash",
    paid_amount: float | None = None,
    potential_payout: float | None = None,
) -> str:
    """Record one live wager and return its internal UUID."""
    _validate_model_identity(model_name, model_type)
    _validate_recommendation_provenance(
        recommended_bet_result_id=recommended_bet_result_id,
        recommendation_evaluation_id=recommendation_evaluation_id,
        candidate_reference_id=candidate_reference_id,
        recommendation_policy_id=recommendation_policy_id,
    )
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
    if funding_type not in {"cash", "bonus", "unresolved"}:
        raise ValueError("funding_type must be cash, bonus, or unresolved.")
    if placed_at is None:
        placed_at = datetime.now(UTC)
    else:
        _require_utc_timestamp(placed_at, label="placed_at")

    bet_id = str(uuid.uuid4())
    row: dict[str, object] = {
        "bet_id": bet_id,
        "source_bet_id": source_bet_id,
        "game_id": game_id,
        "description": description,
        "placed_at": placed_at,
        "market_type": market_type,
        "side": side,
        "line": line,
        "odds": odds,
        "stake": stake,
        "book": book,
        "funding_type": funding_type,
        "paid_amount": paid_amount,
        "potential_payout": potential_payout,
        "reference_provider": reference_provider,
        "reference_provider_event_id": reference_provider_event_id,
        "reference_sportsbook": reference_sportsbook,
        "reference_market_fetched_at": reference_market_fetched_at,
        "reference_sportsbook_updated_at": reference_sportsbook_updated_at,
        "reference_commence_time": reference_commence_time,
        "reference_american_odds": reference_american_odds,
        "reference_line": reference_line,
        "recommended_bet_result_id": recommended_bet_result_id,
        "recommendation_evaluation_id": recommendation_evaluation_id,
        "candidate_reference_id": candidate_reference_id,
        "recommendation_policy_id": recommendation_policy_id,
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
    with _LEDGER_LOCK:
        existing = _read_ledger(repo)
        combined = (
            new_row
            if existing.empty
            else pd.concat(
                [
                    existing.dropna(axis=1, how="all"),
                    new_row.dropna(axis=1, how="all"),
                ],
                ignore_index=True,
            ).reindex(columns=_BET_COLUMNS)
        )
        _write_ledger(combined, repo)
    logger.info("Bet logged: %s %s %s %s @ %s", bet_id, market_type, side, game_id, odds)
    return bet_id


def settle_bet(
    bet_id: str,
    result: BetStatus,
    *,
    settled_at: datetime | None = None,
    paid_amount: float | None = None,
    repo: Path | None = None,
) -> Series:
    """Settle an open wager, optionally preserving exact paid evidence."""
    if result not in {"won", "lost", "push"}:
        raise ValueError(f"Invalid result: {result!r}. Must be 'won', 'lost', or 'push'.")
    settlement_time = settled_at or datetime.now(UTC)
    _require_utc_timestamp(settlement_time, label="settled_at")

    with _LEDGER_LOCK:
        ledger = _read_ledger(repo)
        mask = ledger["bet_id"] == bet_id
        if not mask.any():
            raise ValueError(f"Bet not found: {bet_id}")
        idx: int | str = mask.idxmax()
        bet = ledger.loc[idx]
        if bet["status"] != "open":
            raise ValueError(f"Bet {bet_id} is already settled (status={bet['status']!r}).")
        stake = float(bet["stake"])
        if paid_amount is not None:
            if not isfinite(paid_amount) or paid_amount < 0:
                raise ValueError("paid_amount must be finite and nonnegative.")
            pnl = paid_amount - stake
        else:
            pnl = compute_pnl(stake, int(bet["odds"]), result)
        ledger["settled_at"] = pd.to_datetime(ledger["settled_at"], utc=True)
        ledger.loc[idx, "status"] = result
        ledger.loc[idx, "settled_at"] = settlement_time
        ledger.loc[idx, "paid_amount"] = paid_amount
        ledger.loc[idx, "pnl"] = pnl
        ledger.loc[idx, ["closing_line", "closing_odds", "clv"]] = None
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
) -> DataFrame:
    """Load bets with optional filters."""
    df = _read_ledger(repo)
    if df.empty:
        return df
    if status is not None:
        df = df.loc[df["status"] == status, :]
    if market_type is not None:
        df = df.loc[df["market_type"] == market_type, :]
    if book is not None:
        df = df.loc[df["book"] == book, :]
    game_ids = df["game_id"].astype("string")
    if season is not None:
        df = df.loc[game_ids.str.startswith(season[:4], na=False), :]
        game_ids = df["game_id"].astype("string")
    if week is not None:
        df = df.loc[game_ids.str[5:7].eq(f"{week:02d}").fillna(False), :]
    return df.reset_index(drop=True)

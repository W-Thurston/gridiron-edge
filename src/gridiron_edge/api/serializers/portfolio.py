# src/gridiron_edge/api/serializers/portfolio.py
"""Hand-written serializers for /portfolio/* endpoints."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gridiron_edge.api.meta import ResponseMeta, Unavailable
from gridiron_edge.api.schemas._base import BaseListResponse
from gridiron_edge.api.schemas.portfolio import (
    BankrollCurve,
    BetRow,
    CurveBucket,
    PortfolioSplits,
    PortfolioSummary,
    RecordBetResponse,
    SplitRow,
    TransactionRow,
)

_BetsList = BaseListResponse[BetRow]
_TransactionsList = BaseListResponse[TransactionRow]


def _none_if_nan(value: Any) -> Any:  # noqa: ANN401
    """Return None for missing pandas values and the original value otherwise."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def serialize_portfolio_summary(
    bets: pd.DataFrame, current_bankroll: float, perf: dict
) -> PortfolioSummary:
    """Build the /portfolio/summary response."""
    if bets.empty:
        meta = ResponseMeta().with_blocked("win_pct", *Unavailable.NO_SETTLED_BETS)
        meta = meta.with_blocked("roi_pct", *Unavailable.NO_SETTLED_BETS)
        return PortfolioSummary(
            bankroll=current_bankroll,
            total_bets=0,
            settled_bets=0,
            open_bets=0,
            # pyrefly: ignore [unexpected-keyword]
            response_meta=meta,
        )
    settled_mask = bets["status"] != "open"
    streak_label = _compose_streak_label(
        perf.get("current_streak"), perf.get("current_streak_type")
    )
    meta = ResponseMeta()
    if _none_if_nan(perf.get("win_pct")) is None:
        meta = meta.with_blocked("win_pct", *Unavailable.NO_SETTLED_BETS)
    if _none_if_nan(perf.get("roi_pct")) is None:
        meta = meta.with_blocked("roi_pct", *Unavailable.NO_SETTLED_BETS)
    if _none_if_nan(perf.get("mean_clv")) is None:
        meta = meta.with_blocked("mean_clv", *Unavailable.NO_CLV_DATA)
        meta = meta.with_blocked("pct_positive_clv", *Unavailable.NO_CLV_DATA)
    if _none_if_nan(perf.get("mean_ev_at_bet")) is None:
        meta = meta.with_blocked("mean_ev_at_bet", *Unavailable.NO_MODEL_CONTEXT)
        meta = meta.with_blocked("ev_vs_actual_gap", *Unavailable.NO_MODEL_CONTEXT)
    if streak_label is None:
        meta = meta.with_blocked("current_streak", *Unavailable.NO_STREAK_ACTIVITY)
    return PortfolioSummary(
        bankroll=current_bankroll,
        total_bets=len(bets),
        settled_bets=settled_mask.sum(),
        open_bets=(~settled_mask).sum(),
        wins=perf.get("wins"),
        losses=perf.get("losses"),
        pushes=perf.get("pushes"),
        win_pct=_none_if_nan(perf.get("win_pct")),
        total_staked=_none_if_nan(perf.get("total_staked")),
        total_pnl=_none_if_nan(perf.get("total_pnl")),
        roi_pct=_none_if_nan(perf.get("roi_pct")),
        mean_clv=_none_if_nan(perf.get("mean_clv")),
        pct_positive_clv=_none_if_nan(perf.get("pct_positive_clv")),
        n_clv_bets=perf.get("n_clv_bets"),
        mean_ev_at_bet=_none_if_nan(perf.get("mean_ev_at_bet")),
        ev_vs_actual_gap=_none_if_nan(perf.get("ev_vs_actual_gap")),
        n_model_bets=perf.get("n_model_bets"),
        calibration_health=perf.get("calibration_health"),
        current_streak=streak_label,
        longest_win_streak=perf.get("longest_win_streak"),
        longest_loss_streak=perf.get("longest_loss_streak"),
        response_meta=meta if meta.field_status else None,  # pyrefly: ignore [unexpected-keyword]
    )


def _compose_streak_label(count: int | None, streak_type: str | None) -> str | None:
    """Compose a wire-friendly streak label."""
    if count is None or count == 0 or streak_type in (None, "none", ""):
        return None
    prefix = {"win": "W", "loss": "L", "push": "P"}.get(streak_type)
    return f"{prefix}{abs(count)}" if prefix else None


def serialize_bets(bets: pd.DataFrame) -> _BetsList:
    """Build the /portfolio/bets response."""
    if bets.empty:
        return _BetsList(items=[], total=0)
    rows = [
        BetRow(
            bet_id=_none_if_nan(row.get("bet_id")),
            source_bet_id=_none_if_nan(row.get("source_bet_id")),
            game_id=_none_if_nan(row.get("game_id")),
            description=_none_if_nan(row.get("description")),
            placed_at=str(row["placed_at"]) if pd.notna(row.get("placed_at")) else None,
            market_type=_none_if_nan(row.get("market_type")),
            side=_none_if_nan(row.get("side")),
            line=_none_if_nan(row.get("line")),
            odds=int(row["odds"]) if pd.notna(row.get("odds")) else None,
            stake=float(row["stake"]) if pd.notna(row.get("stake")) else None,
            book=_none_if_nan(row.get("book")),
            funding_type=_none_if_nan(row.get("funding_type")),
            paid_amount=_none_if_nan(row.get("paid_amount")),
            potential_payout=_none_if_nan(row.get("potential_payout")),
            status=_none_if_nan(row.get("status")),
            pnl=_none_if_nan(row.get("pnl")),
            closing_line=_none_if_nan(row.get("closing_line")),
            clv=_none_if_nan(row.get("clv")),
            model_name=_none_if_nan(row.get("model_name")),
            model_type=_none_if_nan(row.get("model_type")),
            recommended_bet_result_id=_none_if_nan(row.get("recommended_bet_result_id")),
            recommendation_evaluation_id=_none_if_nan(row.get("recommendation_evaluation_id")),
            candidate_reference_id=_none_if_nan(row.get("candidate_reference_id")),
            recommendation_policy_id=_none_if_nan(row.get("recommendation_policy_id")),
        )
        for _, row in bets.iterrows()
    ]
    return _BetsList(items=rows, total=len(rows))


def serialize_bankroll_curve(history: pd.DataFrame, period: str | None) -> BankrollCurve:
    """Build the /portfolio/curve response."""
    meta = ResponseMeta()
    if period is None:
        meta = meta.with_blocked("period", *Unavailable.PERIOD_NOT_REQUESTED)
    items = [
        CurveBucket(timestamp=str(row["timestamp"]), bankroll=float(row["running_balance"]))
        for _, row in history.iterrows()
    ]
    return BankrollCurve(
        items=items,
        total=len(items),
        period=period,
        # pyrefly: ignore [unexpected-keyword]
        response_meta=meta if meta.field_status else None,
    )


def serialize_transactions(txns: pd.DataFrame) -> _TransactionsList:
    """Build the /portfolio/transactions response."""
    if txns.empty:
        return _TransactionsList(items=[], total=0)
    rows = [
        TransactionRow(
            txn_id=_none_if_nan(row.get("txn_id")),
            source_transaction_id=_none_if_nan(row.get("source_transaction_id")),
            timestamp=str(row["timestamp"]) if pd.notna(row.get("timestamp")) else None,
            txn_type=_none_if_nan(row.get("txn_type")),
            amount=float(row["amount"]) if pd.notna(row.get("amount")) else None,
            balance_after=float(row["balance_after"])
            if pd.notna(row.get("balance_after"))
            else None,
            reference_id=_none_if_nan(row.get("reference_id")),
            note=_none_if_nan(row.get("note")),
        )
        for _, row in txns.iterrows()
    ]
    return _TransactionsList(items=rows, total=len(rows))


def serialize_splits(splits_df: pd.DataFrame, dimension: str) -> PortfolioSplits:
    """Build the /portfolio/splits response."""
    if splits_df.empty:
        meta = ResponseMeta().with_blocked("items", *Unavailable.NO_SPLIT_DATA)
        return PortfolioSplits(
            items=[],
            total=0,
            dimension=dimension,
            # pyrefly: ignore [unexpected-keyword]
            response_meta=meta,
        )
    rows = [
        SplitRow(
            dimension_value=str(row[dimension]),
            total=int(row["total"]) if pd.notna(row.get("total")) else None,
            wins=int(row["wins"]) if pd.notna(row.get("wins")) else None,
            losses=int(row["losses"]) if pd.notna(row.get("losses")) else None,
            pushes=int(row["pushes"]) if pd.notna(row.get("pushes")) else None,
            win_pct=_none_if_nan(row.get("win_pct")),
            roi=_none_if_nan(row.get("roi")),
        )
        for _, row in splits_df.iterrows()
    ]
    return PortfolioSplits(items=rows, total=len(rows), dimension=dimension)


def serialize_recorded_bet(
    bets: pd.DataFrame, *, bankroll_transaction_id: str
) -> RecordBetResponse:
    """Serialize one newly recorded wager."""
    serialized = serialize_bets(bets)
    if serialized.total != 1 or len(serialized.items) != 1:
        raise ValueError("Recorded wager could not be loaded uniquely.")
    return RecordBetResponse(
        bet=serialized.items[0], bankroll_transaction_id=bankroll_transaction_id
    )

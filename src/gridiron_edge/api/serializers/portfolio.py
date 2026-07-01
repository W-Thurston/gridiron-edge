# src/gridiron_edge/api/serializers/portfolio.py

"""Serializers for /portfolio/* endpoints.

Per D17, one hand-written function per endpoint. Per D18, serializers
own construction of `_meta.field_status`. Per D19, serializers accept
already-loaded DataFrames — they do not touch settings or the filesystem.
"""

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
    SplitRow,
    TransactionRow,
)

# Type aliases for the parameterized list responses. Assigning them to
# names avoids the fragile `BaseListResponse[T](` construction pattern
# and makes intent clearer at call sites.
_BetsList = BaseListResponse[BetRow]
_TransactionsList = BaseListResponse[TransactionRow]


def _none_if_nan(v: Any) -> Any:  # noqa: ANN401
    """Return None for pandas NaN or None; else the value itself."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def serialize_portfolio_summary(
    bets: pd.DataFrame,
    current_bankroll: float,
    perf: dict,
) -> PortfolioSummary:
    """Build the /portfolio/summary response from ledger + bankroll + perf."""
    if bets.empty:
        return PortfolioSummary(
            bankroll=current_bankroll,
            total_bets=0,
            settled_bets=0,
            open_bets=0,
        )

    settled_mask = bets["status"] != "open"

    # Compose streak label and track whether it's null due to no activity.
    streak_label = _compose_streak_label(
        perf.get("current_streak"),
        perf.get("current_streak_type"),
    )

    # Build response_meta for fields that are null due to data limits.
    meta = ResponseMeta()
    if perf.get("mean_clv") is None or (
        isinstance(perf.get("mean_clv"), float) and pd.isna(perf["mean_clv"])
    ):
        meta = meta.with_blocked("mean_clv", *Unavailable.NO_CLV_DATA)
        meta = meta.with_blocked("pct_positive_clv", *Unavailable.NO_CLV_DATA)
    if perf.get("mean_ev_at_bet") is None or (
        isinstance(perf.get("mean_ev_at_bet"), float) and pd.isna(perf["mean_ev_at_bet"])
    ):
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
        # pyrefly: ignore [unexpected-keyword]
        response_meta=meta if meta.field_status else None,
    )


def _compose_streak_label(
    count: int | None,
    streak_type: str | None,
) -> str | None:
    """Compose a wire-friendly streak label from count + type.

    Examples:
        (3, "win")  -> "W3"
        (2, "loss") -> "L2"
        (0, "none") -> None
        (None, _)   -> None
    """
    if count is None or streak_type is None:
        return None
    if count == 0 or streak_type in (None, "none", ""):
        return None
    prefix = {"win": "W", "loss": "L", "push": "P"}.get(streak_type)
    if prefix is None:
        return None
    return f"{prefix}{count}"


def serialize_bets(bets: pd.DataFrame) -> _BetsList:
    """Build the /portfolio/bets list response."""
    if bets.empty:
        return _BetsList(items=[], total=0)

    rows = [
        BetRow(
            bet_id=str(row["bet_id"]) if pd.notna(row.get("bet_id")) else None,
            game_id=str(row["game_id"]) if pd.notna(row.get("game_id")) else None,
            placed_at=str(row["placed_at"]) if pd.notna(row.get("placed_at")) else None,
            market_type=_none_if_nan(row.get("market_type")),
            side=_none_if_nan(row.get("side")),
            line=_none_if_nan(row.get("line")),
            odds=int(row["odds"]) if pd.notna(row.get("odds")) else None,
            stake=float(row["stake"]) if pd.notna(row.get("stake")) else None,
            book=_none_if_nan(row.get("book")),
            status=_none_if_nan(row.get("status")),
            pnl=_none_if_nan(row.get("pnl")),
            closing_line=_none_if_nan(row.get("closing_line")),
            clv=_none_if_nan(row.get("clv")),
            model_name=_none_if_nan(row.get("model_name")),
            model_type=_none_if_nan(row.get("model_type")),
        )
        for _, row in bets.iterrows()
    ]
    return _BetsList(items=rows, total=len(rows))


def serialize_bankroll_curve(
    history: pd.DataFrame,
    period: str | None,
) -> BankrollCurve:
    """Build the /portfolio/curve response from `balance_history()` output.

    Maps the domain-side column `running_balance` to the schema field
    `bankroll` — the API convention is friendlier for the frontend.
    """
    meta = ResponseMeta()
    if period is None:
        meta = meta.with_blocked("period", *Unavailable.PERIOD_NOT_REQUESTED)

    if history.empty:
        return BankrollCurve(
            items=[],
            total=0,
            period=period,
            # pyrefly: ignore [unexpected-keyword]
            response_meta=meta if meta.field_status else None,
        )

    buckets = [
        CurveBucket(
            timestamp=str(row["timestamp"]),
            bankroll=float(row["running_balance"]),
        )
        for _, row in history.iterrows()
    ]
    return BankrollCurve(
        items=buckets,
        total=len(buckets),
        period=period,
        # pyrefly: ignore [unexpected-keyword]
        response_meta=meta if meta.field_status else None,
    )


def serialize_transactions(txns: pd.DataFrame) -> _TransactionsList:
    """Build the /portfolio/transactions list response."""
    if txns.empty:
        return _TransactionsList(items=[], total=0)

    rows = [
        TransactionRow(
            txn_id=str(row["txn_id"]) if pd.notna(row.get("txn_id")) else None,
            timestamp=str(row["timestamp"]) if pd.notna(row.get("timestamp")) else None,
            txn_type=_none_if_nan(row.get("txn_type")),
            amount=float(row["amount"]) if pd.notna(row.get("amount")) else None,
            reference_id=_none_if_nan(row.get("reference_id")),
            note=_none_if_nan(row.get("note")),
        )
        for _, row in txns.iterrows()
    ]
    return _TransactionsList(items=rows, total=len(rows))


def serialize_splits(splits_df: pd.DataFrame, dimension: str) -> PortfolioSplits:
    """Build the /portfolio/splits response from a pre-aggregated DataFrame.

    Input is the output of `performance.record(bets, split_by=dimension)`
    joined with `performance.roi(bets, split_by=dimension)`. Column
    conventions: dimension name is the split column; `wins`, `losses`,
    `pushes`, `total`, `win_pct` from `record`; `roi` from `roi`.
    """
    if splits_df.empty:
        return PortfolioSplits(items=[], total=0, dimension=dimension)

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

# src/gridiron_edge/api/routes/portfolio.py

"""Portfolio endpoints: summary, bets, curve, transactions, splits."""

from __future__ import annotations

from fastapi import APIRouter, Query

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_bankroll_history_df,
    load_bankroll_txns_df,
    load_bets_df,
    load_current_bankroll,
)
from gridiron_edge.api.schemas._base import BaseListResponse
from gridiron_edge.api.schemas.portfolio import (
    BankrollCurve,
    BetRow,
    PortfolioSplits,
    PortfolioSummary,
    TransactionRow,
)
from gridiron_edge.api.serializers.portfolio import (
    serialize_bankroll_curve,
    serialize_bets,
    serialize_portfolio_summary,
    serialize_splits,
    serialize_transactions,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

# Type aliases mirror the serializer module for consistency.
_BetsList = BaseListResponse[BetRow]
_TransactionsList = BaseListResponse[TransactionRow]


@router.get("/summary", response_model=PortfolioSummary)
def get_portfolio_summary(settings: SettingsDep) -> PortfolioSummary:
    """Return bankroll balance and performance rollup."""
    from gridiron_edge.betting.performance import summary as perf_summary

    bets = load_bets_df(settings)
    bankroll = load_current_bankroll(settings)
    perf = perf_summary(bets) if not bets.empty else {}
    return serialize_portfolio_summary(bets, bankroll, perf)


@router.get("/bets", response_model=_BetsList)
def get_portfolio_bets(
    settings: SettingsDep,
    status: str | None = Query(
        default=None,
        description="Filter by bet status: open, won, lost, push.",
    ),
) -> _BetsList:
    """Return the list of bets, optionally filtered by status."""
    bets = load_bets_df(settings, status=status)
    return serialize_bets(bets)


@router.get("/curve", response_model=BankrollCurve)
def get_portfolio_curve(
    settings: SettingsDep,
    period: str | None = Query(
        default=None,
        description="Optional time-window label, e.g. '30d'. Currently informational only.",
    ),
) -> BankrollCurve:
    """Return the bankroll running-balance curve."""
    history = load_bankroll_history_df(settings)
    return serialize_bankroll_curve(history, period)


@router.get("/transactions", response_model=_TransactionsList)
def get_portfolio_transactions(
    settings: SettingsDep,
) -> _TransactionsList:
    """Return the raw bankroll transaction log."""
    txns = load_bankroll_txns_df(settings)
    return serialize_transactions(txns)


@router.get("/splits", response_model=PortfolioSplits)
def get_portfolio_splits(
    settings: SettingsDep,
    dimension: str = Query(
        default="market_type",
        description="Column to split on. One of: market_type, side, book, model_name.",
    ),
) -> PortfolioSplits:
    """Return performance splits grouped by the chosen dimension."""
    from gridiron_edge.betting.performance import record, roi

    bets = load_bets_df(settings)
    if bets.empty:
        return PortfolioSplits(items=[], total=0, dimension=dimension)

    record_df = record(bets, split_by=dimension)
    roi_df = roi(bets, split_by=dimension)
    merged = record_df.merge(roi_df, on=dimension, how="outer")
    return serialize_splits(merged, dimension)

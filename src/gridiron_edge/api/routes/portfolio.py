# src/gridiron_edge/api/routes/portfolio.py
"""Portfolio endpoints: summary, bets, curve, transactions, and splits."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from gridiron_edge.api.deps import SettingsDep
from gridiron_edge.api.loaders import (
    load_bankroll_history_df,
    load_bankroll_txns_df,
    load_bets_df,
    load_current_bankroll,
    load_recorded_bet_df,
    resolve_recommended_bet_recording_evidence,
)
from gridiron_edge.api.schemas._base import BaseListResponse
from gridiron_edge.api.schemas.portfolio import (
    BankrollCurve,
    BetRow,
    PortfolioSplits,
    PortfolioSummary,
    RecordBetRequest,
    RecordBetResponse,
    TransactionRow,
)
from gridiron_edge.api.serializers.portfolio import (
    serialize_bankroll_curve,
    serialize_bets,
    serialize_portfolio_summary,
    serialize_recorded_bet,
    serialize_splits,
    serialize_transactions,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

_BetsList = BaseListResponse[BetRow]
_TransactionsList = BaseListResponse[TransactionRow]
SplitDimension = Literal[
    "market_type",
    "funding_type",
    "side",
    "book",
    "model_name",
    "model_type",
    "confidence_tier",
]


@router.get("/summary", response_model=PortfolioSummary)
def get_portfolio_summary(settings: SettingsDep) -> PortfolioSummary:
    """Return available bankroll balance and performance rollup."""
    from gridiron_edge.betting.performance import summary as perf_summary

    bets = load_bets_df(settings)
    bankroll = load_current_bankroll(settings)
    perf = perf_summary(bets) if not bets.empty else {}
    return serialize_portfolio_summary(bets, bankroll, perf)


@router.get("/bets", response_model=_BetsList)
def get_portfolio_bets(
    settings: SettingsDep,
    status: Annotated[
        str | None,
        Query(description="Filter by bet status: open, won, lost, push."),
    ] = None,
) -> _BetsList:
    """Return bets, optionally filtered by status."""
    bets = load_bets_df(settings, status=status)
    return serialize_bets(bets)


@router.post("/bets", response_model=RecordBetResponse, status_code=201)
def record_portfolio_bet(
    settings: SettingsDep,
    request: RecordBetRequest,
) -> RecordBetResponse:
    """Record a wager locally without placing a sportsbook wager."""
    from gridiron_edge.betting.recording import RecordWagerCommand, record_wager

    recommendation = None
    if request.recommended_bet_result_id is not None:
        try:
            recommendation = resolve_recommended_bet_recording_evidence(
                settings,
                result_id=request.recommended_bet_result_id,
                evaluation_id=request.recommendation_evaluation_id or "",
                candidate_reference_id=request.candidate_reference_id or "",
                policy_id=request.recommendation_policy_id or "",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        recorded = record_wager(
            RecordWagerCommand(
                game_id=request.game_id,
                market_type=request.market_type,
                side=request.side,
                line=request.line,
                odds=request.odds,
                stake=request.stake,
                book=request.book,
                recommendation=recommendation,
            ),
            repo=settings.repo_root,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return serialize_recorded_bet(
        load_recorded_bet_df(settings, bet_id=recorded.bet_id),
        bankroll_transaction_id=recorded.bankroll_transaction_id,
    )


@router.get("/curve", response_model=BankrollCurve)
def get_portfolio_curve(
    settings: SettingsDep,
    period: Annotated[
        str | None,
        Query(
            description=("Optional time-window label, e.g. '30d'. Currently informational only.")
        ),
    ] = None,
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
    dimension: Annotated[
        SplitDimension,
        Query(
            description=(
                "Performance split dimension: market_type, funding_type, "
                "side, book, model_name, model_type, or confidence_tier."
            )
        ),
    ] = "market_type",
) -> PortfolioSplits:
    """Return performance splits grouped by a validated dimension."""
    from gridiron_edge.betting.performance import record, roi

    bets = load_bets_df(settings)
    if bets.empty:
        return serialize_splits(pd.DataFrame(), dimension)

    record_df = record(bets, split_by=dimension)
    roi_df = roi(bets, split_by=dimension)
    merged = record_df.merge(roi_df, on=dimension, how="outer")
    return serialize_splits(merged, dimension)

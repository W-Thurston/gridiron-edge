# src/gridiron_edge/api/schemas/portfolio.py
"""Schemas for /portfolio/* endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse


class PortfolioSummary(BaseResponse):
    """Available bankroll headline plus performance rollup."""

    bankroll: float | None = Field(default=None, description="Current available bankroll balance.")
    total_bets: int | None = None
    settled_bets: int | None = None
    open_bets: int | None = None
    wins: int | None = None
    losses: int | None = None
    pushes: int | None = None
    win_pct: float | None = None
    total_staked: float | None = None
    total_pnl: float | None = None
    roi_pct: float | None = Field(default=None, description="ROI as a percentage.")
    mean_clv: float | None = Field(default=None, description="Mean closing line value.")
    pct_positive_clv: float | None = None
    n_clv_bets: int | None = Field(
        default=None, description="Number of bets with CLV data available."
    )
    mean_ev_at_bet: float | None = None
    ev_vs_actual_gap: float | None = None
    n_model_bets: int | None = None
    calibration_health: str | None = None
    current_streak: str | None = Field(
        default=None, description="Composed streak label, e.g. 'W3' or 'L2'."
    )
    longest_win_streak: int | None = None
    longest_loss_streak: int | None = None


class BetRow(BaseModel):
    """A single row in /portfolio/bets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bet_id: str | None = None
    source_bet_id: str | None = None
    game_id: str | None = None
    description: str | None = None
    placed_at: str | None = None
    market_type: str | None = None
    side: str | None = None
    line: float | None = None
    odds: int | None = None
    stake: float | None = None
    book: str | None = None
    funding_type: str | None = None
    paid_amount: float | None = None
    potential_payout: float | None = None
    status: str | None = None
    pnl: float | None = None
    closing_line: float | None = None
    clv: float | None = None
    model_name: str | None = None
    model_type: str | None = None
    recommended_bet_result_id: str | None = None
    recommendation_evaluation_id: str | None = None
    candidate_reference_id: str | None = None
    recommendation_policy_id: str | None = None


class RecordBetRequest(BaseModel):
    """Recorded wager terms plus an optional complete recommendation identity chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    game_id: str = Field(min_length=1)
    market_type: Literal["moneyline", "spread", "total"]
    side: str = Field(min_length=1)
    line: float | None = None
    odds: int
    stake: float = Field(gt=0)
    book: str = Field(min_length=1)
    recommended_bet_result_id: str | None = None
    recommendation_evaluation_id: str | None = None
    candidate_reference_id: str | None = None
    recommendation_policy_id: str | None = None

    @model_validator(mode="after")
    def validate_recommendation_chain(self) -> RecordBetRequest:
        """Require recommendation identities to be complete or absent."""
        identities = (
            self.recommended_bet_result_id,
            self.recommendation_evaluation_id,
            self.candidate_reference_id,
            self.recommendation_policy_id,
        )
        if any(value is not None for value in identities) and any(
            value is None or not value.strip() for value in identities
        ):
            raise ValueError("Recommendation identities must be entirely absent or complete.")
        return self


class RecordBetResponse(BaseModel):
    """One successfully recorded wager and its bankroll transaction."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    bet: BetRow
    bankroll_transaction_id: str
    message: str = "Wager recorded in Gridiron Edge. No sportsbook wager was placed."


class CurveBucket(BaseModel):
    """A single point in /portfolio/curve."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    timestamp: str
    bankroll: float


class BankrollCurve(BaseListResponse[CurveBucket]):
    """Bankroll over time."""

    period: str | None = Field(default=None, description="Requested period, e.g. '30d'.")


class TransactionRow(BaseModel):
    """A single row in /portfolio/transactions."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    txn_id: str | None = None
    source_transaction_id: str | None = None
    timestamp: str | None = None
    txn_type: str | None = None
    amount: float | None = None
    balance_after: float | None = None
    reference_id: str | None = None
    note: str | None = None


class SplitRow(BaseModel):
    """A single row in /portfolio/splits."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    dimension_value: str
    total: int | None = None
    wins: int | None = None
    losses: int | None = None
    pushes: int | None = None
    win_pct: float | None = None
    roi: float | None = None


class PortfolioSplits(BaseListResponse[SplitRow]):
    """ROI/record splits by a chosen dimension."""

    dimension: str = Field(description="Column grouped on, e.g. 'market_type'.")

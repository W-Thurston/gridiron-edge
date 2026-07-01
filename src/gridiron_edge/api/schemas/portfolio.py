# src/gridiron_edge/api/schemas/portfolio.py

"""Schemas for /portfolio/* endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse


class PortfolioSummary(BaseResponse):
    """Bankroll headline plus performance rollup."""

    bankroll: float | None = Field(default=None, description="Current bankroll balance.")
    total_bets: int | None = Field(default=None)
    settled_bets: int | None = Field(default=None)
    open_bets: int | None = Field(default=None)

    # Record
    wins: int | None = Field(default=None)
    losses: int | None = Field(default=None)
    pushes: int | None = Field(default=None)
    win_pct: float | None = Field(default=None)

    # ROI
    total_staked: float | None = Field(default=None)
    total_pnl: float | None = Field(default=None)
    roi_pct: float | None = Field(default=None, description="ROI as a percentage.")

    # CLV
    mean_clv: float | None = Field(default=None, description="Mean closing line value.")
    pct_positive_clv: float | None = Field(default=None)
    n_clv_bets: int | None = Field(
        default=None,
        description="Number of bets with CLV data available.",
    )

    # EV
    mean_ev_at_bet: float | None = Field(default=None)
    ev_vs_actual_gap: float | None = Field(default=None)
    n_model_bets: int | None = Field(default=None)
    calibration_health: str | None = Field(default=None)

    # Streaks
    current_streak: str | None = Field(
        default=None,
        description="Composed streak label, e.g. 'W3' or 'L2'.",
    )
    longest_win_streak: int | None = Field(default=None)
    longest_loss_streak: int | None = Field(default=None)


class BetRow(BaseModel):
    """A single row in /portfolio/bets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bet_id: str | None = None
    game_id: str | None = None
    placed_at: str | None = None
    market_type: str | None = None
    side: str | None = None
    line: float | None = None
    odds: int | None = None
    stake: float | None = None
    book: str | None = None
    status: str | None = None
    pnl: float | None = None
    closing_line: float | None = None
    clv: float | None = None
    model_name: str | None = None
    model_type: str | None = None


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
    timestamp: str | None = None
    txn_type: str | None = None
    amount: float | None = None
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

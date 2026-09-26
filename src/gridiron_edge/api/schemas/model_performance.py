# src/gridiron_edge/api/schemas/model_performance.py

"""Schemas for /model/performance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseListResponse, BaseResponse


class ModelPerformanceFilters(BaseModel):
    """Echo of applied query parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: str | None = None
    model_name: str | None = None
    model_type: str | None = None
    group_by: str


class ModelQualityBlock(BaseModel):
    """Top-line model-quality metrics from build_latest_run_evaluation_df."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_games: int | None = None
    brier: float | None = None
    log_loss: float | None = None
    accuracy: float | None = None
    ece: float | None = Field(default=None, description="Expected calibration error.")
    roc_auc: float | None = None
    brier_reliability: float | None = None
    brier_resolution: float | None = None
    brier_uncertainty: float | None = None


class BettingPerformanceBlock(BaseModel):
    """Top-line betting-performance metrics scoped to bets with model context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_model_bets: int | None = None
    mean_ev_at_bet: float | None = None
    ev_vs_actual_gap: float | None = None
    mean_clv: float | None = None
    pct_positive_clv: float | None = None
    roi_pct: float | None = None
    calibration_health: str | None = None


class GroupedMetricRow(BaseModel):
    """A single row in the by_group breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_key: str
    n_games: int | None = None
    brier: float | None = None
    accuracy: float | None = None


class ModelPerformance(BaseResponse):
    """Response for GET /model/performance."""

    filters: ModelPerformanceFilters
    model_quality: ModelQualityBlock
    betting_performance: BettingPerformanceBlock
    by_group: list[GroupedMetricRow] = Field(default_factory=list)


class HistoricalMoneylinePerformance(BaseModel):
    """Walk-forward Moneyline quality without fabricated price returns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_type: str
    run_id: str
    evaluated_count: int
    wins: int
    losses: int
    net_wins: int
    accuracy: float | None
    brier: float | None
    log_loss: float | None
    price_evidence_status: Literal["unavailable"]
    unit_return_reason: str


class HistoricalTotalPerformance(BaseModel):
    """Walk-forward Total quality and explicitly assumed-price return."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_type: str
    run_id: str
    decision_count: int
    wins: int
    losses: int
    pushes: int
    no_bets: int
    net_wins: int
    hit_rate_excluding_pushes: float | None
    mae: float | None
    rmse: float | None
    bias: float | None
    net_units: float | None
    roi_per_unit_staked: float | None
    price_evidence_status: Literal["assumed"]
    assumed_american_price: int | None
    methodology: str


class HistoricalSpreadPerformance(BaseModel):
    """Explicit unavailable state for historical Spread performance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["unavailable"] = "unavailable"
    reason: str


class HistoricalPerformanceFinalValues(BaseModel):
    """Final persisted points for compact dashboard presentation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    moneyline_cumulative_net_wins: float
    total_cumulative_net_wins: float
    total_cumulative_units: float


class HistoricalModelPerformance(BaseResponse):
    """Current immutable historical walk-forward report summary."""

    report_id: str
    selected_at: str
    generated_at: str
    first_season: str
    last_season: str
    evidence_row_count: int
    rolling_decision_window: int
    moneyline: HistoricalMoneylinePerformance
    total: HistoricalTotalPerformance
    spread: HistoricalSpreadPerformance
    final_values: HistoricalPerformanceFinalValues


class HistoricalModelPerformancePoint(BaseModel):
    """One persisted chart point from the historical report series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: str
    week: int
    game_id: str
    game_date: str
    moneyline_cumulative_net_wins: float
    moneyline_cumulative_accuracy: float | None
    moneyline_rolling_accuracy_100: float | None
    total_cumulative_net_wins: float
    total_cumulative_accuracy: float | None
    total_rolling_accuracy_100: float | None
    total_cumulative_units: float


class HistoricalModelPerformanceSeries(BaseListResponse[HistoricalModelPerformancePoint]):
    """Verified persisted chart series for the current historical report."""

    report_id: str

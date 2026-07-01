# src/gridiron_edge/api/schemas/model_performance.py

"""Schemas for /model/performance."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class ModelPerformanceFilters(BaseModel):
    """Echo of applied query parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: str | None = None
    model_name: str | None = None
    model_type: str | None = None
    group_by: str


class ModelQualityBlock(BaseModel):
    """Top-line model-quality metrics from build_evaluation_df."""

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

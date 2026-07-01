# tests/unit/api/test_schemas_model_performance.py

"""Unit tests for model_performance schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.schemas.model_performance import (
    BettingPerformanceBlock,
    GroupedMetricRow,
    ModelPerformance,
    ModelPerformanceFilters,
    ModelQualityBlock,
)


class TestModelPerformanceFilters:
    def test_construction(self) -> None:
        f = ModelPerformanceFilters(group_by="season")
        assert f.group_by == "season"
        assert f.season is None

    def test_populated(self) -> None:
        f = ModelPerformanceFilters(
            group_by="week",
            season="2025-2026",
            model_name="win_prob",
            model_type="random_forest",
        )
        assert f.model_name == "win_prob"

    def test_group_by_required(self) -> None:
        with pytest.raises(ValidationError):
            ModelPerformanceFilters()

    def test_is_frozen(self) -> None:
        f = ModelPerformanceFilters(group_by="season")
        with pytest.raises(ValidationError):
            f.group_by = "week"


class TestModelQualityBlock:
    def test_default(self) -> None:
        b = ModelQualityBlock()
        assert b.n_games is None
        assert b.brier is None

    def test_populated(self) -> None:
        b = ModelQualityBlock(n_games=100, brier=0.21, accuracy=0.65)
        assert b.brier == 0.21


class TestBettingPerformanceBlock:
    def test_default(self) -> None:
        b = BettingPerformanceBlock()
        assert b.n_model_bets is None

    def test_populated(self) -> None:
        b = BettingPerformanceBlock(n_model_bets=42, roi_pct=8.3)
        assert b.n_model_bets == 42


class TestGroupedMetricRow:
    def test_populated(self) -> None:
        row = GroupedMetricRow(group_key="2024-2025", n_games=272, brier=0.21)
        assert row.group_key == "2024-2025"

    def test_group_key_required(self) -> None:
        with pytest.raises(ValidationError):
            GroupedMetricRow()


class TestModelPerformance:
    def test_construction(self) -> None:
        resp = ModelPerformance(
            filters=ModelPerformanceFilters(group_by="season"),
            model_quality=ModelQualityBlock(n_games=0),
            betting_performance=BettingPerformanceBlock(n_model_bets=0),
        )
        assert resp.by_group == []
        assert resp.response_meta is None

    def test_wire_shape(self) -> None:
        resp = ModelPerformance(
            filters=ModelPerformanceFilters(group_by="season"),
            model_quality=ModelQualityBlock(n_games=100, brier=0.21),
            betting_performance=BettingPerformanceBlock(n_model_bets=10),
            by_group=[GroupedMetricRow(group_key="2024-2025", n_games=100)],
        )
        dumped = resp.model_dump(by_alias=True, exclude_none=True)
        assert dumped["filters"]["group_by"] == "season"
        assert dumped["model_quality"]["brier"] == 0.21
        assert len(dumped["by_group"]) == 1

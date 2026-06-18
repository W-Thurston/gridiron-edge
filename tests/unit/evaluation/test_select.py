# tests/unit/evaluation/test_select.py
"""Tests for gridiron_edge.evaluation.select — collect_model_metrics, rank_models."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.select import collect_model_metrics, rank_models


class TestCollectModelMetrics:
    def test_returns_empty_list_when_no_data(self, tmp_path: Path) -> None:
        """Models with no archived predictions should be silently skipped."""
        with patch(
            "gridiron_edge.evaluation.metrics.build_evaluation_df",
            return_value=pd.DataFrame(),
        ):
            result: list[dict] = collect_model_metrics(["win_prob_fake"], repo=tmp_path)
            assert result == []

    def test_returns_metrics_dict_when_data_exists(self, tmp_path: Path) -> None:
        eval_df = pd.DataFrame(
            {
                "away_win_prob": [0.6, 0.4, 0.7, 0.3, 0.5],
                "away_team_won": [1, 0, 1, 0, 1],
            }
        )
        with patch(
            "gridiron_edge.evaluation.metrics.build_evaluation_df",
            return_value=eval_df,
        ):
            result: list[dict] = collect_model_metrics(["win_prob_test"], repo=tmp_path)
            assert len(result) == 1
            row: dict = result[0]
            assert "model_key" in row
            assert result[0]["model_key"] == "win_prob_test"

    def test_skips_models_without_data(self, tmp_path: Path) -> None:
        """If one model has data and another doesn't, only the valid one appears."""

        def mock_build(*, model_name: str, model_type: str, repo: Path) -> pd.DataFrame:
            if model_name == "win_prob" and model_type == "good":
                return pd.DataFrame(
                    {
                        "away_win_prob": [0.6, 0.4],
                        "away_team_won": [1, 0],
                    }
                )
            return pd.DataFrame()

        with patch(
            "gridiron_edge.evaluation.metrics.build_evaluation_df",
            side_effect=mock_build,
        ):
            result: list[dict] = collect_model_metrics(
                ["win_prob_good", "win_prob_empty"], repo=tmp_path
            )
            assert len(result) == 1
            assert result[0]["model_key"] == "win_prob_good"


class TestRankModels:
    def test_ranks_by_brier_ascending(self) -> None:
        """Lower Brier = better → should rank first."""
        metrics: list[dict[str, float | str]] = [
            {"model_version": "bad_v1", "brier": 0.30},
            {"model_version": "good_v1", "brier": 0.20},
            {"model_version": "mid_v1", "brier": 0.25},
        ]
        ranked: DataFrame = rank_models(
            metrics,
            criteria_list=["brier"],
            lower_is_better={"brier"},
        )
        assert ranked.iloc[0]["model_version"] == "good_v1"
        assert ranked.iloc[-1]["model_version"] == "bad_v1"

    def test_empty_input_returns_empty(self) -> None:
        result: DataFrame = rank_models(
            [{"model_version": "a", "brier": 0.25}, {"model_version": "b", "brier": 0.20}],
            criteria_list=["brier"],
            lower_is_better={"brier"},
        )
        assert len(result) == 2

    def test_empty_input_raises(self) -> None:
        with pytest.raises(KeyError):
            rank_models([], criteria_list=["brier"], lower_is_better={"brier"})

    def test_single_model_returns_itself(self) -> None:
        metrics: list[dict[str, float | str]] = [{"model_version": "only_v1", "brier": 0.22}]
        ranked: DataFrame = rank_models(
            metrics,
            criteria_list=["brier"],
            lower_is_better={"brier"},
        )
        assert len(ranked) == 1
        assert ranked.iloc[0]["model_version"] == "only_v1"

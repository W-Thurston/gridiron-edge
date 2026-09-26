# tests/unit/evaluation/test_select.py
"""Tests for gridiron_edge.evaluation.select."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.select import (
    collect_forecast_run_metrics,
    collect_latest_forecast_run_metrics,
    latest_backfilled_run_id,
    rank_models,
)


class TestLatestBackfilledRunId:
    def test_returns_none_when_no_events(self, tmp_path: Path) -> None:
        with patch(
            "gridiron_edge.evaluation.forecast_store.load_forecast_events",
            return_value=DataFrame(),
        ):
            assert latest_backfilled_run_id("win_prob", "logistic", repo=tmp_path) is None

    def test_returns_the_most_recently_generated_run(self, tmp_path: Path) -> None:
        events = DataFrame(
            {
                "run_id": ["run-old", "run-old", "run-new"],
                "generated_at": [
                    datetime(2026, 1, 1, tzinfo=UTC),
                    datetime(2026, 1, 1, tzinfo=UTC),
                    datetime(2026, 2, 1, tzinfo=UTC),
                ],
            }
        )
        with patch(
            "gridiron_edge.evaluation.forecast_store.load_forecast_events",
            return_value=events,
        ):
            run_id = latest_backfilled_run_id("win_prob", "logistic", repo=tmp_path)
        assert run_id == "run-new"

    def test_ties_break_by_run_id(self, tmp_path: Path) -> None:
        same_timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        events = DataFrame(
            {
                "run_id": ["run-a", "run-b"],
                "generated_at": [same_timestamp, same_timestamp],
            }
        )
        with patch(
            "gridiron_edge.evaluation.forecast_store.load_forecast_events",
            return_value=events,
        ):
            run_id = latest_backfilled_run_id("win_prob", "logistic", repo=tmp_path)
        assert run_id == "run-b"


class TestCollectLatestForecastRunMetrics:
    def test_skips_models_with_no_backfill_run(self, tmp_path: Path) -> None:
        with patch(
            "gridiron_edge.evaluation.forecast_store.load_forecast_events",
            return_value=DataFrame(),
        ):
            result: list[dict] = collect_latest_forecast_run_metrics(
                ["win_prob_fake"], repo=tmp_path
            )
        assert result == []

    def test_resolves_and_evaluates_the_latest_run(self, tmp_path: Path) -> None:
        events = DataFrame(
            {"run_id": ["run-1"], "generated_at": [datetime(2026, 1, 1, tzinfo=UTC)]}
        )
        evaluation = pd.DataFrame({"away_win_prob": [0.6, 0.4], "away_team_won": [1.0, 0.0]})
        with (
            patch(
                "gridiron_edge.evaluation.forecast_store.load_forecast_events",
                return_value=events,
            ),
            patch(
                "gridiron_edge.evaluation.metrics.build_forecast_run_evaluation_df",
                return_value=evaluation,
            ) as build,
        ):
            result: list[dict] = collect_latest_forecast_run_metrics(
                ["win_prob_test"], repo=tmp_path
            )
        assert result[0]["model_key"] == "win_prob_test"
        assert build.call_args.kwargs["run_id"] == "run-1"


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


class TestCollectForecastRunMetrics:
    def test_uses_exact_runs_and_excludes_ties(self, tmp_path: Path) -> None:
        evaluation = DataFrame({"away_win_prob": [0.8, 0.2, 0.5], "away_team_won": [1.0, 0.0, 0.5]})
        with patch(
            "gridiron_edge.evaluation.metrics.build_forecast_run_evaluation_df",
            return_value=evaluation,
        ) as build:
            result = collect_forecast_run_metrics(
                {("win_prob", "logistic"): "run-1"}, repo=tmp_path
            )
        assert result[0]["model_key"] == "win_prob_logistic"
        assert result[0]["n_games"] == 2
        assert build.call_args.kwargs["run_id"] == "run-1"

    def test_missing_run_error_propagates(self, tmp_path: Path) -> None:
        with (
            patch(
                "gridiron_edge.evaluation.metrics.build_forecast_run_evaluation_df",
                side_effect=ValueError("Backfilled forecast run is unavailable"),
            ),
            pytest.raises(ValueError, match="run is unavailable"),
        ):
            collect_forecast_run_metrics({("win_prob", "logistic"): "missing"}, repo=tmp_path)

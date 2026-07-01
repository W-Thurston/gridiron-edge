# tests/unit/api/test_serializers_model_performance.py

"""Unit tests for model_performance serializer."""

from __future__ import annotations

import pandas as pd

from gridiron_edge.api.serializers.model_performance import (
    serialize_model_performance,
)


def _filters(**kwargs):
    return {
        "season": kwargs.get("season"),
        "model_name": kwargs.get("model_name"),
        "model_type": kwargs.get("model_type"),
        "group_by": kwargs.get("group_by", "season"),
    }


class TestEmptyInputs:
    def test_empty_eval_df_marks_quality_unavailable(self) -> None:
        result = serialize_model_performance(
            df_eval=pd.DataFrame(),
            summary_df=pd.DataFrame(),
            model_bet_summary={},
            filters=_filters(),
        )
        assert result.model_quality.n_games == 0
        assert result.model_quality.brier is None
        assert result.by_group == []
        assert result.response_meta is not None
        assert "model_quality.brier" in result.response_meta.field_status

    def test_empty_bets_marks_betting_unavailable(self) -> None:
        result = serialize_model_performance(
            df_eval=pd.DataFrame(),
            summary_df=pd.DataFrame(),
            model_bet_summary={},
            filters=_filters(),
        )
        assert result.betting_performance.n_model_bets == 0
        assert "betting_performance.roi_pct" in result.response_meta.field_status


class TestPopulatedInputs:
    def test_by_group_serializes(self) -> None:
        summary_df = pd.DataFrame(
            {
                "season": ["2024-2025", "2025-2026"],
                "n_games": [272, 271],
                "brier": [0.209, 0.218],
                "accuracy": [0.643, 0.633],
            },
        )
        # Build a minimal eval df; the exact metric values don't matter
        # here — we just need it non-empty so model_quality populates.
        df_eval = pd.DataFrame(
            {
                "away_win_prob": [0.4, 0.6, 0.5, 0.5],
                "away_team_won": [0, 1, 1, 0],
            },
        )
        result = serialize_model_performance(
            df_eval=df_eval,
            summary_df=summary_df,
            model_bet_summary={
                "n_model_bets": 42,
                "mean_ev_at_bet": 0.048,
                "roi_pct": 8.3,
                "calibration_health": "healthy",
            },
            filters=_filters(group_by="season"),
        )

        assert result.model_quality.n_games == 4
        assert result.betting_performance.n_model_bets == 42
        assert result.betting_performance.roi_pct == 8.3
        assert len(result.by_group) == 2
        assert result.by_group[0].group_key == "2024-2025"
        assert result.by_group[0].brier == 0.209

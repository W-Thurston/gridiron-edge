# tests/unit/models/test_pipeline.py
"""Tests for the prediction pipeline orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.models.game_prediction.pipeline import build_game_predictions


class TestBuildGamePredictions:
    """Tests for build_game_predictions()."""

    def _make_modeling_df(self) -> pd.DataFrame:
        """Minimal modeling DataFrame with two rows per game (home/away)."""
        return pd.DataFrame(
            {
                "GAME_ID": ["G1", "G1", "G2", "G2"],
                "TEAM_A": ["Chiefs", "Ravens", "Bills", "Dolphins"],
                "TEAM_B": ["Ravens", "Chiefs", "Dolphins", "Bills"],
                "YEAR": ["2024-2025"] * 4,
                "WEEK_NUM": [1, 1, 1, 1],
                "HOME_FIELD": [0, 1, 0, 1],
            }
        )

    def test_one_row_per_game(self) -> None:
        """Output has one row per game, not two."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(df, probs, model_version="random_forest")
        assert len(result) == 2

    def test_away_team_perspective(self) -> None:
        """Away team probability matches the HOME_FIELD==0 row."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(df, probs, model_version="random_forest")
        g1 = result[result["game_id"] == "G1"].iloc[0]
        assert g1["away_win_prob"] == pytest.approx(0.45)
        assert g1["home_win_prob"] == pytest.approx(0.55)

    def test_model_version_tagged(self) -> None:
        """All rows have the correct model_version."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(df, probs, model_version="random_forest")
        assert (result["model_version"] == "random_forest").all()

    def test_totals_included_when_provided(self) -> None:
        """model_total column present when totals are passed."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])
        totals = pd.Series([44.0, 44.0, 48.0, 48.0], index=df.index)

        result = build_game_predictions(
            df,
            probs,
            model_version="random_forest",
            totals=totals,
        )
        assert "model_total" in result.columns
        assert result["model_total"].notna().all()

    def test_totals_absent_when_not_provided(self) -> None:
        """model_total column absent when totals are not passed."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(df, probs, model_version="random_forest")
        assert "model_total" not in result.columns

    def test_is_backfilled_flag(self) -> None:
        """is_backfilled flag is set correctly."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result_bf = build_game_predictions(
            df,
            probs,
            model_version="random_forest",
            is_backfilled=True,
        )
        result_live = build_game_predictions(
            df,
            probs,
            model_version="random_forest",
            is_backfilled=False,
        )
        assert result_bf["is_backfilled"].all()
        assert not result_live["is_backfilled"].any()

    def test_required_columns_present(self) -> None:
        """Output contains all base archive columns."""
        df = self._make_modeling_df()
        probs = np.array([0.45, 0.55, 0.60, 0.40])

        result = build_game_predictions(df, probs, model_version="random_forest")
        required = {
            "predicted_at",
            "is_backfilled",
            "model_version",
            "season",
            "week",
            "game_id",
            "game_date",
            "away_team",
            "home_team",
            "away_elo",
            "home_elo",
            "away_win_prob",
            "home_win_prob",
        }
        assert required.issubset(set(result.columns))

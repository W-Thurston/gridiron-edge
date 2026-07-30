# tests/unit/evaluation/test_backfill.py
"""Tests for gridiron_edge.evaluation.backfill - walk-forward dispatch and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridiron_edge.evaluation.backfill import (
    _CURRENT_MODEL_DEFAULTS,
    BackfillMode,
    _resolve_mode,
)
from gridiron_edge.models.game_prediction.predictor import (
    build_regression_predictions,
)


class TestResolveMode:
    def test_elo_defaults_to_current_model(self) -> None:
        assert _resolve_mode("win_prob", "elo", None) == "current-model"

    def test_random_forest_defaults_to_walk_forward(self) -> None:
        assert _resolve_mode("win_prob", "random_forest", None) == "walk-forward"

    def test_logistic_defaults_to_walk_forward(self) -> None:
        assert _resolve_mode("win_prob", "logistic", None) == "walk-forward"

    def test_xgboost_defaults_to_walk_forward(self) -> None:
        assert _resolve_mode("win_prob", "xgboost", None) == "walk-forward"

    def test_total_defaults_to_walk_forward(self) -> None:
        assert _resolve_mode("total", "random_forest", None) == "walk-forward"
        assert _resolve_mode("total", "xgboost", None) == "walk-forward"

    def test_explicit_mode_overrides_default(self) -> None:
        # Even Elo can be forced into walk-forward if caller asks
        assert _resolve_mode("win_prob", "elo", "walk-forward") == "walk-forward"
        # Even RF can be forced into current-model if caller asks
        assert _resolve_mode("win_prob", "random_forest", "current-model") == "current-model"


class TestCurrentModelDefaults:
    def test_elo_is_in_defaults(self) -> None:
        assert ("win_prob", "elo") in _CURRENT_MODEL_DEFAULTS

    def test_ml_models_not_in_defaults(self) -> None:
        assert ("win_prob", "random_forest") not in _CURRENT_MODEL_DEFAULTS
        assert ("win_prob", "logistic") not in _CURRENT_MODEL_DEFAULTS
        assert ("win_prob", "xgboost") not in _CURRENT_MODEL_DEFAULTS
        assert ("total", "random_forest") not in _CURRENT_MODEL_DEFAULTS
        assert ("total", "xgboost") not in _CURRENT_MODEL_DEFAULTS


class TestBuildRegressionPredictions:
    def test_one_row_per_game(self) -> None:
        """Should produce one prediction row per unique game (away perspective)."""
        import numpy as np

        df = pd.DataFrame(
            {
                "GAME_ID": ["g1", "g1", "g2", "g2"],
                "TEAM_A": ["A1", "A2", "A3", "A4"],
                "TEAM_B": ["B1", "B2", "B3", "B4"],
                "YEAR": ["2024-2025"] * 4,
                "WEEK_NUM": [1, 1, 1, 1],
                "HOME_FIELD": [0, 1, 0, 1],
                "GAME_DATE": [
                    "2024-09-05",
                    "2024-09-05",
                    "2024-09-06",
                    "2024-09-06",
                ],
            }
        )
        preds = np.array([45.0, 45.0, 48.0, 48.0])
        result = build_regression_predictions(
            df,
            preds,
        )

        assert len(result) == 2  # one row per game
        assert result["model_total"].iloc[0] == 45.0
        assert result["model_total"].iloc[1] == 48.0

    def test_required_columns_present(self) -> None:
        import numpy as np

        df = pd.DataFrame(
            {
                "GAME_ID": ["g1", "g1"],
                "TEAM_A": ["A1", "A2"],
                "TEAM_B": ["B1", "B2"],
                "YEAR": ["2024-2025", "2024-2025"],
                "WEEK_NUM": [1, 1],
                "HOME_FIELD": [0, 1],
                "GAME_DATE": [
                    "2024-09-05",
                    "2024-09-05",
                ],
            }
        )
        preds = np.array([45.0, 45.0])
        result = build_regression_predictions(
            df,
            preds,
        )

        expected_cols: set[str] = {
            "season",
            "week",
            "game_id",
            "game_date",
            "away_team",
            "home_team",
            "model_total",
        }
        assert expected_cols <= set(result.columns)
        assert "predicted_at" not in result.columns
        assert "is_backfilled" not in result.columns
        assert "model_name" not in result.columns
        assert "model_type" not in result.columns

    def test_preserves_game_identity(self) -> None:
        df = pd.DataFrame(
            {
                "GAME_ID": ["g1", "g1"],
                "GAME_DATE": [
                    "2024-09-05",
                    "2024-09-05",
                ],
                "TEAM_A": ["Chiefs", "Ravens"],
                "TEAM_B": ["Ravens", "Chiefs"],
                "YEAR": [
                    "2024-2025",
                    "2024-2025",
                ],
                "WEEK_NUM": [1, 1],
                "HOME_FIELD": [0, 1],
            }
        )

        result = build_regression_predictions(
            df,
            np.array([45.0, 45.0]),
        )

        assert result["game_date"].iloc[0] == "2024-09-05"
        assert result["away_team"].iloc[0] == "Chiefs"
        assert result["home_team"].iloc[0] == "Ravens"


class TestBackfillModeType:
    def test_type_alias_exists(self) -> None:
        """BackfillMode should be importable for type-checking purposes."""
        # Verify it's a valid type alias
        assert BackfillMode is not None

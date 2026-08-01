# tests/unit/evaluation/test_backfill.py
"""Tests for gridiron_edge.evaluation.backfill - walk-forward dispatch and helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.evaluation.backfill import (
    _CURRENT_MODEL_DEFAULTS,
    BackfillMode,
    _resolve_mode,
    _walk_forward_one_season,
)
from gridiron_edge.models.game_prediction.base import (
    GameModelType,
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


class TestWalkForwardOneSeason:
    """Tests canonical walk-forward prediction dispatch."""

    def _canonical_rows(self) -> pd.DataFrame:
        """Return canonical rows across training and target seasons."""
        return pd.DataFrame(
            {
                "GAME_ID": [
                    "TRAIN_GAME",
                    "TARGET_COMPLETE",
                    "TARGET_INCOMPLETE",
                ],
                "YEAR": [
                    "2023-2024",
                    "2024-2025",
                    "2024-2025",
                ],
                "WEEK_NUM": [1, 1, 2],
                "GAME_DATE": [
                    "2023-09-07",
                    "2024-09-05",
                    "2024-09-12",
                ],
                "AWAY_TEAM": [
                    "Away Train",
                    "Away Complete",
                    "Away Incomplete",
                ],
                "HOME_TEAM": [
                    "Home Train",
                    "Home Complete",
                    "Home Incomplete",
                ],
                "HOME_WIN": [1, 0, 1],
                "ACTUAL_TOTAL": [
                    41.0,
                    47.0,
                    44.0,
                ],
                "MODEL_FEATURE": [
                    1.0,
                    2.0,
                    float("nan"),
                ],
            }
        )

    def _trainer(
        self,
        *,
        task: str,
    ) -> MagicMock:
        """Return a controlled walk-forward trainer."""
        trainer = MagicMock()

        def feature_fn(frame):
            return frame.loc[
                :,
                ["MODEL_FEATURE"],
            ].copy()

        trainer.spec = SimpleNamespace(
            task=task,
            feature_set={GameModelType.RANDOM_FOREST: (SimpleNamespace(feature_fn=feature_fn))},
        )

        trainer.train.return_value = SimpleNamespace(
            parameters={
                "cv_brier": 0.21,
                "cv_mae": 7.5,
            }
        )
        trainer._scaler = None

        return trainer

    @patch("gridiron_edge.evaluation.backfill.build_game_predictions")
    def test_classification_dispatches_canonical_rows(
        self,
        build_predictions_mock: MagicMock,
        tmp_path: Path,
    ) -> None:
        trainer = self._trainer(task="classification")
        trainer._model.predict_proba.return_value = np.array(
            [
                [0.35, 0.65],
            ]
        )

        expected = pd.DataFrame(
            {
                "game_id": ["TARGET_COMPLETE"],
                "home_win_prob": [0.65],
                "away_win_prob": [0.35],
            }
        )
        build_predictions_mock.return_value = expected

        result = _walk_forward_one_season(
            trainer=trainer,
            gm_type=(GameModelType.RANDOM_FOREST),
            df=self._canonical_rows(),
            target_season="2024-2025",
            train_through_season="2023-2024",
            model_name="win_prob",
            model_type="random_forest",
            repo=tmp_path,
        )

        pd.testing.assert_frame_equal(
            result,
            expected,
        )

        trainer.train.assert_called_once()

        train_kwargs = trainer.train.call_args.kwargs
        assert train_kwargs["train_through_season"] == "2023-2024"
        assert train_kwargs["persist"] is False

        predicted_frame = build_predictions_mock.call_args.args[0]
        probabilities = build_predictions_mock.call_args.args[1]

        assert predicted_frame["GAME_ID"].tolist() == ["TARGET_COMPLETE"]
        assert probabilities.tolist() == (pytest.approx([0.65]))

    @patch("gridiron_edge.evaluation.backfill.build_regression_predictions")
    def test_regression_dispatches_canonical_rows(
        self,
        build_predictions_mock: MagicMock,
        tmp_path: Path,
    ) -> None:
        trainer = self._trainer(task="regression")
        trainer._model.predict.return_value = np.array([47.5])

        expected = pd.DataFrame(
            {
                "game_id": ["TARGET_COMPLETE"],
                "model_total": [47.5],
            }
        )
        build_predictions_mock.return_value = expected

        result = _walk_forward_one_season(
            trainer=trainer,
            gm_type=(GameModelType.RANDOM_FOREST),
            df=self._canonical_rows(),
            target_season="2024-2025",
            train_through_season="2023-2024",
            model_name="total",
            model_type="random_forest",
            repo=tmp_path,
        )

        pd.testing.assert_frame_equal(
            result,
            expected,
        )

        predicted_frame = build_predictions_mock.call_args.args[0]
        predictions = build_predictions_mock.call_args.args[1]

        assert predicted_frame["GAME_ID"].tolist() == ["TARGET_COMPLETE"]
        assert predictions.tolist() == (pytest.approx([47.5]))

    def test_empty_target_season_returns_empty(
        self,
        tmp_path: Path,
    ) -> None:
        trainer = self._trainer(task="classification")

        result = _walk_forward_one_season(
            trainer=trainer,
            gm_type=(GameModelType.RANDOM_FOREST),
            df=self._canonical_rows(),
            target_season="2025-2026",
            train_through_season="2024-2025",
            model_name="win_prob",
            model_type="random_forest",
            repo=tmp_path,
        )

        assert result.empty

    def test_incomplete_target_features_are_excluded(
        self,
        tmp_path: Path,
    ) -> None:
        trainer = self._trainer(task="classification")
        trainer._model.predict_proba.return_value = np.array(
            [
                [0.35, 0.65],
            ]
        )

        with patch(
            "gridiron_edge.evaluation.backfill.build_game_predictions",
            return_value=pd.DataFrame({"game_id": ["TARGET_COMPLETE"]}),
        ) as build_predictions_mock:
            _walk_forward_one_season(
                trainer=trainer,
                gm_type=(GameModelType.RANDOM_FOREST),
                df=self._canonical_rows(),
                target_season="2024-2025",
                train_through_season=("2023-2024"),
                model_name="win_prob",
                model_type="random_forest",
                repo=tmp_path,
            )

        valid_frame = build_predictions_mock.call_args.args[0]

        assert valid_frame["GAME_ID"].tolist() == ["TARGET_COMPLETE"]

    def test_walk_forward_rows_exclude_retired_orientation(
        self,
    ) -> None:
        retired = {
            "TEAM_A",
            "TEAM_B",
            "HOME_FIELD",
            "RESULT",
        }

        assert not (retired & set(self._canonical_rows().columns))


class TestBackfillModeType:
    def test_type_alias_exists(self) -> None:
        """BackfillMode should be importable for type-checking purposes."""
        # Verify it's a valid type alias
        assert BackfillMode is not None

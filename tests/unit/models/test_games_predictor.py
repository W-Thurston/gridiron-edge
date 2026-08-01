# tests/unit/models/test_games_predictor.py

"""Tests for GamesPredictor + composite-key registrations + build_game_predictions.

Covers the static surface of the predictor classes plus the module-level
``build_game_predictions`` helper:
    - All 5 composite keys are registered.
    - Each subclass has the right (model_name, model_type, spec).
    - GamesPredictor delegates train() to the right trainer.
    - is_trained() delegates to ArtifactStore with the (name, type) pair.
    - predict_historical() / predict_upcoming() return empty DataFrames
      when artifacts are missing (graceful fallback).
    - build_game_predictions() constructs the standard archive schema.

End-to-end fit-and-predict smoke tests against real modeling data are
deferred to slow integration tests; this unit-test file exercises the
static surface only.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

import numpy as np
import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.pipeline import (
    CANONICAL_FEATURES,
)
from gridiron_edge.models.base import ModelSpec, Trainable
from gridiron_edge.models.game_prediction.base import (
    GameModelType,
)
from gridiron_edge.models.game_prediction.predictor import (
    _TRAINER_FOR_NAME,
    GamesPredictor,
    TotalRandomForestPredictor,
    TotalXGBoostPredictor,
    WinProbLogisticPredictor,
    WinProbRandomForestPredictor,
    WinProbXGBoostPredictor,
    build_game_predictions,
)
from gridiron_edge.models.game_prediction.total import TotalTrainer
from gridiron_edge.models.game_prediction.win_prob import WinProbTrainer
from gridiron_edge.models.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Trainer dispatch table
# ---------------------------------------------------------------------------


class TestTrainerDispatch:
    """``_TRAINER_FOR_NAME`` maps both model_names to the right trainer."""

    def test_win_prob_maps_to_win_prob_trainer(self) -> None:
        assert _TRAINER_FOR_NAME["win_prob"] is WinProbTrainer

    def test_total_maps_to_total_trainer(self) -> None:
        assert _TRAINER_FOR_NAME["total"] is TotalTrainer

    def test_exactly_two_entries(self) -> None:
        assert set(_TRAINER_FOR_NAME.keys()) == {"win_prob", "total"}


# ---------------------------------------------------------------------------
# Composite-key registrations
# ---------------------------------------------------------------------------


_EXPECTED_REGISTRATIONS: list[tuple[str, type[GamesPredictor], str, str]] = [
    ("win_prob_logistic", WinProbLogisticPredictor, "win_prob", "logistic"),
    (
        "win_prob_random_forest",
        WinProbRandomForestPredictor,
        "win_prob",
        "random_forest",
    ),
    ("win_prob_xgboost", WinProbXGBoostPredictor, "win_prob", "xgboost"),
    ("total_random_forest", TotalRandomForestPredictor, "total", "random_forest"),
    ("total_xgboost", TotalXGBoostPredictor, "total", "xgboost"),
]


class TestCompositeRegistrations:
    """All 5 composite keys are registered and resolve to the right class."""

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_registered(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        # Importing predictor.py at test-module load time triggered registration.
        cls = ModelRegistry.get(registry_key)
        assert cls is predictor_cls

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_class_attributes(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        assert predictor_cls.model_name == model_name
        assert predictor_cls.model_type == model_type
        assert predictor_cls.spec.name == registry_key

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_spec_is_trainable(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        assert predictor_cls.spec.trainable is True

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_implements_trainable_protocol(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        instance: GamesPredictor = predictor_cls()
        assert isinstance(instance, Trainable)


# ---------------------------------------------------------------------------
# GamesPredictor accessors
# ---------------------------------------------------------------------------


class TestGamesPredictorAccessors:
    """``_trainer``, ``_game_model_spec``, ``_task`` resolve correctly."""

    def test_win_prob_trainer_instance(self) -> None:
        pred = WinProbRandomForestPredictor()
        trainer = pred._trainer()
        assert isinstance(trainer, WinProbTrainer)

    def test_total_trainer_instance(self) -> None:
        pred = TotalRandomForestPredictor()
        trainer = pred._trainer()
        assert isinstance(trainer, TotalTrainer)

    def test_win_prob_task(self) -> None:
        assert WinProbLogisticPredictor()._task() == "classification"
        assert WinProbRandomForestPredictor()._task() == "classification"
        assert WinProbXGBoostPredictor()._task() == "classification"

    def test_total_task(self) -> None:
        assert TotalRandomForestPredictor()._task() == "regression"
        assert TotalXGBoostPredictor()._task() == "regression"


# ---------------------------------------------------------------------------
# is_trained delegation
# ---------------------------------------------------------------------------


class TestIsTrained:
    """``is_trained`` returns False when the artifact directory is empty."""

    def test_returns_false_when_no_artifact(self, tmp_path: Path) -> None:
        pred = WinProbRandomForestPredictor()
        assert pred.is_trained(repo=tmp_path) is False

    def test_returns_false_for_each_composite(self, tmp_path: Path) -> None:
        for _key, cls, _name, _type in _EXPECTED_REGISTRATIONS:
            assert cls().is_trained(repo=tmp_path) is False


# ---------------------------------------------------------------------------
# predict_historical / predict_upcoming graceful fallback
# ---------------------------------------------------------------------------


class TestPredictGracefulFallback:
    """When no artifact exists, predict methods return an empty DataFrame."""

    def test_predict_historical_classification_empty(self, tmp_path: Path) -> None:
        pred = WinProbRandomForestPredictor()
        empty_games = pd.DataFrame()
        out = pred.predict_historical(empty_games, repo=tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_predict_historical_regression_empty(self, tmp_path: Path) -> None:
        pred = TotalRandomForestPredictor()
        empty_games = pd.DataFrame()
        out = pred.predict_historical(empty_games, repo=tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_predict_upcoming_classification_empty(self, tmp_path: Path) -> None:
        pred = WinProbRandomForestPredictor()
        empty_schedule = pd.DataFrame()
        out = pred.predict_upcoming(empty_schedule, repo=tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_predict_upcoming_regression_empty(self, tmp_path: Path) -> None:
        pred = TotalRandomForestPredictor()
        empty_schedule = pd.DataFrame()
        out = pred.predict_upcoming(empty_schedule, repo=tmp_path)
        assert isinstance(out, pd.DataFrame)
        assert out.empty


# ---------------------------------------------------------------------------
# Type-coverage smoke: every model's spec round-trips through ModelSpec
# ---------------------------------------------------------------------------


class TestModelSpecShape:
    """Every composite spec is a fully formed ``ModelSpec``."""

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_spec_type(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        assert isinstance(predictor_cls.spec, ModelSpec)

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_spec_description_nonempty(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        assert predictor_cls.spec.description.strip() != ""


# ---------------------------------------------------------------------------
# Model type round-trip through GameModelType enum
# ---------------------------------------------------------------------------


class TestModelTypeEnumRoundtrip:
    """Each predictor's ``model_type`` string is a valid ``GameModelType``."""

    @pytest.mark.parametrize(
        ("registry_key", "predictor_cls", "model_name", "model_type"),
        _EXPECTED_REGISTRATIONS,
    )
    def test_model_type_is_valid_enum(
        self,
        registry_key: str,
        predictor_cls: type[GamesPredictor],
        model_name: str,
        model_type: str,
    ) -> None:
        # GameModelType("logistic") / ("random_forest") / ("xgboost") must succeed.
        assert GameModelType(predictor_cls.model_type) in GameModelType


# ---------------------------------------------------------------------------
# Module-level helper: build_game_predictions
# ---------------------------------------------------------------------------


class TestBuildGamePredictions:
    """Tests for direct canonical Home-oriented prediction rows."""

    def _make_modeling_df(self) -> pd.DataFrame:
        """Return two canonical game rows in non-chronological order."""
        return pd.DataFrame(
            {
                "GAME_ID": ["G2", "G1"],
                "AWAY_TEAM": ["Bills", "Chiefs"],
                "HOME_TEAM": ["Dolphins", "Ravens"],
                "YEAR": ["2024-2025", "2024-2025"],
                "WEEK_NUM": [2, 1],
                "GAME_DATE": ["2024-09-12", "2024-09-05"],
                "AWAY_ELO": [1510.0, 1520.0],
                "HOME_ELO": [1500.0, 1480.0],
                "IS_NEUTRAL_SITE": [0, 0],
            }
        )

    def test_one_input_row_produces_one_output_row(self) -> None:
        df = self._make_modeling_df()
        home_win_probs = np.array([0.40, 0.55])

        result = build_game_predictions(
            df,
            home_win_probs,
        )

        assert len(result) == len(df)
        assert result["game_id"].is_unique

    def test_home_probability_is_model_positive_class(self) -> None:
        df = self._make_modeling_df()
        home_win_probs = np.array([0.40, 0.55])

        result = build_game_predictions(
            df,
            home_win_probs,
        )

        g1 = result.loc[result["game_id"] == "G1"].iloc[0]
        g2 = result.loc[result["game_id"] == "G2"].iloc[0]

        assert g1["home_win_prob"] == pytest.approx(0.55)
        assert g1["away_win_prob"] == pytest.approx(0.45)
        assert g2["home_win_prob"] == pytest.approx(0.40)
        assert g2["away_win_prob"] == pytest.approx(0.60)

    def test_probabilities_are_complements(self) -> None:
        result = build_game_predictions(
            self._make_modeling_df(),
            np.array([0.40, 0.55]),
        )

        sums = result["home_win_prob"] + result["away_win_prob"]

        assert sums.tolist() == pytest.approx([1.0, 1.0])

    def test_uses_canonical_team_identity_directly(self) -> None:
        result = build_game_predictions(
            self._make_modeling_df(),
            np.array([0.40, 0.55]),
        )

        g1 = result.loc[result["game_id"] == "G1"].iloc[0]

        assert g1["away_team"] == "Chiefs"
        assert g1["home_team"] == "Ravens"

    def test_neutral_site_preserves_source_orientation(self) -> None:
        df = pd.DataFrame(
            {
                "GAME_ID": ["G_NEUTRAL"],
                "AWAY_TEAM": ["Ravens"],
                "HOME_TEAM": ["Chiefs"],
                "YEAR": ["2024-2025"],
                "WEEK_NUM": [5],
                "IS_NEUTRAL_SITE": [1],
            }
        )

        result = build_game_predictions(
            df,
            np.array([0.65]),
        )

        assert len(result) == 1
        assert result["away_team"].iloc[0] == "Ravens"
        assert result["home_team"].iloc[0] == "Chiefs"
        assert result["home_win_prob"].iloc[0] == pytest.approx(0.65)
        assert result["away_win_prob"].iloc[0] == pytest.approx(0.35)

    def test_output_is_chronologically_stable(self) -> None:
        result = build_game_predictions(
            self._make_modeling_df(),
            np.array([0.40, 0.55]),
        )

        assert result["game_id"].tolist() == ["G1", "G2"]

    def test_preserves_game_date_and_canonical_elo(self) -> None:
        result = build_game_predictions(
            self._make_modeling_df(),
            np.array([0.40, 0.55]),
        )

        g1 = result.loc[result["game_id"] == "G1"].iloc[0]

        assert g1["game_date"] == "2024-09-05"
        assert g1["away_elo"] == pytest.approx(1520.0)
        assert g1["home_elo"] == pytest.approx(1480.0)

    def test_required_archive_columns_are_present(self) -> None:
        result = build_game_predictions(
            self._make_modeling_df(),
            np.array([0.40, 0.55]),
        )

        required = {
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

        assert required <= set(result.columns)
        assert "predicted_at" not in result.columns
        assert "is_backfilled" not in result.columns
        assert "model_name" not in result.columns
        assert "model_type" not in result.columns

    def test_duplicate_game_ids_are_rejected(self) -> None:
        df = self._make_modeling_df()
        df.loc[1, "GAME_ID"] = "G2"

        with pytest.raises(
            ValueError,
            match="duplicate game IDs",
        ):
            build_game_predictions(
                df,
                np.array([0.40, 0.55]),
            )

    def test_probability_count_must_match_rows(self) -> None:
        with pytest.raises(
            ValueError,
            match=("Home-win probability count must match canonical game rows"),
        ):
            build_game_predictions(
                self._make_modeling_df(),
                np.array([0.40]),
            )

    def test_input_is_not_mutated(self) -> None:
        df = self._make_modeling_df()
        expected = df.copy(deep=True)

        build_game_predictions(
            df,
            np.array([0.40, 0.55]),
        )

        pd.testing.assert_frame_equal(df, expected)

    def test_helper_source_excludes_retired_orientation(self) -> None:
        import inspect

        source = inspect.getsource(build_game_predictions)

        assert "TEAM_A" not in source
        assert "TEAM_B" not in source
        assert "HOME_FIELD" not in source


class TestUpcomingClassificationLifecycle:
    """Tests for canonical upcoming Win prediction."""

    def _schedule(self) -> pd.DataFrame:
        """Return two canonical upcoming games."""
        return pd.DataFrame(
            {
                "GAME_ID": [
                    "G_COMPLETE",
                    "G_INCOMPLETE",
                ],
                "YEAR": [
                    "2025-2026",
                    "2025-2026",
                ],
                "WEEK_NUM": [1, 1],
                "AWAY_TEAM": [
                    "Bills",
                    "Chiefs",
                ],
                "HOME_TEAM": [
                    "Dolphins",
                    "Ravens",
                ],
                "IS_NEUTRAL_SITE": [0, 0],
            }
        )

    def _enriched_schedule(self) -> pd.DataFrame:
        """Return canonical feature output with one incomplete row."""
        return pd.DataFrame(
            {
                "GAME_ID": [
                    "G_COMPLETE",
                    "G_INCOMPLETE",
                ],
                "YEAR": [
                    "2025-2026",
                    "2025-2026",
                ],
                "WEEK_NUM": [1, 1],
                "AWAY_TEAM": [
                    "Bills",
                    "Chiefs",
                ],
                "HOME_TEAM": [
                    "Dolphins",
                    "Ravens",
                ],
                "AWAY_ELO": [
                    1510.0,
                    1520.0,
                ],
                "HOME_ELO": [
                    1490.0,
                    1480.0,
                ],
                "MODEL_FEATURE": [
                    1.0,
                    float("nan"),
                ],
            }
        )

    @patch("gridiron_edge.models.game_prediction.predictor.enrich_predictions")
    @patch("gridiron_edge.models.game_prediction.predictor.run_features")
    @patch("gridiron_edge.models.game_prediction.predictor.ArtifactStore")
    def test_upcoming_prediction_uses_home_orientation(
        self,
        store_cls: MagicMock,
        run_features_mock: MagicMock,
        enrich_mock: MagicMock,
        tmp_path: Path,
    ) -> None:
        store = store_cls.return_value
        store.is_trained.return_value = True
        store.load_scaler.return_value = None

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.35, 0.65]])
        store.load.return_value = model

        enriched = self._enriched_schedule()
        run_features_mock.return_value = enriched

        enrich_mock.side_effect = lambda frame, **_kwargs: frame

        predictor = WinProbRandomForestPredictor()

        with patch.object(
            predictor,
            "_feature_fn",
            return_value=(
                lambda frame: frame.loc[
                    :,
                    ["MODEL_FEATURE"],
                ].copy()
            ),
        ):
            result: DataFrame = predictor.predict_upcoming(
                self._schedule(),
                repo=tmp_path,
            )

        assert len(result) == 1

        row = result.iloc[0]

        assert row["GAME_ID"] == "G_COMPLETE"
        assert row["AWAY_TEAM"] == "Bills"
        assert row["HOME_TEAM"] == "Dolphins"

        assert row["HOME_WIN_PROB"] == pytest.approx(0.65)
        assert row["AWAY_WIN_PROB"] == pytest.approx(0.35)

        assert row["HOME_TEAM_WIN_PROB"] == "65.0 %"
        assert row["AWAY_TEAM_WIN_PROB"] == "35.0 %"

        assert row["AWAY_TEAM_ELO"] == pytest.approx(1510.0)
        assert row["HOME_TEAM_ELO"] == pytest.approx(1490.0)

        run_features_mock.assert_called_once()

        call_kwargs = run_features_mock.call_args.kwargs
        assert call_kwargs["feature_names"] == (CANONICAL_FEATURES)

        enrich_mock.assert_called_once()
        enriched_input = enrich_mock.call_args.args[0]

        assert enriched_input["HOME_WIN_PROB"].iloc[0] == pytest.approx(0.65)
        assert enriched_input["AWAY_WIN_PROB"].iloc[0] == pytest.approx(0.35)


class TestBuildRegressionPredictions:
    """Tests for direct canonical Total prediction rows."""

    def _canonical_rows(self) -> pd.DataFrame:
        """Return canonical games in non-chronological order."""
        return pd.DataFrame(
            {
                "GAME_ID": ["G2", "G1"],
                "YEAR": ["2025-2026", "2025-2026"],
                "WEEK_NUM": [2, 1],
                "GAME_DATE": [
                    "2025-09-14",
                    "2025-09-07",
                ],
                "AWAY_TEAM": [
                    "Bills",
                    "Chiefs",
                ],
                "HOME_TEAM": [
                    "Dolphins",
                    "Ravens",
                ],
                "IS_NEUTRAL_SITE": [0, 1],
            }
        )


class TestUpcomingRegressionLifecycle:
    """Tests for independent canonical upcoming Total prediction."""

    def _schedule(self) -> pd.DataFrame:
        """Return two canonical upcoming games."""
        return pd.DataFrame(
            {
                "GAME_ID": [
                    "G_COMPLETE",
                    "G_INCOMPLETE",
                ],
                "YEAR": [
                    "2025-2026",
                    "2025-2026",
                ],
                "WEEK_NUM": [1, 1],
                "AWAY_TEAM": [
                    "Bills",
                    "Chiefs",
                ],
                "HOME_TEAM": [
                    "Dolphins",
                    "Ravens",
                ],
                "IS_NEUTRAL_SITE": [0, 0],
            }
        )

    def _enriched_schedule(self) -> pd.DataFrame:
        """Return canonical feature output with one incomplete row."""
        return pd.DataFrame(
            {
                "GAME_ID": [
                    "G_COMPLETE",
                    "G_INCOMPLETE",
                ],
                "YEAR": [
                    "2025-2026",
                    "2025-2026",
                ],
                "WEEK_NUM": [1, 1],
                "AWAY_TEAM": [
                    "Bills",
                    "Chiefs",
                ],
                "HOME_TEAM": [
                    "Dolphins",
                    "Ravens",
                ],
                "MODEL_FEATURE": [
                    1.0,
                    float("nan"),
                ],
            }
        )

    @patch("gridiron_edge.models.game_prediction.predictor.run_features")
    @patch("gridiron_edge.models.game_prediction.predictor.ArtifactStore")
    def test_total_prediction_runs_independently(
        self,
        store_cls: MagicMock,
        run_features_mock: MagicMock,
        tmp_path: Path,
    ) -> None:
        store = store_cls.return_value
        store.is_trained.return_value = True
        store.load_scaler.return_value = None

        model = MagicMock()
        model.predict.return_value = np.array([47.5])
        store.load.return_value = model

        run_features_mock.return_value = self._enriched_schedule()

        predictor = TotalRandomForestPredictor()

        with patch.object(
            predictor,
            "_feature_fn",
            return_value=(
                lambda frame: frame.loc[
                    :,
                    ["MODEL_FEATURE"],
                ].copy()
            ),
        ):
            result = predictor.predict_upcoming(
                self._schedule(),
                repo=tmp_path,
            )

        assert len(result) == 1

        row = result.iloc[0]

        assert row["GAME_ID"] == "G_COMPLETE"
        assert row["AWAY_TEAM"] == "Bills"
        assert row["HOME_TEAM"] == "Dolphins"
        assert row["model_total"] == pytest.approx(47.5)
        assert row["model_name"] == "total"
        assert row["model_type"] == ("random_forest")

        call_kwargs = run_features_mock.call_args.kwargs
        assert call_kwargs["feature_names"] == (CANONICAL_FEATURES)

        model.predict.assert_called_once()

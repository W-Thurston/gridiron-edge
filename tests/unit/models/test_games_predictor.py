# tests/unit/models/test_games_predictor.py

"""Tests for GamesPredictor + composite-key registrations (Workstream 2 D2b.1).

Covers the static surface of the new predictor classes:
    - All 5 composite keys are registered.
    - Each subclass has the right (model_name, model_type, spec).
    - GamesPredictor delegates train() to the right trainer.
    - is_trained() delegates to ArtifactStore with the (name, type) pair.
    - predict_historical() / predict_upcoming() return empty DataFrames
      when artifacts are missing (graceful fallback).
    - _maybe_predict_totals() returns None when:
        * called from a total predictor (no recursion).
        * the configured total model is not trained.

End-to-end fit-and-predict smoke tests against real modeling data are
deferred to slow integration tests; this unit-test file exercises the
static surface only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.models.base import PredictorSpec, Trainable
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
)
from gridiron_edge.models.game_prediction.total import TotalTrainer
from gridiron_edge.models.game_prediction.win_prob import WinProbTrainer
from gridiron_edge.models.registry import PredictorRegistry

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
        cls = PredictorRegistry.get(registry_key)
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

    def test_default_total_model_type(self) -> None:
        # Default is random_forest for all win_prob variants.
        assert WinProbLogisticPredictor.default_total_model_type == "random_forest"
        assert WinProbRandomForestPredictor.default_total_model_type == "random_forest"
        assert WinProbXGBoostPredictor.default_total_model_type == "random_forest"


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
# _maybe_predict_totals
# ---------------------------------------------------------------------------


class TestMaybePredictTotals:
    """``_maybe_predict_totals`` skips recursion and missing artifacts."""

    def test_total_predictor_returns_none(self, tmp_path: Path) -> None:
        """A total predictor never tries to predict its own totals."""
        pred = TotalRandomForestPredictor()
        empty_df = pd.DataFrame()
        result = pred._maybe_predict_totals(empty_df, repo=tmp_path)
        assert result is None

    def test_win_prob_returns_none_when_total_not_trained(self, tmp_path: Path) -> None:
        """Win-prob silently omits totals when the total model is missing."""
        pred = WinProbRandomForestPredictor()
        empty_df = pd.DataFrame()
        result = pred._maybe_predict_totals(empty_df, repo=tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# Legacy + new registrations coexist (D2b.1 contract)
# ---------------------------------------------------------------------------


class TestCoexistence:
    """During D2b.1, both flat and composite keys must be registered."""

    def test_legacy_keys_still_registered(self) -> None:
        for legacy_key in ("logistic", "random_forest", "xgboost"):
            assert legacy_key in PredictorRegistry.names()

    def test_composite_keys_registered(self) -> None:
        for composite_key in (
            "win_prob_logistic",
            "win_prob_random_forest",
            "win_prob_xgboost",
            "total_random_forest",
            "total_xgboost",
        ):
            assert composite_key in PredictorRegistry.names()


# ---------------------------------------------------------------------------
# Type-coverage smoke: every predictor's spec round-trips through PredictorSpec
# ---------------------------------------------------------------------------


class TestPredictorSpecShape:
    """Every composite spec is a fully-formed ``PredictorSpec``."""

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
        assert isinstance(predictor_cls.spec, PredictorSpec)

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

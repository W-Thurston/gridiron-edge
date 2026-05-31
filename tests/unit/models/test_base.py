# tests/unit/models/test_base.py
"""Tests for gridiron_edge.models.base — PredictorSpec, Predictor, Trainable protocols."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.models.base import Predictor, PredictorSpec, Trainable


class TestPredictorSpec:
    def test_is_frozen(self) -> None:
        spec = PredictorSpec(name="test_v1", description="A test predictor")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "changed"  # type: ignore[misc]

    def test_name_and_description(self) -> None:
        spec = PredictorSpec(name="elo_v1", description="Elo ratings v1")
        assert spec.name == "elo_v1"
        assert spec.description == "Elo ratings v1"

    def test_equality(self) -> None:
        a = PredictorSpec(name="elo_v1", description="Elo v1")
        b = PredictorSpec(name="elo_v1", description="Elo v1")
        assert a == b

    def test_inequality(self) -> None:
        a = PredictorSpec(name="elo_v1", description="Elo v1")
        b = PredictorSpec(name="elo_v2", description="Elo v2")
        assert a != b


class TestPredictorProtocol:
    def test_runtime_checkable(self) -> None:
        """Predictor should be a runtime-checkable Protocol."""

        class DummyPredictor:
            spec = PredictorSpec(name="dummy_v1", description="Dummy")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

        dummy = DummyPredictor()
        assert isinstance(dummy, Predictor)

    def test_class_without_predict_fails_check(self) -> None:
        class NotAPredictor:
            spec = PredictorSpec(name="bad_v1", description="Missing methods")

        assert not isinstance(NotAPredictor(), Predictor)


class TestTrainableProtocol:
    def test_runtime_checkable(self) -> None:
        class DummyTrainable:
            spec = PredictorSpec(name="trainable_v1", description="Trainable dummy")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def train(self, *, repo: Path) -> object:
                return None

            def is_trained(self, *, repo: Path) -> bool:
                return False

        dummy = DummyTrainable()
        assert isinstance(dummy, Trainable)
        assert isinstance(dummy, Predictor)  # Trainable implies Predictor

    def test_predictor_without_train_is_not_trainable(self) -> None:
        class PredictorOnly:
            spec = PredictorSpec(name="pred_v1", description="Predict only")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

        assert isinstance(PredictorOnly(), Predictor)
        assert not isinstance(PredictorOnly(), Trainable)

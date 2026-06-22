# tests/unit/models/test_base.py
"""Tests for gridiron_edge.models.base - PredictorSpec, Predictor, Trainable protocols."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.models.base import Predictor, PredictorSpec, Trainable


class TestPredictorSpec:
    def test_is_frozen(self) -> None:
        spec = PredictorSpec(name="test_a", description="A test predictor")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "changed"  # type: ignore[misc]

    def test_name_and_description(self) -> None:
        spec = PredictorSpec(name="test_a", description="Test predictor A")
        assert spec.name == "test_a"
        assert spec.description == "Test predictor A"

    def test_equality(self) -> None:
        a = PredictorSpec(name="test_a", description="Spec A")
        b = PredictorSpec(name="test_a", description="Spec A")
        assert a == b

    def test_inequality(self) -> None:
        a = PredictorSpec(name="test_a", description="Spec A")
        b = PredictorSpec(name="test_b", description="Spec B")
        assert a != b


class TestPredictorProtocol:
    def test_runtime_checkable(self) -> None:
        """Predictor should be a runtime-checkable Protocol."""

        class DummyPredictor:
            spec = PredictorSpec(name="dummy", description="Dummy")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

        dummy = DummyPredictor()
        assert isinstance(dummy, Predictor)

    def test_class_without_predict_fails_check(self) -> None:
        class NotAPredictor:
            spec = PredictorSpec(name="bad", description="Missing methods")

        assert not isinstance(NotAPredictor(), Predictor)


class TestTrainableProtocol:
    def test_runtime_checkable(self) -> None:
        class DummyTrainable:
            spec = PredictorSpec(name="trainable_dummy", description="Trainable dummy")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def is_trained(self, *, repo: Path | None = None) -> bool:
                return False

        dummy = DummyTrainable()
        assert isinstance(dummy, Trainable)

    def test_predictor_without_is_trained_is_not_trainable(self) -> None:
        class PredictorOnly:
            spec = PredictorSpec(name="predict_only", description="Predict only")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

        assert isinstance(PredictorOnly(), Predictor)
        assert not isinstance(PredictorOnly(), Trainable)

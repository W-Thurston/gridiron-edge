# tests/unit/models/test_base.py
"""Tests for gridiron_edge.models.base model and training protocols."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.models.base import GameModel, ModelSpec, Trainable


class TestModelSpec:
    def test_is_frozen(self) -> None:
        spec = ModelSpec(name="test_a", description="A test model")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "changed"  # type: ignore[misc]

    def test_name_and_description(self) -> None:
        spec = ModelSpec(name="test_a", description="Test model A")
        assert spec.name == "test_a"
        assert spec.description == "Test model A"

    def test_equality(self) -> None:
        a = ModelSpec(name="test_a", description="Spec A")
        b = ModelSpec(name="test_a", description="Spec A")
        assert a == b

    def test_inequality(self) -> None:
        a = ModelSpec(name="test_a", description="Spec A")
        b = ModelSpec(name="test_b", description="Spec B")
        assert a != b


class TestGameModelProtocol:
    def test_runtime_checkable(self) -> None:
        """GameModel should be a runtime-checkable Protocol."""

        class DummyGameModel:
            spec = ModelSpec(name="dummy", description="Dummy")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

        dummy = DummyGameModel()
        assert isinstance(dummy, GameModel)

    def test_class_without_predict_fails_check(self) -> None:
        class NotAGameModel:
            spec = ModelSpec(name="bad", description="Missing methods")

        assert not isinstance(NotAGameModel(), GameModel)


class TestTrainableProtocol:
    def test_runtime_checkable(self) -> None:
        class DummyTrainable:
            spec = ModelSpec(name="trainable_dummy", description="Trainable dummy")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def is_trained(self, *, repo: Path | None = None) -> bool:
                return False

        dummy = DummyTrainable()
        assert isinstance(dummy, Trainable)

    def test_game_model_without_is_trained_is_not_trainable(self) -> None:
        class GameModelOnly:
            spec = ModelSpec(name="predict_only", description="Predict only")

            def predict_historical(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

            def predict_upcoming(self, *, repo: Path) -> pd.DataFrame:
                return pd.DataFrame()

        assert isinstance(GameModelOnly(), GameModel)
        assert not isinstance(GameModelOnly(), Trainable)

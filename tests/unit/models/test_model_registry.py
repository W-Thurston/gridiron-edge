# tests/unit/models/test_model_registry.py
"""Tests for gridiron_edge.models.registry — PredictorRegistry."""

from __future__ import annotations

import pytest

from gridiron_edge.models.base import Predictor

# Trigger registration
import gridiron_edge.models.game_prediction  # noqa: F401
from gridiron_edge.models.registry import PredictorRegistry


class TestPredictorRegistryGet:
    def test_at_least_one_predictor_registered(self) -> None:
        assert len(PredictorRegistry.all()) > 0

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not_a_model"):
            PredictorRegistry.get("not_a_model")

    def test_all_returns_dict(self) -> None:
        all_models: dict[str, type[Predictor]] = PredictorRegistry.all()
        assert isinstance(all_models, dict)

    def test_all_keys_are_strings(self) -> None:
        for name in PredictorRegistry.all():
            assert isinstance(name, str)

    def test_all_values_have_spec(self) -> None:
        for name, cls in PredictorRegistry.all().items():
            instance: Predictor = cls()
            assert hasattr(instance, "spec"), f"{name} missing spec"
            assert instance.spec.name == name, f"{name}: spec.name mismatch"

    def test_all_values_have_predict_methods(self) -> None:
        for name, cls in PredictorRegistry.all().items():
            instance: Predictor = cls()
            assert callable(getattr(instance, "predict_historical", None)), (
                f"{name} missing predict_historical"
            )

    def test_get_roundtrips_with_all(self) -> None:
        """Every key from all() should be retrievable via get()."""
        for name in PredictorRegistry.all():
            cls: type[Predictor] = PredictorRegistry.get(name)
            assert cls().spec.name == name

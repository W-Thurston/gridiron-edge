# tests/unit/models/test_model_registry.py
"""Tests for gridiron_edge.models.registry — PredictorRegistry."""

from __future__ import annotations

import pytest

from gridiron_edge.models.base import GameModel, Predictor

# Trigger registration
from gridiron_edge.models.registry import ModelRegistry, PredictorRegistry


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

    def test_get_roundtrips_with_all(self) -> None:
        """Every key from all() should be retrievable via get()."""
        for name in PredictorRegistry.all():
            cls: type[Predictor] = PredictorRegistry.get(name)
            assert cls().spec.name == name


def test_game_models_have_predict_methods() -> None:
    for _name, cls in ModelRegistry.all().items():
        instance = cls()

        if isinstance(instance, GameModel):
            assert callable(getattr(instance, "predict_historical", None))
            assert callable(getattr(instance, "predict_upcoming", None))


def test_all_registered_models_have_spec() -> None:
    for name, cls in ModelRegistry.all().items():
        instance = cls()
        assert instance.spec.name == name


def test_game_models_register() -> None:
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401

    assert "win_prob_logistic" in ModelRegistry.names()
    assert "win_prob_random_forest" in ModelRegistry.names()
    assert "total_xgboost" in ModelRegistry.names()


def test_prop_models_register() -> None:
    import gridiron_edge.models.prop_prediction.qb_pass_yards
    import gridiron_edge.models.prop_prediction.qb_rush_yards
    import gridiron_edge.models.prop_prediction.rb_rush_yards
    import gridiron_edge.models.prop_prediction.te_rec_yards
    import gridiron_edge.models.prop_prediction.wr_rec_yards  # noqa: F401

    assert "qb_pass_yards" in ModelRegistry.names()
    assert "qb_rush_yards" in ModelRegistry.names()
    assert "rb_rush_yards" in ModelRegistry.names()
    assert "wr_rec_yards" in ModelRegistry.names()
    assert "te_rec_yards" in ModelRegistry.names()


def test_known_model_names_includes_games_and_props() -> None:
    import gridiron_edge.models.game_prediction.predictor
    import gridiron_edge.models.prop_prediction.qb_pass_yards  # noqa: F401

    names = ModelRegistry.known_model_names()

    assert "win_prob" in names
    assert "total" in names
    assert "qb_pass_yards" in names

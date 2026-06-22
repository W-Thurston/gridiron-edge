# tests/unit/models/test_model_registry.py
"""Tests for gridiron_edge.models.registry - PredictorRegistry."""

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


def test_is_trainable_does_not_instantiate_twice() -> None:
    """is_trainable should read spec.trainable, not instantiate."""
    # Game models declare trainable=True; Elo declares trainable=False.
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401
    from gridiron_edge.models.registry import ModelRegistry

    assert ModelRegistry.is_trainable("win_prob_random_forest") is True
    assert ModelRegistry.is_trainable("win_prob_elo") is False


def test_register_rejects_structural_without_flag() -> None:
    """A class that implements Trainable but declares trainable=False must fail."""
    from gridiron_edge.models.base import ModelSpec
    from gridiron_edge.models.registry import ModelRegistry

    class SilentlyTrainable:
        spec = ModelSpec(
            name="silently_trainable_test",
            description="Implements Trainable but spec.trainable=False.",
            trainable=False,
        )

        def train(self, df, *, repo=None):
            return None

        def is_trained(self, *, repo=None) -> bool:
            return False

    with pytest.raises(TypeError, match=r"declares spec\.trainable=False"):
        ModelRegistry.register(SilentlyTrainable)


def test_is_trainable_reads_spec_for_registered_models() -> None:
    """is_trainable should reflect spec.trainable, not protocol detection."""
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401

    assert ModelRegistry.is_trainable("win_prob_random_forest") is True
    assert ModelRegistry.is_trainable("win_prob_elo") is False


def test_trainable_names_reflects_spec_trainable() -> None:
    """trainable_names should match the set of models with spec.trainable=True."""
    import gridiron_edge.models.elo.predictor
    import gridiron_edge.models.game_prediction.predictor  # noqa: F401

    names = ModelRegistry.trainable_names()

    assert "win_prob_random_forest" in names
    assert "win_prob_logistic" in names
    assert "win_prob_xgboost" in names
    assert "total_random_forest" in names
    assert "total_xgboost" in names
    assert "win_prob_elo" not in names


def test_register_rejects_train_methods_without_flag() -> None:
    """A class that implements Trainable but sets trainable=False must fail."""
    from pathlib import Path

    from gridiron_edge.models.base import ModelSpec

    class SilentlyTrainable:
        spec = ModelSpec(
            name="unit10_silently_trainable",
            description="Implements Trainable but spec.trainable=False.",
            trainable=False,
        )

        def is_trained(self, *, repo: Path | None = None) -> bool:
            return False

    with pytest.raises(TypeError, match=r"declares spec\.trainable=False"):
        ModelRegistry.register(SilentlyTrainable)


def test_register_accepts_consistent_declarations() -> None:
    """Consistent declarations are accepted by the registry."""
    from pathlib import Path

    from gridiron_edge.models.base import ModelSpec

    class ConsistentTrainable:
        spec = ModelSpec(
            name="unit10_consistent_trainable",
            description="Trainable=True and implements Trainable.",
            trainable=True,
        )

        def is_trained(self, *, repo: Path | None = None) -> bool:
            return False

    class ConsistentAnalytic:
        spec = ModelSpec(
            name="unit10_consistent_analytic",
            description="Trainable=False and does not implement Trainable.",
            trainable=False,
        )

    ModelRegistry.register(ConsistentTrainable)
    ModelRegistry.register(ConsistentAnalytic)

    assert ModelRegistry.is_trainable("unit10_consistent_trainable") is True
    assert ModelRegistry.is_trainable("unit10_consistent_analytic") is False

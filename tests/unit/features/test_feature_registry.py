# tests/unit/features/test_feature_registry.py
"""Tests for gridiron_edge.features.registry — FeatureRegistry and helpers."""

from __future__ import annotations

from pandas import DataFrame
import pytest
from tests.fixtures.dataframes import make_accessor, make_games, make_modeling_rows

from gridiron_edge.features.base import Feature
from gridiron_edge.features.registry import FeatureRegistry, run_features, validate_ordering

# Ensure feature modules are imported (they register on import)
import gridiron_edge.features.team  # noqa: F401


class TestFeatureRegistryGet:
    def test_known_features_are_registered(self) -> None:
        """All expected feature names should be retrievable."""

        expected: set[str] = {
            "home_field",
            "rest",
            "travel",
            "weather",
            "divisional",
            "epa",
            "primetime",
            "record",
            "schedule_strength",
            "venue_hfa",
            "team_elo",
        }
        for name in expected:
            cls: type[Feature] = FeatureRegistry.get(name)
            assert cls is not None, f"Feature '{name}' not registered"

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not_a_feature"):
            FeatureRegistry.get("not_a_feature")

    def test_registered_features_have_spec(self) -> None:
        for name in ("home_field", "rest", "divisional", "epa"):
            cls: type[Feature] = FeatureRegistry.get(name)
            instance: Feature = cls()
            assert hasattr(instance, "spec")
            assert instance.spec.name == name

    def test_registered_features_have_compute(self) -> None:
        for name in ("home_field", "rest", "divisional"):
            cls: type[Feature] = FeatureRegistry.get(name)
            instance: Feature = cls()
            assert callable(getattr(instance, "compute", None))


class TestValidateOrdering:
    def test_valid_ordering_passes(self) -> None:
        # home_field has no deps, rest has no deps — any order is fine
        validate_ordering(["home_field", "rest"])  # should not raise

    def test_invalid_ordering_raises(self) -> None:
        # schedule_strength depends on elo — putting it before elo should fail
        with pytest.raises((ValueError, KeyError)):
            validate_ordering(["schedule_strength", "elo"])


class TestRunFeatures:
    def test_applies_features_in_order(self) -> None:
        games = make_games(
            [
                {"GAME_ID": "g1", "GAME_LOCATION": "H", "DIV_GAME": 1},
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        df = df.drop(columns=["HOME_FIELD"], errors="ignore")
        acc = make_accessor(games=games)

        result: DataFrame = run_features(
            df=df, feature_names=["home_field", "divisional"], datasets=acc
        )
        assert "HOME_FIELD" in result.columns
        assert "IS_DIV_GAME" in result.columns

# tests/unit/features/test_feature_registry.py
"""Tests for gridiron_edge.features.registry - FeatureRegistry and helpers."""

from __future__ import annotations

import pytest

from gridiron_edge.features.base import Feature
from gridiron_edge.features.registry import FeatureRegistry, validate_ordering

# Ensure feature modules are imported (they register on import)
import gridiron_edge.features.team.divisional
import gridiron_edge.features.team.elo
import gridiron_edge.features.team.epa
import gridiron_edge.features.team.primetime
import gridiron_edge.features.team.record
import gridiron_edge.features.team.rest
import gridiron_edge.features.team.schedule_strength
import gridiron_edge.features.team.travel
import gridiron_edge.features.team.venue_hfa
import gridiron_edge.features.team.weather  # noqa: F401


class TestFeatureRegistryGet:
    def test_known_features_are_registered(self) -> None:
        """All expected feature names should be retrievable."""

        expected: set[str] = {
            "travel",
            "schedule_strength",
            "venue_hfa",
            "home_away_elo",
            "home_away_divisional",
            "home_away_primetime",
            "home_away_weather",
            "home_away_rest",
            "home_away_record",
            "home_away_epa",
        }
        for name in expected:
            cls: type[Feature] = FeatureRegistry.get(name)
            assert cls is not None, f"Feature '{name}' not registered"

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="not_a_feature"):
            FeatureRegistry.get("not_a_feature")

    def test_registered_features_have_spec(self) -> None:
        for name in ("home_away_elo", "home_away_divisional"):
            cls: type[Feature] = FeatureRegistry.get(name)
            instance: Feature = cls()
            assert hasattr(instance, "spec")
            assert instance.spec.name == name

    def test_registered_features_have_compute(self) -> None:
        for name in ("home_away_rest", "home_away_divisional"):
            cls: type[Feature] = FeatureRegistry.get(name)
            instance: Feature = cls()
            assert callable(getattr(instance, "compute", None))


class TestValidateOrdering:
    def test_valid_ordering_passes(self) -> None:
        # home_field has no deps, rest has no deps - any order is fine
        validate_ordering(["home_away_elo", "home_away_divisional"])

    def test_invalid_ordering_raises(self) -> None:
        # schedule_strength depends on elo - putting it before elo should fail
        with pytest.raises((ValueError, KeyError)):
            validate_ordering(["schedule_strength", "elo"])

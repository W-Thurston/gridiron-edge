# tests/unit/features/test_base.py
"""Tests for gridiron_edge.features.base - FeatureSpec and Feature protocol."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from gridiron_edge.features.base import FeatureSpec


class TestFeatureSpec:
    def test_is_frozen(self) -> None:
        spec = FeatureSpec(name="test", produces=["COL_A"])
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "changed"  # type: ignore[misc]

    def test_name_and_produces(self) -> None:
        spec = FeatureSpec(name="my_feature", produces=["X", "Y"])
        assert spec.name == "my_feature"
        assert list(spec.produces) == ["X", "Y"]

    def test_depends_on_defaults_to_empty(self) -> None:
        spec = FeatureSpec(name="f", produces=["A"])
        assert spec.depends_on == ()

    def test_depends_on_accepts_list(self) -> None:
        spec = FeatureSpec(name="f", produces=["A"], depends_on=["home_field", "rest"])
        assert list(spec.depends_on) == ["home_field", "rest"]

    def test_equality(self) -> None:
        a = FeatureSpec(name="f", produces=["X"])
        b = FeatureSpec(name="f", produces=["X"])
        assert a == b

    def test_inequality_different_name(self) -> None:
        a = FeatureSpec(name="f1", produces=["X"])
        b = FeatureSpec(name="f2", produces=["X"])
        assert a != b


class TestFeatureProtocol:
    def test_class_with_spec_and_compute_satisfies_protocol(self) -> None:
        """A class with the right attributes/methods should be a valid Feature."""
        from gridiron_edge.features.base import Feature, FeatureSpec

        class DummyFeature:
            spec = FeatureSpec(name="dummy", produces=["DUMMY_COL"])

            def compute(self, *, df: pd.DataFrame, datasets: object) -> pd.DataFrame:
                df["DUMMY_COL"] = 1
                return df

        # Protocol structural check - if this doesn't raise, it satisfies the protocol
        dummy: Feature = DummyFeature()  # type: ignore[assignment]
        assert dummy.spec.name == "dummy"

# src/gridiron_edge/features/registry.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from .base import Feature

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor


class FeatureRegistry:
    """Registry mapping feature names to their implementing classes.

    Features are registered via the ``@FeatureRegistry.register(name)``
    decorator and retrieved by name at pipeline execution time.
    """

    _features: ClassVar[dict[str, type[Feature]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a feature class under the given name.

        Args:
            name: The string key used to look up this feature at runtime.

        Returns:
            A decorator that registers the wrapped class and returns it unchanged.
        """

        def deco(feature_cls: type[Feature]) -> type[Feature]:
            cls._features[name] = feature_cls
            return feature_cls

        return deco

    @classmethod
    def get(cls, name: str) -> type[Feature]:
        """Retrieve a registered feature class by name.

        Args:
            name: The registered feature key.

        Returns:
            The feature class associated with ``name``.

        Raises:
            KeyError: If ``name`` has not been registered.
        """
        return cls._features[name]


def run_features(
    *,
    df: pd.DataFrame,
    feature_names: Sequence[str],
    datasets: DatasetAccessor,
) -> pd.DataFrame:
    """Apply a sequence of named features to a DataFrame in order.

    Args:
        df: The input modeling DataFrame.
        feature_names: Ordered list of feature keys to apply.
        datasets: A ``DatasetAccessor``-compatible object passed to each feature.

    Returns:
        The DataFrame with all requested features computed and appended.
    """
    out: pd.DataFrame = df
    for name in feature_names:
        out = FeatureRegistry.get(name)().compute(df=out, datasets=datasets)
    return out

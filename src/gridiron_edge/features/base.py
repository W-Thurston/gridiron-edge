# src/gridiron_edge/features/base.py

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata describing a feature's identity and outputs.

    Attributes:
        name: Unique string key used to register and look up this feature.
        produces: Ordered list of column names this feature adds to the
            modeling DataFrame.
    """

    name: str
    produces: Sequence[str]


class Feature(Protocol):
    """Protocol defining the interface all feature implementations must satisfy.

    Any class with a ``spec`` attribute and a ``compute`` method matching
    this signature is a valid ``Feature`` without explicit inheritance.

    Attributes:
        spec: A ``FeatureSpec`` describing the feature's name and outputs.
    """

    spec: FeatureSpec

    def compute(self, *, df: pd.DataFrame, datasets: "DatasetAccessor") -> pd.DataFrame:
        """Compute and append feature columns to the input DataFrame.

        Args:
            df: The modeling DataFrame to augment.
            datasets: A ``DatasetAccessor`` providing access to canonical
                datasets required by this feature.

        Returns:
            The input DataFrame with new feature columns appended.
        """
        ...

# src/gridiron_edge/models/base.py

"""Base types for the prediction model layer.

Defines two protocols:

``Predictor``
    All prediction models must satisfy this. Implements
    ``predict_historical`` and ``predict_upcoming``. Elo models implement
    only this -- they have no training step.

``Trainable``
    Optional extension for models with an explicit training step (logistic
    regression, neural networks, XGBoost). Adds ``train()`` and
    ``is_trained()``. The CLI checks ``isinstance(predictor, Trainable)``
    to decide whether ``gridiron models train`` applies.

Both use structural subtyping via Protocol -- no explicit inheritance needed.
A class is a valid ``Predictor`` or ``Trainable`` if it has the right
methods, regardless of what it inherits from.

Adding a new model requires only:
  1. Implementing the appropriate protocol(s)
  2. Registering with ``PredictorRegistry``
  3. Zero changes to evaluation, archiving, or CLI infrastructure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from gridiron_edge.models.artifact import ModelMetadata


@dataclass(frozen=True)
class PredictorSpec:
    """Metadata describing a predictor's identity.

    Attributes:
        name: Unique string key used to register and look up this predictor.
            Should follow the convention ``{model_type}_v{version}``,
            e.g. ``"elo_v2"``, ``"logistic"``, ``"random_forest"``.
        description: Human-readable description shown in CLI help and
            evaluation output.
        trainable: Whether this predictor has an explicit training step.
            ``False`` for Elo models (no artifact needed), ``True`` for
            ML models. Used by the ``gridiron models list`` command.
    """

    name: str
    description: str
    trainable: bool = False


@runtime_checkable
class Predictor(Protocol):
    """Protocol defining the interface all prediction models must satisfy.

    Any class with a ``spec`` attribute and ``predict_historical`` /
    ``predict_upcoming`` methods matching these signatures is a valid
    ``Predictor`` without explicit inheritance.

    For models with a training step (logistic regression, neural networks),
    ``predict_historical`` should load a pre-trained artifact. The training
    step itself is invoked separately via ``gridiron models train``.

    Attributes:
        spec: A ``PredictorSpec`` describing the predictor's identity.
    """

    spec: PredictorSpec

    def predict_historical(
        self,
        games: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for all historical games.

        Args:
            games: Canonical games DataFrame (``NFL_wk_by_wk_cleaned.csv``).
                Contains completed games only.
            repo: Repository root path.

        Returns:
            DataFrame in prediction archive schema. One row per game.
        """
        ...

    def predict_upcoming(
        self,
        schedule: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for upcoming (unplayed) games.

        Args:
            schedule: Canonical upcoming schedule DataFrame.
            repo: Repository root path.

        Returns:
            DataFrame compatible with ``build_predictions_df()`` output.
        """
        ...


@runtime_checkable
class Trainable(Protocol):
    """Optional protocol for models with an explicit training step.

    Implemented by ML models (logistic regression, XGBoost, neural networks).
    Not implemented by Elo models -- they compute predictions analytically.

    The CLI checks ``isinstance(predictor, Trainable)`` to determine
    whether ``gridiron models train`` applies to a given model version.
    """

    spec: PredictorSpec

    def train(
        self,
        df: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> ModelMetadata:
        """Train the model and save the artifact to the store.

        Implementations should:
          1. Validate feature matrix via ``load_modeling_file``
          2. Split into training and holdout sets
          3. Fit the model on training data
          4. Score on holdout set (Brier score)
          5. Save artifact via ``ArtifactStore.save``
          6. Return populated ``ModelMetadata``

        Args:
            df: Full feature matrix from ``load_modeling_file()``.
            repo: Repository root path.

        Returns:
            ``ModelMetadata`` describing the trained artifact.
        """
        ...

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists for this model version.

        Args:
            repo: Repository root path.

        Returns:
            ``True`` if a trained artifact exists and can be loaded.
        """
        ...

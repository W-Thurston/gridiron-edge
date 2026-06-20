"""Base types for the model layer.

The model layer is organized around domain models:

``Model``
    Minimal root protocol. A model has a ``spec`` describing its identity.

``GameModel``
    A model that can generate game-level predictions from historical games
    or upcoming schedules.

``PropModel``
    A model family for player props. Prop models currently expose training
    through ``PropTrainer`` and will gain richer prediction APIs as the prop
    integration spine matures.

``Trainable``
    Optional capability protocol for models with an explicit training step.

Model discovery is handled by ``ModelRegistry``. The old Predictor naming is
kept as a backward-compatible alias during the migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a model's registry identity.

    Attributes:
        name: Unique registry key. Game model keys use the composite
            ``{model_name}_{model_type}`` convention, e.g.
            ``"win_prob_random_forest"``. Prop model family keys use the
            prop stat name, e.g. ``"qb_pass_yards"``.
        description: Human-readable description shown in CLI output.
        trainable: Whether this model has an explicit training step.
    """

    name: str
    description: str
    trainable: bool = False


@runtime_checkable
class Model(Protocol):
    """Minimal root protocol for any registered model."""

    spec: ModelSpec


@runtime_checkable
class GameModel(Model, Protocol):
    """Protocol for game-level prediction models."""

    def predict_historical(
        self,
        games: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for historical games."""
        ...

    def predict_upcoming(
        self,
        schedule: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for upcoming games."""
        ...


@runtime_checkable
class PropModel(Model, Protocol):
    """Protocol for prop model families.

    Prop models currently train through the PropTrainer interface. This
    protocol intentionally stays light because prop historical/upcoming
    prediction APIs are still evolving as the prop integration spine is
    built out.
    """


@runtime_checkable
class Trainable(Protocol):
    """Optional capability protocol for models with persisted training state.

    Members:
        spec: Model identity.
        is_trained: Whether a saved artifact exists.

    The training call itself (``train(...)``) is intentionally NOT part
    of this protocol because game trainers and prop trainers have
    different training call shapes. The protocol describes the artifact
    lifecycle, not the training workflow.
    """

    spec: ModelSpec

    def is_trained(self, *, repo: Path | None = None) -> bool:
        """Return whether a trained artifact exists."""
        ...


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# Keep old names during the migration. Existing game-side code can continue
# importing Predictor / PredictorSpec while new code shifts to Model /
# ModelSpec. Remove these aliases in a later cleanup pass once all imports
# are migrated.
PredictorSpec = ModelSpec
Predictor = GameModel

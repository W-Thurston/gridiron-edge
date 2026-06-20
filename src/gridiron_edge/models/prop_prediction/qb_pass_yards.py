"""QB passing yards prop model."""

from __future__ import annotations

from gridiron_edge.models.prop_prediction.base import PropModelSpec, PropTrainer
from gridiron_edge.models.registry import ModelRegistry


@ModelRegistry.register
class QBPassYardsTrainer(PropTrainer):
    """QB passing yards prop model."""

    @property
    def spec(self) -> PropModelSpec:
        """QB passing yards model specification."""
        return PropModelSpec(
            name="qb_pass_yards",
            target_col="passing_yards",
            position_filter=["QB"],
            description="QB passing yards prop model",
            clip_hi=600,
        )

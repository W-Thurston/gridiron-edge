"""QB rushing yards prop model."""

from __future__ import annotations

from gridiron_edge.models.prop_prediction.base import PropModelSpec, PropTrainer
from gridiron_edge.models.registry import ModelRegistry


@ModelRegistry.register
class QBRushYardsTrainer(PropTrainer):
    """QB rushing yards prop model."""

    @property
    def spec(self) -> PropModelSpec:
        """QB rushing yards model specification."""
        return PropModelSpec(
            name="qb_rush_yards",
            target_col="rushing_yards",
            position_filter=["QB"],
            description="QB rushing yards prop model",
            clip_hi=200,
        )

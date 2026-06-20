"""WR receiving yards prop model."""

from __future__ import annotations

from gridiron_edge.models.prop_prediction.base import PropModelSpec, PropTrainer
from gridiron_edge.models.registry import ModelRegistry


@ModelRegistry.register
class WRRecYardsTrainer(PropTrainer):
    """WR receiving yards prop model."""

    @property
    def spec(self) -> PropModelSpec:
        """WR receiving yards model specification."""
        return PropModelSpec(
            name="wr_rec_yards",
            target_col="receiving_yards",
            position_filter=["WR"],
            description="WR receiving yards prop model",
            clip_hi=300,
        )

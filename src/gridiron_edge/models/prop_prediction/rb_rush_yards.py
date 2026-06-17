"""RB rushing yards prop model."""

from __future__ import annotations

from gridiron_edge.models.prop_prediction.base import PropModelSpec, PropTrainer


class RBRushYardsTrainer(PropTrainer):
    """RB rushing yards prop model."""

    @property
    def spec(self) -> PropModelSpec:
        """RB rushing yards model specification."""
        return PropModelSpec(
            name="rb_rush_yards",
            target_col="rushing_yards",
            position_filter=["RB", "FB"],
            description="RB rushing yards prop model",
            clip_hi=250,
        )

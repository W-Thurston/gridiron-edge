"""TE receiving yards prop model."""

from __future__ import annotations

from gridiron_edge.models.prop_prediction.base import PropModelSpec, PropTrainer


class TERecYardsTrainer(PropTrainer):
    """TE receiving yards prop model."""

    @property
    def spec(self) -> PropModelSpec:
        """TE receiving yards model specification."""
        return PropModelSpec(
            name="te_rec_yards",
            target_col="receiving_yards",
            position_filter=["TE"],
            description="TE receiving yards prop model",
            clip_hi=250,
        )

# src/gridiron_edge/models/prop_prediction/te_rec_yards.py

"""TE receiving yards prop model."""

from __future__ import annotations

from gridiron_edge.models.prop_prediction.base import PropModelSpec, PropTrainer
from gridiron_edge.models.registry import ModelRegistry


@ModelRegistry.register
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
            exclude_feature_prefixes=(
                # TEs do not throw passes; passing-side rolling stats
                # are structurally undefined for the position. See
                # wr_rec_yards.py for the fuller rationale.
                "passing_",
                "attempts_",
                "completions_",
                "sacks_suffered_",
            ),
        )

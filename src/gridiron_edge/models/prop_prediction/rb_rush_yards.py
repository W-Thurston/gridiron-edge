"""RB rushing yards prop model."""

from __future__ import annotations

from gridiron_edge.models.prop_prediction.base import PropModelSpec, PropTrainer
from gridiron_edge.models.registry import ModelRegistry


@ModelRegistry.register
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
            exclude_feature_prefixes=(
                # RBs do not throw passes; passing-side rolling stats
                # are structurally undefined for the position. Note
                # that receiving_* is intentionally kept — RBs catch
                # passes and receiving features are meaningful signal.
                "passing_",
                "attempts_",
                "completions_",
                "sacks_suffered_",
            ),
        )

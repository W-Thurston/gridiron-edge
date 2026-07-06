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
            exclude_feature_prefixes=(
                # WRs do not throw passes; passing-side rolling stats
                # are structurally undefined for the position. Note
                # that rushing_* is intentionally kept — WRs get rare
                # end-arounds and jet sweeps and the feature is not
                # structurally invalid the way passing_ is.
                "passing_",
                "attempts_",
                "completions_",
                "sacks_suffered_",
            ),
        )

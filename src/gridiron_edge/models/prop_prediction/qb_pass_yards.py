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
            exclude_feature_prefixes=(
                # QBs do not catch passes; receiving-side rolling stats
                # are structurally undefined for the position. Sporadic
                # non-null rows (trick plays, halfback passes recorded
                # against a QB) let these columns squeak under the 50%
                # NaN threshold used during training, but they are
                # ~100% NaN in any given holdout season and collapse
                # the holdout via dropna.
                "receiving_",
                "receptions_",
                "targets_",
                "target_share",
                "air_yards_share",
                "wopr",
                "racr",
                "pacr",
            ),
        )

# tests/unit/features/test_projected_total_plays.py

"""Tests for the market-blind projected-total-plays feature (Domain 13)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.projected_total_plays import (
    PROJECTED_TOTAL_PLAYS,
    ProjectedTotalPlaysFeature,
)

_DATASETS = MagicMock(spec=DatasetAccessor)


def _game_row(**overrides: float) -> DataFrame:
    base = {
        "AWAY_OFF_PLAYS": 62.0,
        "HOME_OFF_PLAYS": 65.0,
        "MARKER": "preserved",
    }
    base.update(overrides)
    return DataFrame([base])


class TestProjectedTotalPlaysFeature:
    def test_registered_as_projected_total_plays(self) -> None:
        assert FeatureRegistry.get("projected_total_plays") is ProjectedTotalPlaysFeature

    def test_depends_on_home_away_epa(self) -> None:
        assert list(ProjectedTotalPlaysFeature.spec.depends_on) == ["home_away_epa"]

    def test_sums_both_teams_pace(self) -> None:
        game = _game_row()

        result = ProjectedTotalPlaysFeature().compute(df=game, datasets=_DATASETS)

        assert result[PROJECTED_TOTAL_PLAYS].iloc[0] == pytest.approx(62.0 + 65.0)

    def test_propagates_nan_from_missing_pace(self) -> None:
        game = _game_row(AWAY_OFF_PLAYS=float("nan"))

        result = ProjectedTotalPlaysFeature().compute(df=game, datasets=_DATASETS)

        assert pd.isna(result[PROJECTED_TOTAL_PLAYS].iloc[0])

    def test_preserves_input_and_unrelated_columns(self) -> None:
        game = _game_row()

        result = ProjectedTotalPlaysFeature().compute(df=game, datasets=_DATASETS)

        assert result["MARKER"].iloc[0] == "preserved"
        assert len(result) == 1

    def test_missing_columns_raises(self) -> None:
        incomplete = DataFrame([{"AWAY_OFF_PLAYS": 60.0}])

        with pytest.raises(ValueError, match="missing required columns"):
            ProjectedTotalPlaysFeature().compute(df=incomplete, datasets=_DATASETS)

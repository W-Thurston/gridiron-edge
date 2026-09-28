# tests/unit/features/test_unit_efficiency.py

"""Tests for unit efficiency differentials (Domain 13 matchup feature)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.unit_efficiency import (
    AWAY_PASS_MATCHUP_EPA,
    AWAY_RUSH_MATCHUP_EPA,
    HOME_PASS_MATCHUP_EPA,
    HOME_RUSH_MATCHUP_EPA,
    UnitEfficiencyDifferentialsFeature,
)

_DATASETS = MagicMock(spec=DatasetAccessor)


def _game_row(**overrides: float) -> DataFrame:
    base = {
        "AWAY_OFF_PASS_EPA_ADJ": 0.20,
        "HOME_OFF_PASS_EPA_ADJ": 0.10,
        "AWAY_DEF_PASS_EPA_ADJ": -0.05,
        "HOME_DEF_PASS_EPA_ADJ": 0.05,
        "AWAY_OFF_RUSH_EPA_ADJ": -0.10,
        "HOME_OFF_RUSH_EPA_ADJ": 0.15,
        "AWAY_DEF_RUSH_EPA_ADJ": 0.02,
        "HOME_DEF_RUSH_EPA_ADJ": -0.08,
        "MARKER": "preserved",
    }
    base.update(overrides)
    return DataFrame([base])


class TestUnitEfficiencyDifferentialsFeature:
    def test_registered_as_unit_efficiency_differentials(self) -> None:
        assert (
            FeatureRegistry.get("unit_efficiency_differentials")
            is UnitEfficiencyDifferentialsFeature
        )

    def test_depends_on_opponent_adjusted_epa(self) -> None:
        spec = UnitEfficiencyDifferentialsFeature.spec
        assert list(spec.depends_on) == ["home_away_opponent_adjusted_epa"]

    def test_computes_matchup_values_correctly(self) -> None:
        game = _game_row()

        result = UnitEfficiencyDifferentialsFeature().compute(df=game, datasets=_DATASETS)

        row = result.iloc[0]
        # AWAY offense (pass) vs HOME defense (pass allowed).
        assert row[AWAY_PASS_MATCHUP_EPA] == pytest.approx(0.20 - 0.05)
        # HOME offense (pass) vs AWAY defense (pass allowed).
        assert row[HOME_PASS_MATCHUP_EPA] == pytest.approx(0.10 - (-0.05))
        assert row[AWAY_RUSH_MATCHUP_EPA] == pytest.approx(-0.10 - (-0.08))
        assert row[HOME_RUSH_MATCHUP_EPA] == pytest.approx(0.15 - 0.02)

    def test_does_not_produce_diff_columns(self) -> None:
        """Diffs are derived later, on demand, by _make_diff_features -
        matching Elo and opponent-adjusted EPA's convention - not persisted
        by this feature itself (a real bug caught by inspecting the
        regenerated modeling file directly: this feature originally
        computed and stored PASS_MATCHUP_EPA_DIFF/RUSH_MATCHUP_EPA_DIFF
        here too, producing dead, redundant columns nothing read, unlike
        every sibling feature)."""
        game = _game_row()

        result = UnitEfficiencyDifferentialsFeature().compute(df=game, datasets=_DATASETS)

        assert "PASS_MATCHUP_EPA_DIFF" not in result.columns
        assert "RUSH_MATCHUP_EPA_DIFF" not in result.columns
        assert set(UnitEfficiencyDifferentialsFeature.spec.produces) == {
            AWAY_PASS_MATCHUP_EPA,
            HOME_PASS_MATCHUP_EPA,
            AWAY_RUSH_MATCHUP_EPA,
            HOME_RUSH_MATCHUP_EPA,
        }

    def test_propagates_nan_from_missing_adjusted_epa(self) -> None:
        """A week-1-like game where opponent-adjusted EPA is itself null."""
        game = _game_row(
            AWAY_OFF_PASS_EPA_ADJ=float("nan"),
            HOME_OFF_PASS_EPA_ADJ=float("nan"),
            AWAY_DEF_PASS_EPA_ADJ=float("nan"),
            HOME_DEF_PASS_EPA_ADJ=float("nan"),
        )

        result = UnitEfficiencyDifferentialsFeature().compute(df=game, datasets=_DATASETS)

        row = result.iloc[0]
        assert pd.isna(row[AWAY_PASS_MATCHUP_EPA])
        assert pd.isna(row[HOME_PASS_MATCHUP_EPA])

    def test_preserves_input_and_unrelated_columns(self) -> None:
        game = _game_row()

        result = UnitEfficiencyDifferentialsFeature().compute(df=game, datasets=_DATASETS)

        assert result["MARKER"].iloc[0] == "preserved"
        assert len(result) == 1

    def test_missing_columns_raises(self) -> None:
        incomplete = DataFrame([{"AWAY_OFF_PASS_EPA_ADJ": 0.1}])

        with pytest.raises(ValueError, match="missing required columns"):
            UnitEfficiencyDifferentialsFeature().compute(df=incomplete, datasets=_DATASETS)

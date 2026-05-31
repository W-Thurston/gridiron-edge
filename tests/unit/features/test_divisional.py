# tests/unit/features/test_divisional.py
"""Tests for gridiron_edge.features.team.divisional — DivisionalFeature."""

from __future__ import annotations

from pandas import DataFrame
from tests.fixtures.dataframes import make_accessor, make_games, make_modeling_rows

from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.divisional import DivisionalFeature


class TestDivisionalFeature:
    def test_divisional_game_flagged_as_1(self) -> None:
        games = make_games([{"GAME_ID": "g1", "DIV_GAME": 1}])
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)

        result: DataFrame = DivisionalFeature().compute(df=df, datasets=acc)
        assert result["IS_DIV_GAME"].iloc[0] == 1

    def test_non_divisional_game_flagged_as_0(self) -> None:
        games = make_games([{"GAME_ID": "g1", "DIV_GAME": 0}])
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)

        result: DataFrame = DivisionalFeature().compute(df=df, datasets=acc)
        assert result["IS_DIV_GAME"].iloc[0] == 0

    def test_preserves_existing_columns(self) -> None:
        games = make_games([{"GAME_ID": "g1", "DIV_GAME": 1}])
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        original_cols: set[str] = set(df.columns)
        acc = make_accessor(games=games)

        result: DataFrame = DivisionalFeature().compute(df=df, datasets=acc)
        assert original_cols <= set(result.columns)

    def test_spec_name_is_divisional(self) -> None:
        assert DivisionalFeature().spec.name == "divisional"

    def test_spec_produces_is_div_game(self) -> None:
        assert "IS_DIV_GAME" in DivisionalFeature().spec.produces

    def test_registered_under_divisional(self) -> None:
        assert FeatureRegistry.get("divisional") is DivisionalFeature

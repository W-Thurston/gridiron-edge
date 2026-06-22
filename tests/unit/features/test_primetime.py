# tests/unit/features/test_primetime.py
"""Tests for gridiron_edge.features.team.primetime - PrimetimeFeature."""

from __future__ import annotations

from pandas import DataFrame
from tests.fixtures.dataframes import make_accessor, make_games, make_modeling_rows

from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.primetime import PrimetimeFeature


class TestIsPrimetimeLogic:
    """Test the _is_primetime helper via the full feature compute path."""

    def test_monday_night_is_primetime(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DAY_OF_WEEK": "Monday",
                    "GAMETIME": "20:15",
                }
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)
        result: DataFrame = PrimetimeFeature().compute(df=df, datasets=acc)
        assert result["IS_PRIMETIME"].iloc[0] == 1

    def test_monday_any_time_is_primetime(self) -> None:
        """All Monday games are primetime regardless of kickoff time."""

        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DAY_OF_WEEK": "Monday",
                    "GAMETIME": "13:00",
                }
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)
        result: DataFrame = PrimetimeFeature().compute(df=df, datasets=acc)
        assert result["IS_PRIMETIME"].iloc[0] == 1

    def test_sunday_night_is_primetime(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DAY_OF_WEEK": "Sunday",
                    "GAMETIME": "20:20",
                }
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)
        result: DataFrame = PrimetimeFeature().compute(df=df, datasets=acc)
        assert result["IS_PRIMETIME"].iloc[0] == 1

    def test_sunday_afternoon_is_not_primetime(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DAY_OF_WEEK": "Sunday",
                    "GAMETIME": "13:00",
                }
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)
        result: DataFrame = PrimetimeFeature().compute(df=df, datasets=acc)
        assert result["IS_PRIMETIME"].iloc[0] == 0

    def test_thursday_night_is_primetime(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DAY_OF_WEEK": "Thursday",
                    "GAMETIME": "20:20",
                }
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)
        result: DataFrame = PrimetimeFeature().compute(df=df, datasets=acc)
        assert result["IS_PRIMETIME"].iloc[0] == 1

    def test_thursday_afternoon_is_not_primetime(self) -> None:
        """Thanksgiving early games are not primetime."""

        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DAY_OF_WEEK": "Thursday",
                    "GAMETIME": "12:30",
                }
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)
        result: DataFrame = PrimetimeFeature().compute(df=df, datasets=acc)
        assert result["IS_PRIMETIME"].iloc[0] == 0

    def test_saturday_night_is_primetime(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DAY_OF_WEEK": "Saturday",
                    "GAMETIME": "20:00",
                }
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)
        result: DataFrame = PrimetimeFeature().compute(df=df, datasets=acc)
        assert result["IS_PRIMETIME"].iloc[0] == 1

    def test_saturday_daytime_is_not_primetime(self) -> None:
        games = make_games(
            [
                {
                    "GAME_ID": "g1",
                    "GAME_DAY_OF_WEEK": "Saturday",
                    "GAMETIME": "12:00",
                }
            ]
        )
        df = make_modeling_rows([{"GAME_ID": "g1"}])
        acc = make_accessor(games=games)
        result: DataFrame = PrimetimeFeature().compute(df=df, datasets=acc)
        assert result["IS_PRIMETIME"].iloc[0] == 0


class TestPrimetimeFeatureSpec:
    def test_spec_name(self) -> None:
        assert PrimetimeFeature().spec.name == "primetime"

    def test_produces_is_primetime(self) -> None:
        assert "IS_PRIMETIME" in PrimetimeFeature().spec.produces

    def test_registered_under_primetime(self) -> None:
        assert FeatureRegistry.get("primetime") is PrimetimeFeature

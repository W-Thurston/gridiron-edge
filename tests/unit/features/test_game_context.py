# tests/unit/features/test_game_context.py
"""Tests for gridiron_edge.features.player.game_context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pytest

from gridiron_edge.features.player.game_context import (
    _DOME_ROOFS,
    _FULL_NAME_TO_ABBREV,
    _TEAM_CODE_MAP,
    _derive_dome,
    _derive_implied_total,
    _derive_is_home,
    _derive_rest_days,
    _derive_spread,
    _drop_raw_game_columns,
    _join_game_data,
    build_game_context_features,
)


def _make_player_logs() -> DataFrame:
    """Build minimal player game logs for 3 weeks."""
    rows: list[dict[str, bool | int | str]] = []
    for week in range(1, 4):
        # KC player (home in game_id)
        rows.append(
            {
                "player_id": "QB1_KC",
                "player_name": "P.Mahomes",
                "position": "QB",
                "team": "KC",
                "opponent_team": "LV",
                "season": 2024,
                "week": week,
                "game_id": f"2024_{week:02d}_LV_KC",
                "is_skill": True,
            }
        )
        # LV player (away in game_id)
        rows.append(
            {
                "player_id": "QB2_LV",
                "player_name": "A.OConnell",
                "position": "QB",
                "team": "LV",
                "opponent_team": "KC",
                "season": 2024,
                "week": week,
                "game_id": f"2024_{week:02d}_LV_KC",
                "is_skill": True,
            }
        )
    return pd.DataFrame(rows)


def _make_games() -> DataFrame:
    """Build matching games data for 3 weeks."""
    rows: list[dict[str, float | str]] = []
    base_dates: list[str] = ["2024-09-08", "2024-09-15", "2024-09-22"]
    for week in range(1, 4):
        rows.append(
            {
                "GAME_ID": f"2024_{week:02d}_LV_KC",
                "VEGAS_LINE": -3.5,
                "OVER_UNDER": 47.0,
                "FAVORITED": "Kansas City Chiefs",
                "ROOF": "outdoors",
                "GAME_DATE": base_dates[week - 1],
            }
        )
    return pd.DataFrame(rows)


def _make_joined() -> DataFrame:
    """Build a pre-joined DataFrame for testing derive functions."""
    player: DataFrame = _make_player_logs()
    games: DataFrame = _make_games()
    return _join_game_data(player, games)


class TestConstants:
    """Verify module-level constants."""

    def test_team_code_map_has_four_entries(self) -> None:
        assert len(_TEAM_CODE_MAP) == 4

    def test_full_name_mapping_covers_32_current_teams(self) -> None:
        # 32 current + 5 historical = 37
        assert len(_FULL_NAME_TO_ABBREV) >= 32

    def test_dome_roofs(self) -> None:
        assert {"dome", "closed"} == _DOME_ROOFS

    def test_historical_names_included(self) -> None:
        assert "Oakland Raiders" in _FULL_NAME_TO_ABBREV
        assert "San Diego Chargers" in _FULL_NAME_TO_ABBREV
        assert "St. Louis Rams" in _FULL_NAME_TO_ABBREV
        assert "Washington Redskins" in _FULL_NAME_TO_ABBREV
        assert "Washington Football Team" in _FULL_NAME_TO_ABBREV


class TestJoinGameData:
    """Verify game data joins correctly to player rows."""

    def test_no_row_duplication(self) -> None:
        player: DataFrame = _make_player_logs()
        games: DataFrame = _make_games()
        result: DataFrame = _join_game_data(player, games)
        assert len(result) == len(player)

    def test_game_columns_present(self) -> None:
        result: DataFrame = _make_joined()
        for col in ["VEGAS_LINE", "OVER_UNDER", "FAVORITED", "ROOF", "GAME_DATE"]:
            assert col in result.columns

    def test_unmatched_rows_get_nan(self) -> None:
        """Player rows with no matching game keep NaN for game columns."""
        player: DataFrame = _make_player_logs()
        # Empty games DataFrame
        games = pd.DataFrame(
            columns=["GAME_ID", "VEGAS_LINE", "OVER_UNDER", "FAVORITED", "ROOF", "GAME_DATE"]
        )
        result: DataFrame = _join_game_data(player, games)
        assert result["VEGAS_LINE"].isna().all()


class TestDeriveIsHome:
    """Verify home/away derivation from game_id."""

    def test_home_team_detected(self) -> None:
        """KC is the home team (4th segment of game_id)."""
        df: DataFrame = _make_joined()
        result: DataFrame = _derive_is_home(df)
        kc: Series = result[result["team"] == "KC"]
        assert kc["is_home"].all()

    def test_away_team_detected(self) -> None:
        """LV is the away team (3rd segment of game_id)."""
        df: DataFrame = _make_joined()
        result: DataFrame = _derive_is_home(df)
        lv: Series = result[result["team"] == "LV"]
        assert not lv["is_home"].any()

    def test_historical_codes_normalized(self) -> None:
        """A player with team=LV should match home in a game_id with OAK."""
        df = pd.DataFrame(
            [
                {
                    "player_id": "P1",
                    "team": "LV",
                    "game_id": "2017_01_DEN_OAK",
                }
            ]
        )
        result: DataFrame = _derive_is_home(df)
        assert result["is_home"].iloc[0] is np.bool_(True)


class TestDeriveSpread:
    """Verify team-perspective spread computation."""

    def test_favorite_gets_negative_spread(self) -> None:
        """KC is FAVORITED → game_spread should be negative."""
        df: DataFrame = _make_joined()
        df = _derive_is_home(df)
        result: DataFrame = _derive_spread(df)
        kc: Series = result[result["team"] == "KC"]
        assert (kc["game_spread"] < 0).all()

    def test_underdog_gets_positive_spread(self) -> None:
        """LV is the underdog → game_spread should be positive."""
        df: DataFrame = _make_joined()
        df = _derive_is_home(df)
        result: DataFrame = _derive_spread(df)
        lv: Series = result[result["team"] == "LV"]
        assert (lv["game_spread"] > 0).all()

    def test_spread_magnitude_correct(self) -> None:
        """abs(game_spread) should equal abs(VEGAS_LINE) for both teams."""
        df: DataFrame = _make_joined()
        df = _derive_is_home(df)
        result: DataFrame = _derive_spread(df)
        assert result["game_spread"].abs().eq(3.5).all()

    def test_pickem_spread_is_zero(self) -> None:
        """When VEGAS_LINE=0 and FAVORITED=NaN, spread is 0."""
        df = pd.DataFrame(
            [
                {
                    "player_id": "P1",
                    "team": "KC",
                    "game_id": "2024_01_LV_KC",
                    "VEGAS_LINE": 0.0,
                    "FAVORITED": None,
                }
            ]
        )
        result: DataFrame = _derive_spread(df)
        assert result["game_spread"].iloc[0] == 0.0

    def test_nan_vegas_line_produces_nan_spread(self) -> None:
        """Missing VEGAS_LINE → NaN game_spread."""
        df = pd.DataFrame(
            [
                {
                    "player_id": "P1",
                    "team": "KC",
                    "game_id": "2024_01_LV_KC",
                    "VEGAS_LINE": np.nan,
                    "FAVORITED": None,
                }
            ]
        )
        result: DataFrame = _derive_spread(df)
        assert pd.isna(result["game_spread"].iloc[0])


class TestDeriveImpliedTotal:
    """Verify implied team total formula."""

    def test_favorite_implied_total(self) -> None:
        """KC favored by 3.5, total 47: implied = (47 - (-3.5)) / 2 = 25.25."""
        df: DataFrame = _make_joined()
        df = _derive_is_home(df)
        df = _derive_spread(df)
        result: DataFrame = _derive_implied_total(df)
        kc: Series = result[result["team"] == "KC"]
        assert kc["implied_team_total"].iloc[0] == pytest.approx(25.25)

    def test_underdog_implied_total(self) -> None:
        """LV underdog by 3.5, total 47: implied = (47 - 3.5) / 2 = 21.75."""
        df: DataFrame = _make_joined()
        df = _derive_is_home(df)
        df = _derive_spread(df)
        result: DataFrame = _derive_implied_total(df)
        lv: Series = result[result["team"] == "LV"]
        assert lv["implied_team_total"].iloc[0] == pytest.approx(21.75)

    def test_implied_totals_sum_to_over_under(self) -> None:
        """Both teams' implied totals should sum to over_under."""
        df: DataFrame = _make_joined()
        df = _derive_is_home(df)
        df = _derive_spread(df)
        result: DataFrame = _derive_implied_total(df)
        wk1: Series = result[result["week"] == 1]
        total = wk1["implied_team_total"].sum()
        assert total == pytest.approx(47.0)

    def test_over_under_column_present(self) -> None:
        df: DataFrame = _make_joined()
        df = _derive_is_home(df)
        df = _derive_spread(df)
        result: DataFrame = _derive_implied_total(df)
        assert "over_under" in result.columns


class TestDeriveDome:
    """Verify dome detection from ROOF values."""

    def test_dome_is_dome(self) -> None:
        df = pd.DataFrame({"ROOF": ["dome"]})
        result: DataFrame = _derive_dome(df)
        assert result["is_dome"].iloc[0] is np.bool_(True)

    def test_closed_is_dome(self) -> None:
        df = pd.DataFrame({"ROOF": ["closed"]})
        result: DataFrame = _derive_dome(df)
        assert result["is_dome"].iloc[0] is np.bool_(True)

    def test_outdoors_is_not_dome(self) -> None:
        df = pd.DataFrame({"ROOF": ["outdoors"]})
        result: DataFrame = _derive_dome(df)
        assert result["is_dome"].iloc[0] is np.bool_(False)

    def test_open_is_not_dome(self) -> None:
        df = pd.DataFrame({"ROOF": ["open"]})
        result: DataFrame = _derive_dome(df)
        assert result["is_dome"].iloc[0] is np.bool_(False)

    def test_nan_defaults_to_false(self) -> None:
        df = pd.DataFrame({"ROOF": [None]})
        result: DataFrame = _derive_dome(df)
        assert result["is_dome"].iloc[0] is np.bool_(False)


class TestDeriveRestDays:
    """Verify rest day computation."""

    def test_week1_is_nan(self) -> None:
        """First game in the dataset has no previous game → NaN rest."""
        df: DataFrame = _make_joined()
        df["GAME_DATE"] = ["2024-09-08"] * 2 + ["2024-09-15"] * 2 + ["2024-09-22"] * 2
        result: DataFrame = _derive_rest_days(df)
        wk1: Series = result[result["week"] == 1]
        assert wk1["rest_days"].isna().all()

    def test_weekly_rest_is_7(self) -> None:
        """Games one week apart → 7 days rest."""
        df: DataFrame = _make_joined()
        df["GAME_DATE"] = ["2024-09-08"] * 2 + ["2024-09-15"] * 2 + ["2024-09-22"] * 2
        result: DataFrame = _derive_rest_days(df)
        wk2_kc: Series = result[(result["week"] == 2) & (result["team"] == "KC")]
        assert wk2_kc["rest_days"].iloc[0] == 7

    def test_per_team_independent(self) -> None:
        """Each team's rest is computed from its own schedule."""
        # KC plays weeks 1,2; LV plays weeks 1,3 (bye week 2)
        rows: list[dict[str, int | str]] = [
            {
                "player_id": "P1",
                "team": "KC",
                "season": 2024,
                "week": 1,
                "game_id": "2024_01_LV_KC",
                "GAME_DATE": "2024-09-08",
            },
            {
                "player_id": "P2",
                "team": "LV",
                "season": 2024,
                "week": 1,
                "game_id": "2024_01_LV_KC",
                "GAME_DATE": "2024-09-08",
            },
            {
                "player_id": "P1",
                "team": "KC",
                "season": 2024,
                "week": 2,
                "game_id": "2024_02_KC_DEN",
                "GAME_DATE": "2024-09-15",
            },
            {
                "player_id": "P2",
                "team": "LV",
                "season": 2024,
                "week": 3,
                "game_id": "2024_03_LV_LAC",
                "GAME_DATE": "2024-09-22",
            },
        ]
        df = pd.DataFrame(rows)
        result: DataFrame = _derive_rest_days(df)
        kc_wk2: Series = result[(result["team"] == "KC") & (result["week"] == 2)]
        lv_wk3: Series = result[(result["team"] == "LV") & (result["week"] == 3)]
        assert kc_wk2["rest_days"].iloc[0] == 7
        assert lv_wk3["rest_days"].iloc[0] == 14  # bye week


class TestDropRawGameColumns:
    """Verify intermediate columns are cleaned up."""

    def test_raw_columns_removed(self) -> None:
        df: DataFrame = _make_joined()
        result: DataFrame = _drop_raw_game_columns(df)
        for col in ["VEGAS_LINE", "OVER_UNDER", "FAVORITED", "ROOF", "GAME_DATE"]:
            assert col not in result.columns


class TestBuildGameContextFeatures:
    """Verify the public entry point."""

    def test_raises_when_no_player_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="player game logs"):
            build_game_context_features(repo=tmp_path)

    def test_raises_when_no_games_data(self, tmp_path: Path) -> None:
        data_dir: Path = tmp_path / "data" / "cleaned"
        data_dir.mkdir(parents=True)
        _make_player_logs().to_parquet(data_dir / "player_game_logs.parquet")
        with pytest.raises(FileNotFoundError, match="games data"):
            build_game_context_features(repo=tmp_path)

    def test_output_has_all_context_columns(self, tmp_path: Path) -> None:
        data_dir: Path = tmp_path / "data" / "cleaned"
        data_dir.mkdir(parents=True)
        _make_player_logs().to_parquet(data_dir / "player_game_logs.parquet")
        _make_games().to_csv(data_dir / "NFL_wk_by_wk_cleaned.csv", index=False)

        result: DataFrame = build_game_context_features(repo=tmp_path)
        expected: set[str] = {
            "is_home",
            "game_spread",
            "over_under",
            "implied_team_total",
            "is_dome",
            "rest_days",
        }
        assert expected.issubset(set(result.columns))

    def test_no_raw_game_columns_in_output(self, tmp_path: Path) -> None:
        data_dir: Path = tmp_path / "data" / "cleaned"
        data_dir.mkdir(parents=True)
        _make_player_logs().to_parquet(data_dir / "player_game_logs.parquet")
        _make_games().to_csv(data_dir / "NFL_wk_by_wk_cleaned.csv", index=False)

        result: DataFrame = build_game_context_features(repo=tmp_path)
        for col in ["VEGAS_LINE", "OVER_UNDER", "FAVORITED", "ROOF", "GAME_DATE"]:
            assert col not in result.columns

    def test_no_row_duplication(self, tmp_path: Path) -> None:
        data_dir: Path = tmp_path / "data" / "cleaned"
        data_dir.mkdir(parents=True)
        player: DataFrame = _make_player_logs()
        player.to_parquet(data_dir / "player_game_logs.parquet")
        _make_games().to_csv(data_dir / "NFL_wk_by_wk_cleaned.csv", index=False)

        result: DataFrame = build_game_context_features(repo=tmp_path)
        assert len(result) == len(player)

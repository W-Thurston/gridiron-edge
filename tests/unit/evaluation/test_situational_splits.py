# tests/unit/evaluation/test_situational_splits.py

"""Unit tests for evaluation/situational_splits.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

LONG_TO_SHORT = {
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "Buffalo Bills": "BUF",
    "Miami Dolphins": "MIA",
    "Baltimore Ravens": "BAL",
    "Cleveland Browns": "CLE",
}


def _make_player_game_logs() -> pd.DataFrame:
    """5 player-game rows for one QB, across different contexts."""
    return pd.DataFrame(
        [
            # Game 1: KC vs LAC at home, KC favored, indoor, KC wins → Mahomes home
            {
                "player_id": "00-0033873",
                "player_name": "P.Mahomes",
                "team": "KC",
                "game_id": "2024_01_LAC_KC",
                "season": 2024,
                "week": 1,
                "passing_yards": 280,
            },
            # Game 2: KC at LAC (away), LAC favored, outdoors
            {
                "player_id": "00-0033873",
                "player_name": "P.Mahomes",
                "team": "KC",
                "game_id": "2024_02_KC_LAC",
                "season": 2024,
                "week": 2,
                "passing_yards": 220,
            },
            # Game 3: KC vs BUF at home, KC favored, indoor
            {
                "player_id": "00-0033873",
                "player_name": "P.Mahomes",
                "team": "KC",
                "game_id": "2024_03_BUF_KC",
                "season": 2024,
                "week": 3,
                "passing_yards": 310,
            },
            # Game 4: KC at BAL (away), BAL favored, outdoors
            {
                "player_id": "00-0033873",
                "player_name": "P.Mahomes",
                "team": "KC",
                "game_id": "2024_04_KC_BAL",
                "season": 2024,
                "week": 4,
                "passing_yards": 200,
            },
            # Game 5: KC vs MIA at home, KC favored, outdoors
            {
                "player_id": "00-0033873",
                "player_name": "P.Mahomes",
                "team": "KC",
                "game_id": "2024_05_MIA_KC",
                "season": 2024,
                "week": 5,
                "passing_yards": 290,
            },
        ]
    )


def _make_games() -> pd.DataFrame:
    """5 games matching the player game logs."""
    return pd.DataFrame(
        [
            # Game 1: KC vs LAC at KC. KC wins. Home. KC favored. Dome.
            {
                "GAME_ID": "2024_01_LAC_KC",
                "GAME_LOCATION": "H",
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Los Angeles Chargers",
                "ROOF": "dome",
                "FAVORITED": "Kansas City Chiefs",
            },
            # Game 2: KC at LAC. LAC wins. LAC home. KC away. LAC favored. Outdoors.
            {
                "GAME_ID": "2024_02_KC_LAC",
                "GAME_LOCATION": "H",
                "WINNER": "Los Angeles Chargers",
                "LOSER": "Kansas City Chiefs",
                "ROOF": "outdoors",
                "FAVORITED": "Los Angeles Chargers",
            },
            # Game 3: KC vs BUF at KC. KC wins. KC home. KC favored. Dome.
            {
                "GAME_ID": "2024_03_BUF_KC",
                "GAME_LOCATION": "H",
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Buffalo Bills",
                "ROOF": "dome",
                "FAVORITED": "Kansas City Chiefs",
            },
            # Game 4: KC at BAL. BAL wins. BAL home. KC away. BAL favored. Outdoors.
            {
                "GAME_ID": "2024_04_KC_BAL",
                "GAME_LOCATION": "H",
                "WINNER": "Baltimore Ravens",
                "LOSER": "Kansas City Chiefs",
                "ROOF": "outdoors",
                "FAVORITED": "Baltimore Ravens",
            },
            # Game 5: KC vs MIA at KC. KC wins. KC home. KC favored. Outdoors.
            {
                "GAME_ID": "2024_05_MIA_KC",
                "GAME_LOCATION": "H",
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Miami Dolphins",
                "ROOF": "outdoors",
                "FAVORITED": "Kansas City Chiefs",
            },
        ]
    )


class TestComputePlayerSituationalSplits:
    def test_empty_inputs_return_empty(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            pd.DataFrame(),
            pd.DataFrame(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )
        assert result.empty

    def test_unknown_stat_type_returns_empty(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "unknown_stat",
        )
        assert result.empty

    def test_season_cohort_computes_full_sample(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )

        season = result.loc[
            (result["player_id"] == "00-0033873") & (result["cohort"] == "season"),
            :,
        ]
        assert len(season) == 1
        assert season.iloc[0]["sample_size"] == 5
        # Mean of [280, 220, 310, 200, 290] = 260
        assert season.iloc[0]["mean_value"] == 260.0

    def test_home_cohort_partitions_correctly(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )

        home = result.loc[
            (result["player_id"] == "00-0033873") & (result["cohort"] == "home"),
            :,
        ]
        # Games 1, 3, 5 were home for KC (games 2, 4 were away).
        assert home.iloc[0]["sample_size"] == 3
        # Mean of [280, 310, 290] = 293.333...
        assert home.iloc[0]["mean_value"] == pytest.approx(293.333, abs=0.01)

    def test_away_cohort_partitions_correctly(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )

        away = result.loc[
            (result["player_id"] == "00-0033873") & (result["cohort"] == "away"),
            :,
        ]
        # Games 2, 4 were away for KC.
        assert away.iloc[0]["sample_size"] == 2
        # Mean of [220, 200] = 210
        assert away.iloc[0]["mean_value"] == 210.0

    def test_favored_cohort_partitions_correctly(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )

        favored = result.loc[
            (result["player_id"] == "00-0033873") & (result["cohort"] == "favored"),
            :,
        ]
        # KC favored in games 1, 3, 5.
        assert favored.iloc[0]["sample_size"] == 3
        assert favored.iloc[0]["mean_value"] == pytest.approx(293.333, abs=0.01)

    def test_underdog_cohort_partitions_correctly(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )

        underdog = result.loc[
            (result["player_id"] == "00-0033873") & (result["cohort"] == "underdog"),
            :,
        ]
        # KC underdog in games 2, 4.
        assert underdog.iloc[0]["sample_size"] == 2
        assert underdog.iloc[0]["mean_value"] == 210.0

    def test_indoor_cohort_partitions_correctly(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )

        indoor = result.loc[
            (result["player_id"] == "00-0033873") & (result["cohort"] == "indoor"),
            :,
        ]
        # Games 1, 3 were dome.
        assert indoor.iloc[0]["sample_size"] == 2
        # Mean of [280, 310] = 295
        assert indoor.iloc[0]["mean_value"] == 295.0

    def test_outdoor_cohort_partitions_correctly(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )

        outdoor = result.loc[
            (result["player_id"] == "00-0033873") & (result["cohort"] == "outdoor"),
            :,
        ]
        # Games 2, 4, 5 were outdoors.
        assert outdoor.iloc[0]["sample_size"] == 3
        # Mean of [220, 200, 290] = 236.667
        assert outdoor.iloc[0]["mean_value"] == pytest.approx(236.667, abs=0.01)

    def test_l4_cohort_returns_last_4_games(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )

        l4 = result.loc[
            (result["player_id"] == "00-0033873") & (result["cohort"] == "l4"),
            :,
        ]
        assert l4.iloc[0]["sample_size"] == 4
        # Last 4 games sorted by (season, week): games 2, 3, 4, 5
        # Mean of [220, 310, 200, 290] = 255
        assert l4.iloc[0]["mean_value"] == 255.0

    def test_stat_column_not_in_logs_returns_empty(self) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        # Player game logs missing passing_yards column
        logs = _make_player_game_logs()
        logs = logs.drop(columns=["passing_yards"])

        result = compute_player_situational_splits(
            logs,
            _make_games(),
            LONG_TO_SHORT,
            "qb_pass_yards",
        )
        assert result.empty

    def test_no_matching_game_ids_returns_empty(self) -> None:
        """Games CSV has no game_ids matching any player game log."""
        from gridiron_edge.evaluation.situational_splits import (
            compute_player_situational_splits,
        )

        games = _make_games().copy()
        games["GAME_ID"] = games["GAME_ID"] + "_MISMATCH"

        result = compute_player_situational_splits(
            _make_player_game_logs(),
            games,
            LONG_TO_SHORT,
            "qb_pass_yards",
        )
        assert result.empty


class TestWriteSituationalSplits:
    def test_writes_to_expected_path(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            write_situational_splits,
        )

        df = pd.DataFrame(
            [
                {"player_id": "KC1", "cohort": "season", "sample_size": 5, "mean_value": 260.0},
            ]
        )

        path = write_situational_splits(df, "qb_pass_yards", tmp_path)

        assert path.exists()
        assert path.name == "qb_pass_yards.parquet"
        assert path.parent == tmp_path / "data" / "output" / "props" / "situational_splits"

    def test_overwrites_same_stat_type(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            write_situational_splits,
        )

        df1 = pd.DataFrame(
            [
                {
                    "player_id": "KC1",
                    "cohort": "season",
                    "sample_size": 3,
                    "mean_value": 200.0,
                }
            ]
        )
        df2 = pd.DataFrame(
            [
                {
                    "player_id": "KC1",
                    "cohort": "season",
                    "sample_size": 5,
                    "mean_value": 260.0,
                }
            ]
        )

        write_situational_splits(df1, "qb_pass_yards", tmp_path)
        path = write_situational_splits(df2, "qb_pass_yards", tmp_path)

        loaded = pd.read_parquet(path)
        assert loaded.iloc[0]["mean_value"] == 260.0


class TestLoadSituationalSplits:
    def test_empty_when_missing(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            load_situational_splits,
        )

        result = load_situational_splits("qb_pass_yards", tmp_path)
        assert result.empty

    def test_loads_written_data(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.situational_splits import (
            load_situational_splits,
            write_situational_splits,
        )

        df = pd.DataFrame(
            [
                {"player_id": "KC1", "cohort": "season", "sample_size": 5, "mean_value": 260.0},
            ]
        )
        write_situational_splits(df, "qb_pass_yards", tmp_path)

        loaded = load_situational_splits("qb_pass_yards", tmp_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["mean_value"] == 260.0

# tests/unit/evaluation/test_opponent_allowed.py

"""Unit tests for evaluation/opponent_allowed.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _make_player_game_logs() -> pd.DataFrame:
    """Two QBs, three games. Simple scenario for testing.

    - Mahomes (KC) plays LAC in game 1: 300 pass yards
    - Mahomes (KC) plays BAL in game 2: 200 pass yards
    - Mahomes (KC) plays LAC in game 3: 250 pass yards
    - Herbert (LAC) plays KC in game 1: 240 pass yards
    - Herbert (LAC) plays BUF in game 2: 280 pass yards
    - Herbert (LAC) plays KC in game 3: 220 pass yards

    So LAC's defense allowed to QBs: 300 (game 1) + 250 (game 3) = avg 275.
    KC's defense allowed to QBs: 240 (game 1) + 220 (game 3) = avg 230.
    BAL's defense allowed to QBs: 200.
    BUF's defense allowed to QBs: 280.
    """
    return pd.DataFrame(
        [
            # Game 1: KC vs LAC.
            {
                "player_id": "P1",
                "player_name": "P.Mahomes",
                "team": "KC",
                "opponent_team": "LAC",
                "position": "QB",
                "season": 2024,
                "week": 1,
                "game_id": "2024_01_LAC_KC",
                "passing_yards": 300,
                "rushing_yards": 20,
                "receiving_yards": 0,
            },
            {
                "player_id": "P2",
                "player_name": "J.Herbert",
                "team": "LAC",
                "opponent_team": "KC",
                "position": "QB",
                "season": 2024,
                "week": 1,
                "game_id": "2024_01_LAC_KC",
                "passing_yards": 240,
                "rushing_yards": 15,
                "receiving_yards": 0,
            },
            # Game 2: KC vs BAL.
            {
                "player_id": "P1",
                "player_name": "P.Mahomes",
                "team": "KC",
                "opponent_team": "BAL",
                "position": "QB",
                "season": 2024,
                "week": 2,
                "game_id": "2024_02_BAL_KC",
                "passing_yards": 200,
                "rushing_yards": 10,
                "receiving_yards": 0,
            },
            {
                "player_id": "P2",
                "player_name": "J.Herbert",
                "team": "LAC",
                "opponent_team": "BUF",
                "position": "QB",
                "season": 2024,
                "week": 2,
                "game_id": "2024_02_BUF_LAC",
                "passing_yards": 280,
                "rushing_yards": 12,
                "receiving_yards": 0,
            },
            # Game 3: KC vs LAC.
            {
                "player_id": "P1",
                "player_name": "P.Mahomes",
                "team": "KC",
                "opponent_team": "LAC",
                "position": "QB",
                "season": 2024,
                "week": 3,
                "game_id": "2024_03_LAC_KC",
                "passing_yards": 250,
                "rushing_yards": 15,
                "receiving_yards": 0,
            },
            {
                "player_id": "P2",
                "player_name": "J.Herbert",
                "team": "LAC",
                "opponent_team": "KC",
                "position": "QB",
                "season": 2024,
                "week": 3,
                "game_id": "2024_03_LAC_KC",
                "passing_yards": 220,
                "rushing_yards": 10,
                "receiving_yards": 0,
            },
        ]
    )


class TestComputeOpponentAllowed:
    def test_empty_input_returns_empty(self) -> None:
        from gridiron_edge.evaluation.opponent_allowed import (
            compute_opponent_allowed,
        )

        result = compute_opponent_allowed(pd.DataFrame())
        assert result.empty

    def test_missing_columns_returns_empty(self) -> None:
        from gridiron_edge.evaluation.opponent_allowed import (
            compute_opponent_allowed,
        )

        df = pd.DataFrame([{"player_id": "P1", "team": "KC"}])
        result = compute_opponent_allowed(df)
        assert result.empty

    def test_computes_season_cohort(self) -> None:
        from gridiron_edge.evaluation.opponent_allowed import (
            compute_opponent_allowed,
        )

        result = compute_opponent_allowed(_make_player_game_logs())

        # LAC defense's season avg (games 1 and 3): (300 + 250) / 2 = 275
        lac_qb_season = result.loc[
            (result["opponent_team"] == "LAC")
            & (result["position"] == "QB")
            & (result["stat_type"] == "qb_pass_yards")
            & (result["cohort"] == "season"),
            :,
        ]
        assert len(lac_qb_season) == 1
        assert lac_qb_season.iloc[0]["avg_allowed"] == 275.0
        assert lac_qb_season.iloc[0]["sample_size"] == 2

        # KC defense's season avg (games 1 and 3): (240 + 220) / 2 = 230
        kc_qb_season = result.loc[
            (result["opponent_team"] == "KC")
            & (result["position"] == "QB")
            & (result["stat_type"] == "qb_pass_yards")
            & (result["cohort"] == "season"),
            :,
        ]
        assert kc_qb_season.iloc[0]["avg_allowed"] == 230.0

    def test_ranks_by_stinginess(self) -> None:
        """Rank 1 = stingiest = lowest avg_allowed."""
        from gridiron_edge.evaluation.opponent_allowed import (
            compute_opponent_allowed,
        )

        result = compute_opponent_allowed(_make_player_game_logs())

        # QB pass yards season rankings:
        # BAL: 200 (rank 1)
        # KC:  230 (rank 2)
        # LAC: 275 (rank 3)
        # BUF: 280 (rank 4)
        qb_season = result.loc[
            (result["stat_type"] == "qb_pass_yards") & (result["cohort"] == "season"),
            :,
        ]

        ranks = dict(
            zip(qb_season["opponent_team"], qb_season["rank_against_position"], strict=False)
        )
        assert ranks["BAL"] == 1
        assert ranks["KC"] == 2
        assert ranks["LAC"] == 3
        assert ranks["BUF"] == 4

    def test_l5_cohort_uses_last_5_games(self) -> None:
        from gridiron_edge.evaluation.opponent_allowed import (
            compute_opponent_allowed,
        )

        result = compute_opponent_allowed(_make_player_game_logs())

        # In our fixture, each defense plays only 2 or 1 games.
        # L5 falls back to what's available.
        lac_qb_l5 = result.loc[
            (result["opponent_team"] == "LAC")
            & (result["stat_type"] == "qb_pass_yards")
            & (result["cohort"] == "l5"),
            :,
        ]
        assert len(lac_qb_l5) == 1
        # LAC played 2 games; all are within L5 window
        assert lac_qb_l5.iloc[0]["avg_allowed"] == 275.0
        assert lac_qb_l5.iloc[0]["sample_size"] == 2

    def test_only_latest_season_included(self) -> None:
        """Older seasons are ignored."""
        from gridiron_edge.evaluation.opponent_allowed import (
            compute_opponent_allowed,
        )

        old = _make_player_game_logs().copy()
        old["season"] = 2023
        old["passing_yards"] = 999  # Different value

        combined = pd.concat(
            [old, _make_player_game_logs()],
            ignore_index=True,
        )

        result = compute_opponent_allowed(combined)

        # Result should be based on 2024 data only.
        lac_qb = result.loc[
            (result["opponent_team"] == "LAC")
            & (result["stat_type"] == "qb_pass_yards")
            & (result["cohort"] == "season"),
            "avg_allowed",
        ].iloc[0]
        assert lac_qb == 275.0  # From 2024 data, not the 999 old data

    def test_stat_type_position_mapping(self) -> None:
        """rb_rush_yards only aggregates RBs, not QBs."""
        from gridiron_edge.evaluation.opponent_allowed import (
            compute_opponent_allowed,
        )

        # Same fixture but ensure the QB rows don't contribute to rb_rush_yards
        result = compute_opponent_allowed(_make_player_game_logs())

        # No RBs in fixture, so no rb_rush_yards rows should exist
        rb_rows = result.loc[result["stat_type"] == "rb_rush_yards", :]
        assert rb_rows.empty


class TestWriteOpponentAllowed:
    def test_writes_to_expected_path(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.opponent_allowed import (
            write_opponent_allowed,
        )

        df = pd.DataFrame(
            [
                {
                    "opponent_team": "KC",
                    "position": "QB",
                    "stat_type": "qb_pass_yards",
                    "cohort": "season",
                    "avg_allowed": 230.0,
                    "sample_size": 5,
                    "rank_against_position": 2,
                }
            ]
        )

        path = write_opponent_allowed(df, tmp_path)
        assert path.exists()
        assert path.name == "opponent_allowed.parquet"
        assert path.parent == tmp_path / "data" / "output" / "props"


class TestLoadOpponentAllowed:
    def test_empty_when_missing(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.opponent_allowed import (
            load_opponent_allowed,
        )

        result = load_opponent_allowed(tmp_path)
        assert result.empty

    def test_loads_written_data(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.opponent_allowed import (
            load_opponent_allowed,
            write_opponent_allowed,
        )

        df = pd.DataFrame(
            [
                {
                    "opponent_team": "KC",
                    "position": "QB",
                    "stat_type": "qb_pass_yards",
                    "cohort": "season",
                    "avg_allowed": 230.0,
                    "sample_size": 5,
                    "rank_against_position": 2,
                }
            ]
        )
        write_opponent_allowed(df, tmp_path)

        loaded = load_opponent_allowed(tmp_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["avg_allowed"] == 230.0

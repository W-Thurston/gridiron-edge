# tests/unit/api/test_serializers_compare.py

"""Unit tests for compare serializers."""

from __future__ import annotations

import pandas as pd

from gridiron_edge.api.serializers.compare import serialize_compare_teams

LONG_TO_SHORT = {
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
}


def _make_elo() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1620.0,
            },
            {
                "NFL_TEAM": "Los Angeles Chargers",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1520.0,
            },
        ]
    )


def _make_games() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "GAME_ID",
            "WEEK_NUM",
            "GAME_DATE",
            "AWAY_TEAM",
            "HOME_TEAM",
            "AWAY_SCORE",
            "HOME_SCORE",
            "IS_NEUTRAL_SITE",
            "YEAR",
        ]
    )


def _make_percentiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team_abbr": "KC",
                "season": "2026-2027",
                "week": 1,
                "rating_pct": 0.75,
                "avg_wins_pct": 0.80,
                "make_playoffs_pct": 0.85,
                "win_sb_pct": 0.90,
            },
            {
                "team_abbr": "LAC",
                "season": "2026-2027",
                "week": 1,
                "rating_pct": 0.25,
                "avg_wins_pct": 0.20,
                "make_playoffs_pct": 0.15,
                "win_sb_pct": 0.10,
            },
        ]
    )


class TestCompareTeamsPercentiles:
    def test_populates_rating_percentiles(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=_make_percentiles(),
        )
        by_key = {row.key: row for row in result.stats}
        assert by_key["rating"].team_a_pct == 0.75
        assert by_key["rating"].team_b_pct == 0.25

    def test_populates_avg_wins_percentiles(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=_make_percentiles(),
        )
        by_key = {row.key: row for row in result.stats}
        assert by_key["avg_wins"].team_a_pct == 0.80
        assert by_key["avg_wins"].team_b_pct == 0.20

    def test_populates_make_playoffs_percentiles(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=_make_percentiles(),
        )
        by_key = {row.key: row for row in result.stats}
        assert by_key["make_playoffs"].team_a_pct == 0.85
        assert by_key["make_playoffs"].team_b_pct == 0.15

    def test_populates_win_sb_percentiles(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=_make_percentiles(),
        )
        by_key = {row.key: row for row in result.stats}
        assert by_key["win_sb"].team_a_pct == 0.90
        assert by_key["win_sb"].team_b_pct == 0.10

    def test_empty_percentiles_leaves_pct_fields_null(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=pd.DataFrame(),
        )
        by_key = {row.key: row for row in result.stats}
        assert by_key["rating"].team_a_pct is None
        assert by_key["rating"].team_b_pct is None
        assert by_key["avg_wins"].team_a_pct is None
        assert by_key["make_playoffs"].team_a_pct is None
        assert by_key["win_sb"].team_a_pct is None

    def test_non_rankable_rows_pct_always_null(self) -> None:
        """Rows like record don't have percentile fields populated."""
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=_make_percentiles(),
        )
        by_key = {row.key: row for row in result.stats}
        assert by_key["record"].team_a_pct is None
        assert by_key["off_rating"].team_a_pct is None

    def test_percentile_ranks_row_removed(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=_make_percentiles(),
        )
        keys = {row.key for row in result.stats}
        assert "percentile_ranks" not in keys

    def test_percentile_ranks_meta_pending_removed(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=pd.DataFrame(),
        )
        assert result.response_meta is not None
        assert "percentile_ranks" not in result.response_meta.field_status


class TestSerializeComparePlayerOpponentAllowed:
    def _valid_row(self) -> dict:
        return {
            "predicted_at": "2024-08-01T00:00:00+00:00",
            "is_backfilled": True,
            "season": 2024,
            "week": 1,
            "game_id": "2024_01_LAC_KC",
            "player_id": "P1",
            "player_name": "P.Mahomes",
            "position": "QB",
            "team": "KC",
            "stat_type": "qb_pass_yards",
            "model_name": "qb_pass_yards",
            "model_type": "elasticnet",
            "predicted_mean": 275.0,
            "predicted_std": 45.0,
            "lo_90": 200.0,
            "hi_90": 350.0,
        }

    def test_populates_defense_rows_when_data_present(self) -> None:
        from gridiron_edge.api.serializers.compare import serialize_compare_player

        opponent_allowed = {
            "season": {"avg_allowed": 275.0, "sample_size": 5, "rank_against_position": 3},
            "l4": {"avg_allowed": 265.0, "sample_size": 5, "rank_against_position": 2},
        }

        result = serialize_compare_player(
            self._valid_row(),
            opponent_allowed=opponent_allowed,
        )

        by_key = {row.key: row for row in result.stats}
        assert by_key["avg_allowed"].defense_value == 275.0
        assert by_key["rank_against_position"].defense_value == 3
        assert by_key["last_4_games_avg"].defense_value == 265.0

    def test_removes_blocker_when_data_present(self) -> None:
        from gridiron_edge.api.serializers.compare import serialize_compare_player

        opponent_allowed = {
            "season": {"avg_allowed": 275.0, "sample_size": 5, "rank_against_position": 3},
        }

        result = serialize_compare_player(
            self._valid_row(),
            opponent_allowed=opponent_allowed,
        )

        fs = result.response_meta.field_status
        # 3 defense rows no longer blocked.
        assert "avg_allowed" not in fs
        assert "rank_against_position" not in fs
        assert "last_4_games_avg" not in fs
        # red_zone_rate_allowed still blocked.
        assert "red_zone_rate_allowed" in fs

    def test_none_leaves_blockers_intact(self) -> None:
        from gridiron_edge.api.serializers.compare import serialize_compare_player

        result = serialize_compare_player(self._valid_row(), opponent_allowed=None)

        fs = result.response_meta.field_status
        assert "avg_allowed" in fs
        assert "rank_against_position" in fs
        assert "last_4_games_avg" in fs
        assert "red_zone_rate_allowed" in fs

    def test_empty_dict_leaves_blockers_intact(self) -> None:
        from gridiron_edge.api.serializers.compare import serialize_compare_player

        result = serialize_compare_player(self._valid_row(), opponent_allowed={})

        fs = result.response_meta.field_status
        assert "avg_allowed" in fs
        assert "rank_against_position" in fs
        assert "last_4_games_avg" in fs


class TestCompareTeamsCohortSplits:
    def test_populates_cohort_splits(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=pd.DataFrame(),
            cohort_splits={
                "KC": {"season": {"off_epa_per_play": 0.15, "sample_size": 4}},
                "LAC": {"season": {"off_epa_per_play": 0.10, "sample_size": 4}},
            },
        )
        assert result.cohort_splits is not None
        assert result.cohort_splits["KC"]["season"]["off_epa_per_play"] == 0.15

    def test_none_leaves_pending_marker(self) -> None:
        result = serialize_compare_teams(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            team_a_short="KC",
            team_b_short="LAC",
            season="2026-2027",
            as_of_week=1,
            percentiles=pd.DataFrame(),
            cohort_splits=None,
        )
        assert result.cohort_splits is None
        assert "cohort_splits" in result.response_meta.field_status

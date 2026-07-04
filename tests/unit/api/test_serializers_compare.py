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
            "WINNER",
            "LOSER",
            "GAME_LOCATION",
            "PTS_WINNER",
            "PTS_LOSER",
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

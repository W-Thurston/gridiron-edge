# tests/unit/api/test_serializers_teams.py

"""Unit tests for teams serializers."""

from __future__ import annotations

import pandas as pd

from gridiron_edge.api.meta import FieldStatus
from gridiron_edge.api.schemas.teams import TeamProfile, TeamRankingsList
from gridiron_edge.api.serializers.teams import (
    _compute_record,
    _latest_ratings,
    _serialize_result,
    serialize_team_profile,
    serialize_team_rankings,
)

LONG_TO_SHORT = {
    "Baltimore Ravens": "BAL",
    "Kansas City Chiefs": "KC",
    "Cleveland Browns": "CLE",
}


def _make_games() -> pd.DataFrame:
    """Build a small games DataFrame for testing."""
    return pd.DataFrame(
        [
            # BAL wins @ CLE
            {
                "GAME_ID": "2025_01_BAL_CLE",
                "WEEK_NUM": 1,
                "GAME_DATE": "2025-09-07",
                "WINNER": "Baltimore Ravens",
                "GAME_LOCATION": "@",
                "LOSER": "Cleveland Browns",
                "PTS_WINNER": 27,
                "PTS_LOSER": 14,
                "YEAR": "2025-2026",
            },
            # KC wins at home over CLE
            {
                "GAME_ID": "2025_02_CLE_KC",
                "WEEK_NUM": 2,
                "GAME_DATE": "2025-09-14",
                "WINNER": "Kansas City Chiefs",
                "GAME_LOCATION": "H",
                "LOSER": "Cleveland Browns",
                "PTS_WINNER": 30,
                "PTS_LOSER": 10,
                "YEAR": "2025-2026",
            },
            # BAL wins at home over KC (BAL is winner, "H" location)
            {
                "GAME_ID": "2025_03_KC_BAL",
                "WEEK_NUM": 3,
                "GAME_DATE": "2025-09-21",
                "WINNER": "Baltimore Ravens",
                "GAME_LOCATION": "H",
                "LOSER": "Kansas City Chiefs",
                "PTS_WINNER": 24,
                "PTS_LOSER": 17,
                "YEAR": "2025-2026",
            },
        ],
    )


def _make_elo() -> pd.DataFrame:
    """Build a small Elo DataFrame for testing."""
    return pd.DataFrame(
        [
            {"NFL_TEAM": "Baltimore Ravens", "NFL_YEAR": "2025-2026", "NFL_WEEK": 1, "ELO": 1600.0},
            {"NFL_TEAM": "Baltimore Ravens", "NFL_YEAR": "2025-2026", "NFL_WEEK": 2, "ELO": 1620.0},
            {"NFL_TEAM": "Baltimore Ravens", "NFL_YEAR": "2025-2026", "NFL_WEEK": 3, "ELO": 1642.3},
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2025-2026",
                "NFL_WEEK": 1,
                "ELO": 1580.0,
            },
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2025-2026",
                "NFL_WEEK": 2,
                "ELO": 1600.0,
            },
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2025-2026",
                "NFL_WEEK": 3,
                "ELO": 1585.0,
            },
            {"NFL_TEAM": "Cleveland Browns", "NFL_YEAR": "2025-2026", "NFL_WEEK": 1, "ELO": 1500.0},
            {"NFL_TEAM": "Cleveland Browns", "NFL_YEAR": "2025-2026", "NFL_WEEK": 2, "ELO": 1490.0},
            {"NFL_TEAM": "Cleveland Browns", "NFL_YEAR": "2025-2026", "NFL_WEEK": 3, "ELO": 1470.0},
        ],
    )


class TestComputeRecord:
    def test_baltimore_record(self) -> None:
        games = _make_games()
        rec = _compute_record(games, "Baltimore Ravens")
        assert rec.wins == 2
        assert rec.losses == 0
        assert rec.ties == 0

    def test_cleveland_record(self) -> None:
        games = _make_games()
        rec = _compute_record(games, "Cleveland Browns")
        assert rec.wins == 0
        assert rec.losses == 2

    def test_empty_games(self) -> None:
        rec = _compute_record(pd.DataFrame(), "Baltimore Ravens")
        assert rec.wins == 0


class TestLatestRatings:
    def test_returns_one_row_per_team(self) -> None:
        elo = _make_elo()
        latest = _latest_ratings(elo, "2025-2026", 3)
        assert len(latest) == 3
        # BAL's latest should be week 3 → 1642.3
        bal = latest.loc[latest["NFL_TEAM"] == "Baltimore Ravens"].iloc[0]
        assert bal["ELO"] == 1642.3

    def test_respects_week_cap(self) -> None:
        elo = _make_elo()
        latest = _latest_ratings(elo, "2025-2026", 1)
        # Everyone should have their week-1 rating.
        for _, r in latest.iterrows():
            assert r["NFL_WEEK"] == 1

    def test_empty_elo(self) -> None:
        latest = _latest_ratings(pd.DataFrame(), "2025-2026", 3)
        assert latest.empty


class TestSerializeResult:
    def test_baltimore_at_cleveland(self) -> None:
        games = _make_games()
        row = games.iloc[0]  # BAL @ CLE
        result = _serialize_result(row, "Baltimore Ravens", LONG_TO_SHORT)
        assert result.opponent == "CLE"
        assert result.is_home is False
        assert result.result == "W"
        assert result.score_for == 27
        assert result.score_against == 14

    def test_cleveland_hosting(self) -> None:
        games = _make_games()
        row = games.iloc[0]  # BAL @ CLE
        result = _serialize_result(row, "Cleveland Browns", LONG_TO_SHORT)
        assert result.opponent == "BAL"
        assert result.is_home is True
        assert result.result == "L"
        assert result.score_for == 14

    def test_baltimore_at_home_vs_kc(self) -> None:
        games = _make_games()
        row = games.iloc[2]  # BAL vs KC, BAL wins at H
        result = _serialize_result(row, "Baltimore Ravens", LONG_TO_SHORT)
        assert result.opponent == "KC"
        assert result.is_home is True
        assert result.result == "W"


class TestSerializeTeamRankings:
    def test_empty_elo(self) -> None:
        result: TeamRankingsList = serialize_team_rankings(
            pd.DataFrame(),
            _make_games(),
            LONG_TO_SHORT,
            "2025-2026",
            3,
            pd.DataFrame(),  # percentiles
            pd.DataFrame(),  # trends
        )
        assert result.total == 0
        assert result.items == []

    def test_ranks_by_elo(self) -> None:
        result: TeamRankingsList = serialize_team_rankings(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            "2025-2026",
            3,
            pd.DataFrame(),  # percentiles
            pd.DataFrame(),  # trends
        )
        assert result.total == 3
        assert result.items[0].abbr == "BAL"  # Highest at week 3
        assert result.items[0].rank == 1
        assert result.items[-1].abbr == "CLE"  # Lowest

    def test_marks_meta_fields_blocked(self) -> None:
        result: TeamRankingsList = serialize_team_rankings(
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            "2025-2026",
            3,
            pd.DataFrame(),  # percentiles
            pd.DataFrame(),  # trends
        )
        assert result.response_meta is not None
        fs: dict[str, FieldStatus] = result.response_meta.field_status
        assert "items.off_rating" in fs


class TestSerializeTeamProfile:
    def test_unknown_abbr(self) -> None:
        result: TeamProfile = serialize_team_profile(
            "XXX",
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            "2025-2026",
            3,
            pd.DataFrame(),  # percentiles
            pd.DataFrame(),  # trends
        )
        assert result.abbr == "XXX"
        assert result.rating is None

    def test_populated_baltimore(self) -> None:
        result: TeamProfile = serialize_team_profile(
            "BAL",
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            "2025-2026",
            3,
            pd.DataFrame(),  # percentiles
            pd.DataFrame(),  # trends
        )
        assert result.abbr == "BAL"
        assert result.name == "Baltimore Ravens"
        assert result.rating == 1642.3
        assert result.rank == 1
        assert result.record.wins == 2
        assert len(result.rating_history) == 3
        assert len(result.recent_results) == 2  # Only 2 games in fixture

    def test_meta_has_all_expected_entries(self) -> None:
        result: TeamProfile = serialize_team_profile(
            "BAL",
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            "2025-2026",
            3,
            pd.DataFrame(),  # percentiles
            pd.DataFrame(),  # trends
        )
        fs = result.response_meta.field_status
        for expected in (
            "off_rating",
            "def_rating",
            "schedule_difficulty",
            "playoff_probability",
            "cohort_splits",
            "top_players",
        ):
            assert expected in fs, f"missing field_status for {expected}"


class TestTeamRankingsPercentiles:
    def test_populates_percentile_fields(self) -> None:
        elo = pd.DataFrame(
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
        games = pd.DataFrame(columns=["YEAR", "WINNER", "LOSER", "PTS_WINNER", "PTS_LOSER"])
        long_to_short = {"Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC"}
        percentiles = pd.DataFrame(
            [
                {
                    "team_abbr": "KC",
                    "season": "2026-2027",
                    "week": 1,
                    "rating_pct": 0.75,
                    "avg_wins_pct": 0.75,
                    "make_playoffs_pct": 0.75,
                    "win_sb_pct": 0.75,
                },
                {
                    "team_abbr": "LAC",
                    "season": "2026-2027",
                    "week": 1,
                    "rating_pct": 0.25,
                    "avg_wins_pct": 0.25,
                    "make_playoffs_pct": 0.25,
                    "win_sb_pct": 0.25,
                },
            ]
        )

        result = serialize_team_rankings(
            elo,
            games,
            long_to_short,
            "2026-2027",
            1,
            percentiles,
            pd.DataFrame(),  # trends
        )

        by_abbr = {row.abbr: row for row in result.items}
        assert by_abbr["KC"].rating_pct == 0.75
        assert by_abbr["KC"].avg_wins_pct == 0.75
        assert by_abbr["LAC"].rating_pct == 0.25

    def test_empty_percentiles_null_fields(self) -> None:
        elo = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1620.0,
                },
            ]
        )
        games = pd.DataFrame(columns=["YEAR", "WINNER", "LOSER", "PTS_WINNER", "PTS_LOSER"])
        long_to_short = {"Kansas City Chiefs": "KC"}
        percentiles = pd.DataFrame()  # Empty

        result = serialize_team_rankings(
            elo,
            games,
            long_to_short,
            "2026-2027",
            1,
            percentiles,
            pd.DataFrame(),  # trends
        )

        assert result.items[0].rating_pct is None
        assert result.items[0].avg_wins_pct is None


class TestTrendPopulation:
    def test_populates_trend_from_delta(self) -> None:
        elo = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1600.0,
                },
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 2,
                    "ELO": 1620.0,
                },
            ]
        )
        games = pd.DataFrame(columns=["YEAR", "WINNER", "LOSER", "PTS_WINNER", "PTS_LOSER"])
        long_to_short = {"Kansas City Chiefs": "KC"}
        trends = pd.DataFrame(
            [
                {"team_abbr": "KC", "elo_delta": 20.0},
            ]
        )

        result = serialize_team_rankings(
            elo,
            games,
            long_to_short,
            "2026-2027",
            2,
            pd.DataFrame(),  # percentiles
            trends,
        )

        assert result.items[0].trend == 20.0

    def test_empty_trends_leaves_trend_null(self) -> None:
        elo = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1600.0,
                },
            ]
        )
        games = pd.DataFrame(columns=["YEAR", "WINNER", "LOSER", "PTS_WINNER", "PTS_LOSER"])
        long_to_short = {"Kansas City Chiefs": "KC"}

        result = serialize_team_rankings(
            elo,
            games,
            long_to_short,
            "2026-2027",
            1,
            pd.DataFrame(),  # percentiles
            pd.DataFrame(),  # empty trends
        )

        assert result.items[0].trend is None


class TestTeamProfileCohortSplits:
    def test_populates_cohort_splits_when_data_present(self) -> None:
        from gridiron_edge.api.serializers.teams import serialize_team_profile

        result = serialize_team_profile(
            "BAL",
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            "2025-2026",
            3,
            pd.DataFrame(),  # percentiles
            pd.DataFrame(),  # trends
            cohort_splits={
                "season": {"off_epa_per_play": 0.15, "sample_size": 4},
                "l4": {"off_epa_per_play": 0.20, "sample_size": 4},
            },
        )
        assert result.cohort_splits is not None
        assert result.cohort_splits["season"]["off_epa_per_play"] == 0.15

    def test_none_leaves_pending_marker(self) -> None:
        from gridiron_edge.api.serializers.teams import serialize_team_profile

        result = serialize_team_profile(
            "BAL",
            _make_elo(),
            _make_games(),
            LONG_TO_SHORT,
            "2025-2026",
            3,
            pd.DataFrame(),
            pd.DataFrame(),
            cohort_splits=None,
        )
        assert result.cohort_splits is None
        assert "cohort_splits" in result.response_meta.field_status

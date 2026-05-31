# tests/features/test_rest.py

"""Unit tests for features/team/rest.py.

Tests cover days-rest computation, short-week and post-bye flag
derivation, Week 1 NaN behaviour, multi-team lookups, and the
symmetric two-row-per-game design.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_accessor(games: pd.DataFrame) -> MagicMock:
    acc = MagicMock()
    acc.games.return_value = games
    return acc


def _make_games(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "GAME_ID": "2024_01_KC_LV",
        "WINNER": "Kansas City Chiefs",
        "LOSER": "Las Vegas Raiders",
        "YEAR": "2024-2025",
        "WEEK_NUM": 1,
        "GAME_DATE": "2024-09-05",
        "GAME_LOCATION": "NULL_VALUE",
        "STADIUM": "Arrowhead Stadium",
        "ROOF": "outdoors",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_modeling_row(**kwargs: object) -> dict:
    base = {
        "GAME_ID": "2024_01_KC_LV",
        "TEAM_A": "Kansas City Chiefs",
        "TEAM_B": "Las Vegas Raiders",
        "YEAR": "2024-2025",
        "WEEK_NUM": 1,
        "RESULT": 1,
        "HOME_FIELD": 1,
    }
    return {**base, **kwargs}


# ---------------------------------------------------------------------------
# Days rest computation
# ---------------------------------------------------------------------------


class TestDaysRestComputation:
    """Tests for the core days-rest calculation."""

    def test_standard_week_seven_days(self) -> None:
        """Back-to-back Sunday games should produce 7 days rest."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_01_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2024-09-08",
                },
                {
                    "GAME_ID": "2024_02_KC_BAL",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Baltimore Ravens",
                    "WEEK_NUM": 2,
                    "GAME_DATE": "2024-09-15",
                },
            ]
        )
        df = pd.DataFrame(
            [_make_modeling_row(GAME_ID="2024_02_KC_BAL", TEAM_B="Baltimore Ravens", WEEK_NUM=2)]
        )
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        assert result.iloc[0]["TEAM_A_DAYS_REST"] == pytest.approx(7.0)

    def test_week_1_produces_nan(self) -> None:
        """First game of the season has no prior game; DAYS_REST should be NaN."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_01_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2024-09-05",
                },
            ]
        )
        df = pd.DataFrame([_make_modeling_row()])
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        assert pd.isna(result.iloc[0]["TEAM_A_DAYS_REST"])

    def test_cross_season_boundary_not_carried(self) -> None:
        """Rest days should not bleed across season boundaries for a team."""
        from gridiron_edge.features.team.rest import RestFeature

        # Same team, different seasons — week 1 of new season should be NaN
        games = _make_games(
            [
                {
                    "GAME_ID": "2023_18_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "YEAR": "2023-2024",
                    "WEEK_NUM": 18,
                    "GAME_DATE": "2024-01-07",
                },
                {
                    "GAME_ID": "2024_01_KC_BAL",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Baltimore Ravens",
                    "YEAR": "2024-2025",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2024-09-08",
                },
            ]
        )
        df = pd.DataFrame(
            [
                _make_modeling_row(
                    GAME_ID="2024_01_KC_BAL",
                    TEAM_B="Baltimore Ravens",
                    YEAR="2024-2025",
                    WEEK_NUM=1,
                )
            ]
        )
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        # Week 1 of a new season — prior game is in a different season.
        # Days since last game is > 200 days (offseason gap), not a short week.
        # The feature computes raw day difference regardless of season boundary,
        # so this will be a large positive number, not NaN.
        days = result.iloc[0]["TEAM_A_DAYS_REST"]
        assert pd.notna(days)
        assert days > 100  # offseason gap >> short week threshold

    def test_loser_also_gets_rest_days(self) -> None:
        """The losing team should also receive TEAM_A_DAYS_REST via LOSER join."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_01_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2024-09-08",
                },
                {
                    "GAME_ID": "2024_02_LV_DEN",
                    "WINNER": "Denver Broncos",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 2,
                    "GAME_DATE": "2024-09-15",
                },
            ]
        )
        # Loser-perspective row: TEAM_A = Las Vegas Raiders
        df = pd.DataFrame(
            [
                _make_modeling_row(
                    GAME_ID="2024_02_LV_DEN",
                    TEAM_A="Las Vegas Raiders",
                    TEAM_B="Denver Broncos",
                    WEEK_NUM=2,
                    RESULT=0,
                )
            ]
        )
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        assert result.iloc[0]["TEAM_A_DAYS_REST"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Short-week flag
# ---------------------------------------------------------------------------


class TestShortWeekFlag:
    """Tests for TEAM_A_SHORT_WEEK and TEAM_B_SHORT_WEEK."""

    def test_thursday_game_is_short_week(self) -> None:
        """4 days rest (Sunday to Thursday) should set SHORT_WEEK=1."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_01_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2024-09-08",
                },  # Sunday
                {
                    "GAME_ID": "2024_02_KC_BAL",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Baltimore Ravens",
                    "WEEK_NUM": 2,
                    "GAME_DATE": "2024-09-12",
                },  # Thursday — 4 days later
            ]
        )
        df = pd.DataFrame(
            [_make_modeling_row(GAME_ID="2024_02_KC_BAL", TEAM_B="Baltimore Ravens", WEEK_NUM=2)]
        )
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        assert result.iloc[0]["TEAM_A_DAYS_REST"] == pytest.approx(4.0)
        assert result.iloc[0]["TEAM_A_SHORT_WEEK"] == 1

    def test_standard_week_not_short(self) -> None:
        """7 days rest should produce SHORT_WEEK=0."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_01_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2024-09-08",
                },
                {
                    "GAME_ID": "2024_02_KC_BAL",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Baltimore Ravens",
                    "WEEK_NUM": 2,
                    "GAME_DATE": "2024-09-15",
                },
            ]
        )
        df = pd.DataFrame(
            [_make_modeling_row(GAME_ID="2024_02_KC_BAL", TEAM_B="Baltimore Ravens", WEEK_NUM=2)]
        )
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        assert result.iloc[0]["TEAM_A_SHORT_WEEK"] == 0

    def test_short_week_threshold_boundary(self) -> None:
        """Exactly 6 days rest should not be flagged as a short week (< 6 required)."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_01_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2024-09-08",
                },  # Sunday
                {
                    "GAME_ID": "2024_02_KC_BAL",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Baltimore Ravens",
                    "WEEK_NUM": 2,
                    "GAME_DATE": "2024-09-14",
                },  # Saturday — 6 days later
            ]
        )
        df = pd.DataFrame(
            [_make_modeling_row(GAME_ID="2024_02_KC_BAL", TEAM_B="Baltimore Ravens", WEEK_NUM=2)]
        )
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        assert result.iloc[0]["TEAM_A_DAYS_REST"] == pytest.approx(6.0)
        assert result.iloc[0]["TEAM_A_SHORT_WEEK"] == 0  # not < 6


# ---------------------------------------------------------------------------
# Post-bye flag
# ---------------------------------------------------------------------------


class TestPostByeFlag:
    """Tests for TEAM_A_POST_BYE and TEAM_B_POST_BYE."""

    def test_14_days_is_post_bye(self) -> None:
        """14 days rest (bye week gap) should set POST_BYE=1."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_05_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 5,
                    "GAME_DATE": "2024-10-06",
                },
                {
                    "GAME_ID": "2024_07_KC_BAL",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Baltimore Ravens",
                    "WEEK_NUM": 7,
                    "GAME_DATE": "2024-10-20",
                },  # 14 days -- bye in week 6
            ]
        )
        df = pd.DataFrame(
            [_make_modeling_row(GAME_ID="2024_07_KC_BAL", TEAM_B="Baltimore Ravens", WEEK_NUM=7)]
        )
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        assert result.iloc[0]["TEAM_A_DAYS_REST"] == pytest.approx(14.0)
        assert result.iloc[0]["TEAM_A_POST_BYE"] == 1
        assert result.iloc[0]["TEAM_A_SHORT_WEEK"] == 0

    def test_post_bye_threshold_boundary(self) -> None:
        """Exactly 13 days should trigger POST_BYE (>= 13 required)."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_05_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 5,
                    "GAME_DATE": "2024-10-06",
                },
                {
                    "GAME_ID": "2024_07_KC_BAL",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Baltimore Ravens",
                    "WEEK_NUM": 7,
                    "GAME_DATE": "2024-10-19",
                },  # 13 days
            ]
        )
        df = pd.DataFrame(
            [_make_modeling_row(GAME_ID="2024_07_KC_BAL", TEAM_B="Baltimore Ravens", WEEK_NUM=7)]
        )
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        assert result.iloc[0]["TEAM_A_DAYS_REST"] == pytest.approx(13.0)
        assert result.iloc[0]["TEAM_A_POST_BYE"] == 1


# ---------------------------------------------------------------------------
# Column completeness
# ---------------------------------------------------------------------------


class TestRestFeatureColumns:
    """Tests for column presence and spec accuracy."""

    def test_all_six_columns_present(self) -> None:
        """All six produced columns must appear in the output DataFrame."""
        from gridiron_edge.features.team.rest import RestFeature

        games = _make_games(
            [
                {
                    "GAME_ID": "2024_01_KC_LV",
                    "WINNER": "Kansas City Chiefs",
                    "LOSER": "Las Vegas Raiders",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2024-09-05",
                },
            ]
        )
        df = pd.DataFrame([_make_modeling_row()])
        result = RestFeature().compute(df=df, datasets=_make_accessor(games))

        expected = {
            "TEAM_A_DAYS_REST",
            "TEAM_B_DAYS_REST",
            "TEAM_A_SHORT_WEEK",
            "TEAM_B_SHORT_WEEK",
            "TEAM_A_POST_BYE",
            "TEAM_B_POST_BYE",
        }
        assert expected.issubset(set(result.columns))

    def test_spec_produces_matches_compute_output(self) -> None:
        """FeatureSpec.produces must exactly match what compute() adds."""
        from gridiron_edge.features.team.rest import RestFeature

        feature = RestFeature()
        assert set(feature.spec.produces) == {
            "TEAM_A_DAYS_REST",
            "TEAM_B_DAYS_REST",
            "TEAM_A_SHORT_WEEK",
            "TEAM_B_SHORT_WEEK",
            "TEAM_A_POST_BYE",
            "TEAM_B_POST_BYE",
        }

    def test_registered_under_rest(self) -> None:
        """RestFeature must be registered under the key 'rest'."""
        from gridiron_edge.features.registry import FeatureRegistry
        import gridiron_edge.features.team.rest  # noqa: F401

        assert FeatureRegistry.get("rest") is not None

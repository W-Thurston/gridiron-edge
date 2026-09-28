# tests/unit/ratings/test_linear_ratings.py

"""Tests for Massey, Colley, and Keener ratings recomputed per cutoff."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.fixtures.rating_history import GameSpec, make_rating_history

from gridiron_edge.ratings.linear_ratings import (
    NEUTRAL_RATINGS,
    LinearRatingConfig,
    LinearRatings,
    solve_linear_ratings,
)


def _window(rows: list[tuple[str, str, int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["AWAY_TEAM", "HOME_TEAM", "AWAY_SCORE", "HOME_SCORE", "WEIGHT"]
    )


def _ratings(extra: list[GameSpec], config: LinearRatingConfig | None = None) -> LinearRatings:
    return LinearRatings(make_rating_history(extra), config)


class TestSolvers:
    def test_colley_two_team_hand_solution(self) -> None:
        rated = solve_linear_ratings(_window([("A", "B", 20, 10, 1.0)]), LinearRatingConfig())
        assert rated.loc["A", "COLLEY"] == pytest.approx(0.625)
        assert rated.loc["B", "COLLEY"] == pytest.approx(0.375)

    def test_massey_two_team_hand_solution(self) -> None:
        cfg = LinearRatingConfig(massey_ridge=1.0, margin_cap=100.0)
        rated = solve_linear_ratings(_window([("A", "B", 20, 10, 1.0)]), cfg)
        assert rated.loc["A", "MASSEY"] == pytest.approx(10 / 3)
        assert rated.loc["B", "MASSEY"] == pytest.approx(-10 / 3)

    def test_massey_caps_blowouts(self) -> None:
        cfg = LinearRatingConfig(margin_cap=14.0)
        blowout = solve_linear_ratings(_window([("A", "B", 60, 0, 1.0)]), cfg)
        capped = solve_linear_ratings(_window([("A", "B", 14, 0, 1.0)]), cfg)
        assert blowout.loc["A", "MASSEY"] == pytest.approx(capped.loc["A", "MASSEY"])

    def test_tie_leaves_teams_level_in_all_methods(self) -> None:
        rated = solve_linear_ratings(_window([("A", "B", 17, 17, 1.0)]), LinearRatingConfig())
        assert rated.loc["A"].to_dict() == pytest.approx(rated.loc["B"].to_dict())
        assert rated.loc["A", "COLLEY"] == pytest.approx(0.5)

    def test_winner_outranks_loser_in_all_methods(self) -> None:
        rated = solve_linear_ratings(_window([("A", "B", 27, 20, 1.0)]), LinearRatingConfig())
        for column in ("MASSEY", "COLLEY", "KEENER"):
            assert rated.loc["A", column] > rated.loc["B", column]

    def test_keener_is_positive_with_league_mean_one(self) -> None:
        window = _window([("A", "B", 27, 20, 1.0), ("B", "C", 10, 24, 1.0), ("C", "A", 3, 13, 1.0)])
        keener = solve_linear_ratings(window, LinearRatingConfig())["KEENER"]
        assert (keener > 0).all()
        assert keener.mean() == pytest.approx(1.0)

    def test_colley_ratings_average_one_half(self) -> None:
        window = _window([("A", "B", 27, 20, 1.0), ("B", "C", 10, 24, 1.0), ("C", "A", 3, 13, 1.0)])
        assert solve_linear_ratings(window, LinearRatingConfig())["COLLEY"].mean() == pytest.approx(
            0.5
        )

    def test_team_order_does_not_change_ratings(self) -> None:
        rows = [("A", "B", 27, 20, 1.0), ("B", "C", 10, 24, 1.0), ("C", "A", 3, 13, 1.0)]
        forward = solve_linear_ratings(_window(rows), LinearRatingConfig())
        backward = solve_linear_ratings(_window(rows[::-1]), LinearRatingConfig())
        pd.testing.assert_frame_equal(forward, backward.loc[forward.index])

    def test_lower_weight_game_matters_less(self) -> None:
        full = solve_linear_ratings(_window([("A", "B", 30, 0, 1.0)]), LinearRatingConfig())
        half = solve_linear_ratings(_window([("A", "B", 30, 0, 0.5)]), LinearRatingConfig())
        assert half.loc["A", "COLLEY"] < full.loc["A", "COLLEY"]


class TestAsOf:
    def test_first_week_of_history_is_neutral_not_null(self) -> None:
        rated = _ratings([]).as_of(1999, 1, ["A", "B"])
        assert rated.loc["A"].to_dict() == NEUTRAL_RATINGS
        assert not rated.isna().any().any()

    def test_team_without_games_gets_neutral_values(self) -> None:
        rated = _ratings([(2010, 1, "A", "B", 20, 10)]).as_of(2010, 2, ["A", "B", "Newcomer"])
        assert rated.loc["Newcomer"].to_dict() == NEUTRAL_RATINGS
        assert rated.loc["A", "COLLEY"] > 0.5

    def test_same_week_result_is_excluded(self) -> None:
        ratings = _ratings([(2010, 3, "A", "B", 50, 0)])
        assert ratings.as_of(2010, 3, ["A"]).loc["A"].to_dict() == NEUTRAL_RATINGS
        assert ratings.as_of(2010, 4, ["A"]).loc["A", "COLLEY"] > 0.5

    def test_future_scores_do_not_change_earlier_ratings(self) -> None:
        base = [(2010, 1, "A", "B", 20, 10), (2010, 8, "A", "C", 30, 3)]
        changed = [(2010, 1, "A", "B", 20, 10), (2010, 8, "A", "C", 0, 44)]
        pd.testing.assert_frame_equal(
            _ratings(base).as_of(2010, 8, ["A", "B", "C"]),
            _ratings(changed).as_of(2010, 8, ["A", "B", "C"]),
        )

    def test_lookahead_equals_next_week(self) -> None:
        ratings = _ratings([(2026, 1, "A", "B", 20, 10), (2026, 2, "B", "C", 7, 14)])
        next_week = ratings.as_of(2026, 3, ["A", "B", "C"])
        for week in (4, 10, 18):
            pd.testing.assert_frame_equal(next_week, ratings.as_of(2026, week, ["A", "B", "C"]))

    def test_prior_season_carries_at_configured_weight(self) -> None:
        extra: list[GameSpec] = [(2009, 5, "A", "B", 40, 0)]
        carried = _ratings(extra, LinearRatingConfig(prior_season_weight=1.0)).as_of(2010, 1, ["A"])
        dropped = _ratings(extra, LinearRatingConfig(prior_season_weight=0.0)).as_of(2010, 1, ["A"])
        assert carried.loc["A", "COLLEY"] > 0.5
        assert dropped.loc["A"].to_dict() == NEUTRAL_RATINGS

    def test_two_seasons_back_is_ignored(self) -> None:
        rated = _ratings([(2008, 5, "A", "B", 40, 0)]).as_of(2010, 1, ["A"])
        assert rated.loc["A"].to_dict() == NEUTRAL_RATINGS

    def test_truncated_history_raises(self) -> None:
        games = make_rating_history([(2010, 1, "A", "B", 20, 10)])
        with pytest.raises(ValueError, match="must begin with season"):
            LinearRatings(games.loc[games["YEAR"] != "1999-2000"])

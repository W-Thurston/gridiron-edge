# tests/unit/ratings/test_glicko2.py

"""Tests for team Glicko-2 recomputed from complete history."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from tests.fixtures.rating_history import make_rating_history

from gridiron_edge.ratings.glicko2 import (
    GLICKO2_SCALE,
    Glicko2Config,
    Glicko2History,
    Glicko2State,
    expected_score,
    glicko2_step,
)


class TestGlicko2Step:
    def test_matches_published_worked_example(self) -> None:
        """Glickman's example: 1500/200/0.06 vs three opponents, tau 0.5."""
        scale = GLICKO2_SCALE
        results = [
            ((1400 - 1500) / scale, 30 / scale, 1.0),
            ((1550 - 1500) / scale, 100 / scale, 0.0),
            ((1700 - 1500) / scale, 300 / scale, 0.0),
        ]
        mu, phi, sigma = glicko2_step(0.0, 200 / scale, 0.06, results, tau=0.5)
        assert mu * scale + 1500 == pytest.approx(1464.06, abs=0.01)
        assert phi * scale == pytest.approx(151.52, abs=0.01)
        assert sigma == pytest.approx(0.05999, abs=0.00001)

    def test_no_games_only_widens_rd(self) -> None:
        mu, phi, sigma = glicko2_step(0.3, 0.8, 0.06, [], tau=0.5)
        assert mu == 0.3
        assert sigma == 0.06
        assert phi == pytest.approx(math.sqrt(0.8**2 + 0.06**2))

    def test_expected_score_is_symmetric_for_equal_teams(self) -> None:
        team = Glicko2State(1500.0, 80.0, 0.06)
        assert expected_score(team, team) == pytest.approx(0.5)


def _history(extra: list | None = None, config: Glicko2Config | None = None) -> Glicko2History:
    return Glicko2History(make_rating_history(extra), config)


class TestGlicko2History:
    def test_before_any_game_everyone_is_initial(self) -> None:
        state = _history().state_as_of(1999, 1, ["X"])["X"]
        assert state == Glicko2State(1500.0, 350.0, 0.06)

    def test_winner_gains_and_loser_drops_with_no_same_week_leakage(self) -> None:
        history = _history([(2010, 3, "X", "Y", 30, 10)])
        before = history.state_as_of(2010, 3, ["X", "Y"])
        after = history.state_as_of(2010, 4, ["X", "Y"])
        assert before["X"].rating == before["Y"].rating
        assert after["X"].rating > after["Y"].rating
        assert after["X"].rd < before["X"].rd

    def test_tie_moves_equal_teams_not_at_all(self) -> None:
        history = _history([(2010, 3, "X", "Y", 17, 17)])
        after = history.state_as_of(2010, 4, ["X", "Y"])
        assert after["X"].rating == pytest.approx(after["Y"].rating)

    def test_bye_week_widens_rd(self) -> None:
        history = _history([(2010, 3, "X", "Y", 30, 10), (2010, 4, "Z", "Y", 10, 20)])
        after_game = history.state_as_of(2010, 4, ["X"])["X"]
        after_bye = history.state_as_of(2010, 5, ["X"])["X"]
        assert after_bye.rd > after_game.rd
        assert after_bye.rating == after_game.rating

    def test_lookahead_beyond_last_completed_game_is_frozen(self) -> None:
        history = _history([(2026, 1, "X", "Y", 30, 10), (2026, 2, "X", "Z", 21, 20)])
        next_week = history.state_as_of(2026, 3, ["X", "Y", "Z"])
        for week in (4, 9, 18):
            assert history.state_as_of(2026, week, ["X", "Y", "Z"]) == next_week

    def test_new_season_applies_offseason_regression_and_wider_rd(self) -> None:
        history = _history([(2025, 4, "X", "Y", 40, 3)])
        end = history.state_as_of(2025, 22, ["X"])["X"]
        next_season = history.state_as_of(2026, 1, ["X"])["X"]
        assert abs(next_season.rating - 1500) < abs(end.rating - 1500)
        assert next_season.rd > end.rd

    def test_lookahead_past_next_season_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot look ahead"):
            _history([(2025, 4, "X", "Y", 40, 3)]).state_as_of(2028, 1, ["X"])

    def test_expansion_team_enters_at_expansion_rating(self) -> None:
        history = _history([(2002, 1, "Houston Texans", "Y", 20, 10)])
        assert history.state_as_of(2002, 1, ["Houston Texans"])["Houston Texans"].rating == 1300.0
        assert history.state_as_of(2002, 1, ["Houston Texans"])["Houston Texans"].rd == 350.0

    def test_replay_is_deterministic(self) -> None:
        extra = [(2010, 3, "X", "Y", 30, 10), (2010, 4, "Y", "Z", 3, 6)]
        a = _history(extra).state_as_of(2010, 5, ["X", "Y", "Z"])
        b = _history(extra).state_as_of(2010, 5, ["X", "Y", "Z"])
        assert a == b

    def test_future_scores_do_not_change_earlier_state(self) -> None:
        base = [(2010, 3, "X", "Y", 30, 10), (2010, 9, "X", "Z", 21, 20)]
        changed = [(2010, 3, "X", "Y", 30, 10), (2010, 9, "X", "Z", 0, 45)]
        assert _history(base).state_as_of(2010, 9, ["X", "Y", "Z"]) == _history(
            changed
        ).state_as_of(2010, 9, ["X", "Y", "Z"])

    def test_truncated_history_raises(self) -> None:
        games = make_rating_history([(2010, 3, "X", "Y", 30, 10)])
        with pytest.raises(ValueError, match="must begin with season"):
            Glicko2History(games.loc[games["YEAR"] != "1999-2000"])

    def test_empty_history_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Glicko2History(pd.DataFrame(columns=make_rating_history().columns))

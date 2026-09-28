# tests/unit/ratings/test_qb_elo.py

"""Tests for the starting-QB rating recomputed from complete history."""

from __future__ import annotations

import pytest
from tests.fixtures.rating_history import GameSpec, make_qb_starters, make_rating_history

from gridiron_edge.ratings.qb_elo import QbEloConfig, QbEloHistory, QbState


def _history(
    extra: list[GameSpec],
    *,
    overrides: dict[str, tuple[str, str]] | None = None,
    config: QbEloConfig | None = None,
) -> QbEloHistory:
    games = make_rating_history(extra)
    return QbEloHistory(games, make_qb_starters(games, overrides), config)


class TestQbEloHistory:
    def test_debut_qb_is_zero_with_no_starts(self) -> None:
        state = _history([]).qb_state_as_of(2010, 1, ["never_started"])
        assert state["never_started"] == QbState(0.0, 0)

    def test_winning_qb_gains_losing_qb_drops_equally_for_two_debuts(self) -> None:
        history = _history([(2010, 3, "X", "Y", 30, 10)])
        after = history.qb_state_as_of(2010, 4, ["qb_X", "qb_Y"])
        assert after["qb_X"].rating > 0 > after["qb_Y"].rating
        assert after["qb_X"].rating == pytest.approx(-after["qb_Y"].rating)
        assert after["qb_X"].starts == after["qb_Y"].starts == 1

    def test_own_result_never_enters_pre_game_value(self) -> None:
        history = _history([(2010, 3, "X", "Y", 30, 10)])
        assert history.qb_state_as_of(2010, 3, ["qb_X"])["qb_X"] == QbState(0.0, 0)

    def test_debut_moves_faster_than_a_veteran(self) -> None:
        early = [(2010, week, "X", "Z", 24, 20) for week in range(1, 13)]
        history = _history([*early, (2010, 14, "X", "Y", 30, 10)])
        rookie_move = history.qb_state_as_of(2010, 15, ["qb_Y"])["qb_Y"].rating
        veteran_before = history.qb_state_as_of(2010, 14, ["qb_X"])["qb_X"].rating
        veteran_after = history.qb_state_as_of(2010, 15, ["qb_X"])["qb_X"].rating
        assert abs(rookie_move) > abs(veteran_after - veteran_before)

    def test_rating_follows_the_qb_across_teams(self) -> None:
        history = _history(
            [(2010, 1, "X", "Y", 40, 3), (2010, 2, "Z", "W", 20, 20)],
            overrides={"2010_02_Z_W": ("qb_X", "qb_W")},
        )
        moved = history.qb_state_as_of(2010, 3, ["qb_X"])["qb_X"]
        assert moved.starts == 2
        assert moved.rating > 0

    def test_offseason_regresses_qb_rating_toward_zero(self) -> None:
        history = _history([(2025, 4, "X", "Y", 40, 3)])
        end = history.qb_state_as_of(2025, 22, ["qb_X"])["qb_X"]
        next_season = history.qb_state_as_of(2026, 1, ["qb_X"])["qb_X"]
        assert 0 < next_season.rating < end.rating
        assert next_season.starts == end.starts

    def test_lookahead_beyond_last_completed_game_is_frozen(self) -> None:
        history = _history([(2026, 1, "X", "Y", 30, 10), (2026, 2, "X", "Z", 21, 20)])
        ids = ["qb_X", "qb_Y", "qb_Z"]
        next_week = history.qb_state_as_of(2026, 3, ids)
        for week in (4, 9, 18):
            assert history.qb_state_as_of(2026, week, ids) == next_week

    def test_future_scores_do_not_change_earlier_state(self) -> None:
        base = [(2010, 3, "X", "Y", 30, 10), (2010, 9, "X", "Z", 21, 20)]
        changed = [(2010, 3, "X", "Y", 30, 10), (2010, 9, "X", "Z", 0, 45)]
        ids = ["qb_X", "qb_Y", "qb_Z"]
        assert _history(base).qb_state_as_of(2010, 9, ids) == _history(changed).qb_state_as_of(
            2010, 9, ids
        )

    def test_predictions_cover_every_game_with_pre_game_probability(self) -> None:
        history = _history([(2010, 3, "X", "Y", 30, 10)])
        frame = history.prediction_frame()
        assert len(frame) == 27 + 1 + 1  # filler seasons, latest-season filler, one extra
        row = frame.loc[frame["game_id"] == "2010_03_X_Y"].iloc[0]
        assert row["away_win_probability"] == pytest.approx(0.5)
        assert row["away_result"] == 1.0

    def test_completed_game_without_starter_raises(self) -> None:
        games = make_rating_history([(2010, 3, "X", "Y", 30, 10)])
        starters = make_qb_starters(games).astype({"AWAY_QB_ID": object})
        starters.loc[starters["GAME_ID"] == "2010_03_X_Y", "AWAY_QB_ID"] = None
        with pytest.raises(ValueError, match="no recorded starting QB"):
            QbEloHistory(games, starters)

    def test_missing_starter_columns_raise(self) -> None:
        games = make_rating_history()
        with pytest.raises(ValueError, match="missing required columns"):
            QbEloHistory(games, make_qb_starters(games).drop(columns=["HOME_QB_ID"]))

    def test_truncated_history_raises(self) -> None:
        games = make_rating_history([(2010, 3, "X", "Y", 30, 10)])
        games = games.loc[games["YEAR"] != "1999-2000"]
        with pytest.raises(ValueError, match="must begin with season"):
            QbEloHistory(games, make_qb_starters(games))

    def test_lookahead_past_next_season_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot look ahead"):
            _history([]).qb_state_as_of(2029, 1, ["qb_X"])

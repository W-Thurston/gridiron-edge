# tests/unit/evaluation/test_market_agreement.py

"""Tests for the model-versus-closing-market agreement tables (``DECISIONS.md`` D85)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gridiron_edge.evaluation.market_agreement import (
    FamilyAgreement,
    agreement_tables,
    build_report,
    prediction_frame,
    total_agreement,
    win_agreement,
)

pytestmark = pytest.mark.unit


def _games(count: int, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    expected_margin = rng.normal(0.0, 6.0, count)
    margin = expected_margin + rng.normal(0.0, 13.5, count)
    line = np.round(rng.normal(46.0, 4.0, count) * 2) / 2
    total = line + rng.normal(0.0, 12.5, count)
    home = np.round((total + margin) / 2)
    away = np.round((total - margin) / 2)
    return pd.DataFrame(
        {
            "GAME_ID": [f"g{i}" for i in range(count)],
            "YEAR": ["2020-2021"] * count,
            "AWAY_SCORE": away,
            "HOME_SCORE": home,
            "VEGAS_LINE": expected_margin,
            "OVER_UNDER": line,
        }
    )


def _events_win(games: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"game_id": games["GAME_ID"], "home_win_prob": probabilities})


def _market_probability(games: pd.DataFrame) -> np.ndarray:
    from math import erf, sqrt

    return np.array([0.5 * (1 + erf(m / 13.5 / sqrt(2))) for m in games["VEGAS_LINE"]])


def test_a_model_that_only_adds_noise_has_a_slope_near_zero_and_is_not_validated() -> None:
    games = _games(4000)
    rng = np.random.default_rng(1)
    noisy = np.clip(_market_probability(games) + rng.normal(0.0, 0.06, len(games)), 0.02, 0.98)

    result = win_agreement("noise", games, _events_win(games, noisy))

    assert abs(result.slope) < 0.25
    assert result.slope_low < 0.05
    assert not result.validated
    assert result.model_score > result.market_score
    assert result.score_label == "Brier"
    assert sum(bin_.games for bin_ in result.bins) == result.games


def test_a_model_that_is_right_where_the_market_is_noisy_has_a_slope_near_one() -> None:
    count = 8000
    rng = np.random.default_rng(3)
    true_margin = rng.normal(0.0, 6.0, count)
    from math import erf, sqrt

    true_p = np.array([0.5 * (1 + erf(m / 13.5 / sqrt(2))) for m in true_margin])
    won = rng.random(count) < true_p
    games = pd.DataFrame(
        {
            "GAME_ID": [f"g{i}" for i in range(count)],
            "YEAR": ["2020-2021"] * count,
            "AWAY_SCORE": np.where(won, 17.0, 24.0),
            "HOME_SCORE": np.where(won, 24.0, 17.0),
            # The market's expected margin is the truth plus noise; the model sees the truth.
            "VEGAS_LINE": true_margin + rng.normal(0.0, 3.0, count),
            "OVER_UNDER": 45.0,
        }
    )

    result = win_agreement("informed", games, _events_win(games, true_p))

    assert result.slope == pytest.approx(1.0, abs=0.15)
    assert result.validated
    assert result.model_score < result.market_score


def test_total_noise_is_not_validated_and_the_line_has_the_lower_error() -> None:
    games = _games(4000, seed=5)
    rng = np.random.default_rng(2)
    events = pd.DataFrame(
        {
            "game_id": games["GAME_ID"],
            "model_total": games["OVER_UNDER"] + rng.normal(0.0, 3.0, len(games)),
        }
    )

    result = total_agreement("noise", games, events)

    assert not result.validated
    assert result.score_label == "MAE"
    assert result.model_score > result.market_score
    assert abs(result.slope) < 0.3


def test_a_total_that_is_right_where_the_line_is_noisy_beats_the_line() -> None:
    count = 6000
    rng = np.random.default_rng(7)
    line = np.full(count, 45.0)
    model_total = line + rng.normal(0.0, 3.0, count)
    actual = model_total + rng.normal(0.0, 12.0, count)
    games = pd.DataFrame(
        {
            "GAME_ID": [f"g{i}" for i in range(count)],
            "YEAR": ["2020-2021"] * count,
            "AWAY_SCORE": actual / 2,
            "HOME_SCORE": actual / 2,
            "VEGAS_LINE": 0.0,
            "OVER_UNDER": line,
        }
    )
    events = pd.DataFrame({"game_id": games["GAME_ID"], "model_total": model_total})

    result = total_agreement("informed", games, events)

    assert result.slope == pytest.approx(1.0, abs=0.12)
    assert result.validated
    assert result.model_score < result.market_score


def test_seasons_before_the_first_and_unscored_games_are_excluded() -> None:
    games = _games(500, seed=9)
    games.loc[:99, "YEAR"] = "2005-2006"
    games.loc[100:149, "AWAY_SCORE"] = np.nan
    probabilities = np.full(len(games), 0.6)

    result = win_agreement("x", games, _events_win(games, probabilities))

    assert result.games <= 350


def test_missing_game_columns_and_unknown_kinds_are_rejected() -> None:
    games = _games(50).drop(columns=["VEGAS_LINE"])
    with pytest.raises(ValueError, match="missing columns"):
        win_agreement("x", games, _events_win(games.assign(VEGAS_LINE=0.0), np.full(50, 0.5)))
    with pytest.raises(ValueError, match="Unknown family kind"):
        agreement_tables(_games(10), {"x": ("spread", pd.DataFrame({"game_id": []}))})


def test_a_model_that_never_disagrees_has_no_slope() -> None:
    games = _games(200, seed=4)

    with pytest.raises(ValueError, match="never disagrees"):
        win_agreement("same", games, _events_win(games, _market_probability(games)))


def test_prediction_frame_keeps_the_latest_event_per_game() -> None:
    events = pd.DataFrame(
        {
            "game_id": ["a", "a", "b"],
            "generated_at": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-01-01"], utc=True),
            "home_win_prob": [0.4, 0.7, 0.5],
        }
    )

    frame = prediction_frame(events, "home_win_prob")

    assert frame.set_index("game_id")["home_win_prob"].to_dict() == {"a": 0.7, "b": 0.5}
    with pytest.raises(ValueError, match="No forecast events"):
        prediction_frame(events.iloc[0:0], "home_win_prob")


def test_the_report_states_the_rule_and_the_verdict() -> None:
    failing = FamilyAgreement("Win - A", "win", 100, 0.22, 0.21, "Brier", 0.02, -0.1, 0.14, ())
    passing = FamilyAgreement("Total - B", "total", 100, 10.0, 10.5, "MAE", 0.9, 0.6, 1.2, ())

    none_pass = build_report([failing], generated_on=date(2026, 10, 4))
    one_passes = build_report([failing, passing], generated_on=date(2026, 10, 4))

    assert "excludes 0 and includes 1" in none_pass
    assert "chosen after seeing the data" in none_pass
    assert "No family passes the rule" in none_pass
    assert (
        "| Win - A | 100 | Brier 0.2200 | Brier 0.2100 | 0.020 | -0.100 to 0.140 | no |"
        in none_pass
    )
    assert "Families that pass the rule: Total - B." in one_passes

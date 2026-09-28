# tests/unit/features/test_home_away_record_feature.py

"""Tests for canonical Away/Home record features."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.record import (
    _PYTHAGOREAN_EXPONENT,
    HomeAwayRecordFeature,
)


def _datasets(
    games: DataFrame,
) -> MagicMock:
    """Return a controlled historical-games accessor."""
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.games.return_value = games.copy()
    return datasets


def _historical_game(
    *,
    game_id: str,
    week: int,
    away_team: str,
    home_team: str,
    away_score: int,
    home_score: int,
    year: str = "2025-2026",
) -> dict[str, object]:
    """Return one completed canonical historical game."""
    return {
        "GAME_ID": game_id,
        "YEAR": year,
        "WEEK_NUM": week,
        "GAME_DATE": f"2025-09-{week + 1:02d}",
        "AWAY_TEAM": away_team,
        "HOME_TEAM": home_team,
        "AWAY_SCORE": away_score,
        "HOME_SCORE": home_score,
    }


def _history() -> DataFrame:
    """Return prior results for the target teams."""
    return DataFrame(
        [
            _historical_game(
                game_id="away-week-1",
                week=1,
                away_team="Away Team",
                home_team="Opponent One",
                away_score=24,
                home_score=17,
            ),
            _historical_game(
                game_id="away-week-2",
                week=2,
                away_team="Opponent Two",
                home_team="Away Team",
                away_score=20,
                home_score=27,
            ),
            _historical_game(
                game_id="home-week-3",
                week=3,
                away_team="Home Team",
                home_team="Opponent Three",
                away_score=14,
                home_score=21,
            ),
        ]
    )


def _target() -> DataFrame:
    """Return one canonical week-four target game."""
    return DataFrame(
        {
            "GAME_ID": ["target-week-4"],
            "YEAR": ["2025-2026"],
            "WEEK_NUM": [4],
            "AWAY_TEAM": ["Away Team"],
            "HOME_TEAM": ["Home Team"],
            "MARKER": ["preserved"],
        }
    )


def test_registered_as_home_away_record() -> None:
    assert FeatureRegistry.get("home_away_record") is HomeAwayRecordFeature


def test_spec_contains_canonical_outputs() -> None:
    assert HomeAwayRecordFeature.spec.produces == [
        "AWAY_WINS",
        "AWAY_LOSSES",
        "AWAY_WIN_PCT",
        "AWAY_WIN_STREAK",
        "AWAY_LOSS_STREAK",
        "HOME_WINS",
        "HOME_LOSSES",
        "HOME_WIN_PCT",
        "HOME_WIN_STREAK",
        "HOME_LOSS_STREAK",
        "AWAY_FORM_TREND",
        "AWAY_PYTHAG_WIN_PCT",
        "AWAY_PYTHAG_LUCK_GAP",
        "AWAY_ONE_SCORE_WIN_PCT",
        "AWAY_ONE_SCORE_GAMES",
        "HOME_FORM_TREND",
        "HOME_PYTHAG_WIN_PCT",
        "HOME_PYTHAG_LUCK_GAP",
        "HOME_ONE_SCORE_WIN_PCT",
        "HOME_ONE_SCORE_GAMES",
    ]


def test_computes_distinct_away_and_home_records() -> None:
    result = HomeAwayRecordFeature().compute(
        df=_target(),
        datasets=_datasets(_history()),
    )

    row = result.iloc[0]

    assert row["AWAY_WINS"] == pytest.approx(2.0)
    assert row["AWAY_LOSSES"] == pytest.approx(0.0)
    assert row["AWAY_WIN_PCT"] == pytest.approx(1.0)
    assert row["AWAY_WIN_STREAK"] == 2
    assert row["AWAY_LOSS_STREAK"] == 0

    assert row["HOME_WINS"] == pytest.approx(0.0)
    assert row["HOME_LOSSES"] == pytest.approx(1.0)
    assert row["HOME_WIN_PCT"] == pytest.approx(0.0)
    assert row["HOME_WIN_STREAK"] == 0
    assert row["HOME_LOSS_STREAK"] == 1


def test_current_and_future_weeks_do_not_leak() -> None:
    extra = DataFrame(
        [
            _historical_game(
                game_id="current-week",
                week=4,
                away_team="Away Team",
                home_team="Current Opponent",
                away_score=0,
                home_score=40,
            ),
            _historical_game(
                game_id="future-week",
                week=5,
                away_team="Away Team",
                home_team="Future Opponent",
                away_score=0,
                home_score=40,
            ),
        ]
    )

    history = pd.concat(
        [
            _history(),
            extra,
        ],
        ignore_index=True,
    )

    result = HomeAwayRecordFeature().compute(
        df=_target(),
        datasets=_datasets(history),
    )

    row = result.iloc[0]

    assert row["AWAY_WINS"] == pytest.approx(2.0)
    assert row["AWAY_LOSSES"] == pytest.approx(0.0)
    assert row["AWAY_WIN_STREAK"] == 2


def test_tie_counts_half_and_resets_streaks() -> None:
    tie = DataFrame(
        [
            _historical_game(
                game_id="away-week-3-tie",
                week=3,
                away_team="Away Team",
                home_team="Tie Opponent",
                away_score=21,
                home_score=21,
            )
        ]
    )

    history = pd.concat(
        [
            _history(),
            tie,
        ],
        ignore_index=True,
    )

    result = HomeAwayRecordFeature().compute(
        df=_target(),
        datasets=_datasets(history),
    )

    row = result.iloc[0]

    assert row["AWAY_WINS"] == pytest.approx(2.5)
    assert row["AWAY_LOSSES"] == pytest.approx(0.5)
    assert row["AWAY_WIN_PCT"] == pytest.approx(2.5 / 3.0)
    assert row["AWAY_WIN_STREAK"] == 0
    assert row["AWAY_LOSS_STREAK"] == 0


def test_record_resets_by_season() -> None:
    history = _history()
    history["YEAR"] = "2024-2025"

    result = HomeAwayRecordFeature().compute(
        df=_target(),
        datasets=_datasets(history),
    )

    row = result.iloc[0]

    assert row["AWAY_WINS"] == 0
    assert row["AWAY_LOSSES"] == 0
    assert row["AWAY_WIN_PCT"] == pytest.approx(0.0)
    assert row["AWAY_WIN_STREAK"] == 0
    assert row["AWAY_LOSS_STREAK"] == 0


def test_preserves_input_and_unrelated_columns() -> None:
    target = _target()
    expected = target.copy(deep=True)

    result = HomeAwayRecordFeature().compute(
        df=target,
        datasets=_datasets(_history()),
    )

    pd.testing.assert_frame_equal(
        target,
        expected,
    )
    assert result["MARKER"].tolist() == ["preserved"]


@pytest.mark.parametrize(
    "column",
    [
        "GAME_ID",
        "YEAR",
        "WEEK_NUM",
        "AWAY_TEAM",
        "HOME_TEAM",
    ],
)
def test_missing_target_column_is_rejected(
    column: str,
) -> None:
    target = _target().drop(columns=[column])

    with pytest.raises(
        ValueError,
        match=(f"Home/away game frame is missing required columns: {column}"),
    ):
        HomeAwayRecordFeature().compute(
            df=target,
            datasets=_datasets(_history()),
        )


def test_duplicate_historical_game_ids_are_rejected() -> None:
    history = pd.concat(
        [
            _history(),
            _history().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game IDs",
    ):
        HomeAwayRecordFeature().compute(
            df=_target(),
            datasets=_datasets(history),
        )


def test_week_one_uses_zero_record_defaults() -> None:
    target = _target()
    target["WEEK_NUM"] = 1

    row = (
        HomeAwayRecordFeature()
        .compute(
            df=target,
            datasets=_datasets(_history()),
        )
        .iloc[0]
    )

    assert row["AWAY_WINS"] == pytest.approx(0.0)
    assert row["AWAY_LOSSES"] == pytest.approx(0.0)
    assert row["AWAY_WIN_PCT"] == pytest.approx(0.0)
    assert row["HOME_WINS"] == pytest.approx(0.0)
    assert row["HOME_LOSSES"] == pytest.approx(0.0)
    assert row["HOME_WIN_PCT"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Same-season performance metrics
# ---------------------------------------------------------------------------


def _pythag(points_for: float, points_against: float) -> float:
    pf = points_for**_PYTHAGOREAN_EXPONENT
    pa = points_against**_PYTHAGOREAN_EXPONENT
    return pf / (pf + pa)


def _team_history(
    scores: list[tuple[int, int]],
    *,
    team: str = "Away Team",
    year: str = "2025-2026",
) -> DataFrame:
    """One team playing at home against distinct opponents, weeks 1..n."""
    return DataFrame(
        [
            _historical_game(
                game_id=f"{year}-{team}-{week}",
                week=week,
                away_team=f"Opponent {week}",
                home_team=team,
                away_score=points_against,
                home_score=points_for,
                year=year,
            )
            for week, (points_for, points_against) in enumerate(scores, start=1)
        ]
    )


def _performance(scores: list[tuple[int, int]], *, week: int | None = None) -> pd.Series:
    target = _target()
    target["WEEK_NUM"] = week if week is not None else len(scores) + 1
    return (
        HomeAwayRecordFeature()
        .compute(df=target, datasets=_datasets(_team_history(scores)))
        .iloc[0]
    )


def test_performance_metrics_for_distinct_sides() -> None:
    row = HomeAwayRecordFeature().compute(df=_target(), datasets=_datasets(_history())).iloc[0]

    # Away Team: won 24-17 and 27-20 (both one-score margins of 7).
    assert row["AWAY_FORM_TREND"] == pytest.approx(0.0)
    assert row["AWAY_PYTHAG_WIN_PCT"] == pytest.approx(_pythag(51, 37))
    assert row["AWAY_PYTHAG_LUCK_GAP"] == pytest.approx(1.0 - _pythag(51, 37))
    assert row["AWAY_ONE_SCORE_WIN_PCT"] == pytest.approx(1.0)
    assert row["AWAY_ONE_SCORE_GAMES"] == 2

    # Home Team: lost 14-21.
    assert row["HOME_PYTHAG_WIN_PCT"] == pytest.approx(_pythag(14, 21))
    assert row["HOME_PYTHAG_LUCK_GAP"] == pytest.approx(0.0 - _pythag(14, 21))
    assert row["HOME_ONE_SCORE_WIN_PCT"] == pytest.approx(0.0)
    assert row["HOME_ONE_SCORE_GAMES"] == 1


def test_week_one_performance_uses_neutral_values() -> None:
    row = _performance([(30, 10)], week=1)  # week-1 target has no prior games

    assert row["AWAY_FORM_TREND"] == pytest.approx(0.0)
    assert row["AWAY_PYTHAG_WIN_PCT"] == pytest.approx(0.5)
    assert row["AWAY_PYTHAG_LUCK_GAP"] == pytest.approx(0.0)
    assert row["AWAY_ONE_SCORE_WIN_PCT"] == pytest.approx(0.5)
    assert row["AWAY_ONE_SCORE_GAMES"] == 0


def test_form_trend_is_zero_with_four_or_fewer_games() -> None:
    row = _performance([(30, 10), (10, 30), (24, 20), (13, 27)])

    assert row["AWAY_FORM_TREND"] == pytest.approx(0.0)


def test_form_trend_compares_recent_games_with_season_mean() -> None:
    # Margins +10, +10, +10, +10, -6: last four = 6.0, season mean = 6.8.
    row = _performance([(20, 10), (20, 10), (20, 10), (20, 10), (10, 16)])

    assert row["AWAY_FORM_TREND"] == pytest.approx(6.0 - 6.8)


def test_one_score_boundary_and_ties() -> None:
    # Margin exactly 8 (win) counts; 9 does not; a 17-17 tie counts as 0.5.
    row = _performance([(28, 20), (29, 20), (17, 17)])

    assert row["AWAY_ONE_SCORE_GAMES"] == 2
    assert row["AWAY_ONE_SCORE_WIN_PCT"] == pytest.approx(0.75)


def test_luck_gap_is_actual_minus_pythagorean() -> None:
    # 2-0 on tiny margins: actual 1.0, Pythagorean barely above .5.
    row = _performance([(21, 20), (22, 21)])

    assert row["AWAY_PYTHAG_LUCK_GAP"] == pytest.approx(1.0 - _pythag(43, 41))
    assert row["AWAY_PYTHAG_LUCK_GAP"] > 0.4


def test_performance_metrics_reset_by_season_and_ignore_future_weeks() -> None:
    history = pd.concat(
        [
            _team_history([(50, 0), (50, 0)], year="2024-2025"),
            _team_history([(20, 17)]),
            _team_history([(0, 40)]).assign(WEEK_NUM=5, GAME_ID="future"),
        ],
        ignore_index=True,
    )
    target = _target()
    target["WEEK_NUM"] = 2
    row = HomeAwayRecordFeature().compute(df=target, datasets=_datasets(history)).iloc[0]

    assert row["AWAY_PYTHAG_WIN_PCT"] == pytest.approx(_pythag(20, 17))
    assert row["AWAY_ONE_SCORE_GAMES"] == 1

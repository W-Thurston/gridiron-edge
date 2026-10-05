# tests/unit/evaluation/test_weekly_recap_season.py

"""Tests for the season recap rolled up from the selected weekly recaps (``DECISIONS.md`` D86)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
import math
import statistics

import pandas as pd
from pandas import DataFrame
import pytest
from tests.fixtures.weekly_recap import BUILT_AT, SEASON, recap_week

from gridiron_edge.evaluation.metrics import expected_calibration_error
from gridiron_edge.evaluation.weekly_recap import (
    WeeklyRecap,
    WeeklyRecapRefusalError,
    build_weekly_recap,
    recap_metrics,
)
from gridiron_edge.evaluation.weekly_recap_season import (
    INTERVAL_Z,
    SeasonRecap,
    SeasonWeekStatus,
    build_season_recap,
    season_recap_from_payload,
    season_recap_id,
    season_recap_payload,
)

pytestmark = pytest.mark.unit

_SIDES = ("away", "home")


def _game(
    week: int,
    role: str,
    *,
    p: float,
    home_won: bool | None,
    market_p: float,
    spread: tuple[float, float] | None,
    total: tuple[float, float] | None,
    starter: dict[str, object] | None = None,
) -> dict[str, object]:
    """One recap game row with only the columns the season recap reads.

    ``spread`` and ``total`` are (model error, market error); projected scores are missing.
    """
    y = None if home_won is None else float(home_won)
    row: dict[str, object] = {
        "week": week,
        "forecast_role": role,
        "actual_home_win": home_won,
        "win_status": "available" if home_won is not None else "outcome_tie",
        "home_win_prob": p,
        "closing_home_win_prob": market_p,
        "win_model_brier": None if y is None else (p - y) ** 2,
        "win_market_brier": None if y is None else (market_p - y) ** 2,
        "win_closer": None,
        "spread_model_error": None if spread is None else spread[0],
        "spread_market_error": None if spread is None else spread[1],
        "spread_closer": None,
        "total_model_error": None if total is None else total[0],
        "total_market_error": None if total is None else total[1],
        "total_closer": None,
        "away_score": 20.0,
        "home_score": 24.0,
        "projected_away_score": None,
        "projected_home_score": None,
        "market_implied_away_score": None,
        "market_implied_home_score": None,
        "projected_score_model_abs_error": None,
        "projected_score_market_abs_error": None,
        "projected_score_closer": None,
    }
    for side in _SIDES:
        row |= {
            f"{side}_starter_status": "not_recorded",
            f"{side}_starter_record_source": None,
            f"{side}_actual_qb_source": "nflverse_schedule",
            f"{side}_assumed_correct": None,
            f"{side}_nflverse_correct": None,
            f"{side}_carry_forward_correct": None,
        }
    return row | (starter or {})


def _scored_home_starter(*, assumed: bool) -> dict[str, object]:
    return {
        "home_starter_status": "available",
        "home_starter_record_source": "product_record",
        "home_assumed_correct": assumed,
        "home_nflverse_correct": True,
        "home_carry_forward_correct": False,
    }


def _frame(rows: list[dict[str, object]]) -> DataFrame:
    frame = DataFrame.from_records(rows)
    frame["actual_home_win"] = frame["actual_home_win"].astype("boolean")
    for flag in ("assumed_correct", "nflverse_correct", "carry_forward_correct"):
        for side in _SIDES:
            frame[f"{side}_{flag}"] = frame[f"{side}_{flag}"].astype("boolean")
    for column in frame.columns:
        if column.endswith(("_error", "_brier", "_prob", "_score")):
            frame[column] = pd.to_numeric(frame[column]).astype("float64")
    return frame


@pytest.fixture(scope="module")
def base_recap() -> WeeklyRecap:
    inputs = recap_week()
    return build_weekly_recap(
        season=SEASON,
        week=4,
        schedule=inputs.schedule,
        product=inputs.product,
        forecast_events=inputs.events,
        games=inputs.games,
        player_game_logs=inputs.logs,
        sidecar_captures=inputs.captures,
        built_at=BUILT_AT,
    ).recap


def _recap(base: WeeklyRecap, week: int, role: str, frame: DataFrame) -> WeeklyRecap:
    return replace(
        base,
        week=week,
        forecast_role=role,
        recap_id=f"{week:064x}",
        metrics=recap_metrics(frame),
    )


# Week 1 (live): two decided games. Week 2 (development): two decided games and a tie.
WEEK_1 = [
    _game(
        1,
        "live",
        p=0.72,
        home_won=True,
        market_p=0.62,
        spread=(-2.0, 4.0),
        total=(3.0, -1.0),
        starter=_scored_home_starter(assumed=True),
    ),
    _game(1, "live", p=0.42, home_won=False, market_p=0.52, spread=(6.0, -5.0), total=(-4.0, 2.0)),
]
WEEK_2 = [
    _game(
        2,
        "development",
        p=0.86,
        home_won=True,
        market_p=0.74,
        spread=(1.0, 3.0),
        total=(0.0, 5.0),
        starter=_scored_home_starter(assumed=False),
    ),
    _game(
        2,
        "development",
        p=0.56,
        home_won=False,
        market_p=0.46,
        spread=(-7.0, -3.0),
        total=(2.0, -6.0),
    ),
    _game(2, "development", p=0.5, home_won=None, market_p=0.5, spread=None, total=None),
]


@pytest.fixture
def weeks(base_recap: WeeklyRecap) -> list[tuple[WeeklyRecap, DataFrame]]:
    first, second = _frame(WEEK_1), _frame(WEEK_2)
    return [
        (_recap(base_recap, 1, "live", first), first),
        (_recap(base_recap, 2, "development", second), second),
    ]


def _season(
    weeks: list[tuple[WeeklyRecap, DataFrame]], finished: tuple[int, ...] = (1, 2, 3)
) -> SeasonRecap:
    return build_season_recap(
        season=SEASON, recaps=weeks, finished_weeks=finished, built_at=BUILT_AT
    )


def test_each_week_reads_exactly_as_its_recap(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)

    first, second = recap.weeks[0], recap.weeks[1]
    assert first.metrics == weeks[0][0].metrics
    assert second.metrics == weeks[1][0].metrics
    assert first.metrics is not None and second.metrics is not None
    assert first.metrics.win.model.brier == pytest.approx((0.28**2 + 0.42**2) / 2)
    assert first.metrics.win.market.brier == pytest.approx((0.38**2 + 0.52**2) / 2)
    assert second.metrics.win.model.count == 2
    assert second.metrics.spread.model.mae == pytest.approx(4.0)
    assert (first.status, first.forecast_role, first.games, first.live_games) == (
        SeasonWeekStatus.RECAPPED,
        "live",
        2,
        2,
    )
    assert (second.games, second.live_games) == (3, 0)


def test_season_to_date_pools_every_game_through_the_week(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)

    first, second = recap.weeks[0], recap.weeks[1]
    assert first.to_date == first.metrics
    assert second.to_date is not None
    # Pooled over the four decided games, not the mean of the two weekly values.
    assert second.to_date.win.model.brier == pytest.approx(
        (0.28**2 + 0.42**2 + 0.14**2 + 0.56**2) / 4
    )
    assert second.to_date.spread.model.mae == pytest.approx((2 + 6 + 1 + 7) / 4)
    assert second.to_date.spread.market.mae == pytest.approx((4 + 5 + 3 + 3) / 4)
    assert recap.all_games.metrics == second.to_date
    assert (recap.all_games.weeks, recap.all_games.games) == (2, 5)


def test_the_live_only_scope_keeps_live_rows(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)

    second = recap.weeks[1]
    assert second.live_metrics is not None
    assert second.live_metrics.win.model.count == 0
    assert second.live_metrics.win.model.brier is None
    assert second.live_to_date == recap.weeks[0].metrics
    assert recap.live_games.metrics == weeks[0][0].metrics
    assert (recap.live_games.weeks, recap.live_games.games) == (1, 2)


def test_paired_differences_carry_a_standard_error_and_interval(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)

    model = (0.28**2, 0.42**2, 0.14**2, 0.56**2)
    market = (0.38**2, 0.52**2, 0.26**2, 0.46**2)
    differences = [m - k for m, k in zip(model, market, strict=True)]
    error = statistics.stdev(differences) / math.sqrt(4)
    win = recap.all_games.paired.win
    assert win.count == 4
    assert win.mean == pytest.approx(statistics.fmean(differences))
    assert win.standard_error == pytest.approx(error)
    assert win.interval_low == pytest.approx(win.mean - INTERVAL_Z * error)
    assert win.interval_high == pytest.approx(win.mean + INTERVAL_Z * error)
    # |model| - |market|: 2-4, 6-5, 1-3, 7-3.
    assert recap.all_games.paired.spread.mean == pytest.approx(0.25)
    # |model| - |market|: 3-1, 4-2, 0-5, 2-6.
    assert recap.all_games.paired.total.mean == pytest.approx(-1.25)


def test_a_paired_difference_without_games_or_spread_is_empty(
    weeks: list[tuple[WeeklyRecap, DataFrame]], base_recap: WeeklyRecap
) -> None:
    recap = _season(weeks)
    assert recap.all_games.paired.projected_score.count == 0
    assert recap.all_games.paired.projected_score.mean is None

    single = _frame(WEEK_1[:1])
    one_game = _season([(_recap(base_recap, 1, "live", single), single)], finished=(1,))
    assert one_game.all_games.paired.win.count == 1
    assert one_game.all_games.paired.win.mean is not None
    assert one_game.all_games.paired.win.standard_error is None
    assert one_game.all_games.paired.win.interval_low is None


def test_calibration_uses_ten_fixed_bins_with_counts_and_the_helper_ece(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)

    model = recap.all_games.calibration.model
    market = recap.all_games.calibration.market
    assert len(model.bins) == 10
    assert [round(b.lower, 1) for b in model.bins] == [i / 10 for i in range(10)]
    # The tie is excluded: four decided games per side.
    assert (model.count, market.count) == (4, 4)
    assert [b.count for b in model.bins] == [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    assert [b.count for b in market.bins] == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    assert model.bins[0].mean_forecast is None
    assert model.bins[0].observed_rate is None
    assert model.bins[7].mean_forecast == pytest.approx(0.72)
    assert model.bins[7].observed_rate == 1.0
    assert model.ece == pytest.approx((0.28 + 0.42 + 0.14 + 0.56) / 4)
    assert model.ece == pytest.approx(
        expected_calibration_error(
            pd.Series([0.72, 0.42, 0.86, 0.56]), pd.Series([1.0, 0.0, 1.0, 0.0])
        )
    )
    live = recap.live_games.calibration.model
    assert live.count == 2
    assert sum(b.count for b in live.bins) == 2


def test_starter_accuracy_accumulates_over_the_season(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)

    starters = recap.all_games.starters
    assert (starters.slots, starters.scored, starters.not_recorded) == (10, 2, 8)
    assert (starters.assumed.correct, starters.assumed.count) == (1, 2)
    assert (starters.nflverse.correct, starters.carry_forward.correct) == (2, 0)
    assert recap.live_games.starters.assumed.accuracy == 1.0


def test_a_finished_week_without_a_recap_is_not_built(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)

    third = recap.weeks[2]
    assert third.week == 3
    assert third.status is SeasonWeekStatus.NOT_BUILT
    assert third.recap_id is None
    assert third.metrics is None and third.to_date is None and third.live_to_date is None
    assert recap.weekly_recap_ids == (f"{1:064x}", f"{2:064x}")
    assert recap.all_games.weeks == 2


def test_identity_tracks_the_weekly_recap_ids_and_excludes_built_at(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)
    later = build_season_recap(
        season=SEASON,
        recaps=weeks,
        finished_weeks=(1, 2, 3),
        built_at=BUILT_AT + timedelta(hours=1),
    )
    reselected = [weeks[0], (replace(weeks[1][0], recap_id="f" * 64), weeks[1][1])]

    assert later.season_recap_id == recap.season_recap_id
    assert recap.season_recap_id == season_recap_id(recap)
    assert _season(reselected).season_recap_id != recap.season_recap_id
    assert _season(weeks, finished=(1, 2)).season_recap_id != recap.season_recap_id


def test_the_manifest_round_trips_through_json(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    recap = _season(weeks)
    payload = json.loads(json.dumps(season_recap_payload(recap), allow_nan=False))

    assert season_recap_from_payload(payload) == recap


def test_a_manifest_whose_content_does_not_match_its_id_is_rejected(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    payload = season_recap_payload(_season(weeks))
    payload["season"] = "2027-2028"

    with pytest.raises(ValueError, match="ID does not match"):
        season_recap_from_payload(json.loads(json.dumps(payload)))


def test_a_manifest_with_an_unknown_field_is_rejected(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    payload = season_recap_payload(_season(weeks)) | {"extra": 1}

    with pytest.raises(ValueError, match="fields do not match"):
        season_recap_from_payload(json.loads(json.dumps(payload)))


def test_a_season_without_a_selected_recap_is_refused() -> None:
    with pytest.raises(WeeklyRecapRefusalError, match="No weekly recap is selected"):
        build_season_recap(season=SEASON, recaps=[], finished_weeks=(1,), built_at=BUILT_AT)


def test_recaps_of_another_season_or_a_repeated_week_are_rejected(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    other = [(replace(weeks[0][0], season="2025-2026"), weeks[0][1])]
    with pytest.raises(ValueError, match="covers 2025-2026"):
        _season(other)
    with pytest.raises(ValueError, match="Two weekly recaps"):
        _season([weeks[0], weeks[0]])


def test_a_recap_whose_metrics_do_not_match_its_frame_is_rejected(
    weeks: list[tuple[WeeklyRecap, DataFrame]],
) -> None:
    tampered = [(weeks[0][0], weeks[1][1]), weeks[1]]

    with pytest.raises(ValueError, match="do not match its own game frame"):
        _season(tampered)

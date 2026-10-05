# tests/unit/evaluation/test_weekly_recap.py

"""Tests for the pure weekly-recap builder (``DECISIONS.md`` D86)."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
import math

import pandas as pd
import pytest
from tests.fixtures.weekly_recap import (
    BUILT_AT,
    CARRIED_RUN,
    GAME_A,
    GAME_B,
    GAME_C,
    GAME_D,
    PRODUCT_RUN,
    SEASON,
    WEEK,
    RecapWeek,
    recap_week,
)

from gridiron_edge.evaluation.live_forecast_closeout import close_live_forecasts
from gridiron_edge.evaluation.market_agreement import MARGIN_STD
from gridiron_edge.evaluation.weekly_recap import (
    BuiltWeeklyRecap,
    WeeklyRecapRefusalError,
    build_weekly_recap,
    unplayed_game_ids,
    weekly_recap_from_payload,
    weekly_recap_id,
    weekly_recap_payload,
)

pytestmark = pytest.mark.unit


def _phi(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _build(inputs: RecapWeek | None = None, **overrides: object) -> BuiltWeeklyRecap:
    inputs = inputs or recap_week()
    arguments: dict[str, object] = {
        "season": SEASON,
        "week": WEEK,
        "schedule": inputs.schedule,
        "product": inputs.product,
        "forecast_events": inputs.events,
        "games": inputs.games,
        "player_game_logs": inputs.logs,
        "sidecar_captures": inputs.captures,
        "built_at": BUILT_AT,
    }
    return build_weekly_recap(**(arguments | overrides))  # type: ignore[arg-type]


def _row(built: BuiltWeeklyRecap, game_id: str) -> pd.Series:
    return built.games.set_index("game_id").loc[game_id]


def test_every_scheduled_game_is_a_row_in_kickoff_order() -> None:
    built = _build()

    assert built.games["game_id"].tolist() == [GAME_B, GAME_A, GAME_C, GAME_D]
    assert str(built.games["kickoff"].iloc[0]) == "2026-10-02 00:15:00+00:00"
    assert built.recap.counts.scheduled_games == 4
    assert built.recap.counts.in_product == 3
    assert built.recap.counts.not_in_product == 1


def test_a_fully_available_game_reads_home_minus_away_against_both_markets() -> None:
    row = _row(_build(), GAME_A)

    assert row["product_row_status"] == "in_product"
    assert not row["carried"]
    for family in ("win", "spread", "total", "projected_score"):
        assert row[f"{family}_status"] == "available"
    # model_spread -3 means home favored by 3.
    assert row["projected_home_margin"] == 3.0
    assert row["forecast_market_status"] == "available"
    assert row["forecast_home_margin_line"] == 3.0
    assert row["forecast_total_line"] == 44.5
    assert row["closing_home_margin"] == 2.5
    assert row["closing_home_win_prob"] == pytest.approx(_phi(2.5 / MARGIN_STD))
    assert row["closing_home_win_prob"] > 0.5
    # Implied scores: home (45.5 + 2.5) / 2, away (45.5 - 2.5) / 2.
    assert row["market_implied_home_score"] == 24.0
    assert row["market_implied_away_score"] == 21.5
    assert row["actual_margin"] == 7.0
    assert row["actual_total"] == 41.0
    assert bool(row["actual_home_win"])
    assert row["spread_model_error"] == -4.0
    assert row["spread_market_error"] == -4.5
    assert row["spread_closer"] == "model"
    assert row["total_model_error"] == 3.0
    assert row["total_market_error"] == 4.5
    assert row["win_model_brier"] == pytest.approx(0.16)
    assert row["win_market_brier"] == pytest.approx((1 - _phi(2.5 / MARGIN_STD)) ** 2)
    assert row["win_closer"] == "model"
    assert row["projected_home_score"] == 23.5
    assert row["projected_away_score"] == 20.5
    assert row["projected_score_model_abs_error"] == 2.0
    assert row["projected_score_market_abs_error"] == 2.25


def test_a_carried_row_records_its_own_run() -> None:
    built = _build()
    row = _row(built, GAME_B)

    assert row["carried"]
    assert row["forecast_run_id"] == CARRIED_RUN
    assert row["forecast_role"] == "live"
    assert built.recap.carried_run_ids == (CARRIED_RUN,)
    assert built.recap.product_run_id == PRODUCT_RUN
    assert built.recap.counts.carried_rows == 1


def test_a_tie_keeps_the_forecast_but_has_no_win_outcome() -> None:
    row = _row(_build(), GAME_B)

    assert row["win_status"] == "outcome_tie"
    assert row["home_win_prob"] == 0.55
    assert pd.isna(row["actual_home_win"])
    assert pd.isna(row["win_model_brier"])
    assert pd.isna(row["win_closer"])
    assert row["spread_status"] == "available"


def test_a_scheduled_game_missing_from_the_product_is_a_row_with_a_status() -> None:
    row = _row(_build(), GAME_C)

    assert row["product_row_status"] == "not_in_product"
    for family in ("win", "spread", "total", "projected_score"):
        assert row[f"{family}_status"] == "not_in_product"
    assert row["forecast_market_status"] == "not_in_product"
    assert pd.isna(row["carried"])
    assert pd.isna(row["home_win_prob"])
    assert pd.isna(row["spread_closer"])
    # The market side is still scored.
    assert row["spread_market_error"] == 2.0
    assert row["away_score"] == 30.0


def test_a_missing_forecast_and_missing_closing_lines_carry_statuses() -> None:
    row = _row(_build(), GAME_D)

    assert row["total_status"] == "forecast_missing"
    assert row["projected_score_status"] == "forecast_missing"
    assert pd.isna(row["model_total"])
    assert pd.isna(row["projected_home_score"])
    assert row["closing_spread_status"] == "closing_line_missing"
    assert row["closing_total_status"] == "closing_line_missing"
    assert pd.isna(row["closing_home_win_prob"])
    assert pd.isna(row["market_implied_home_score"])
    assert row["spread_model_error"] == -1.0
    assert pd.isna(row["spread_market_error"])
    assert pd.isna(row["spread_closer"])


def test_week_metrics_match_hand_values() -> None:
    metrics = _build().recap.metrics

    assert metrics.win.model.count == 2
    assert metrics.win.model.brier == pytest.approx((0.16 + 0.49) / 2)
    assert metrics.win.model.accuracy == 0.5
    assert metrics.win.market.count == 2
    assert metrics.win.market.brier == pytest.approx(
        ((1 - _phi(2.5 / MARGIN_STD)) ** 2 + _phi(-1 / MARGIN_STD) ** 2) / 2
    )
    assert (metrics.win.closer.model, metrics.win.closer.market) == (1, 0)
    assert metrics.spread.model.count == 3
    assert metrics.spread.model.mae == pytest.approx(2.0)
    assert metrics.spread.model.bias == pytest.approx(-2.0)
    assert metrics.spread.model.rmse == pytest.approx(math.sqrt(6.0))
    assert metrics.spread.market.count == 3
    assert metrics.spread.market.mae == pytest.approx(9.5 / 3)
    assert metrics.spread.market.bias == pytest.approx(-5.5 / 3)
    assert (metrics.spread.closer.model, metrics.spread.closer.market) == (2, 0)
    assert metrics.total.model.count == 2
    assert metrics.total.model.mae == pytest.approx(1.5)
    assert metrics.total.market.count == 3
    assert metrics.total.market.bias == pytest.approx((4.5 + 1.0 - 7.0) / 3)
    # Team scores: A and B for the model (four scores), A, B and C for the market (six).
    assert metrics.projected_score.model.count == 4
    assert metrics.projected_score.model.mae == pytest.approx(1.25)
    assert metrics.projected_score.model.bias == pytest.approx(0.75)
    assert metrics.projected_score.market.count == 6
    assert metrics.projected_score.market.mae == pytest.approx((4.5 + 0 + 2 + 1 + 4.5 + 2.5) / 6)
    assert (metrics.projected_score.closer.model, metrics.projected_score.closer.market) == (2, 0)


def test_model_metrics_equal_the_closeout() -> None:
    week = recap_week()
    metrics = _build(week).recap.metrics
    closeout = close_live_forecasts(
        product=week.product, forecast_events=week.events, games=week.games
    )

    assert metrics.win.model.count == closeout.win.evaluated_count
    assert metrics.win.model.brier == pytest.approx(closeout.win.brier, abs=1e-12)
    assert metrics.win.model.log_loss == pytest.approx(closeout.win.log_loss, abs=1e-12)
    assert metrics.win.model.accuracy == closeout.win.accuracy
    for ours, theirs in (
        (metrics.spread.model, closeout.spread),
        (metrics.total.model, closeout.total),
    ):
        assert ours.count == theirs.evaluated_count
        assert ours.mae == pytest.approx(theirs.mae, abs=1e-12)
        assert ours.rmse == pytest.approx(theirs.rmse, abs=1e-12)
        assert ours.bias == pytest.approx(theirs.bias, abs=1e-12)


def test_a_product_without_market_columns_reads_not_recorded() -> None:
    week = recap_week()
    product = week.product.drop(columns=["spread_line", "total_line", "ingested_at"])
    built = _build(replace(week, product=product))

    in_product = built.games.loc[built.games["product_row_status"] == "in_product"]
    assert set(in_product["forecast_market_status"]) == {"not_recorded"}
    assert in_product["forecast_home_margin_line"].isna().all()
    assert built.recap.metrics.spread.market.count == 3


def test_the_role_and_product_identity_are_recorded() -> None:
    recap = _build().recap

    assert recap.forecast_role == "live"
    assert recap.product_id == "weekly_product"
    assert recap.product_schema_version == 2
    assert recap.conventions.margin_std == MARGIN_STD


def test_identity_excludes_built_at_and_follows_content() -> None:
    first = _build().recap
    later = _build(built_at=BUILT_AT + timedelta(days=1)).recap
    week = recap_week()
    corrected = week.games.copy()
    corrected.loc[corrected["GAME_ID"] == GAME_C, "VEGAS_LINE"] = -1.5

    assert later.recap_id == first.recap_id
    assert later.built_at != first.built_at
    assert _build(replace(week, games=corrected)).recap.recap_id != first.recap_id
    assert weekly_recap_id(first) == first.recap_id


def test_the_manifest_round_trips_through_json() -> None:
    recap = _build().recap
    payload = json.loads(json.dumps(weekly_recap_payload(recap)))

    assert weekly_recap_from_payload(payload) == recap


def test_a_manifest_whose_content_does_not_match_its_id_is_rejected() -> None:
    payload = weekly_recap_payload(_build().recap)
    payload["product_id"] = "another_product"

    with pytest.raises(ValueError, match="does not match its content"):
        weekly_recap_from_payload(json.loads(json.dumps(payload)))


def test_a_manifest_with_an_unknown_field_is_rejected() -> None:
    payload = weekly_recap_payload(_build().recap)
    payload["extra"] = 1

    with pytest.raises(ValueError, match="fields do not match"):
        weekly_recap_from_payload(payload)


def test_an_unplayed_game_refuses_the_week() -> None:
    week = recap_week()
    schedule = week.schedule.copy()
    schedule.loc[schedule["game_id"] == GAME_D, ["away_score", "home_score", "result"]] = None

    assert unplayed_game_ids(schedule) == (GAME_D,)
    with pytest.raises(WeeklyRecapRefusalError, match=f"1 of 4 games have no result yet: {GAME_D}"):
        _build(replace(week, schedule=schedule))


def test_a_score_conflict_between_raw_and_cleaned_refuses_the_week() -> None:
    week = recap_week()
    games = week.games.copy()
    games.loc[games["GAME_ID"] == GAME_A, "HOME_SCORE"] = 21.0

    with pytest.raises(WeeklyRecapRefusalError, match=f"disagree on scores: {GAME_A}"):
        _build(replace(week, games=games))


def test_a_played_game_missing_from_the_cleaned_file_refuses_the_week() -> None:
    week = recap_week()
    games = week.games.loc[week.games["GAME_ID"] != GAME_C]

    with pytest.raises(WeeklyRecapRefusalError, match="not in the cleaned games file"):
        _build(replace(week, games=games))


def test_a_product_game_that_is_not_scheduled_refuses_the_week() -> None:
    week = recap_week()
    schedule = week.schedule.loc[week.schedule["game_id"] != GAME_D]

    with pytest.raises(WeeklyRecapRefusalError, match=f"not scheduled: {GAME_D}"):
        _build(replace(week, schedule=schedule))


def test_a_component_without_its_forecast_event_refuses_the_week() -> None:
    week = recap_week()
    events = week.events.loc[week.events["event_id"] != f"total-{GAME_A}"]

    with pytest.raises(WeeklyRecapRefusalError, match=f"no matching forecast event: {GAME_A}"):
        _build(replace(week, events=events))


def test_a_product_of_another_week_is_rejected() -> None:
    with pytest.raises(ValueError, match="not 2026-2027 week 5"):
        _build(week=5)

# tests/unit/evaluation/test_starting_qb_projection.py

"""Tests for the projected starting-QB resolver and its backtest."""

from __future__ import annotations

from datetime import date

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.starting_qb_projection import (
    CARRY_FORWARD,
    DEPTH_QB1,
    PRIMARY,
    VARIANT,
    RuleEvaluation,
    bar_verdict,
    build_team_games,
    clears_bar,
    error_breakdown,
    evaluate_rules,
    leakage_gap,
    score_rules,
    snapshot_timing,
)
from gridiron_edge.evaluation.starting_qb_projection_report import build_report

pytestmark = pytest.mark.unit


def _games() -> DataFrame:
    rows = []
    starters = {
        ("KC", 1): "A",
        ("KC", 2): "A",
        ("KC", 3): "B",
        ("DEN", 1): "X",
        ("DEN", 2): "X",
        ("DEN", 3): "X",
    }
    for week in (1, 2, 3):
        rows.append(
            {
                "game_id": f"2020_{week:02d}_DEN_KC",
                "season": 2020,
                "week": week,
                "game_type": "REG",
                "home_team": "KC",
                "away_team": "DEN",
                "home_qb_id": starters[("KC", week)],
                "away_qb_id": starters[("DEN", week)],
            }
        )
    rows.append({**rows[0], "game_id": "2020_19_DEN_KC", "week": 19, "game_type": "WC"})
    return DataFrame(rows)


def _chart(season: int, team: str, week: int, ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "season": season,
            "club_code": team,
            "week": float(week),
            "game_type": "REG",
            "formation": "Offense",
            "depth_position": "QB",
            "depth_team": str(rank),
            "gsis_id": player,
        }
        for rank, player in enumerate(ids, start=1)
    ]


def _sources() -> tuple[dict[int, DataFrame], dict[int, DataFrame]]:
    chart_rows: list[dict[str, object]] = []
    for week in (1, 2, 3):
        chart_rows += _chart(2020, "KC", week, ["A", "B"])
        chart_rows += _chart(2020, "DEN", week, ["X", "Y"])
    chart_rows += [
        {**_chart(2020, "KC", 1, ["Z"])[0], "formation": "Defense"},
        {**_chart(2020, "KC", 1, ["Z"])[0], "depth_position": "RB"},
    ]
    injuries = DataFrame(
        [
            {
                "season": 2020,
                "game_type": "REG",
                "team": "KC",
                "week": 3.0,
                "gsis_id": "A",
                "report_status": "Out",
            }
        ]
    )
    return {2020: DataFrame(chart_rows)}, {2020: injuries}


def test_team_games_carry_the_previous_starter_chart_and_status() -> None:
    charts, injuries = _sources()

    frame = build_team_games(_games(), charts, injuries)

    kc = frame[frame["team"] == "KC"].set_index("week")
    assert len(frame) == 6
    assert kc.loc[1, "prev"] is None or pd.isna(kc.loc[1, "prev"])
    assert kc.loc[3, "prev"] == "A"
    assert kc.loc[3, "actual"] == "B"
    assert kc.loc[3, "chart"] == [(1, "A"), (2, "B")]
    assert kc.loc[3, "statuses"] == {"A": "Out"}
    assert kc.loc[1, "statuses"] == {}


def test_scoring_requires_a_chart_and_a_previous_starter_in_the_scored_seasons() -> None:
    charts, injuries = _sources()
    frame = build_team_games(_games(), charts, injuries)

    scored = score_rules(frame.assign(season=2008))
    assert scored.empty

    scored = score_rules(frame)
    assert len(scored) == 4  # weeks 2 and 3 for both teams; week 1 has no previous starter
    kc3 = scored[(scored["team"] == "KC") & (scored["week"] == 3)].iloc[0]
    assert not kc3[f"{CARRY_FORWARD}_correct"]  # previous starter A, actual B
    assert not kc3[f"{DEPTH_QB1}_correct"]  # chart QB1 is A
    assert kc3[f"{PRIMARY}_correct"]  # A is out, so B
    assert kc3[f"{VARIANT}_correct"]
    assert kc3["primary_pick"] == "B"


def _scored_rows(correct: list[tuple[int, int, bool, bool]]) -> DataFrame:
    """Rows of (season, week, carry-forward correct, primary correct)."""
    return DataFrame(
        {
            "season": [r[0] for r in correct],
            "week": [r[1] for r in correct],
            f"{CARRY_FORWARD}_correct": [r[2] for r in correct],
            f"{DEPTH_QB1}_correct": [r[2] for r in correct],
            f"{PRIMARY}_correct": [r[3] for r in correct],
            f"{VARIANT}_correct": [r[3] for r in correct],
        }
    )


def _synthetic(primary_wins: int, carry_wins: int, seasons: int = 5) -> DataFrame:
    rows: list[tuple[int, int, bool, bool]] = []
    for season in range(2010, 2010 + seasons):
        for index in range(100):
            rows.append((season, 5, index < carry_wins, index < primary_wins))
    return _scored_rows(rows)


def test_a_rule_that_is_clearly_better_clears_the_bar() -> None:
    scored = _synthetic(primary_wins=96, carry_wins=88, seasons=8)

    primary = next(r for r in evaluate_rules(scored) if r.rule == PRIMARY)

    assert primary.gain > 0
    assert clears_bar(primary)


def test_bar_requires_the_gain_the_interval_and_the_season_share() -> None:
    def result(gain: float, low: float, won: float) -> RuleEvaluation:
        return RuleEvaluation(PRIMARY, 0.9, gain, low, gain + 0.01, won)

    assert clears_bar(result(0.02, 0.005, 0.7))
    assert not clears_bar(result(0.0192, 0.014, 0.94))
    assert not clears_bar(result(0.03, -0.001, 0.9))
    assert not clears_bar(result(0.03, 0.01, 0.69))


def test_bootstrap_interval_is_reproducible_and_contains_the_gain() -> None:
    scored = _synthetic(primary_wins=94, carry_wins=88)

    first = evaluate_rules(scored)
    second = evaluate_rules(scored)
    primary = next(r for r in first if r.rule == PRIMARY)

    assert first == second
    assert primary.gain_low <= primary.gain <= primary.gain_high
    base = next(r for r in first if r.rule == CARRY_FORWARD)
    assert (base.gain, base.gain_low, base.gain_high) == (0.0, 0.0, 0.0)


def test_guard_fires_and_week_one_is_excluded_from_the_decision() -> None:
    later = [(2010 + s, 5, i < 80, i < 92) for s in range(8) for i in range(100)]
    week_one = [(2010 + s, 1, i < 40, True) for s in range(8) for i in range(100)]
    scored = _scored_rows(later + week_one)

    verdict = bar_verdict(scored)

    assert leakage_gap(scored) == pytest.approx(0.08)
    assert verdict.flagged
    assert verdict.all_weeks.gain > verdict.later_weeks.gain
    assert verdict.later_weeks.gain == pytest.approx(0.12)
    assert verdict.passes == clears_bar(verdict.later_weeks)


def test_guard_stays_quiet_when_week_one_matches_later_weeks() -> None:
    rows = [(2010 + s, w, i < 80, i < 90) for s in range(8) for w in (1, 2) for i in range(100)]
    verdict = bar_verdict(_scored_rows(rows))

    assert not verdict.flagged
    assert verdict.passes == clears_bar(verdict.all_weeks)


def test_error_breakdown_describes_the_misses() -> None:
    scored = DataFrame(
        {
            "season": [2010, 2010],
            "week": [1, 5],
            "actual": ["B", "Z"],
            "prev": ["B", "A"],
            "chart": [[(1, "A"), (2, "B")], [(1, "A"), (2, "C")]],
            f"{PRIMARY}_correct": [False, False],
        }
    )

    breakdown = error_breakdown(scored)

    assert breakdown["misses"] == 2.0
    assert breakdown["share_actual_in_top_three"] == 0.5
    assert breakdown["share_actual_was_carry_forward"] == 0.5
    assert breakdown["share_week_1"] == 0.5


def _snapshots() -> DataFrame:
    def rows(when: str, ids: list[str]) -> list[dict[str, object]]:
        return [
            {"dt": when, "team": "KC", "pos_abb": "QB", "pos_rank": rank, "gsis_id": player}
            for rank, player in enumerate(ids, start=1)
        ]

    return DataFrame(
        rows("2025-09-01T08:00:00Z", ["A", "B"])
        + rows("2025-09-04T08:00:00Z", ["B", "A"])
        + [
            {
                "dt": "2025-09-01T08:00:00Z",
                "team": "KC",
                "pos_abb": "RB",
                "pos_rank": 1,
                "gsis_id": "R",
            }
        ]
    )


def _timing_games() -> DataFrame:
    return DataFrame(
        {
            "game_id": ["g1"],
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "gameday": ["2025-09-07"],
            "gametime": ["13:00"],
            "home_team": ["KC"],
            "away_team": ["DEN"],
            "home_qb_id": ["B"],
            "away_qb_id": ["X"],
        }
    )


def test_snapshot_timing_only_sees_charts_before_the_lead_time() -> None:
    injuries = DataFrame(columns=["game_type", "team", "week", "gsis_id", "report_status"])

    early = snapshot_timing(_timing_games(), _snapshots(), injuries, season=2025, lead_hours=[96])
    late = snapshot_timing(_timing_games(), _snapshots(), injuries, season=2025, lead_hours=[24])

    # Kickoff 17:00 UTC on 2025-09-07: 96 hours earlier sees only the first chart (A first).
    assert early.loc[0, "chart_only_accuracy"] == 0.0
    # 24 hours earlier sees the second chart (B first), the actual starter.
    assert late.loc[0, "chart_only_accuracy"] == 1.0
    # DEN has no snapshots, so only KC is scored.
    assert late.loc[0, "team_games"] == 1
    assert not late.loc[0, "report_time_limited"]


def test_snapshot_timing_limits_reports_by_modification_time_when_present() -> None:
    injuries = DataFrame(
        {
            "game_type": ["REG"],
            "team": ["KC"],
            "week": [1],
            "gsis_id": ["B"],
            "report_status": ["Out"],
            "date_modified": ["2025-09-06T12:00:00Z"],
        }
    )

    early = snapshot_timing(_timing_games(), _snapshots(), injuries, season=2025, lead_hours=[96])
    late = snapshot_timing(_timing_games(), _snapshots(), injuries, season=2025, lead_hours=[24])

    assert late.loc[0, "report_time_limited"]
    # Kickoff is 17:00Z on 2025-09-07. At 24 hours (cutoff 2025-09-06 17:00Z) the report filed
    # at 12:00Z is visible and removes B, so A is named.
    assert late.loc[0, "chart_and_report_accuracy"] == 0.0
    assert late.loc[0, "chart_only_accuracy"] == 1.0
    # At 96 hours it has not been filed yet.
    assert early.loc[0, "chart_and_report_accuracy"] == early.loc[0, "chart_only_accuracy"]


def test_report_states_the_verdict_and_both_readings() -> None:
    rows = [(2010 + s, 5, i < 80, i < 92) for s in range(8) for i in range(100)]
    rows += [(2010 + s, 1, i < 40, True) for s in range(8) for i in range(100)]
    scored = _scored_rows(rows).assign(
        actual="B", prev="A", chart=[[(1, "A")]] * len(rows), statuses=[{}] * len(rows)
    )
    timing = {
        "All weeks": DataFrame(
            {
                "lead_hours": [24],
                "scored": [10],
                "chart_only_accuracy": [0.9],
                "chart_and_report_accuracy": [0.95],
            }
        )
    }

    text = build_report(scored, bar_verdict(scored), timing, generated_on=date(2026, 10, 3))

    assert "**fired**" in text
    assert "Decision reading: weeks after week 1." in text
    assert "Bar cleared" in text or "Bar not cleared" in text
    assert "Generated 2026-10-03" in text
    assert "### All weeks" in text

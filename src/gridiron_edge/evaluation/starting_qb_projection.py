# src/gridiron_edge/evaluation/starting_qb_projection.py

"""Projected starting-QB resolver and its backtest against carry-forward.

``DECISIONS.md`` D65 names each team's previous starter as the live starter.
This module scores a rules-based alternative built from the depth chart and the
weekly availability status against the quarterback who actually started.

The resolver is a pure function that returns the ordered candidate list with the
inputs it used, so later work (hot-swapping a quarterback, a scenario engine)
can reuse the list. Nothing here reads or writes anything but its arguments.

Public API
----------
Candidate                 dataclass  - one chart quarterback with its status
ordered_candidates        chart + statuses -> candidate list
project_starter           candidate list -> projected starter id
build_team_games          sources -> one row per team-game
score_rules               team-game frame -> per-rule correctness
evaluate_rules            correctness -> accuracies, intervals, bar verdict
snapshot_timing           2025 snapshots -> accuracy by lead time
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

CARRY_FORWARD: Final[str] = "carry_forward"
DEPTH_QB1: Final[str] = "depth_qb1"
PRIMARY: Final[str] = "primary"
VARIANT: Final[str] = "variant"
RULES: Final[tuple[str, ...]] = (CARRY_FORWARD, DEPTH_QB1, PRIMARY, VARIANT)

#: Statuses that remove a quarterback from consideration in the primary rule.
UNAVAILABLE_PRIMARY: Final[frozenset[str]] = frozenset({"Out", "Doubtful"})
#: The variant also skips Questionable; reported for information only.
UNAVAILABLE_VARIANT: Final[frozenset[str]] = UNAVAILABLE_PRIMARY | {"Questionable"}

#: Bar a rule must clear against carry-forward to justify a feature unit.
MIN_GAIN: Final[float] = 0.02
MIN_SEASONS_WON: Final[float] = 0.70
BOOTSTRAP_DRAWS: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 20261003
INTERVAL: Final[tuple[float, float]] = (5.0, 95.0)

FIRST_SCORED_SEASON: Final[int] = 2009
LAST_OLD_FORMAT_SEASON: Final[int] = 2024
LEAKAGE_GAP: Final[float] = 0.05

_CHART_WINDOW: Final[int] = 3


@dataclass(frozen=True)
class Candidate:
    """One chart quarterback with the inputs the resolver used.

    Attributes:
        gsis_id: Player identifier.
        chart_rank: Depth chart rank (1 is the top quarterback).
        status: Reported game status, or ``None`` when not on the report.
        available: Whether the status leaves the player eligible.
    """

    gsis_id: str
    chart_rank: int
    status: str | None
    available: bool


def ordered_candidates(
    chart: Sequence[tuple[int, str]],
    statuses: Mapping[str, str | None],
    unavailable: frozenset[str],
) -> list[Candidate]:
    """Order chart quarterbacks by rank and attach their availability.

    Args:
        chart: ``(chart_rank, gsis_id)`` pairs for one team and week.
        statuses: Reported game status by player id.
        unavailable: Statuses that make a player ineligible.

    Returns:
        Candidates by ascending rank; a repeated player keeps its best rank and
        ties keep their input order.
    """
    seen: set[str] = set()
    candidates: list[Candidate] = []
    for rank, player_id in sorted(chart, key=lambda item: item[0]):
        if player_id in seen:
            continue
        seen.add(player_id)
        status: str | None = statuses.get(player_id)
        candidates.append(Candidate(player_id, rank, status, status not in unavailable))
    return candidates


def project_starter(candidates: Sequence[Candidate]) -> str | None:
    """Return the first available candidate, else the top of the chart, else ``None``."""
    for candidate in candidates:
        if candidate.available:
            return candidate.gsis_id
    return candidates[0].gsis_id if candidates else None


def _chart_lookup(depth_charts: Mapping[int, DataFrame]) -> dict[tuple[int, str, int], list]:
    """Index old-format regular-season offensive quarterbacks by team and week."""
    lookup: dict[tuple[int, str, int], list] = {}
    for season, frame in depth_charts.items():
        if season > LAST_OLD_FORMAT_SEASON or "depth_position" not in frame:
            continue
        rows = frame[
            (frame["game_type"] == "REG")
            & (frame["formation"] == "Offense")
            & (frame["depth_position"] == "QB")
            & frame["gsis_id"].notna()
        ]
        for team, week, rank, player in zip(
            rows["club_code"],
            rows["week"].astype(int),
            pd.to_numeric(rows["depth_team"]).astype(int),
            rows["gsis_id"],
            strict=True,
        ):
            lookup.setdefault((season, team, week), []).append((rank, player))
    return lookup


def _status_lookup(injuries: Mapping[int, DataFrame]) -> dict[tuple[int, str, int], dict]:
    """Index regular-season game statuses by team and week."""
    lookup: dict[tuple[int, str, int], dict] = {}
    for season, frame in injuries.items():
        rows = frame[(frame["game_type"] == "REG") & frame["gsis_id"].notna()]
        for team, week, player, status in zip(
            rows["team"],
            rows["week"].astype(int),
            rows["gsis_id"],
            rows["report_status"],
            strict=True,
        ):
            lookup.setdefault((season, team, week), {})[player] = status
    return lookup


def build_team_games(
    games: DataFrame,
    depth_charts: Mapping[int, DataFrame],
    injuries: Mapping[int, DataFrame],
) -> DataFrame:
    """Build one row per regular-season team-game with its starter and chart.

    Args:
        games: Raw nflverse games with ``home_qb_id`` and ``away_qb_id``.
        depth_charts: Old-format depth charts keyed by season.
        injuries: Injury reports keyed by season.

    Returns:
        A frame with ``season``, ``week``, ``team``, ``actual`` (the starter),
        ``prev`` (the carry-forward starter), ``chart`` and ``statuses``.
    """
    regular = games[games["game_type"] == "REG"]
    sides: list[DataFrame] = []
    for side in ("home", "away"):
        part = regular[["game_id", "season", "week", f"{side}_team", f"{side}_qb_id"]].copy()
        part.columns = pd.Index(["game_id", "season", "week", "team", "actual"])
        sides.append(part)
    frame = pd.concat(sides, ignore_index=True).dropna(subset=["actual"])
    frame = frame.sort_values(["team", "season", "week"], kind="stable").reset_index(drop=True)
    frame["prev"] = frame.groupby("team")["actual"].shift(1)

    charts = _chart_lookup(depth_charts)
    statuses = _status_lookup(injuries)
    keys = list(zip(frame["season"], frame["team"], frame["week"], strict=True))
    frame["chart"] = pd.Series(
        [charts.get(key, []) for key in keys], index=frame.index, dtype=object
    )
    frame["statuses"] = pd.Series(
        [statuses.get(key, {}) for key in keys], index=frame.index, dtype=object
    )
    return frame


def _pick(row: Series, rule: str) -> str | None:
    """Return the quarterback a rule names for one team-game."""
    if rule == CARRY_FORWARD:
        return row["prev"] if isinstance(row["prev"], str) else None
    if rule == DEPTH_QB1:
        ordered = ordered_candidates(row["chart"], {}, frozenset())
        return ordered[0].gsis_id if ordered else None
    unavailable = UNAVAILABLE_PRIMARY if rule == PRIMARY else UNAVAILABLE_VARIANT
    return project_starter(ordered_candidates(row["chart"], row["statuses"], unavailable))


def score_rules(team_games: DataFrame) -> DataFrame:
    """Score every rule on the scored team-games.

    Scored rows are regular-season games from 2009 through 2024 that have a depth
    chart and a previous starter.

    Returns:
        The scored rows with one boolean ``<rule>_correct`` column per rule and a
        ``picks`` column recording the primary rule's pick.
    """
    scored = team_games[
        team_games["season"].between(FIRST_SCORED_SEASON, LAST_OLD_FORMAT_SEASON)
        & team_games["chart"].map(bool)
        & team_games["prev"].notna()
    ].copy()
    if scored.empty:
        for rule in RULES:
            scored[f"{rule}_correct"] = pd.Series(dtype=bool)
        scored["primary_pick"] = pd.Series(dtype=object)
        return scored
    for rule in RULES:
        picks = pd.Series(
            [_pick(row, rule) for _, row in scored.iterrows()], index=scored.index, dtype=object
        )
        scored[f"{rule}_correct"] = (picks == scored["actual"]).to_numpy()
        if rule == PRIMARY:
            scored["primary_pick"] = picks
    return scored


@dataclass(frozen=True)
class RuleEvaluation:
    """Accuracy and the paired comparison of one rule with carry-forward."""

    rule: str
    accuracy: float
    gain: float
    gain_low: float
    gain_high: float
    seasons_won: float


def _bootstrap_gain(correct: Series, baseline: Series, seasons: Series) -> tuple[float, float]:
    """Season-stratified paired bootstrap interval of the accuracy gain."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diff = correct.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    groups = [np.flatnonzero(seasons.to_numpy() == s) for s in sorted(seasons.unique())]
    totals = np.zeros(BOOTSTRAP_DRAWS)
    for indices in groups:
        draws = rng.integers(0, len(indices), size=(BOOTSTRAP_DRAWS, len(indices)))
        totals += diff[indices][draws].sum(axis=1)
    means = totals / len(diff)
    low, high = np.percentile(means, INTERVAL)
    return float(low), float(high)


def evaluate_rules(scored: DataFrame) -> list[RuleEvaluation]:
    """Compare every rule with carry-forward on the scored rows."""
    baseline = scored[f"{CARRY_FORWARD}_correct"]
    by_season = scored.groupby("season")
    results: list[RuleEvaluation] = []
    for rule in RULES:
        correct = scored[f"{rule}_correct"]
        low, high = _bootstrap_gain(correct, baseline, scored["season"])
        season_gain = (
            by_season[f"{rule}_correct"].mean() - by_season[f"{CARRY_FORWARD}_correct"].mean()
        )
        results.append(
            RuleEvaluation(
                rule=rule,
                accuracy=float(correct.mean()),
                gain=float(correct.mean() - baseline.mean()),
                gain_low=low,
                gain_high=high,
                seasons_won=float((season_gain > 0).mean()),
            )
        )
    return results


def clears_bar(result: RuleEvaluation) -> bool:
    """Whether a rule clears the pre-declared gain, interval and season bar."""
    return result.gain >= MIN_GAIN and result.gain_low > 0 and result.seasons_won >= MIN_SEASONS_WON


@dataclass(frozen=True)
class BarVerdict:
    """The bar applied to the primary rule, with the leakage guard.

    Attributes:
        week_1_gap: Week-1 minus later-week accuracy of the primary rule.
        flagged: Whether the gap exceeds ``LEAKAGE_GAP``; a flagged chart is
            excluded from the bar and both readings are reported.
        all_weeks: The primary rule on every scored week.
        later_weeks: The primary rule on weeks after week 1.
        passes: The bar on weeks after week 1 when flagged, else on every week.
    """

    week_1_gap: float
    flagged: bool
    all_weeks: RuleEvaluation
    later_weeks: RuleEvaluation
    passes: bool


def bar_verdict(scored: DataFrame) -> BarVerdict:
    """Apply the pre-declared bar to the primary rule.

    When the week-1 leakage guard fires, week 1 is excluded from the decision
    and both readings are reported (``PLAN.md`` Unit 9).
    """
    gap = leakage_gap(scored)
    all_weeks = next(r for r in evaluate_rules(scored) if r.rule == PRIMARY)
    later_weeks = next(r for r in evaluate_rules(scored[scored["week"] > 1]) if r.rule == PRIMARY)
    flagged = gap > LEAKAGE_GAP
    return BarVerdict(
        week_1_gap=gap,
        flagged=flagged,
        all_weeks=all_weeks,
        later_weeks=later_weeks,
        passes=clears_bar(later_weeks if flagged else all_weeks),
    )


def leakage_gap(scored: DataFrame) -> float:
    """Return week-1 minus later-week accuracy of the primary rule."""
    correct = scored[f"{PRIMARY}_correct"]
    return float(correct[scored["week"] == 1].mean() - correct[scored["week"] > 1].mean())


def error_breakdown(scored: DataFrame) -> dict[str, float]:
    """Describe the primary rule's misses."""
    misses = scored[~scored[f"{PRIMARY}_correct"]]
    if misses.empty:
        return {
            "misses": 0.0,
            "share_actual_in_top_three": 0.0,
            "share_actual_was_carry_forward": 0.0,
            "share_week_1": 0.0,
        }

    def in_window(row: Series) -> bool:
        ids = [p for _, p in sorted(row["chart"], key=lambda item: item[0])[:_CHART_WINDOW]]
        return row["actual"] in ids

    return {
        "misses": float(len(misses)),
        "share_actual_in_top_three": float(misses.apply(in_window, axis=1).mean()),
        "share_actual_was_carry_forward": float((misses["actual"] == misses["prev"]).mean()),
        "share_week_1": float((misses["week"] == 1).mean()),
    }


def snapshot_timing(
    games: DataFrame,
    snapshots: DataFrame,
    injuries: DataFrame,
    *,
    season: int,
    lead_hours: Sequence[int],
    weeks: Sequence[int] | None = None,
) -> DataFrame:
    """Score the resolver from timestamped chart snapshots at fixed lead times.

    For each regular-season team-game, the latest chart snapshot on or before
    ``kickoff - lead`` is used, so no later chart is visible. Two readings are
    scored: the chart alone (fully as-of), and the chart plus the injury report.
    Injury files without ``date_modified`` (2025 onward) cannot be restricted in
    time, so their report is the whole week's final report and that reading is
    optimistic about what was known at the lead time.

    Args:
        games: Raw nflverse games with ``gameday`` and ``gametime`` (US Eastern).
        snapshots: Depth chart in the timestamped format for ``season``.
        injuries: Injury reports for ``season``.
        season: Season to score.
        lead_hours: Hours before kickoff at which to take the information.
        weeks: Weeks to score; every week when ``None``.

    Returns:
        One row per lead time with the game counts, both accuracies, and the
        share of games with no snapshot yet.
    """
    regular = games[(games["game_type"] == "REG") & (games["season"] == season)]
    if weeks is not None:
        regular = regular[regular["week"].isin(list(weeks))]
    kickoff = (
        pd.to_datetime(regular["gameday"].astype(str) + " " + regular["gametime"].astype(str))
        .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    qbs = snapshots[snapshots["pos_abb"] == "QB"].copy()
    qbs["when"] = pd.to_datetime(qbs["dt"], utc=True)
    reports = injuries[injuries["game_type"] == "REG"].copy()
    time_limited = "date_modified" in reports.columns
    if time_limited:
        reports["when"] = pd.to_datetime(reports["date_modified"], utc=True)
    by_team = dict(iter(qbs.groupby("team")))

    rows: list[dict[str, float | bool]] = []
    for hours in lead_hours:
        chart_correct = report_correct = missing = total = 0
        for (_, game), start in zip(regular.iterrows(), kickoff, strict=True):
            if pd.isna(start):
                continue
            cutoff = start - pd.Timedelta(hours=hours)
            for side in ("home", "away"):
                actual = game[f"{side}_qb_id"]
                team = game[f"{side}_team"]
                if not isinstance(actual, str) or team not in by_team:
                    continue
                total += 1
                frame = by_team[team]
                visible = frame[frame["when"] <= cutoff]
                if visible.empty:
                    missing += 1
                    continue
                latest = visible[visible["when"] == visible["when"].max()]
                chart = [
                    (int(rank), player)
                    for rank, player in zip(latest["pos_rank"], latest["gsis_id"], strict=True)
                    if isinstance(player, str)
                ]
                seen = reports[
                    (reports["team"] == team) & (reports["week"].astype(int) == int(game["week"]))
                ]
                if time_limited:
                    seen = seen[seen["when"] <= cutoff]
                statuses = dict(zip(seen["gsis_id"], seen["report_status"], strict=True))
                chart_pick = project_starter(ordered_candidates(chart, {}, UNAVAILABLE_PRIMARY))
                report_pick = project_starter(
                    ordered_candidates(chart, statuses, UNAVAILABLE_PRIMARY)
                )
                chart_correct += int(chart_pick == actual)
                report_correct += int(report_pick == actual)
        scored_games = total - missing
        rows.append(
            {
                "lead_hours": hours,
                "team_games": total,
                "scored": scored_games,
                "chart_only_accuracy": chart_correct / scored_games if scored_games else np.nan,
                "chart_and_report_accuracy": (
                    report_correct / scored_games if scored_games else np.nan
                ),
                "report_time_limited": time_limited,
                "share_without_snapshot": missing / total if total else np.nan,
            }
        )
    return DataFrame(rows)

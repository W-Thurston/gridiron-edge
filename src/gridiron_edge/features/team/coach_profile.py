# src/gridiron_edge/features/team/coach_profile.py

"""Canonical Away/Home head-coach profile features.

Head-coach identity, unlike starting-QB identity, is known for every
scheduled game before it is played - a season's coaching staff is set in
the offseason, so nflverse's ``home_coach``/``away_coach`` fields are
populated even for not-yet-played games (see ``DECISIONS.md``). This
needs no carry-forward policy: every metric here reads the QB/coach
identity sidecar's own value directly.

Career win % is recomputed fresh from completed-game results (mirroring
``features/team/record.py``'s pattern, but keyed on coach identity across
any team they've coached - a career, not same-season, aggregation).
Tenure and the new-coaching-staff flag are season-level: computed from
consecutive-season (team, coach) continuity in the sidecar, which also
means an interim coach who takes over mid-season (and therefore has no
prior-season history under that team) correctly gets tenure 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.models.game_prediction.game_schema import (
    away_feature_name,
    home_feature_name,
)

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Prior for a known head coach with no prior games as a head coach in the data
# (a first-time coach). Win share of first-time coaches' debut games, seasons
# 2000-2022 only (1999 debuts are left-censored, 2023+ is holdout): 0.4055
# over 127 debuts. See DECISIONS.md D73.
DEBUT_HC_WIN_PCT: Final[float] = 0.4055

COACH_PROFILE_METRICS: Final[tuple[str, ...]] = (
    "hc_win_pct_career",
    "hc_tenure",
)

_HOME_AWAY_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
)

_SIDECAR_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_COACH",
    "HOME_COACH",
)

_GAMES_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_SCORE",
    "HOME_SCORE",
)

_AWAY_METRIC_COLS: Final[list[str]] = [
    away_feature_name(metric.upper()) for metric in COACH_PROFILE_METRICS
]
_HOME_METRIC_COLS: Final[list[str]] = [
    home_feature_name(metric.upper()) for metric in COACH_PROFILE_METRICS
]
_AWAY_NEW_STAFF_FLAG: Final[str] = away_feature_name("NEW_COACHING_STAFF_FLAG")
_HOME_NEW_STAFF_FLAG: Final[str] = home_feature_name("NEW_COACHING_STAFF_FLAG")


def _require_columns(frame: DataFrame, required: tuple[str, ...], *, label: str) -> None:
    """Require an established input schema."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _season_numbers(year_text: Series) -> Series:
    """Convert canonical season labels ("2024-2025") to starting-season ints."""
    if year_text.isna().any():
        raise ValueError("YEAR must not contain nulls.")

    stripped = year_text.astype(str).str.strip()
    if stripped.eq("").any():
        raise ValueError("YEAR must not contain empty values.")

    season_text = stripped.str.split("-", n=1).str[0]
    try:
        return season_text.astype(int)
    except ValueError as exc:
        raise ValueError("YEAR must begin with a numeric season.") from exc


def _build_coach_result_history(sidecar: DataFrame, games: DataFrame) -> DataFrame:
    """One row per (coach, completed game) with RESULT in {1.0, 0.5, 0.0}."""
    _require_columns(sidecar, _SIDECAR_REQUIRED_COLUMNS, label="QB/coach sidecar")
    _require_columns(games, _GAMES_REQUIRED_COLUMNS, label="Historical games")

    joined = sidecar.loc[:, ["GAME_ID", "AWAY_COACH", "HOME_COACH"]].merge(
        games.loc[:, list(_GAMES_REQUIRED_COLUMNS)],
        on="GAME_ID",
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        return DataFrame(columns=["COACH", "SEASON", "WEEK_NUM", "GAME_ID", "RESULT"])

    away_won = joined["AWAY_SCORE"] > joined["HOME_SCORE"]
    home_won = joined["HOME_SCORE"] > joined["AWAY_SCORE"]

    away_results = joined.loc[:, ["GAME_ID", "YEAR", "WEEK_NUM", "AWAY_COACH"]].rename(
        columns={"AWAY_COACH": "COACH"}
    )
    away_results["RESULT"] = np.select([away_won, home_won], [1.0, 0.0], default=0.5)

    home_results = joined.loc[:, ["GAME_ID", "YEAR", "WEEK_NUM", "HOME_COACH"]].rename(
        columns={"HOME_COACH": "COACH"}
    )
    home_results["RESULT"] = np.select([home_won, away_won], [1.0, 0.0], default=0.5)

    results = pd.concat([away_results, home_results], ignore_index=True)
    results = results.loc[results["COACH"].notna(), :].copy()
    results["SEASON"] = _season_numbers(results["YEAR"])

    return results.loc[:, ["COACH", "SEASON", "WEEK_NUM", "GAME_ID", "RESULT"]].sort_values(
        ["COACH", "SEASON", "WEEK_NUM", "GAME_ID"],
        kind="stable",
        ignore_index=True,
    )


def _hc_win_pct_entering(
    history: DataFrame,
    *,
    coach: object,
    season: int,
    week: int,
) -> float:
    """Return career win % for one coach entering one game, or NaN with no prior games."""
    # pyrefly: ignore [no-matching-overload]
    if pd.isna(coach):
        return float("nan")

    prior = history.loc[
        (history["COACH"] == coach)
        & (
            (history["SEASON"] < season)
            | ((history["SEASON"] == season) & (history["WEEK_NUM"] < week))
        ),
        "RESULT",
    ]
    if prior.empty:
        return float("nan")

    return float(prior.sum()) / float(len(prior))


def _build_target_hc_win_pct(history: DataFrame, targets: DataFrame) -> DataFrame:
    """Compute career win % for explicit (coach, season, week) targets."""
    normalized_targets = targets.drop_duplicates(
        subset=["COACH", "SEASON", "WEEK_NUM"],
        keep="first",
        ignore_index=True,
    )

    rows: list[dict[str, object]] = [
        {
            "COACH": target.COACH,
            "SEASON": target.SEASON,
            "WEEK_NUM": target.WEEK_NUM,
            "HC_WIN_PCT_CAREER": _hc_win_pct_entering(
                history,
                coach=target.COACH,
                # pyrefly: ignore [bad-argument-type]
                season=int(target.SEASON),
                # pyrefly: ignore [bad-argument-type]
                week=int(target.WEEK_NUM),
            ),
        }
        for target in normalized_targets.itertuples(index=False)
    ]

    return DataFrame(rows, columns=["COACH", "SEASON", "WEEK_NUM", "HC_WIN_PCT_CAREER"])


def _known_team_coach_seasons(sidecar: DataFrame) -> set[tuple[str, object, int]]:
    """Return every (team, coach, season) triple with at least one sidecar row."""
    away = sidecar.loc[:, ["YEAR", "AWAY_TEAM", "AWAY_COACH"]].rename(
        columns={"AWAY_TEAM": "TEAM", "AWAY_COACH": "COACH"}
    )
    home = sidecar.loc[:, ["YEAR", "HOME_TEAM", "HOME_COACH"]].rename(
        columns={"HOME_TEAM": "TEAM", "HOME_COACH": "COACH"}
    )
    combined = pd.concat([away, home], ignore_index=True)
    combined = combined.loc[combined["COACH"].notna(), :].copy()
    combined["SEASON"] = _season_numbers(combined["YEAR"])

    return set(
        zip(
            combined["TEAM"].tolist(),
            combined["COACH"].tolist(),
            combined["SEASON"].tolist(),
            strict=True,
        )
    )


def _tenure_entering(
    known_triples: set[tuple[str, object, int]],
    *,
    team: str,
    coach: object,
    season: int,
) -> int | float:
    """Return consecutive seasons (including this one) this coach has led this team."""
    # pyrefly: ignore [no-matching-overload]
    if pd.isna(coach):
        return float("nan")

    tenure = 0
    probe_season = season
    while (team, coach, probe_season) in known_triples:
        tenure += 1
        probe_season -= 1
    return tenure


@FeatureRegistry.register("home_away_coach_profile")
class HomeAwayCoachProfileFeature:
    """Join Away/Home head-coach career win % and tenure metrics."""

    spec = FeatureSpec(
        name="home_away_coach_profile",
        produces=[
            *_AWAY_METRIC_COLS,
            *_HOME_METRIC_COLS,
            _AWAY_NEW_STAFF_FLAG,
            _HOME_NEW_STAFF_FLAG,
        ],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach Away and Home head-coach profile metrics.

        Every input row remains in the output. A side with no resolvable
        coach identity produces null win %/tenure and a zero new-staff
        flag.
        """
        _require_columns(df, _HOME_AWAY_INPUT_COLUMNS, label="Home/away game frame")

        source = df.copy().drop(
            columns=[
                *_AWAY_METRIC_COLS,
                *_HOME_METRIC_COLS,
                _AWAY_NEW_STAFF_FLAG,
                _HOME_NEW_STAFF_FLAG,
            ],
            errors="ignore",
        )
        source["_COACH_PROFILE_ORDER"] = range(len(source))
        source["SEASON"] = _season_numbers(source["YEAR"])

        sidecar = datasets.qb_coach_sidecar()
        if sidecar.empty:
            fill_values = {
                column: float("nan") for column in (*_AWAY_METRIC_COLS, *_HOME_METRIC_COLS)
            }
            fill_values[_AWAY_NEW_STAFF_FLAG] = 0
            fill_values[_HOME_NEW_STAFF_FLAG] = 0
            result = source.assign(**fill_values)
            return result.drop(columns=["_COACH_PROFILE_ORDER", "SEASON"]).reset_index(drop=True)

        _require_columns(sidecar, _SIDECAR_REQUIRED_COLUMNS, label="QB/coach sidecar")

        coach_lookup = sidecar.loc[:, ["GAME_ID", "AWAY_COACH", "HOME_COACH"]]
        resolved = source.merge(
            coach_lookup,
            how="left",
            on="GAME_ID",
            sort=False,
            validate="one_to_one",
        )

        games = datasets.games()
        win_pct_history = _build_coach_result_history(sidecar, games)

        away_wp_targets = resolved.loc[:, ["SEASON", "WEEK_NUM", "AWAY_COACH"]].rename(
            columns={"AWAY_COACH": "COACH"}
        )
        home_wp_targets = resolved.loc[:, ["SEASON", "WEEK_NUM", "HOME_COACH"]].rename(
            columns={"HOME_COACH": "COACH"}
        )
        wp_targets = pd.concat([away_wp_targets, home_wp_targets], ignore_index=True)
        win_pct = _build_target_hc_win_pct(win_pct_history, wp_targets)

        result = resolved.merge(
            win_pct.rename(
                columns={
                    "COACH": "AWAY_COACH",
                    "HC_WIN_PCT_CAREER": away_feature_name("HC_WIN_PCT_CAREER"),
                }
            ),
            how="left",
            on=["AWAY_COACH", "SEASON", "WEEK_NUM"],
            sort=False,
            validate="many_to_one",
        ).merge(
            win_pct.rename(
                columns={
                    "COACH": "HOME_COACH",
                    "HC_WIN_PCT_CAREER": home_feature_name("HC_WIN_PCT_CAREER"),
                }
            ),
            how="left",
            on=["HOME_COACH", "SEASON", "WEEK_NUM"],
            sort=False,
            validate="many_to_one",
        )

        for side_coach, side_col in (
            ("AWAY_COACH", away_feature_name("HC_WIN_PCT_CAREER")),
            ("HOME_COACH", home_feature_name("HC_WIN_PCT_CAREER")),
        ):
            # A known coach with no prior games gets the debut prior; an
            # unresolved coach stays missing so an identity gap fails closed.
            result.loc[result[side_coach].notna() & result[side_col].isna(), side_col] = (
                DEBUT_HC_WIN_PCT
            )

        known_triples = _known_team_coach_seasons(sidecar)
        result[away_feature_name("HC_TENURE")] = [
            _tenure_entering(known_triples, team=team, coach=coach, season=int(season))
            for team, coach, season in zip(
                result["AWAY_TEAM"], result["AWAY_COACH"], result["SEASON"], strict=True
            )
        ]
        result[home_feature_name("HC_TENURE")] = [
            _tenure_entering(known_triples, team=team, coach=coach, season=int(season))
            for team, coach, season in zip(
                result["HOME_TEAM"], result["HOME_COACH"], result["SEASON"], strict=True
            )
        ]

        result[_AWAY_NEW_STAFF_FLAG] = (result[away_feature_name("HC_TENURE")] == 1).astype(int)
        result[_HOME_NEW_STAFF_FLAG] = (result[home_feature_name("HC_TENURE")] == 1).astype(int)

        return (
            result.sort_values("_COACH_PROFILE_ORDER", kind="stable")
            .drop(
                columns=[
                    "_COACH_PROFILE_ORDER",
                    "SEASON",
                    "AWAY_COACH",
                    "HOME_COACH",
                ]
            )
            .reset_index(drop=True)
        )

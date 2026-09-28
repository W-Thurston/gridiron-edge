# src/gridiron_edge/features/team/opponent_adjusted_epa.py

"""Opponent-adjusted EPA/play via a ridge regression on play-level PBP.

Raw EPA/play conflates a team's own quality with the quality of the
opponents it faced. Untangling the two requires solving for every team's
offensive and defensive strength simultaneously across the whole league -
a team's adjusted rating depends on the strength of the teams it played,
which in turn depends on the strength of the teams they played, and so on.
A ridge regression on play-level EPA, with one offense term and one
defense term per team, solves this joint system directly; the L2 penalty
also shrinks thin-sample teams toward the league average instead of
letting a handful of plays produce an overconfident rating.

Recomputed fresh from play-level PBP for each (season, week) as-of cutoff
rather than persisted as an incremental ledger like Elo - see
``DECISIONS.md`` D63 for why, and the condition under which that would
change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import Ridge

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.models.game_prediction.game_schema import (
    away_feature_name,
    home_feature_name,
)

# ``map_short_to_long`` lives under transform/clean/, the layer below
# features/, because that's where every other nflverse short-code mapping
# in this codebase is applied (transform/clean/epa.py's own aggregate_epa
# does the same conversion before writing epa_by_game.parquet). This
# feature is the first to read raw PBP directly rather than that
# already-mapped intermediate, so it needs the same conversion itself
# rather than duplicating the mapping table a second time.
from gridiron_edge.transform.clean._nflverse_common import map_short_to_long

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Ridge L2 strength. A fixed, deliberate placeholder rather than a
# per-fit cross-validated value: refitting with CV on every distinct
# (season, week) cutoff would multiply the per-fit cost several-fold,
# compounding the performance risk DECISIONS.md D63 flags. Unit 7's
# promotion sweep is the empirical check on whether this value clears the
# acceptance bar; tuning it is the natural first response if it doesn't.
RIDGE_ALPHA: Final[float] = 1.0

_PLAY_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "posteam",
    "defteam",
    "epa",
)

_SOURCE_PLAY_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "posteam",
    "defteam",
    "pass",
    "rush",
    "epa",
)

_TARGET_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "team",
)

# Maximum regular-season week. Playoff plays are excluded from the
# season-to-date sample by default, mirroring epa/C1's rationale in
# ``features/team/epa.py``: a short, structurally different playoff slate
# would otherwise distort an in-progress fit for the following season's
# early weeks.
_MAX_REG_SEASON_WEEK: Final[int] = 18

ADJUSTED_EPA_METRICS: Final[tuple[str, ...]] = (
    "off_pass_epa_adj",
    "off_rush_epa_adj",
    "def_pass_epa_adj",
    "def_rush_epa_adj",
)


def _fit_opponent_adjusted_epa(
    plays: DataFrame,
    *,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, tuple[float, float]]:
    """Fit one ridge regression solving offense and defense strength jointly.

    Args:
        plays: One row per play, with ``posteam`` (offense), ``defteam``
            (defense), and ``epa`` (the play's expected-points-added).
            Every team appearing as ``posteam`` or ``defteam`` gets a
            coefficient, even if it only appears on one side.
        alpha: Ridge L2 strength. Defaults to ``RIDGE_ALPHA``; exposed as a
            parameter for testing the shrinkage behavior directly.

    Returns:
        Mapping of team -> (offense_coefficient, defense_coefficient).
        Coefficients are relative within their own group (offense
        coefficients are only meaningfully compared to other offense
        coefficients, likewise for defense) - the overall level is not
        uniquely identified, matching the usual centering ambiguity of a
        two-way fixed-effects regression.

    Raises:
        ValueError: If required columns are missing, or ``plays`` is empty.
    """
    missing = sorted(set(_PLAY_REQUIRED_COLUMNS) - set(plays.columns))
    if missing:
        raise ValueError("Play-level frame is missing required columns: " + ", ".join(missing))
    if plays.empty:
        raise ValueError("Cannot fit opponent-adjusted EPA on an empty play sample.")

    teams: list[str] = sorted(set(plays["posteam"]) | set(plays["defteam"]))

    offense = pd.get_dummies(plays["posteam"]).reindex(columns=teams, fill_value=0)
    defense = pd.get_dummies(plays["defteam"]).reindex(columns=teams, fill_value=0)
    design = pd.concat(
        [offense.add_prefix("off_"), defense.add_prefix("def_")],
        axis=1,
    )

    target: Series = plays["epa"].astype(float)

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design.to_numpy(dtype=float), target.to_numpy())

    n_teams = len(teams)
    offense_coefs = model.coef_[:n_teams]
    defense_coefs = model.coef_[n_teams:]

    return {
        team: (float(offense_coefs[index]), float(defense_coefs[index]))
        for index, team in enumerate(teams)
    }


def _build_target_opponent_adjusted_epa(
    plays: DataFrame,
    targets: DataFrame,
    *,
    alpha: float = RIDGE_ALPHA,
    exclude_playoffs: bool = True,
) -> DataFrame:
    """Compute season-to-date opponent-adjusted EPA for explicit targets.

    Fits one ridge regression per distinct (season, week) cutoff - not per
    team, not per row - reusing it for every team requested at that cutoff.
    The sample for each fit is every scrimmage play from the start of that
    season through the week strictly before the cutoff (no lookahead, no
    cross-season carryover); a cutoff at week 1 has no prior plays and
    produces ``NaN``, matching the existing rolling-EPA week-1 convention.

    Args:
        plays: Play-level PBP with ``season``, ``week``, ``posteam``,
            ``defteam``, ``pass``, ``rush``, ``epa``.
        targets: One row per (season, week, team) identity that needs a
            value. Duplicate identities are computed once.
        alpha: Ridge L2 strength, forwarded to ``_fit_opponent_adjusted_epa``.
        exclude_playoffs: If ``True`` (default), playoff plays (week > 18)
            are excluded from the season-to-date sample.

    Returns:
        DataFrame with columns ``season``, ``week``, ``team``, and the four
        adjusted-EPA metrics in ``ADJUSTED_EPA_METRICS``. A team with no
        season-to-date plays at its cutoff (week 1, or a team that hasn't
        played yet) gets ``NaN`` for all four.

    Raises:
        ValueError: If ``plays`` or ``targets`` is missing required columns.
    """
    missing_plays = sorted(set(_SOURCE_PLAY_COLUMNS) - set(plays.columns))
    if missing_plays:
        raise ValueError("Play-level PBP is missing required columns: " + ", ".join(missing_plays))
    missing_targets = sorted(set(_TARGET_REQUIRED_COLUMNS) - set(targets.columns))
    if missing_targets:
        raise ValueError(
            "Opponent-adjusted EPA targets are missing required columns: "
            + ", ".join(missing_targets)
        )

    normalized_targets = targets.drop_duplicates(
        subset=["season", "week", "team"],
        keep="first",
        ignore_index=True,
    )

    source = plays.copy()
    if exclude_playoffs:
        source = source.loc[source["week"] <= _MAX_REG_SEASON_WEEK, :].copy()

    distinct_cutoffs = normalized_targets.loc[:, ["season", "week"]].drop_duplicates(
        ignore_index=True,
    )

    rows: list[dict[str, object]] = []
    for cutoff in distinct_cutoffs.itertuples(index=False):
        cutoff_season = int(str(cutoff.season))
        cutoff_week = int(str(cutoff.week))

        teams_needed = normalized_targets.loc[
            (normalized_targets["season"] == cutoff.season)
            & (normalized_targets["week"] == cutoff.week),
            "team",
        ].unique()

        season_to_date = source.loc[
            (source["season"] == cutoff_season) & (source["week"] < cutoff_week),
            :,
        ]

        pass_coefs: dict[str, tuple[float, float]] = {}
        rush_coefs: dict[str, tuple[float, float]] = {}
        if not season_to_date.empty:
            pass_plays = season_to_date.loc[season_to_date["pass"] == 1, :]
            rush_plays = season_to_date.loc[season_to_date["rush"] == 1, :]
            if not pass_plays.empty:
                pass_coefs = _fit_opponent_adjusted_epa(pass_plays, alpha=alpha)
            if not rush_plays.empty:
                rush_coefs = _fit_opponent_adjusted_epa(rush_plays, alpha=alpha)

        for team in teams_needed:
            off_pass, def_pass = pass_coefs.get(team, (float("nan"), float("nan")))
            off_rush, def_rush = rush_coefs.get(team, (float("nan"), float("nan")))
            rows.append(
                {
                    "season": cutoff_season,
                    "week": cutoff_week,
                    "team": team,
                    "off_pass_epa_adj": off_pass,
                    "off_rush_epa_adj": off_rush,
                    "def_pass_epa_adj": def_pass,
                    "def_rush_epa_adj": def_rush,
                }
            )

    return DataFrame(
        rows,
        columns=["season", "week", "team", *ADJUSTED_EPA_METRICS],
    )


_HOME_AWAY_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
)

# Raw PBP columns needed to fit the regression - excludes everything
# ``ingest.nflverse.pbp._KEEP_COLUMNS`` carries that this feature doesn't
# use, since PBP is large enough (~540MB cached) that a narrow column
# projection matters.
_PBP_COLUMNS_NEEDED: Final[list[str]] = [
    "season",
    "week",
    "play_type",
    "posteam",
    "defteam",
    "pass",
    "rush",
    "epa",
]

# Mirrors transform/clean/epa.py's aggregate_epa filter exactly, since
# this feature reads raw PBP rather than that module's already-filtered
# output.
_SCRIMMAGE_PLAY_TYPES: Final[frozenset[str]] = frozenset(["pass", "run"])

_AWAY_ADJUSTED_EPA_COLS: Final[list[str]] = [
    away_feature_name(metric.upper()) for metric in ADJUSTED_EPA_METRICS
]
_HOME_ADJUSTED_EPA_COLS: Final[list[str]] = [
    home_feature_name(metric.upper()) for metric in ADJUSTED_EPA_METRICS
]


def _require_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require an established input schema."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _season_numbers(frame: DataFrame) -> Series:
    """Convert canonical season labels ("2024-2025") to starting-season ints."""
    if frame["YEAR"].isna().any():
        raise ValueError("YEAR must not contain nulls.")

    year_text = frame["YEAR"].astype(str).str.strip()
    if year_text.eq("").any():
        raise ValueError("YEAR must not contain empty values.")

    season_text = year_text.str.split("-", n=1).str[0]

    try:
        return season_text.astype(int)
    except ValueError as exc:
        raise ValueError("YEAR must begin with a numeric season.") from exc


@FeatureRegistry.register("home_away_opponent_adjusted_epa")
class HomeAwayOpponentAdjustedEpaFeature:
    """Join season-to-date opponent-adjusted EPA for canonical Away/Home teams."""

    spec = FeatureSpec(
        name="home_away_opponent_adjusted_epa",
        produces=[
            *_AWAY_ADJUSTED_EPA_COLS,
            *_HOME_ADJUSTED_EPA_COLS,
        ],
    )

    def __init__(
        self,
        *,
        alpha: float = RIDGE_ALPHA,
        exclude_playoffs: bool = True,
    ) -> None:
        """Configure the regression's shrinkage strength and playoff handling."""
        self.alpha = alpha
        self.exclude_playoffs = exclude_playoffs

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach Away and Home season-to-date opponent-adjusted EPA.

        Every input row remains in the output. A game with no
        season-to-date plays for a team (week 1, or no cached PBP at all)
        produces null adjusted-EPA values for that team's side.
        """
        _require_columns(
            df,
            _HOME_AWAY_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        source = df.copy().drop(
            columns=[*_AWAY_ADJUSTED_EPA_COLS, *_HOME_ADJUSTED_EPA_COLS],
            errors="ignore",
        )
        source["_OPP_ADJ_EPA_ORDER"] = range(len(source))
        source["season"] = _season_numbers(source)

        plays = datasets.pbp(columns=_PBP_COLUMNS_NEEDED)

        if plays.empty:
            result = source.assign(
                **{
                    column: float("nan")
                    for column in (*_AWAY_ADJUSTED_EPA_COLS, *_HOME_ADJUSTED_EPA_COLS)
                }
            )
            return result.drop(columns=["_OPP_ADJ_EPA_ORDER", "season"]).reset_index(drop=True)

        scrimmage = plays.loc[
            plays["play_type"].isin(_SCRIMMAGE_PLAY_TYPES)
            & plays["epa"].notna()
            & ((plays["pass"] == 1) | (plays["rush"] == 1)),
            :,
        ].copy()
        scrimmage["posteam"] = scrimmage["posteam"].map(map_short_to_long)
        scrimmage["defteam"] = scrimmage["defteam"].map(map_short_to_long)

        away_targets = source.loc[:, ["season", "WEEK_NUM", "AWAY_TEAM"]].rename(
            columns={"WEEK_NUM": "week", "AWAY_TEAM": "team"}
        )
        home_targets = source.loc[:, ["season", "WEEK_NUM", "HOME_TEAM"]].rename(
            columns={"WEEK_NUM": "week", "HOME_TEAM": "team"}
        )
        targets = pd.concat([away_targets, home_targets], ignore_index=True)

        adjusted = _build_target_opponent_adjusted_epa(
            scrimmage,
            targets,
            alpha=self.alpha,
            exclude_playoffs=self.exclude_playoffs,
        )

        away_rename = {"team": "AWAY_TEAM", "week": "WEEK_NUM"} | {
            metric: away_feature_name(metric.upper()) for metric in ADJUSTED_EPA_METRICS
        }
        result = source.merge(
            adjusted.rename(columns=away_rename),
            how="left",
            on=["season", "WEEK_NUM", "AWAY_TEAM"],
            sort=False,
            validate="many_to_one",
        )

        home_rename = {"team": "HOME_TEAM", "week": "WEEK_NUM"} | {
            metric: home_feature_name(metric.upper()) for metric in ADJUSTED_EPA_METRICS
        }
        result = result.merge(
            adjusted.rename(columns=home_rename),
            how="left",
            on=["season", "WEEK_NUM", "HOME_TEAM"],
            sort=False,
            validate="many_to_one",
        )

        return (
            result.sort_values("_OPP_ADJ_EPA_ORDER", kind="stable")
            .drop(columns=["_OPP_ADJ_EPA_ORDER", "season"])
            .reset_index(drop=True)
        )

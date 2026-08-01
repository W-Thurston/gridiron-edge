# src/gridiron_edge/features/team/epa.py

"""Canonical Away/Home rolling EPA features.

Computes pregame rolling EPA statistics from prior games and joins them
to one canonical game row for the designated Away and Home teams.

The rolling source is shifted by one game to prevent current-game
lookahead. Alternate rolling windows use the same implementation during
model tuning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

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

# Default rolling window - will be tuned in evaluation.
# 4 games is a reasonable NFL default: enough to reduce noise,
# short enough to capture current-season form.
DEFAULT_ROLLING_WINDOW: Final[int] = 4

# EPA columns from epa_by_game.parquet that we roll over.
# This is the single source of truth for EPA metric names - both the model
# layer (_shared.py) and tree tuning (tree.py) derive their column lists
# from this constant rather than maintaining independent copies.
EPA_COLS: Final[list[str]] = [
    # --- Offensive ---
    "off_epa_per_play",
    "off_pass_epa",
    "off_rush_epa",
    "off_success_rate",
    "off_pass_success_rate",
    "off_rush_success_rate",
    "off_explosive_rate",
    "off_third_down_pct",
    "off_redzone_td_pct",
    "off_turnover_rate",
    "off_sack_rate",
    "off_plays",
    "off_yards_per_play",
    "off_redzone_attempts",
    "off_int_rate",
    "off_penalty_rate",
    "off_avg_score_diff",
    "off_close_game_pct",
    # --- Defensive ---
    "def_epa_per_play",
    "def_pass_epa",
    "def_rush_epa",
    "def_success_rate",
    "def_pass_success_rate",
    "def_rush_success_rate",
    "def_explosive_rate",
    "def_third_down_pct",
    "def_redzone_td_pct",
    "def_turnover_rate",
    "def_sack_rate",
    "def_plays",
    "def_yards_per_play",
    "def_redzone_attempts",
    "def_int_rate",
    "def_penalty_rate",
    "def_avg_score_diff",
    "def_close_game_pct",
]

# Private alias kept for internal use within this module
_EPA_COLS: list[str] = EPA_COLS

# Canonical Away and Home EPA output columns.
_AWAY_EPA_COLS: Final[list[str]] = [away_feature_name(column.upper()) for column in EPA_COLS]
_HOME_EPA_COLS: Final[list[str]] = [home_feature_name(column.upper()) for column in EPA_COLS]

_HOME_AWAY_EPA_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
)

_EPA_SOURCE_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "week",
    "team",
)

# Maximum regular-season week. Used by ``_build_rolling_epa`` to
# optionally exclude prior-season playoff games from the rolling
# window (epa/C1).
_MAX_REG_SEASON_WEEK: Final[int] = 18


def _build_rolling_epa(
    epa_by_game: pd.DataFrame,
    *,
    window: int,
    exclude_playoffs: bool = True,
) -> pd.DataFrame:
    """Compute rolling EPA features for every (team, season, week) triple.

    For each game a team plays in week W of season Y, computes the rolling
    mean of each EPA metric over the previous ``window`` games played by
    that team. Games are ordered chronologically by (season, week).

    Pre-game only: the rolling window at week W uses games through week W-1,
    ensuring no leakage.

    Args:
        epa_by_game: Game-level EPA aggregation with columns
            ``game_id``, ``season``, ``week``, ``team``, plus EPA metric cols.
        window: Number of prior games to include in each rolling window.
        exclude_playoffs: If ``True`` (default), playoff games are
            excluded from the rolling window source. Early-season
            features therefore reflect only prior regular-season form,
            not the structurally different playoff slate. Set to
            ``False`` to include all completed games in the rolling
            window. See ``epa/C1``.

    Returns:
        DataFrame with columns ``season``, ``week``, ``team``,
        plus rolled EPA columns prefixed with ``rolling_``.
        One row per (season, week, team) matching the input.
    """
    df: DataFrame = epa_by_game.copy()

    # Optionally drop playoff games so they do not contribute to the
    # rolling window of any subsequent (regular- or post-season) game.
    if exclude_playoffs:
        df = df.loc[df["week"] <= _MAX_REG_SEASON_WEEK, :].copy()

    # Sort chronologically within each team
    df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)

    # Compute rolling means grouped by team.
    # shift(1) ensures we use prior games only (no current game leakage).
    rolled_parts: list[DataFrame] = []
    for _team, group in df.groupby("team", sort=False):
        sorted_group: DataFrame = group.sort_values(["season", "week"]).copy()
        available_cols: list[str] = [c for c in _EPA_COLS if c in sorted_group.columns]
        missing_cols: list[str] = [c for c in _EPA_COLS if c not in sorted_group.columns]

        if available_cols:
            rolled_vals = (
                sorted_group[available_cols].shift(1).rolling(window=window, min_periods=1).mean()
            )
            sorted_group[[f"rolling_{c}" for c in available_cols]] = rolled_vals

        for col in missing_cols:
            sorted_group[f"rolling_{col}"] = float("nan")

        rolled_parts.append(sorted_group)

    rolled: DataFrame = pd.concat(rolled_parts, ignore_index=True)

    # Keep only the keys and rolled columns
    rolling_cols: list[str] = [f"rolling_{c}" for c in _EPA_COLS]
    return rolled.loc[:, ["game_id", "season", "week", "team", *rolling_cols]].copy()


def _require_home_away_epa_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require the identity columns used by canonical EPA joins."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _validate_home_away_epa_identity(
    epa: DataFrame,
) -> None:
    """Reject ambiguous team-season-week EPA identities."""
    duplicated = epa.duplicated(
        subset=[
            "season",
            "week",
            "team",
        ],
        keep=False,
    )
    if not duplicated.any():
        return

    duplicate_rows = (
        epa.loc[
            duplicated,
            [
                "season",
                "week",
                "team",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "season",
                "week",
                "team",
            ],
            kind="stable",
        )
    )

    identities = [
        (f"{row['team']}/{row['season']}/{row['week']}") for _, row in duplicate_rows.iterrows()
    ]

    raise ValueError("EPA source contains duplicate identities: " + ", ".join(identities))


def _canonical_epa_lookup(
    rolled: DataFrame,
    *,
    team_column: str,
    prefix: str,
) -> DataFrame:
    """Project rolling EPA columns onto one canonical game side."""
    renamed = rolled.rename(
        columns={
            "team": team_column,
            "week": "WEEK_NUM",
            **{f"rolling_{column}": (f"{prefix}{column.upper()}") for column in EPA_COLS},
        }
    )

    feature_columns = [f"{prefix}{column.upper()}" for column in EPA_COLS]

    return renamed.loc[
        :,
        [
            "season",
            "WEEK_NUM",
            team_column,
            *feature_columns,
        ],
    ].copy()


def _season_numbers(
    frame: DataFrame,
) -> Series:
    """Convert canonical season labels to starting-season integers."""
    if frame["YEAR"].isna().any():
        raise ValueError("YEAR must not contain nulls.")

    year_text = frame["YEAR"].astype(str).str.strip()
    if year_text.eq("").any():
        raise ValueError("YEAR must not contain empty values.")

    season_text = year_text.str.split(
        "-",
        n=1,
    ).str[0]

    try:
        return season_text.astype(int)
    except ValueError as exc:
        raise ValueError("YEAR must begin with a numeric season.") from exc


@FeatureRegistry.register("home_away_epa")
class HomeAwayEpaFeature:
    """Join pregame rolling EPA for canonical Away and Home teams."""

    spec = FeatureSpec(
        name="home_away_epa",
        produces=[
            *_AWAY_EPA_COLS,
            *_HOME_EPA_COLS,
        ],
    )

    def __init__(
        self,
        window: int = DEFAULT_ROLLING_WINDOW,
        *,
        exclude_playoffs: bool = True,
    ) -> None:
        """Configure rolling EPA behavior."""
        if window < 1:
            raise ValueError("window must be at least 1.")

        self.window = window
        self.exclude_playoffs = exclude_playoffs

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach Away and Home pregame EPA without removing games."""
        _require_home_away_epa_columns(
            df,
            _HOME_AWAY_EPA_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        source = df.copy().drop(
            columns=[
                *_AWAY_EPA_COLS,
                *_HOME_EPA_COLS,
            ],
            errors="ignore",
        )
        source["_EPA_INPUT_ORDER"] = range(len(source))

        epa_raw: DataFrame = datasets.epa_by_game()

        if epa_raw.empty:
            result = source.assign(
                **{
                    column: float("nan")
                    for column in (
                        *_AWAY_EPA_COLS,
                        *_HOME_EPA_COLS,
                    )
                }
            )
            return result.drop(columns=["_EPA_INPUT_ORDER"]).reset_index(drop=True)

        _require_home_away_epa_columns(
            epa_raw,
            _EPA_SOURCE_IDENTITY_COLUMNS,
            label="EPA source",
        )
        _validate_home_away_epa_identity(epa_raw)

        rolled = _build_rolling_epa(
            epa_raw,
            window=self.window,
            exclude_playoffs=(self.exclude_playoffs),
        )

        source["season"] = _season_numbers(source)

        away_lookup = _canonical_epa_lookup(
            rolled,
            team_column="AWAY_TEAM",
            prefix="AWAY_",
        )
        result = source.merge(
            away_lookup,
            how="left",
            on=[
                "season",
                "WEEK_NUM",
                "AWAY_TEAM",
            ],
            sort=False,
            validate="many_to_one",
        )

        home_lookup = _canonical_epa_lookup(
            rolled,
            team_column="HOME_TEAM",
            prefix="HOME_",
        )
        result = result.merge(
            home_lookup,
            how="left",
            on=[
                "season",
                "WEEK_NUM",
                "HOME_TEAM",
            ],
            sort=False,
            validate="many_to_one",
        )

        return (
            result.sort_values(
                "_EPA_INPUT_ORDER",
                kind="stable",
            )
            .drop(
                columns=[
                    "_EPA_INPUT_ORDER",
                    "season",
                ]
            )
            .reset_index(drop=True)
        )

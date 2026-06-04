# src/gridiron_edge/transform/clean/epa.py

"""Aggregate play-by-play data to game-level team EPA statistics.

Transforms raw PBP data into a compact game-level table with one row
per team per game. This is the intermediate artifact that EPA features
read from — much faster to load than raw PBP (~500KB vs ~540MB).

Output schema (``epa_by_game.parquet``)::

    game_id             str     "2025_01_PHI_GB"
    season              int     2025
    week                int     1-22
    team                str     "PHI"  (short code, matches nflverse)
    is_offense          int     1 (always — one row per team as offense)
    off_epa_per_play    float   Offensive EPA/play (pass + rush)
    off_pass_epa        float   Offensive passing EPA/play
    off_rush_epa        float   Offensive rushing EPA/play
    off_success_rate    float   Fraction of offensive plays with EPA > 0
    off_plays           int     Total offensive plays (pass + rush)
    off_explosive_rate  float   Fraction of offensive plays gaining 20+ yds
                                (pass) or 10+ yds (rush)
    off_pass_success_rate   float   Pass plays with EPA > 0 as fraction
    off_rush_success_rate   float   Rush plays with EPA > 0 as fraction
    off_third_down_pct      float   3rd-down conversion rate
    off_redzone_td_pct      float   TD rate on red zone plays (yardline <= 20)
    off_turnover_rate       float   Interceptions + fumbles lost per play
    off_sack_rate           float   Sacks per dropback
    def_epa_per_play    float   Defensive EPA/play allowed
    def_pass_epa        float   Defensive passing EPA/play allowed
    def_rush_epa        float   Defensive rushing EPA/play allowed
    def_success_rate    float   Fraction of opponent plays with EPA > 0
    def_plays           int     Total defensive plays faced
    def_explosive_rate  float   Fraction of opponent plays gaining 20+ yds
                                (pass) or 10+ yds (rush) -- big-play
                                vulnerability
    def_pass_success_rate   float   Opponent pass success rate allowed
    def_rush_success_rate   float   Opponent rush success rate allowed
    def_third_down_pct      float   Opponent 3rd-down conversion rate allowed
    def_redzone_td_pct      float   Opponent red zone TD rate allowed
    def_turnover_rate       float   Opponent turnovers forced per play
    def_sack_rate           float   Sacks generated per opponent dropback

Note on EPA reliability: nflfastR EPA model is reliable from 2006 onward.
Seasons before 2006 will have NaN EPA values and are handled gracefully.

Temporal integrity: this module aggregates completed plays only. It does
not produce future-looking values — rolling window computation happens
in the feature layer (``features/team/epa.py``).
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas import DataFrame, Series
from pandas.api.typing import DataFrameGroupBy

from gridiron_edge.core.settings import get_settings
from gridiron_edge.ingest.nflverse.pbp import load_pbp
from gridiron_edge.transform.clean._nflverse_common import map_short_to_long

logger: Logger = logging.getLogger(__name__)

# Play types that count as offensive scrimmage plays for EPA
_SCRIMMAGE_PLAY_TYPES: frozenset[str] = frozenset(["pass", "run"])


def _safe_mean(series: pd.Series) -> float:
    """Return mean of non-null values, or NaN if all null."""
    valid: Series = series.dropna()
    return valid.mean() if len(valid) > 0 else float("nan")


def _agg_side(df: pd.DataFrame, *, is_offense: bool) -> pd.DataFrame:
    """Aggregate EPA metrics for one side of the ball per game.

    Args:
        df: PBP DataFrame filtered to scrimmage plays with valid EPA.
            Must contain ``game_id``, ``season``, ``week``, ``posteam``,
            ``defteam``, ``pass``, ``rush``, ``epa``, ``success``,
            ``yards_gained``, ``yardline_100``,
            ``score_differential``, and ``penalty`` columns.

        is_offense: If ``True``, group by ``posteam`` (offensive team).
            If ``False``, group by ``defteam`` (defensive team).

    Returns:
        DataFrame with one row per (game_id, team) containing EPA metrics.
        Prefix is ``off_`` for offense, ``def_`` for defense.
    """
    team_col: Literal["defteam", "posteam"] = "posteam" if is_offense else "defteam"
    prefix: Literal["def", "off"] = "off" if is_offense else "def"

    group_keys: list[str] = ["game_id", "season", "week", team_col]

    pass_mask: Series[bool] = df["pass"] == 1
    rush_mask: Series[bool] = df["rush"] == 1

    grouped: DataFrameGroupBy = df.groupby(group_keys)

    total_epa: Series = grouped["epa"].mean().rename(f"{prefix}_epa_per_play")
    pass_epa: Series = df[pass_mask].groupby(group_keys)["epa"].mean().rename(f"{prefix}_pass_epa")
    rush_epa: Series = df[rush_mask].groupby(group_keys)["epa"].mean().rename(f"{prefix}_rush_epa")
    success_rate: Series = grouped["success"].mean().rename(f"{prefix}_success_rate")
    n_plays: Series[int] = grouped["epa"].count().rename(f"{prefix}_plays")

    # Explosive play rate: pass gaining 20+ yds OR rush gaining 10+ yds.
    # Thresholds are standard NFL analytics definitions, not arbitrary bins.
    # The output is a continuous rate in [0, 1] -- no categorisation applied.
    explosive: Series[bool] = (pass_mask & (df["yards_gained"] >= 20)) | (
        rush_mask & (df["yards_gained"] >= 10)
    )
    explosive_rate: Series = (
        df.assign(_explosive=explosive.astype(int))
        .groupby(group_keys)["_explosive"]
        .mean()
        .rename(f"{prefix}_explosive_rate")
    )

    # -- Pass/rush success rate split --
    pass_success_rate: Series = (
        df[pass_mask].groupby(group_keys)["success"].mean().rename(f"{prefix}_pass_success_rate")
    )
    rush_success_rate: Series = (
        df[rush_mask].groupby(group_keys)["success"].mean().rename(f"{prefix}_rush_success_rate")
    )

    # -- 3rd-down conversion rate --
    third_down_plays: DataFrame = df.loc[df["down"] == 3, :]
    third_down_converted: Series[bool] = (third_down_plays["first_down"] == 1) | (
        third_down_plays["touchdown"] == 1
    )
    third_down_pct: Series = (
        third_down_plays.assign(_converted=third_down_converted.astype(int))
        .groupby(group_keys)["_converted"]
        .mean()
        .rename(f"{prefix}_third_down_pct")
    )

    # -- Red zone TD rate (per red zone play, yardline_100 <= 20) --
    redzone_plays: DataFrame = df.loc[df["yardline_100"] <= 20, :]
    redzone_td_pct: Series = (
        redzone_plays.groupby(group_keys)["touchdown"].mean().rename(f"{prefix}_redzone_td_pct")
    )

    # -- Turnover rate (interceptions + fumbles lost per play) --
    turnover: Series[bool] = (df["interception"] == 1) | (df["fumble_lost"] == 1)
    turnover_rate: Series = (
        df.assign(_turnover=turnover.astype(int))
        .groupby(group_keys)["_turnover"]
        .mean()
        .rename(f"{prefix}_turnover_rate")
    )

    # -- Sack rate (sacks per dropback) --
    sack_rate: Series = (
        df[pass_mask].groupby(group_keys)["sack"].mean().rename(f"{prefix}_sack_rate")
    )

    # -- CPOE: completion probability over expected (passing plays only) --
    cpoe_avg: Series = df[pass_mask].groupby(group_keys)["cpoe"].mean().rename(f"{prefix}_cpoe")

    # -- Yards per play --
    yards_per_play: Series = grouped["yards_gained"].mean().rename(f"{prefix}_yards_per_play")

    # -- Red zone attempts (plays inside opponent 20) --
    redzone_attempts: Series = (
        df.assign(_rz=(df["yardline_100"] <= 20).astype(int))
        .groupby(group_keys)["_rz"]
        .sum()
        .rename(f"{prefix}_redzone_attempts")
    )

    # -- Interception rate (per pass play) --
    int_rate: Series = (
        df[pass_mask].groupby(group_keys)["interception"].mean().rename(f"{prefix}_int_rate")
    )

    # -- Penalty rate (penalties per play) --
    penalty_rate: Series = grouped["penalty"].mean().rename(f"{prefix}_penalty_rate")

    # -- Average score differential (game script context) --
    avg_score_diff: Series = grouped["score_differential"].mean().rename(f"{prefix}_avg_score_diff")

    # -- Close game rate (% of plays within 7 points) --
    close_game_pct: Series = (
        df.assign(_close=(df["score_differential"].abs() <= 7).astype(int))
        .groupby(group_keys)["_close"]
        .mean()
        .rename(f"{prefix}_close_game_pct")
    )

    result: DataFrame = (
        pd.concat(
            [
                total_epa,
                pass_epa,
                rush_epa,
                success_rate,
                pass_success_rate,
                rush_success_rate,
                explosive_rate,
                third_down_pct,
                redzone_td_pct,
                turnover_rate,
                sack_rate,
                n_plays,
                cpoe_avg,
                yards_per_play,
                redzone_attempts,
                int_rate,
                penalty_rate,
                avg_score_diff,
                close_game_pct,
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={team_col: "team"})
    )

    return result


def aggregate_epa(
    seasons: list[int] | None = None,
    *,
    repo: Path | None = None,
) -> Path:
    """Aggregate PBP data to game-level EPA stats and write to Parquet.

    Reads from the cached PBP raw files and produces a compact game-level
    table at ``data/cleaned/epa_by_game.parquet``.

    Args:
        seasons: Season years to aggregate. If ``None``, aggregates all
            cached PBP seasons. Incremental updates pass only the current
            season.
        repo: Repository root.

    Returns:
        Absolute path to the written ``epa_by_game.parquet`` file.
    """
    resolved_repo: Path = repo or get_settings().repo_root

    columns_needed: list[str] = [
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "play_type",
        "pass",
        "rush",
        "epa",
        "success",
        "yards_gained",
        "down",
        "first_down",
        "touchdown",
        "interception",
        "fumble_lost",
        "yardline_100",
        "sack",
        "cpoe",
        "score_differential",
        "penalty",
    ]

    pbp: DataFrame = load_pbp(seasons=seasons, repo=resolved_repo, columns=columns_needed)

    if pbp.empty:
        logger.warning("No PBP data found — run 'gridiron ingest pbp' first.")
        out_path: Path = resolved_repo / "data" / "cleaned" / "epa_by_game.parquet"
        return out_path

    # Filter to scrimmage plays with valid EPA
    scrimmage: DataFrame = pbp.loc[
        pbp["play_type"].isin(_SCRIMMAGE_PLAY_TYPES)
        & pbp["epa"].notna()
        & ((pbp["pass"] == 1) | (pbp["rush"] == 1)),
        :,
    ].copy()

    n_raw: int = len(pbp)
    n_scrimmage: int = len(scrimmage)
    logger.info(
        "EPA aggregation: %d raw plays -> %d scrimmage plays with EPA",
        n_raw,
        n_scrimmage,
    )

    off_df: DataFrame = _agg_side(scrimmage, is_offense=True)
    def_df: DataFrame = _agg_side(scrimmage, is_offense=False)

    # Join offense and defense on (game_id, season, week, team)
    result: DataFrame = off_df.merge(
        def_df,
        on=["game_id", "season", "week", "team"],
        how="outer",
    )

    # Map nflverse short team codes to canonical long names so the
    # EPA features can join against the modeling file on team name.
    result["team"] = result["team"].map(map_short_to_long)

    result = result.sort_values(["season", "week", "team"]).reset_index(drop=True)

    # Write to cleaned/ — incremental updates merge with existing
    out_path = resolved_repo / "data" / "cleaned" / "epa_by_game.parquet"

    if seasons is not None and out_path.exists():
        # Incremental: remove old rows for these seasons, append new
        existing: DataFrame = pd.read_parquet(out_path)
        existing = existing.loc[~existing["season"].isin(seasons), :].copy()
        result = pd.concat([existing, result], ignore_index=True)
        result = result.sort_values(["season", "week", "team"]).reset_index(drop=True)

    result.to_parquet(out_path, index=False)

    size_kb: float = out_path.stat().st_size / 1024
    logger.info(
        "EPA by game written: %d rows, %d games -> %s (%.0f KB)",
        len(result),
        result["game_id"].nunique(),
        out_path,
        size_kb,
    )

    return out_path

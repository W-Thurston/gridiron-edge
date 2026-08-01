# src/gridiron_edge/features/player/matchup.py

"""Opponent matchup features for prop models.

Computes per-team defensive allowances by position group, then applies
rolling averages and rankings. Each player-game row gets tagged with
how generous/tough their opponent's defense has been against their
position recently.

Key features produced:
- ``opp_{stat}_allowed_L{window}`` - rolling avg of what the defense allows
- ``opp_{stat}_rank_L{window}`` - rank (1=toughest, 32=most generous)

Position-filter semantics (matchup/C1):
    ``_MATCHUP_STATS`` filters players into strict position buckets when
    aggregating what each defense allows. WR receiving yards are tracked
    separately from TE receiving yards, even though both positions can
    catch passes. This is deliberate: defenses see WR sets and TE sets
    differently (slot/perimeter vs in-line/H-back), so a defense that
    is generous against TEs is not necessarily generous against WRs.

    Edge cases not modeled here:
        - A WR who lines up in the slot in a hybrid offense and a TE who
          splits wide on the same play. The position label drives the
          bucket, not the snap alignment.
        - Multi-position players. A player listed as RB who occasionally
          takes WR snaps still counts toward RB rushing allowances only.

    A future enhancement could replace the strict position filter with
    snap-alignment data once that's ingested. Until then, the current
    semantics produce stable, interpretable matchup features.

Usage::

    from gridiron_edge.features.player.matchup import build_matchup_features

    df = build_matchup_features()
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

# Default rolling window for defensive allowances.
DEFAULT_MATCHUP_WINDOW: Final[int] = 6

# Defensive allowance stats to compute, keyed by position group.
# Each entry: (position_filter, stat_column, output_name)
_MATCHUP_STATS: Final[list[tuple[list[str], str, str]]] = [
    # Passing (allowed to QBs)
    (["QB"], "passing_yards", "pass_yards"),
    (["QB"], "passing_tds", "pass_tds"),
    (["QB"], "passing_epa", "pass_epa"),
    (["QB"], "passing_interceptions", "pass_int_forced"),
    (["QB"], "sacks_suffered", "sacks"),
    # Rushing (allowed to RBs + FBs)
    (["RB", "FB"], "rushing_yards", "rush_yards"),
    (["RB", "FB"], "rushing_tds", "rush_tds"),
    (["RB", "FB"], "rushing_epa", "rush_epa"),
    # WR receiving
    (["WR"], "receiving_yards", "wr_rec_yards"),
    (["WR"], "receiving_tds", "wr_rec_tds"),
    (["WR"], "targets", "wr_targets"),
    # TE receiving
    (["TE"], "receiving_yards", "te_rec_yards"),
    (["TE"], "receiving_tds", "te_rec_tds"),
    (["TE"], "targets", "te_targets"),
]


def _compute_def_allowed_per_game(player_logs: DataFrame) -> DataFrame:
    """Aggregate what each defense allows per position group per game.

    Groups by (opponent_team, season, week) for each stat/position combo,
    summing across all players of that position in the game.

    Returns:
        DataFrame with columns: season, week, team (the defense),
        and one column per matchup stat (e.g. ``pass_yards_allowed``).
    """
    # Collect all per-game defensive allowance stats
    frames: list[DataFrame] = []

    for positions, stat_col, output_name in _MATCHUP_STATS:
        if stat_col not in player_logs.columns:
            logger.debug("Skipping %s - column not in data", stat_col)
            continue

        # Explicit copy() after boolean filtering to avoid potential
        # SettingWithCopyWarning if downstream code ever mutates `filtered`
        # (matchup/H1). Defensive even if no current call site mutates.
        filtered = player_logs[player_logs["position"].isin(positions)].copy()
        agg = (
            filtered.groupby(["opponent_team", "season", "week"])[stat_col]
            .sum()
            .rename(f"{output_name}_allowed")
            .reset_index()
            .rename(columns={"opponent_team": "team"})
        )
        frames.append(agg)

    if not frames:
        msg = "No matchup stats could be computed"
        raise ValueError(msg)

    # Merge all stats into a single DataFrame
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["team", "season", "week"], how="outer")

    return result.sort_values(["team", "season", "week"])


def _rolling_def_allowed(
    def_allowed: DataFrame,
    *,
    window: int,
) -> DataFrame:
    """Compute shifted rolling averages of defensive allowances.

    Uses shift(1) to prevent lookahead - a team's defensive profile
    for week N reflects only games through week N-1.

    Also computes per-week rankings (1=toughest, 32=most generous)
    based on the rolling averages.
    """
    allowed_cols = [c for c in def_allowed.columns if c.endswith("_allowed")]
    def_allowed = def_allowed.sort_values(["team", "season", "week"]).copy()

    grouped = def_allowed.groupby(["team", "season"])

    for col in allowed_cols:
        roll_col = f"opp_{col}_L{window}"
        def_allowed[roll_col] = grouped[col].transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
        )

    return def_allowed


def _rank_defenses(def_rolling: DataFrame, *, window: int) -> DataFrame:
    """Rank defenses per (season, week) for each stat.

    Rank 1 = fewest yards/points allowed (toughest defense).
    Rank 32 = most yards/points allowed (most generous).

    Vectorized via a single groupby().rank() over all roll_cols at once,
    replacing the per-column loop (matchup/H2).
    """
    roll_cols = [c for c in def_rolling.columns if f"_L{window}" in c]
    if not roll_cols:
        return def_rolling

    # Single groupby; rank all columns at once. Each column in the result
    # corresponds to the same-named source column with rank semantics applied.
    ranks = def_rolling.groupby(["season", "week"])[roll_cols].rank(
        method="min",
        ascending=True,
        na_option="bottom",
    )

    rename_map: dict[str, str] = {
        col: col.replace(f"_L{window}", f"_rank_L{window}") for col in roll_cols
    }
    # pyrefly: ignore [no-matching-overload]
    ranks = ranks.rename(columns=rename_map)
    return pd.concat([def_rolling, ranks], axis=1)


def build_matchup_features(
    *,
    df: DataFrame | None = None,
    window: int | None = None,
    repo: Path | None = None,
) -> DataFrame:
    """Build opponent matchup features and join to player game logs.

    Each player-game row gets columns describing how their opponent's
    defense has performed against their position group recently.

    Args:
        df: Pre-loaded player game logs.  If ``None``, loads from disk.
        window: Rolling window size for defensive averages. Defaults to 6.
        repo: Repository root.

    Returns:
        Player game logs with matchup feature columns appended.

    Raises:
        FileNotFoundError: If cleaned player game logs don't exist.
    """
    resolved_window: int = window or DEFAULT_MATCHUP_WINDOW

    if df is None:
        resolved_repo: Path = repo or get_settings().repo_root

        path: Path = resolved_repo / "data" / "cleaned" / "player_game_logs.parquet"
        if not path.exists():
            msg: str = f"Cleaned player game logs not found at {path}"
            raise FileNotFoundError(msg)

        player_logs: DataFrame = pd.read_parquet(path)
    else:
        player_logs = df.copy()
    logger.info("Loaded %d player-game rows for matchup features", len(player_logs))

    # Compute what each defense allows per game.
    def_allowed: DataFrame = _compute_def_allowed_per_game(player_logs)
    logger.info(
        "Defensive allowances: %d team-game rows, %d stats",
        len(def_allowed),
        len([c for c in def_allowed.columns if c.endswith("_allowed")]),
    )

    # Compute shifted rolling averages without lookahead.
    def_rolling: DataFrame = _rolling_def_allowed(def_allowed, window=resolved_window)

    # Rank defenses.
    def_rolling = _rank_defenses(def_rolling, window=resolved_window)

    # Join defense values to player logs through opponent_team.
    matchup_cols: list[str] = [c for c in def_rolling.columns if f"_L{resolved_window}" in c]
    join_cols: list[str] = ["team", "season", "week", *matchup_cols]

    result: DataFrame = player_logs.merge(
        # pyrefly: ignore [no-matching-overload]
        def_rolling[join_cols].rename(columns={"team": "opponent_team"}),
        on=["opponent_team", "season", "week"],
        how="left",
    )

    n_matchup: int = len(matchup_cols)
    n_matched: int = result[matchup_cols[0]].notna().sum() if matchup_cols else 0
    logger.info(
        "Joined %d matchup features. Matched: %d / %d rows (%.1f%%)",
        n_matchup,
        n_matched,
        len(result),
        n_matched / len(result) * 100 if len(result) > 0 else 0,
    )

    return result

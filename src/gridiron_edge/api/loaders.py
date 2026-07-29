# src/gridiron_edge/api/loaders.py

"""Thin loader wrappers for API consumption.

Bridges domain data loaders (datasets/, betting/, evaluation/) into
FastAPI-friendly functions that take Settings and return DataFrames or
domain objects.

Per D19, every wrapper passes `settings.repo_root` explicitly to the
underlying loader — the domain-loader fallback to `get_settings()` is
never used from the API path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import Settings


def load_bets_df(settings: Settings, *, status: str | None = None) -> pd.DataFrame:
    """Return bets from the ledger, optionally filtered by status."""
    from gridiron_edge.betting.ledger import load_bets

    return load_bets(status=status, repo=settings.repo_root)


def load_bankroll_txns_df(
    settings: Settings,
    *,
    txn_type: str | None = None,
) -> pd.DataFrame:
    """Return the raw bankroll transaction log."""
    from gridiron_edge.betting.bankroll import load_transactions

    return load_transactions(txn_type=txn_type, repo=settings.repo_root)


def load_bankroll_history_df(settings: Settings) -> pd.DataFrame:
    """Return the running-balance curve as a DataFrame.

    Columns: timestamp, txn_type, amount, signed_amount, running_balance.
    """
    from gridiron_edge.betting.bankroll import balance_history

    return balance_history(repo=settings.repo_root)


def load_current_bankroll(settings: Settings) -> float:
    """Return the current bankroll balance."""
    from gridiron_edge.betting.bankroll import current_balance

    return current_balance(repo=settings.repo_root)


def resolve_current_week(settings: Settings) -> tuple[int, int, str]:
    """Resolve the current NFL (season, week) from the schedule.

    Returns (season, week, source_label). If the schedule is unavailable
    or empty, falls back to (current_nfl_season(), 1, "fallback").
    """
    from gridiron_edge.core.settings import current_nfl_season
    from gridiron_edge.datasets.loaders import load_schedule_upcoming

    season = current_nfl_season()

    try:
        schedule = load_schedule_upcoming(settings.repo_root)
    except FileNotFoundError:
        return (season, 1, "fallback")

    if schedule.empty:
        return (season, 1, "fallback")

    # The schedule is upcoming games; the first row after sorting is the
    # nearest upcoming week, which we treat as "current" for scheduling.
    # NOTE: column names assumed to be `season` and `week`. If the actual
    # columns differ, this is the one line to update.
    sort_cols = [c for c in ("season", "week") if c in schedule.columns]
    if not sort_cols:
        return (season, 1, "fallback")

    latest = schedule.sort_values(sort_cols).iloc[0]
    return (int(latest["season"]), int(latest["week"]), "schedule")


def load_evaluation_df(
    settings: Settings,
    *,
    model_name: str | None = None,
    model_type: str | None = None,
    season: str | None = None,
) -> DataFrame:
    """Return the evaluation DataFrame (predictions joined to outcomes)."""
    from gridiron_edge.evaluation.metrics import build_evaluation_df

    return build_evaluation_df(
        model_name=model_name,
        model_type=model_type,
        season=season,
        repo=settings.repo_root,
    )


def load_games_df(settings: Settings) -> pd.DataFrame:
    """Return the cleaned historical games table."""
    from gridiron_edge.datasets.loaders import load_games

    return load_games(settings.repo_root)


def load_elo_state_df(settings: Settings) -> pd.DataFrame:
    """Return the full Elo history (team, season, week, ELO)."""
    from gridiron_edge.datasets.loaders import load_elo_state

    return load_elo_state(settings.repo_root)


def load_team_name_map(settings: Settings) -> dict[str, str]:
    """Return the long → short team name mapping as a dict.

    Example: {"Baltimore Ravens": "BAL", "Kansas City Chiefs": "KC", ...}
    """
    from gridiron_edge.datasets.loaders import load_teams_long_short

    df = load_teams_long_short(settings.repo_root)
    return dict(zip(df["NFL_LONG_NAME"], df["NFL_SHORT_NAME"], strict=True))


def resolve_current_season_week(settings: Settings) -> tuple[str, int]:
    """Resolve the current (season, week) for default views.

    Normally the latest completed game. But when the completed archive
    ends on a season-ending game (week 22 = Super Bowl) and an upcoming
    schedule exists, prefer the upcoming schedule's earliest week — so
    the offseason lands on next season's Week 1 rather than replaying the
    just-finished Super Bowl.

    Returns ("", 0) if games is empty.
    """
    from gridiron_edge.datasets.loaders import (
        load_games,
        load_schedule_upcoming,
    )

    games = load_games(settings.repo_root)
    if games.empty:
        return ("", 0)

    games_sorted = games.sort_values(["YEAR", "WEEK_NUM"])
    latest = games_sorted.iloc[-1]
    latest_season = str(latest["YEAR"])
    latest_week = int(latest["WEEK_NUM"])

    # Season complete (SB played) → look forward to the upcoming slate.
    if latest_week >= 22:
        try:
            upcoming = load_schedule_upcoming(settings.repo_root)
        except FileNotFoundError:
            upcoming = None

        if upcoming is not None and not upcoming.empty:
            # Upcoming schedule columns: confirm the season/week columns.
            season_col = next((c for c in ("YEAR", "season") if c in upcoming.columns), None)
            week_col = next((c for c in ("WEEK_NUM", "week") if c in upcoming.columns), None)
            if season_col and week_col:
                up_sorted = upcoming.sort_values([season_col, week_col])
                up_first = up_sorted.iloc[0]
                return (str(up_first[season_col]), int(up_first[week_col]))

    return (latest_season, latest_week)


@dataclass(frozen=True)
class ProjectionGridData:
    """Static sources needed to serialize the weekly projection grid.

    The loader reads and season-scopes source artifacts only. It does not
    classify team-week states, infer byes, or construct API response rows;
    those responsibilities belong to the projections serializer.
    """

    probabilities: DataFrame
    schedule: DataFrame
    games: DataFrame
    long_to_short: dict[str, str]
    season: str
    completed_through_week: int
    schedule_available: bool


def compute_elo_deltas(
    elo_state: DataFrame,
    long_to_short: dict[str, str],
) -> DataFrame:
    """Compute per-team Elo delta from prior NFL week within same season.

    For the latest (season, week) in the Elo state table, computes:
        elo_delta = current_week_elo - prior_week_elo

    Prior week is `NFL_WEEK - 1` within the same `NFL_YEAR`. Teams with
    no prior-week Elo (Week 1 of a season, or fresh checkouts) get null.

    The Elo state table stores long team names (e.g. "Arizona Cardinals").
    This function converts to short codes (e.g. "ARI") using the provided
    map for join compatibility with the projections summary CSV.

    Args:
        elo_state: DataFrame with columns NFL_TEAM (long), NFL_YEAR, NFL_WEEK, ELO.
        long_to_short: Mapping from long team names to short codes.

    Returns:
        DataFrame with columns team_abbr, elo_delta. One row per team.
        Empty if elo_state is empty.
    """
    if elo_state.empty:
        return pd.DataFrame(columns=["team_abbr", "elo_delta"])

    # Resolve latest (season, week) — max NFL_WEEK for latest NFL_YEAR.
    latest_year = str(elo_state["NFL_YEAR"].max())
    year_rows = elo_state.loc[elo_state["NFL_YEAR"] == latest_year, :]
    latest_week = int(year_rows["NFL_WEEK"].max())

    # Week 1 → no prior week within same season → return null deltas.
    if latest_week == 1:
        current = year_rows.loc[
            year_rows["NFL_WEEK"] == latest_week,
            ["NFL_TEAM"],
        ].copy()
        current["team_abbr"] = current["NFL_TEAM"].map(long_to_short).fillna(current["NFL_TEAM"])
        current["elo_delta"] = None
        return current.loc[:, ["team_abbr", "elo_delta"]].copy()

    # Current-week Elo.
    current = year_rows.loc[
        year_rows["NFL_WEEK"] == latest_week,
        ["NFL_TEAM", "ELO"],
    ].rename(columns={"ELO": "current_elo"})

    # Prior-week Elo.
    prior = year_rows.loc[
        year_rows["NFL_WEEK"] == latest_week - 1,
        ["NFL_TEAM", "ELO"],
    ].rename(columns={"ELO": "prior_elo"})

    # Join on long team name.
    merged = current.merge(prior, on="NFL_TEAM", how="left")
    merged["elo_delta"] = merged["current_elo"] - merged["prior_elo"]

    # Convert to short codes.
    merged["team_abbr"] = merged["NFL_TEAM"].map(long_to_short).fillna(merged["NFL_TEAM"])

    return merged.loc[:, ["team_abbr", "elo_delta"]].copy()


def load_projections_summary_df(
    settings: Settings,
) -> tuple[pd.DataFrame, str | None, int | None]:
    """Load the projections summary CSV, joined with Elo deltas.

    Also reads the projections_metadata.json sidecar if present.

    Reads projections_summary.csv and joins per-team Elo delta from the
    Elo state table (prior NFL week within same season). Populates the
    ``elo_delta`` column on the returned DataFrame.

    Returns:
        Tuple of (dataframe, csv_mtime_iso, n_simulations).
        - dataframe: projections with delta column merged.
        - csv_mtime_iso: last-modified time of the CSV as ISO string.
        - n_simulations: from the metadata JSON, or None if unavailable.

        Returns (empty_df, None, None) if the CSV doesn't exist.
    """
    path: Path = settings.repo_root / "data" / "output" / "temp" / "projections_summary.csv"
    metadata_path: Path = (
        settings.repo_root / "data" / "output" / "temp" / "projections_metadata.json"
    )

    if not path.exists():
        return pd.DataFrame(), None, None

    df = pd.read_csv(path)
    mtime: str = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()

    # Read metadata sidecar if it exists.
    n_simulations: int | None = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            n_simulations = metadata.get("n_simulations")
        except (json.JSONDecodeError, OSError):
            # Corrupt metadata → treat as missing.
            n_simulations = None

    # Compute per-team Elo deltas and join.
    elo_state: DataFrame = load_elo_state_df(settings)
    long_to_short: dict[str, str] = load_team_name_map(settings)
    deltas: DataFrame = compute_elo_deltas(elo_state, long_to_short)

    if deltas.empty:
        df["elo_delta"] = None
    else:
        df = df.merge(
            deltas,
            left_on="TEAM",
            right_on="team_abbr",
            how="left",
        )
        df = df.drop(columns=["team_abbr"], errors="ignore")

    return df, mtime, n_simulations


def load_projection_grid_data(
    settings: Settings,
) -> ProjectionGridData:
    """Load and season-scope the static weekly-grid source artifacts.

    Sources:
        - ``data/output/temp/season_grid.csv`` for Week 1-18 win
          probabilities.
        - The cleaned upcoming schedule for matchup, venue perspective,
          date, time, and confirmed bye detection.
        - The cleaned games dataset for completed regular-season results.
        - The unified team-name mapping for long-name to abbreviation
          resolution.

    This function performs file loading and season scoping only. It does
    not construct the 32 x 18 response, classify byes, or infer actual
    outcomes.

    Missing probability, schedule, or games artifacts are treated as
    unavailable source states rather than fatal API errors.

    Returns:
        A ``ProjectionGridData`` container with season-scoped sources.
    """
    from gridiron_edge.datasets.loaders import (
        load_games,
        load_schedule_upcoming,
    )

    probability_path = settings.repo_root / "data" / "output" / "temp" / "season_grid.csv"

    probabilities = pd.read_csv(probability_path) if probability_path.exists() else pd.DataFrame()

    schedule_available = True
    try:
        schedule = load_schedule_upcoming(settings.repo_root)
    except FileNotFoundError:
        schedule = pd.DataFrame()
        schedule_available = False

    season = ""
    if not schedule.empty and "YEAR" in schedule.columns:
        season_values = schedule["YEAR"].dropna().astype(str)
        if not season_values.empty:
            season = sorted(season_values.unique())[-1]

    if not season:
        try:
            season, _ = resolve_current_season_week(settings)
        except FileNotFoundError:
            season = ""

    if not schedule.empty and season and {"YEAR", "WEEK_NUM"}.issubset(schedule.columns):
        schedule = schedule.loc[
            (schedule["YEAR"].astype(str) == season) & schedule["WEEK_NUM"].between(1, 18)
        ].copy()
    elif not schedule.empty:
        schedule = pd.DataFrame()

    try:
        games = load_games(settings.repo_root)
    except FileNotFoundError:
        games = pd.DataFrame()

    if not games.empty and season and {"YEAR", "WEEK_NUM"}.issubset(games.columns):
        games = games.loc[
            (games["YEAR"].astype(str) == season) & games["WEEK_NUM"].between(1, 18)
        ].copy()
    elif not games.empty:
        games = pd.DataFrame()

    completed_through_week = 0 if games.empty else int(games["WEEK_NUM"].max())

    return ProjectionGridData(
        probabilities=probabilities,
        schedule=schedule,
        games=games,
        long_to_short=load_team_name_map(settings),
        season=season,
        completed_through_week=completed_through_week,
        schedule_available=schedule_available,
    )


def load_team_percentiles_df(
    settings: Settings,
) -> pd.DataFrame:
    """Load the latest team percentile artifact.

    Reads the most recent file from ``data/output/rankings/percentiles/``.
    Returns empty DataFrame if no artifact exists yet (e.g., before the
    first sim run has completed).

    Returns:
        DataFrame with columns team_abbr, season, week, rating_pct,
        avg_wins_pct, make_playoffs_pct, win_sb_pct.
    """
    from gridiron_edge.evaluation.percentiles import load_latest_team_percentiles

    return load_latest_team_percentiles(settings.repo_root)


def load_prop_situational_splits(
    settings: Settings,
    *,
    player_id: str,
    stat_type: str,
) -> dict | None:
    """Load situational splits for one (player_id, stat_type).

    Reads the per-stat-type artifact and filters to the player. Returns
    a nested dict of cohort → {sample_size, mean_value}. Returns None
    if no artifact exists for the stat_type, or an empty dict if the
    artifact exists but has no rows for the player.

    Args:
        settings: API settings.
        player_id: Player identifier.
        stat_type: Stat family (e.g. "qb_pass_yards").

    Returns:
        Nested dict: {"season": {"sample_size": 5, "mean_value": 260.0}, ...}
        Empty dict if artifact exists but player not found.
        None if the artifact file doesn't exist at all.
    """
    from gridiron_edge.evaluation.situational_splits import load_situational_splits

    df = load_situational_splits(stat_type, settings.repo_root)
    if df.empty:
        return None

    player_rows = df.loc[df["player_id"] == player_id, :]
    if player_rows.empty:
        return {}

    # Nest into dict: cohort → {sample_size, mean_value}
    result: dict = {}
    for _, row in player_rows.iterrows():
        cohort = str(row["cohort"])
        sample_size = _none_if_nan_int(row["sample_size"])
        mean_value = _none_if_nan_float(row["mean_value"])
        result[cohort] = {
            "sample_size": sample_size,
            "mean_value": mean_value,
        }

    return result


# Maps friendly stat keys → player_game_logs.parquet column names.
# Extend as views need more stats (rush_tds, receptions, pass_tds, ...).
_PLAYER_STAT_COLUMNS: dict[str, str] = {
    "pass_yards": "passing_yards",
    "rush_yards": "rushing_yards",
    "rec_yards": "receiving_yards",
}


def player_stat_columns() -> list:
    """Return the valid stat keys for the player-history endpoint."""
    return list(_PLAYER_STAT_COLUMNS.keys())


def load_player_history(
    settings: Settings,
    *,
    player_id: str,
    stat: str,
    season: int | None = None,
    limit: int | None = None,
) -> dict | None:
    """Load a player's per-game stat series for one season.

    Reads player_game_logs.parquet, filters to the player + season
    (regular season only), and projects the one stat column mapped from
    the ``stat`` key. Bars are raw per-game values — no cohort filtering.

    Args:
        settings: API settings, source of repo_root.
        player_id: Player identifier (e.g. "00-0039793").
        stat: Friendly stat key (e.g. "rush_yards"). Must be a key of
            ``_PLAYER_STAT_COLUMNS``.
        season: Season int (e.g. 2024). Defaults to the player's latest
            season present in the logs.
        limit: If set, return only the last N games (most recent weeks).

    Returns:
        Dict with keys: player_id, player_name, stat, season, rows
        (list of {week, value, opponent, game_id, is_home}). Rows sorted
        by week ascending. Returns None if the stat key is unknown or
        the player has no rows.

    Notes:
        player_game_logs season is an int (1999..2024) and the logs are
        stale relative to game-side data. The endpoint defaults to the
        latest season the *player* actually has, not the league-current
        season.
    """
    column = _PLAYER_STAT_COLUMNS.get(stat)
    if column is None:
        return None

    logs_path: Path = settings.repo_root / "data" / "cleaned" / "player_game_logs.parquet"
    if not logs_path.exists():
        return None

    df: DataFrame = pd.read_parquet(logs_path)
    player_rows = df.loc[df["player_id"] == player_id, :]
    if player_rows.empty:
        return None

    # Regular season only for a clean weekly series.
    if "season_type" in player_rows.columns:
        player_rows = player_rows.loc[player_rows["season_type"] == "REG", :]
    if player_rows.empty:
        return None

    # Default season = the player's latest season present.
    resolved_season: int = season if season is not None else int(player_rows["season"].max())
    season_rows = player_rows.loc[player_rows["season"] == resolved_season, :].sort_values("week")
    if season_rows.empty:
        return None

    if limit is not None and limit > 0:
        season_rows = season_rows.tail(limit)

    player_name = str(season_rows.iloc[0]["player_name"])

    rows: list[dict] = []
    for _, r in season_rows.iterrows():
        rows.append(
            {
                "week": int(r["week"]),
                "value": _none_if_nan_float(r[column]),
                "opponent": str(r["opponent_team"]),
                "game_id": str(r["game_id"]),
                "is_home": bool(r["is_home"]),
            }
        )

    return {
        "player_id": player_id,
        "player_name": player_name,
        "stat": stat,
        "season": resolved_season,
        "rows": rows,
    }


def load_players_list(
    settings: Settings,
    *,
    season: int | None = None,
) -> dict | None:
    """Load skill players active in a season, deduped to latest team.

    Reads player_game_logs.parquet, filters to skill positions + REG
    season, and takes each player's most-recent game row for their
    current team/position/name. Sorted by name.

    Args:
        settings: API settings.
        season: Season int; defaults to the latest present in the logs.

    Returns:
        Dict {season, rows: [{player_id, player_name, position, team}]}.
        None if the logs are missing/empty.
    """
    logs_path: Path = settings.repo_root / "data" / "cleaned" / "player_game_logs.parquet"
    if not logs_path.exists():
        return None

    df: DataFrame = pd.read_parquet(logs_path)
    if df.empty:
        return None

    if "season_type" in df.columns:
        df = df.loc[df["season_type"] == "REG", :]
    if "is_skill" in df.columns:
        df = df.loc[df["is_skill"], :]
    if df.empty:
        return None

    resolved_season: int = season if season is not None else int(df["season"].max())
    scope = df.loc[df["season"] == resolved_season, :]
    if scope.empty:
        return None

    # Each player's most-recent game row = current team/position/name.
    latest = scope.sort_values(["player_id", "week"]).groupby("player_id", group_keys=False).tail(1)

    rows: list[dict] = [
        {
            "player_id": str(r["player_id"]),
            "player_name": str(r["player_name"]),
            "position": str(r["position"]),
            "team": str(r["team"]),
        }
        for _, r in latest.sort_values("player_name").iterrows()
    ]

    return {"season": resolved_season, "rows": rows}


def _is_home_from_game_id(game_id: str, team: str) -> bool:
    """Home if team matches the HOME slot of 'YYYY_WW_AWAY_HOME'."""
    parts = game_id.split("_")
    if len(parts) != 4:
        return False
    return parts[3] == team


def _none_if_nan_int(v: Any) -> int | None:  # noqa: ANN401
    """Return int or None for NaN."""
    if pd.isna(v):
        return None
    return int(v)  # type: ignore[arg-type]


def _none_if_nan_float(v: Any) -> float | None:  # noqa: ANN401
    """Return float or None for NaN."""
    if pd.isna(v):
        return None
    return float(v)  # type: ignore[arg-type]


# Upcoming weeks only ever have elo predictions (trained models need a
# feature matrix that doesn't exist for unplayed games). When the champion
# has no rows for a (season, week), fall back to elo so upcoming weeks
# still serve.
_UPCOMING_FALLBACK_MODEL_TYPE: str = "elo"


def _resolve_win_prob_archive(
    settings: Settings,
    *,
    season: str | None,
    week: int | None,
) -> tuple[DataFrame, str]:
    """Load win_prob predictions, champion-first with an elo fallback.

    Returns (archive, model_type_used). Tries the current champion; if
    that yields no rows for the requested scope (e.g. an upcoming week,
    which only elo can predict), retries with elo. Empty frame + champion
    type if neither has rows.
    """
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.evaluation.champion_resolver import resolve_current_champion

    _, champion_type = resolve_current_champion("win_prob", repo=settings.repo_root)

    archive: DataFrame = load_prediction_log(
        season=season,
        week=week,
        model_name="win_prob",
        model_type=champion_type,
        repo=settings.repo_root,
    )
    if not archive.empty:
        return archive, champion_type

    # Champion has no rows for this scope — try elo (upcoming-week model).
    if champion_type != _UPCOMING_FALLBACK_MODEL_TYPE:
        fallback: DataFrame = load_prediction_log(
            season=season,
            week=week,
            model_name="win_prob",
            model_type=_UPCOMING_FALLBACK_MODEL_TYPE,
            repo=settings.repo_root,
        )
        if not fallback.empty:
            return fallback, _UPCOMING_FALLBACK_MODEL_TYPE

    return archive, champion_type  # empty


def load_games_for_week(
    settings: Settings,
    *,
    season: str,
    week: int,
) -> pd.DataFrame:
    """Load champion-model predictions for all games in (season, week).

    Filters the prediction archive to the current win_prob champion's
    output. Joins to the games table for schedule truth (game_date,
    game_time, venue) and converts team names to short codes.

    Args:
        settings: API settings, source of repo_root.
        season: Season label, e.g. "2026-2027".
        week: Week number.

    Returns:
        DataFrame with one row per game. Columns:
            game_id, game_date, week, season,
            away_team, home_team (short codes),
            home_win_prob, away_win_prob,
            model_spread, model_total,
            projected_home_score, projected_away_score,
            confidence_tier,
            win_prob_lo, win_prob_hi (uncertainty band).
        Empty DataFrame if no games match.

    Raises:
        ChampionNotFoundError: If the champion manifest is missing or
            has no win_prob entry.
    """
    archive, _ = _resolve_win_prob_archive(settings, season=season, week=week)
    if archive.empty:
        return archive

    games: DataFrame = load_games_df(settings)
    long_to_short: dict[str, str] = load_team_name_map(settings)

    return _finalize_games_frame(archive, games, long_to_short)


def load_game(
    settings: Settings,
    *,
    game_id: str,
) -> dict | None:
    """Load champion-model prediction for one game.

    Same champion-filtering and enrichment as ``load_games_for_week``,
    but returns a dict for a single game rather than a DataFrame.

    Args:
        settings: API settings, source of repo_root.
        game_id: Composite game_id, e.g. "2026_01_KC_LAC".

    Returns:
        Dict of the fields listed in ``load_games_for_week``'s docstring,
        or ``None`` if the game_id is not in the archive.

    Raises:
        ChampionNotFoundError: If the champion manifest is missing or
            has no win_prob entry.
    """
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.evaluation.champion_resolver import resolve_current_champion

    _, champion_type = resolve_current_champion("win_prob", repo=settings.repo_root)

    def _load_for_game(mtype: str) -> DataFrame:
        a = load_prediction_log(
            model_name="win_prob",
            model_type=mtype,
            repo=settings.repo_root,
        )
        return a.loc[a["game_id"] == game_id, :].copy() if not a.empty else a

    archive = _load_for_game(champion_type)
    if archive.empty and champion_type != _UPCOMING_FALLBACK_MODEL_TYPE:
        archive = _load_for_game(_UPCOMING_FALLBACK_MODEL_TYPE)
    if archive.empty:
        return None

    games: DataFrame = load_games_df(settings)
    long_to_short: dict[str, str] = load_team_name_map(settings)

    enriched: DataFrame = _finalize_games_frame(archive, games, long_to_short)
    if enriched.empty:
        return None

    return enriched.iloc[0].to_dict()


def load_edges_for_week(
    settings: Settings,
    *,
    season: str,
    week: int,
    min_ev: float = 0.0,
    bankroll: float | None = None,
    kelly_multiplier: float = 0.25,
) -> pd.DataFrame:
    """Load ranked edges for (season, week) using the champion model.

    Resolves the current win_prob champion, loads its predictions
    filtered by (season, week), joins to the current DK odds snapshot,
    computes edges via ``market.recommendations.build_edge_report``, and
    ranks by EV. Team names are converted to short codes.

    Args:
        settings: API settings, source of repo_root.
        season: Season label, e.g. "2026-2027".
        week: Week number.
        min_ev: Minimum EV threshold. Rows with ev <= min_ev are excluded.
        bankroll: Bankroll for Kelly stake sizing. When None, edge rows
            retain full-Kelly fractions but do not populate kelly_stake.
        kelly_multiplier: Fraction of full Kelly (e.g. 0.25 for quarter).

    Returns:
        DataFrame with columns from ``_REPORT_COLUMNS`` in
        ``market.recommendations``, ranked by EV descending. Team name
        columns hold short codes.

    Raises:
        ChampionNotFoundError: If the champion manifest is missing or
            has no win_prob entry.
        OddsUnavailableError: If the current odds snapshot is missing
            or empty.
    """
    from gridiron_edge.api.exceptions import OddsUnavailableError
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.evaluation.champion_resolver import resolve_current_champion
    from gridiron_edge.ingest.odds.store import load_current_odds
    from gridiron_edge.market.recommendations import build_edge_report, rank_edges
    from gridiron_edge.models.game_prediction.post_process import (
        get_margin_std,
        get_total_std,
    )

    _, model_type = resolve_current_champion("win_prob", repo=settings.repo_root)

    predictions: DataFrame = load_prediction_log(
        season=season,
        week=week,
        model_name="win_prob",
        model_type=model_type,
        repo=settings.repo_root,
    )
    if predictions.empty:
        return pd.DataFrame()

    odds: DataFrame | None = load_current_odds(repo=settings.repo_root)
    if odds is None or odds.empty:
        raise OddsUnavailableError(
            f"No current odds snapshot at "
            f"{settings.repo_root}/data/odds/dk_odds_current.parquet. "
            f"Run `gridiron ingest fetch-odds` to refresh."
        )

    margin_std: float = get_margin_std("win_prob", model_type)
    total_std: float = get_total_std("total", model_type, default=13.0)

    edge_report: DataFrame = build_edge_report(
        predictions,
        odds,
        margin_std=margin_std,
        total_std=total_std,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
    )
    if edge_report.empty:
        return pd.DataFrame()

    ranked: DataFrame = rank_edges(edge_report, min_ev=min_ev)
    if ranked.empty:
        return pd.DataFrame()

    long_to_short: dict[str, str] = load_team_name_map(settings)
    ranked["away_team"] = ranked["away_team"].map(long_to_short).fillna(ranked["away_team"])
    ranked["home_team"] = ranked["home_team"].map(long_to_short).fillna(ranked["home_team"])

    return ranked


def _finalize_games_frame(
    archive: pd.DataFrame,
    games: pd.DataFrame,
    long_to_short: dict[str, str],
) -> pd.DataFrame:
    """Convert archive rows to the API-facing games shape.

    Shared by ``load_games_for_week`` and ``load_game``. Joins the
    archive to the games table for schedule truth, converts long team
    names to short codes, and selects the API-relevant columns.
    """
    # Join to games for schedule truth. Left join preserves archive
    # rows that might reference upcoming games not yet in the games
    # table (games table is populated post-week).
    merged: DataFrame = archive.merge(
        games[["GAME_ID", "YEAR", "WEEK_NUM", "GAME_DATE"]],
        left_on="game_id",
        right_on="GAME_ID",
        how="left",
        suffixes=("", "_games"),
    )

    # Prefer games table for game_date when available (schedule truth);
    # fall back to archive's game_date.
    merged["game_date"] = merged["GAME_DATE"].fillna(merged["game_date"])

    # Team name conversion to short codes.
    merged["away_team"] = merged["away_team"].map(long_to_short).fillna(merged["away_team"])
    merged["home_team"] = merged["home_team"].map(long_to_short).fillna(merged["home_team"])

    columns: list[str] = [
        "game_id",
        "game_date",
        "week",
        "season",
        "away_team",
        "home_team",
        "home_win_prob",
        "away_win_prob",
        "model_spread",
        "model_total",
        "projected_home_score",
        "projected_away_score",
        "confidence_tier",
        "win_prob_lo",
        "win_prob_hi",
    ]
    return merged.loc[:, columns].copy()


def _parse_season_int(season: str) -> int:
    """Parse the leading int year from a season label.

    Examples:
        "2026-2027" -> 2026
        "2026"      -> 2026
    """
    try:
        return int(season.split("-")[0])
    except (ValueError, IndexError, AttributeError) as exc:
        msg: str = f"Cannot parse season {season!r}; expected 'YYYY' or 'YYYY-YYYY+1'."
        raise ValueError(msg) from exc


def load_props_for_week(
    settings: Settings,
    *,
    season: str,
    week: int,
    stat_type: str | None = None,
    position: str | None = None,
) -> pd.DataFrame:
    """Load champion-model prop predictions for (season, week).

    Iterates registered prop stat families (or just ``stat_type`` when
    provided), resolves each family's current champion via
    :func:`resolve_current_champion`, and filters the prop archive to
    that champion's rows. Families without a resolved champion are
    silently skipped.

    Args:
        settings: API settings, source of repo_root.
        season: Season label, e.g. "2026-2027".
        week: Week number.
        stat_type: Optional single-family filter. When set, only this
            family is processed.
        position: Optional position filter applied to the returned
            rows (e.g. "QB", "RB").

    Returns:
        DataFrame with columns from ``_ARCHIVE_COLUMNS`` in
        ``evaluation.prop_archive``, filtered to the champion for each
        family. Empty DataFrame if no families produced rows after
        filtering.

    Raises:
        ChampionNotFoundError: If zero families resolved a champion.
            Not raised when families resolve but their archive is empty.
    """
    from gridiron_edge.evaluation.champion_resolver import (
        ChampionNotFoundError,
        resolve_current_champion,
    )
    from gridiron_edge.evaluation.prop_archive import load_prop_archive
    from gridiron_edge.models.catalog import PROP_STAT_FAMILIES

    season_int: int = _parse_season_int(season)
    families: list[str] = [stat_type] if stat_type else PROP_STAT_FAMILIES

    resolved_frames: list[pd.DataFrame] = []
    resolved_families: list[str] = []

    for family in families:
        try:
            _, model_type = resolve_current_champion(
                family,
                repo=settings.repo_root,
            )
        except ChampionNotFoundError:
            continue

        family_rows: DataFrame = load_prop_archive(
            repo=settings.repo_root,
            stat_type=family,
            season=season_int,
        )
        if family_rows.empty:
            resolved_families.append(family)
            continue

        filtered = family_rows.loc[
            (family_rows["model_type"] == model_type) & (family_rows["week"] == week),
            :,
        ]
        if position is not None:
            filtered = filtered.loc[filtered["position"] == position, :]

        resolved_families.append(family)
        if not filtered.empty:
            resolved_frames.append(filtered)

    if not resolved_families:
        raise ChampionNotFoundError(
            "No prop champions registered. Run `gridiron full-retrain` or "
            "`gridiron props champion --write-manifest`."
        )

    if not resolved_frames:
        return pd.DataFrame()

    return pd.concat(resolved_frames, ignore_index=True)


def load_prop(
    settings: Settings,
    *,
    game_id: str,
    player_id: str,
    stat_type: str,
) -> dict | None:
    """Load champion-model prop prediction for one (game_id, player_id, stat_type).

    Args:
        settings: API settings, source of repo_root.
        game_id: Composite game_id.
        player_id: Player identifier.
        stat_type: Prop stat family (e.g. "qb_pass_yards").

    Returns:
        Dict with archive-schema fields, or None if the composite
        doesn't match any archived prediction under the current champion.

    Raises:
        ChampionNotFoundError: If the champion manifest has no entry
            for ``stat_type``.
    """
    from gridiron_edge.evaluation.champion_resolver import resolve_current_champion
    from gridiron_edge.evaluation.prop_archive import load_prop_archive

    _, model_type = resolve_current_champion(
        stat_type,
        repo=settings.repo_root,
    )

    archive: DataFrame = load_prop_archive(
        repo=settings.repo_root,
        stat_type=stat_type,
    )
    if archive.empty:
        return None

    match = archive.loc[
        (archive["game_id"] == game_id)
        & (archive["player_id"] == player_id)
        & (archive["model_type"] == model_type),
        :,
    ]
    if match.empty:
        return None

    return match.iloc[0].to_dict()


def load_opponent_allowed_for_prop(
    settings: Settings,
    *,
    opponent_team: str,
    position: str,
    stat_type: str,
) -> dict[str, dict]:
    """Load opponent-allowed data for a specific defense vs position vs stat.

    Reads the opponent_allowed.parquet artifact and filters to matching
    rows. Returns nested dict of cohort → aggregates.

    Args:
        settings: API settings.
        opponent_team: Opponent's short team code (e.g. "LAC").
        position: Position matching the stat_type (e.g. "QB").
        stat_type: Stat family (e.g. "qb_pass_yards").

    Returns:
        Dict mapping cohort name → dict of stats. Empty if no artifact
        exists or no rows match.

    Example:
        {
            "season": {"avg_allowed": 275.0, "sample_size": 2,
                       "rank_against_position": 3},
            "l4": {"avg_allowed": 275.0, "sample_size": 2,
                   "rank_against_position": 3},
        }
    """
    from gridiron_edge.evaluation.opponent_allowed import load_opponent_allowed

    df = load_opponent_allowed(settings.repo_root)
    if df.empty:
        return {}

    match = df.loc[
        (df["opponent_team"] == opponent_team)
        & (df["position"] == position)
        & (df["stat_type"] == stat_type),
        :,
    ]

    if match.empty:
        return {}

    result: dict[str, dict] = {}
    for _, row in match.iterrows():
        cohort = str(row["cohort"])
        result[cohort] = {
            "avg_allowed": _none_if_nan_float(row["avg_allowed"]),
            "sample_size": _none_if_nan_int(row["sample_size"]),
            "rank_against_position": _none_if_nan_int(row["rank_against_position"]),
        }

    return result


def load_team_cohort_splits_df(settings: Settings) -> pd.DataFrame:
    """Load the team cohort splits artifact.

    Returns:
        DataFrame with team_abbr, cohort, and metric columns from
        team_cohort_splits.parquet. Empty DataFrame if the artifact
        doesn't exist.
    """
    from gridiron_edge.evaluation.team_cohort_splits import load_team_cohort_splits

    return load_team_cohort_splits(settings.repo_root)


def format_team_cohort_splits(
    df: pd.DataFrame,
    team_abbr: str,
) -> dict[str, dict] | None:
    """Format cohort splits DataFrame into a nested dict for one team.

    Args:
        df: DataFrame from load_team_cohort_splits_df.
        team_abbr: Team short code (e.g. "KC").

    Returns:
        Nested dict {cohort: {metric_name: value, ...}} or None if the
        team isn't in the DataFrame or df is empty.
    """
    if df.empty:
        return None

    team_rows = df.loc[df["team_abbr"] == team_abbr, :]
    if team_rows.empty:
        return None

    result: dict[str, dict] = {}
    for _, row in team_rows.iterrows():
        cohort = str(row["cohort"])
        # Collect all non-identity columns for this cohort row.
        cohort_data: dict = {}
        for col in row.index:
            if col in ("team_abbr", "cohort"):
                continue
            val = row[col]
            # Convert numpy/pandas types to Python primitives.
            if pd.isna(val):
                cohort_data[str(col)] = None
            elif hasattr(val, "item"):
                cohort_data[str(col)] = val.item()
            else:
                cohort_data[str(col)] = val
        result[cohort] = cohort_data

    return result


def load_team_metadata(settings: Settings) -> pd.DataFrame:
    """Load the unified team metadata CSV via the datasets registry.

    Returns:
        DataFrame with columns NFL_LONG_NAME, NFL_SHORT_NAME, city,
        name, conf, div, primary_color, secondary_color. Empty if
        the file does not exist yet.
    """
    from gridiron_edge.datasets.registry import dataset_path

    path: Path = dataset_path(settings.repo_root, "team_metadata")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def team_metadata_lookup(settings: Settings) -> dict[str, dict]:
    """Return dict of long_name → metadata dict.

    Metadata dict has keys: city, name, conference, division,
    primary_color, secondary_color. All None if team not found.
    """
    df: DataFrame = load_team_metadata(settings)
    if df.empty:
        return {}

    def _clean(v: Any) -> str | None:  # noqa: ANN401
        """Return None for NaN, else str value."""
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return str(v)

    return {
        str(row["NFL_LONG_NAME"]): {
            "city": _clean(row.get("city")),
            "name": _clean(row.get("name")),
            "conference": _clean(row.get("conf")),
            "division": _clean(row.get("div")),
            "primary_color": _clean(row.get("primary_color")),
            "secondary_color": _clean(row.get("secondary_color")),
        }
        for _, row in df.iterrows()
    }


def load_defense_allowed(
    settings: Settings,
    *,
    team: str,
    stat_type: str,
) -> tuple[str, dict[str, dict]]:
    """Load a team's allowed aggregates for a stat_type, all cohorts.

    Derives the position from stat_type and reuses the opponent-allowed
    filter. Keyed on team directly (not a prop) so arbitrary team +
    stat_type combinations work for the Compare Player-vs-Defense
    independent-team picker.

    Args:
        settings: API settings.
        team: Defense's short team code (e.g. "SF").
        stat_type: Stat family (e.g. "rb_rush_yards").

    Returns:
        Tuple of (position, cohorts_dict). position is "" if the
        stat_type is unknown. cohorts_dict is cohort → aggregates,
        empty if no rows match.
    """
    from gridiron_edge.evaluation.opponent_allowed import STAT_POSITION_MAP

    position = STAT_POSITION_MAP.get(stat_type, "")
    if not position:
        return ("", {})

    cohorts = load_opponent_allowed_for_prop(
        settings,
        opponent_team=team,
        position=position,
        stat_type=stat_type,
    )
    return (position, cohorts)

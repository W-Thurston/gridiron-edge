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
    """Resolve the current (season, week) from the games table.

    Uses the latest completed game. Returns ("", 0) if games is empty.
    """
    from gridiron_edge.datasets.loaders import load_games

    games = load_games(settings.repo_root)
    if games.empty:
        return ("", 0)

    games_sorted = games.sort_values(["YEAR", "WEEK_NUM"])
    latest = games_sorted.iloc[-1]
    return (str(latest["YEAR"]), int(latest["WEEK_NUM"]))


def load_projections_summary_df(
    settings: Settings,
) -> tuple[pd.DataFrame, str | None]:
    """Load the projections summary CSV.

    Returns:
        Tuple of (dataframe, csv_mtime_iso). The mtime is the CSV file's
        last-modified time as an ISO string, useful for staleness display.
        Returns (empty_df, None) if the CSV doesn't exist.
    """
    from datetime import UTC, datetime

    path = settings.repo_root / "data" / "output" / "temp" / "projections_summary.csv"

    if not path.exists():
        return pd.DataFrame(), None

    df = pd.read_csv(path)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    return df, mtime

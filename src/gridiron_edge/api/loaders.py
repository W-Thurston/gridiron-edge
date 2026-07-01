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

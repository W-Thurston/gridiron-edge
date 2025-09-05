from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class SnapshotPolicy:
    name: str  # "EARLY_WED_10ET" | "T_MINUS_24H_ET"


def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize(ET)
    return ts.tz_convert(UTC)


def to_et(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.tz_convert(ET)


def previous_weekday_at(
    dt_et: pd.Timestamp, weekday: int, hour: int, minute: int = 0
) -> pd.Timestamp:
    """Return the most recent given weekday (Mon=0..Sun=6) at HH:MM ET before dt_et."""
    dt = dt_et.tz_convert(ET)
    days_back = (dt.weekday() - weekday) % 7
    candidate = (dt - pd.Timedelta(days=days_back)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    if candidate > dt:
        candidate -= pd.Timedelta(days=7)
    return candidate.tz_convert(ET)


def assign_snapshot_utc_weekwise(games_week_df: pd.DataFrame, policy: SnapshotPolicy) -> pd.Series:
    """
    Assign per-game snapshot timestamps in UTC for a group (single season+week).
    Requires column 'kickoff_et' tz-aware.
    """
    if len(games_week_df) == 0:
        return pd.Series(dtype="datetime64[ns, UTC]")
    if games_week_df["kickoff_et"].dt.tz is None:
        raise ValueError("kickoff_et must be timezone-aware ET")

    if policy.name == "EARLY_WED_10ET":
        earliest = games_week_df["kickoff_et"].min()
        snap_et = previous_weekday_at(
            earliest, weekday=2, hour=10
        )  # Wed 10:00 ET before earliest kickoff
        return pd.Series([snap_et.tz_convert(UTC)] * len(games_week_df), index=games_week_df.index)

    if policy.name == "T_MINUS_24H_ET":
        return (games_week_df["kickoff_et"] - pd.Timedelta(hours=24)).dt.tz_convert(UTC)

    raise ValueError(f"Unknown snapshot policy: {policy.name}")

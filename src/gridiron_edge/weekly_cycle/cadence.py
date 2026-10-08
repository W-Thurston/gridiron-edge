# src/gridiron_edge/weekly_cycle/cadence.py

"""When each weekly-cycle step is due (``DECISIONS.md`` D82)."""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from typing import Final

#: Closeout starts at this local wall-clock time once the finished week's last game is over.
CLOSEOUT_LOCAL_HOUR: Final[int] = 6
#: A finished game is assumed over this long after its kickoff when timing the closeout.
GAME_LENGTH_ALLOWANCE: Final[timedelta] = timedelta(hours=4)
#: Minimum gap between attempts of a step that failed, was blocked or found nothing yet.
RETRY_AFTER: Final[timedelta] = timedelta(hours=1)
#: How long after a closeout that soft-failed its recap the cycle keeps retrying the recap hourly;
#: set before any retry was observed (``DECISIONS.md`` D91).
RECAP_RETRY_WINDOW: Final[timedelta] = timedelta(days=7)
#: A closeout still waiting this long after its window opened needs a person; set before any
#: closeout had waited (``DECISIONS.md`` D92).
ALERT_WAITING_AFTER: Final[timedelta] = timedelta(hours=12)
#: How often an alert that stays open is sent again (``DECISIONS.md`` D92).
ALERT_REMINDER_INTERVAL: Final[timedelta] = timedelta(hours=24)
#: The timer fires with jitter, so a step is due slightly before its nominal interval.
TICK_SLACK: Final[timedelta] = timedelta(minutes=5)

#: ``(hours to the next unstarted kickoff above, hours between starter checks)``, in order.
#: These tiers mirror the odds collector's polling policy and are a starting point chosen
#: before any evidence about how often nflverse refreshes these files; the watch receipts
#: record whether each check saw a change so they can be set from data (``DECISIONS.md`` D82).
WATCH_TIERS: Final[tuple[tuple[float, float], ...]] = (
    (72.0, 24.0),
    (48.0, 12.0),
    (24.0, 6.0),
    (6.0, 3.0),
)
WATCH_FINAL_INTERVAL_HOURS: Final[float] = 1.0


def watch_interval(hours_to_kickoff: float) -> timedelta:
    """Return the gap between starter checks at ``hours_to_kickoff`` before the next kickoff."""
    for above, interval in WATCH_TIERS:
        if hours_to_kickoff > above:
            return timedelta(hours=interval)
    return timedelta(hours=WATCH_FINAL_INTERVAL_HOURS)


def is_due(last: datetime | None, interval: timedelta, *, now: datetime) -> bool:
    """Return whether ``interval`` has elapsed since ``last`` (up to the tick slack)."""
    return last is None or now - last >= interval - TICK_SLACK


def next_due(last: datetime | None, interval: timedelta, *, now: datetime) -> datetime:
    """Return when a step last run at ``last`` is next due (``now`` when it is already)."""
    if last is None:
        return now
    return max(now, last + interval - TICK_SLACK)


def closeout_earliest(last_kickoff: datetime, *, tz: tzinfo | None) -> datetime:
    """Return the first 06:00 local time at or after the finished week's last game is over."""
    over = (last_kickoff + GAME_LENGTH_ALLOWANCE).astimezone(tz)
    start = over.replace(hour=CLOSEOUT_LOCAL_HOUR, minute=0, second=0, microsecond=0)
    if start < over:
        start += timedelta(days=1)
    return start

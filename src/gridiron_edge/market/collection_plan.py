"""Pure scheduler-neutral planning for weekly quote collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

import pandas as pd
from pandas import DataFrame

SCHEMA_VERSION = 1
EASTERN = ZoneInfo("America/New_York")
_REQUIRED_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "game_id",
    "game_date",
    "game_time",
)


class CollectionPlanStatus(StrEnum):
    """Availability of a weekly collection plan."""

    AVAILABLE = "available"
    SCHEDULE_UNAVAILABLE = "schedule_unavailable"


class CollectionReason(StrEnum):
    """Policy band that proposed one collection."""

    BASELINE = "baseline"
    APPROACH = "approach"
    NEAR_KICKOFF = "near_kickoff"


@dataclass(frozen=True, slots=True)
class QuoteCollectionPolicy:
    """Configurable weekly provider budget and ramp guideline."""

    weekly_poll_limit: int = 34
    credit_cost_per_poll: int = 3
    baseline_interval_hours: int = 12
    approach_window_hours: int = 24
    approach_interval_hours: int = 3
    near_window_hours: int = 6
    near_interval_hours: int = 1

    def __post_init__(self) -> None:
        """Validate the collection budget and ramp intervals."""
        positive = (
            self.weekly_poll_limit,
            self.credit_cost_per_poll,
            self.baseline_interval_hours,
            self.approach_window_hours,
            self.approach_interval_hours,
            self.near_window_hours,
            self.near_interval_hours,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Collection policy values must be positive.")
        if self.near_window_hours >= self.approach_window_hours:
            raise ValueError("near_window_hours must be less than approach_window_hours.")
        if not (
            self.near_interval_hours <= self.approach_interval_hours <= self.baseline_interval_hours
        ):
            raise ValueError(
                "Collection intervals must increase from near to approach to baseline."
            )


@dataclass(frozen=True, slots=True)
class KickoffGroup:
    """Games sharing one exact UTC kickoff instant."""

    commence_time: datetime
    game_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlannedQuoteCollection:
    """One proposed provider collection with its schedule context."""

    scheduled_at: datetime
    next_kickoff: datetime
    hours_to_next_kickoff: float
    reason: CollectionReason


@dataclass(frozen=True, slots=True)
class WeeklyQuoteCollectionPlan:
    """Immutable reviewable plan for one NFL season-and-week scope."""

    schema_version: int
    status: CollectionPlanStatus
    season: str
    week: int
    created_at: datetime
    plan_start: datetime
    policy: QuoteCollectionPolicy
    kickoff_groups: tuple[KickoffGroup, ...]
    polls: tuple[PlannedQuoteCollection, ...]
    planned_poll_count: int
    planned_credit_cost: int
    remaining_poll_capacity: int
    omitted_candidate_count: int


def build_weekly_quote_collection_plan(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
    plan_start: datetime,
    created_at: datetime,
    policy: QuoteCollectionPolicy | None = None,
) -> WeeklyQuoteCollectionPlan:
    """Build one deterministic collection plan without provider access or I/O."""
    resolved_policy = policy or QuoteCollectionPolicy()
    normalized_season = season.strip()
    if not normalized_season:
        raise ValueError("season must not be empty.")
    if week < 1 or week > 22:
        raise ValueError("week must be between 1 and 22.")
    plan_start = _require_utc(plan_start, label="plan_start")
    created_at = _require_utc(created_at, label="created_at")
    groups = derive_kickoff_groups(schedule, season=normalized_season, week=week)
    if not groups:
        plan = WeeklyQuoteCollectionPlan(
            schema_version=SCHEMA_VERSION,
            status=CollectionPlanStatus.SCHEDULE_UNAVAILABLE,
            season=normalized_season,
            week=week,
            created_at=created_at,
            plan_start=plan_start,
            policy=resolved_policy,
            kickoff_groups=(),
            polls=(),
            planned_poll_count=0,
            planned_credit_cost=0,
            remaining_poll_capacity=resolved_policy.weekly_poll_limit,
            omitted_candidate_count=0,
        )
        validate_weekly_quote_collection_plan(plan)
        return plan

    candidates = _generate_candidates(groups, plan_start=plan_start, policy=resolved_policy)
    selected = _allocate_candidates(
        candidates,
        groups=groups,
        limit=resolved_policy.weekly_poll_limit,
    )
    polls = tuple(sorted(selected, key=lambda item: (item.scheduled_at, item.next_kickoff)))
    plan = WeeklyQuoteCollectionPlan(
        schema_version=SCHEMA_VERSION,
        status=CollectionPlanStatus.AVAILABLE,
        season=normalized_season,
        week=week,
        created_at=created_at,
        plan_start=plan_start,
        policy=resolved_policy,
        kickoff_groups=groups,
        polls=polls,
        planned_poll_count=len(polls),
        planned_credit_cost=len(polls) * resolved_policy.credit_cost_per_poll,
        remaining_poll_capacity=resolved_policy.weekly_poll_limit - len(polls),
        omitted_candidate_count=len(candidates) - len(polls),
    )
    validate_weekly_quote_collection_plan(plan)
    return plan


def derive_kickoff_groups(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
) -> tuple[KickoffGroup, ...]:
    """Derive exact UTC kickoff groups from Eastern nflverse schedule values."""
    missing = sorted(set(_REQUIRED_SCHEDULE_COLUMNS) - set(schedule.columns))
    if missing:
        raise ValueError("Schedule is missing required columns: " + ", ".join(missing))
    rows = schedule.loc[
        (schedule["season"].astype(str) == season) & (schedule["week"] == week),
        list(_REQUIRED_SCHEDULE_COLUMNS),
    ].copy()
    if rows.empty:
        return ()
    identifiers = rows["game_id"]
    if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
        raise ValueError("Scoped schedule game_id values must be nonempty.")
    if identifiers.astype(str).duplicated().any():
        raise ValueError("Scoped schedule game_id values must be unique.")

    grouped: dict[datetime, list[str]] = {}
    for row in rows.itertuples(index=False):
        kickoff = _schedule_kickoff_utc(row.game_date, row.game_time)
        grouped.setdefault(kickoff, []).append(str(row.game_id).strip())
    return tuple(
        KickoffGroup(commence_time=kickoff, game_ids=tuple(sorted(game_ids)))
        for kickoff, game_ids in sorted(grouped.items())
    )


def validate_weekly_quote_collection_plan(  # noqa: PLR0912
    plan: WeeklyQuoteCollectionPlan,
) -> None:
    """Validate generated or deserialized plan semantics."""
    if plan.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported collection plan schema_version: {plan.schema_version}.")
    _require_utc(plan.created_at, label="created_at")
    _require_utc(plan.plan_start, label="plan_start")
    if plan.planned_poll_count != len(plan.polls):
        raise ValueError("planned_poll_count does not match polls.")
    if plan.planned_poll_count > plan.policy.weekly_poll_limit:
        raise ValueError("Collection plan exceeds weekly_poll_limit.")
    expected_cost = plan.planned_poll_count * plan.policy.credit_cost_per_poll
    if plan.planned_credit_cost != expected_cost:
        raise ValueError("planned_credit_cost does not match poll count and policy.")
    if plan.remaining_poll_capacity != plan.policy.weekly_poll_limit - len(plan.polls):
        raise ValueError("remaining_poll_capacity is inconsistent.")
    kickoffs = tuple(group.commence_time for group in plan.kickoff_groups)
    if kickoffs != tuple(sorted(set(kickoffs))):
        raise ValueError("kickoff_groups must be uniquely ordered.")
    for group in plan.kickoff_groups:
        _require_utc(group.commence_time, label="kickoff commence_time")
        if not group.game_ids or group.game_ids != tuple(sorted(set(group.game_ids))):
            raise ValueError("Kickoff group game_ids must be nonempty, unique, and sorted.")
    scheduled = tuple(poll.scheduled_at for poll in plan.polls)
    if scheduled != tuple(sorted(set(scheduled))):
        raise ValueError("Plan polls must be uniquely ordered by scheduled_at.")
    kickoff_set = set(kickoffs)
    for poll in plan.polls:
        _require_utc(poll.scheduled_at, label="poll scheduled_at")
        _require_utc(poll.next_kickoff, label="poll next_kickoff")
        if poll.scheduled_at < plan.plan_start:
            raise ValueError("Plan poll occurs before plan_start.")
        if poll.next_kickoff not in kickoff_set:
            raise ValueError("Plan poll references an unknown kickoff group.")
        if poll.scheduled_at >= poll.next_kickoff:
            raise ValueError("Every plan poll must precede its next kickoff.")
        hours = (poll.next_kickoff - poll.scheduled_at).total_seconds() / 3600
        if abs(hours - poll.hours_to_next_kickoff) > 1e-9:
            raise ValueError("hours_to_next_kickoff is inconsistent.")
    if plan.status is CollectionPlanStatus.SCHEDULE_UNAVAILABLE:
        if plan.kickoff_groups or plan.polls:
            raise ValueError("Unavailable plans cannot contain kickoffs or polls.")
    elif not plan.kickoff_groups:
        raise ValueError("Available plans require kickoff groups.")


def _schedule_kickoff_utc(game_date: object, game_time: object) -> datetime:
    """Combine nflverse Eastern date/time values and convert to UTC."""
    # pyrefly: ignore [no-matching-overload]
    if pd.isna(game_date) or pd.isna(game_time):
        raise ValueError("Scoped schedule kickoff date and time must be present.")
    try:
        parsed_date = date.fromisoformat(str(game_date))
        parsed_time = time.fromisoformat(str(game_time))
    except ValueError as exc:
        raise ValueError("Scoped schedule kickoff date and time must be valid.") from exc
    if parsed_time.tzinfo is not None:
        raise ValueError("Schedule game_time must be an Eastern local wall time.")
    return datetime.combine(parsed_date, parsed_time, tzinfo=EASTERN).astimezone(UTC)


def _generate_candidates(
    groups: tuple[KickoffGroup, ...],
    *,
    plan_start: datetime,
    policy: QuoteCollectionPolicy,
) -> tuple[PlannedQuoteCollection, ...]:
    """Generate unique candidates against the next future kickoff group."""
    last_kickoff = groups[-1].commence_time
    candidates: dict[datetime, PlannedQuoteCollection] = {}
    current = plan_start
    while current < last_kickoff:
        next_group = next((group for group in groups if group.commence_time > current), None)
        if next_group is None:
            break
        hours = (next_group.commence_time - current).total_seconds() / 3600
        if hours <= 1:
            current = next_group.commence_time
            continue
        if hours <= policy.near_window_hours:
            reason = CollectionReason.NEAR_KICKOFF
            interval = policy.near_interval_hours
        elif hours <= policy.approach_window_hours:
            reason = CollectionReason.APPROACH
            interval = policy.approach_interval_hours
        else:
            reason = CollectionReason.BASELINE
            interval = policy.baseline_interval_hours
        candidates[current] = PlannedQuoteCollection(
            scheduled_at=current,
            next_kickoff=next_group.commence_time,
            hours_to_next_kickoff=hours,
            reason=reason,
        )
        current = min(current + timedelta(hours=interval), next_group.commence_time)
    return tuple(sorted(candidates.values(), key=lambda item: item.scheduled_at))


def _allocate_candidates(
    candidates: tuple[PlannedQuoteCollection, ...],
    *,
    groups: tuple[KickoffGroup, ...],
    limit: int,
) -> tuple[PlannedQuoteCollection, ...]:
    """Allocate a bounded deterministic mix of baseline and near evidence."""
    if len(candidates) <= limit:
        return candidates
    chosen: dict[datetime, PlannedQuoteCollection] = {}
    for group in groups:
        group_candidates = [item for item in candidates if item.next_kickoff == group.commence_time]
        baseline = [item for item in group_candidates if item.reason is CollectionReason.BASELINE]
        if baseline and len(chosen) < limit:
            item = min(baseline, key=lambda candidate: candidate.scheduled_at)
            chosen[item.scheduled_at] = item
        if group_candidates and len(chosen) < limit:
            item = min(group_candidates, key=lambda candidate: candidate.hours_to_next_kickoff)
            chosen[item.scheduled_at] = item
    remaining = sorted(
        (item for item in candidates if item.scheduled_at not in chosen),
        key=lambda item: (
            item.hours_to_next_kickoff,
            item.next_kickoff,
            item.scheduled_at,
            item.reason.value,
        ),
    )
    for item in remaining:
        if len(chosen) >= limit:
            break
        chosen[item.scheduled_at] = item
    return tuple(chosen.values())


def _require_utc(value: datetime, *, label: str) -> datetime:
    """Require a timezone-aware UTC datetime and normalize its timezone."""
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value.astimezone(UTC)

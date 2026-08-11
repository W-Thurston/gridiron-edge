"""Atomic JSON persistence for weekly quote collection plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import cast
from uuid import uuid4

from gridiron_edge.market.collection_plan import (
    SCHEMA_VERSION,
    CollectionPlanStatus,
    CollectionReason,
    KickoffGroup,
    PlannedQuoteCollection,
    QuoteCollectionPolicy,
    WeeklyQuoteCollectionPlan,
    validate_weekly_quote_collection_plan,
)

CURRENT_COLLECTION_PLAN_SCHEMA_VERSION = 1
_CURRENT_COLLECTION_PLAN_FILENAME = "current.json"


@dataclass(frozen=True, slots=True)
class CurrentCollectionPlanSelection:
    """Explicit selection of one operational weekly collection plan."""

    schema_version: int
    season: str
    week: int
    selected_at: datetime


def current_collection_plan_path(*, repo: Path) -> Path:
    """Return the explicit current collection-plan selection path."""
    return repo / "data" / "odds" / "collection_plans" / _CURRENT_COLLECTION_PLAN_FILENAME


def select_current_collection_plan(
    *, season: str, week: int, selected_at: datetime, repo: Path
) -> CurrentCollectionPlanSelection:
    """Explicitly select one existing validated weekly collection plan."""
    selected_at = _require_utc(selected_at, label="selected_at")
    plan = read_collection_plan(season=season, week=week, repo=repo)
    if plan.season != season or plan.week != week:
        raise ValueError("Selected collection plan scope does not match its artifact.")
    selection = CurrentCollectionPlanSelection(
        CURRENT_COLLECTION_PLAN_SCHEMA_VERSION, season, week, selected_at
    )
    _validate_current_selection(selection)
    path = current_collection_plan_path(repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_selection_payload(selection), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return selection


def read_current_collection_plan_selection(*, repo: Path) -> CurrentCollectionPlanSelection:
    """Read and validate the explicit current collection-plan selection."""
    payload = json.loads(current_collection_plan_path(repo=repo).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Current collection-plan selection must contain a JSON object.")
    if set(payload) != {"schema_version", "season", "week", "selected_at"}:
        raise ValueError("Current collection-plan selection keys do not match the schema.")
    selection = CurrentCollectionPlanSelection(
        int(cast(int, payload["schema_version"])),
        str(payload["season"]),
        int(cast(int, payload["week"])),
        _datetime(payload["selected_at"]),
    )
    _validate_current_selection(selection)
    return selection


def load_current_collection_plan(*, repo: Path) -> WeeklyQuoteCollectionPlan:
    """Load the explicitly selected and revalidated weekly collection plan."""
    selection = read_current_collection_plan_selection(repo=repo)
    plan = read_collection_plan(season=selection.season, week=selection.week, repo=repo)
    if plan.season != selection.season or plan.week != selection.week:
        raise ValueError("Selected collection plan scope does not match the selection.")
    return plan


def collection_plan_path(*, season: str, week: int, repo: Path) -> Path:
    """Return the deterministic artifact path for one weekly plan."""
    normalized = season.strip()
    if not normalized or not all(
        character.isalnum() or character in {"-", "_"} for character in normalized
    ):
        raise ValueError("season must be a safe nonempty path component.")
    if week < 1 or week > 22:
        raise ValueError("week must be between 1 and 22.")
    return (
        repo
        / "data"
        / "odds"
        / "collection_plans"
        / f"season={normalized}"
        / f"week={week:02d}.json"
    )


def write_collection_plan(plan: WeeklyQuoteCollectionPlan, *, repo: Path) -> Path:
    """Validate and atomically persist one versioned collection plan."""
    validate_weekly_quote_collection_plan(plan)
    path = collection_plan_path(season=plan.season, week=plan.week, repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = _to_payload(plan)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_collection_plan(*, season: str, week: int, repo: Path) -> WeeklyQuoteCollectionPlan:
    """Read and validate one exact versioned collection plan."""
    path = collection_plan_path(season=season, week=week, repo=repo)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Collection plan JSON root must be an object.")
    plan = _from_payload(cast(dict[str, object], payload))
    validate_weekly_quote_collection_plan(plan)
    return plan


def _to_payload(plan: WeeklyQuoteCollectionPlan) -> dict[str, object]:
    """Serialize one plan with explicit ISO timestamps and enum values."""
    return {
        "schema_version": plan.schema_version,
        "status": plan.status.value,
        "season": plan.season,
        "week": plan.week,
        "created_at": plan.created_at.isoformat().replace("+00:00", "Z"),
        "plan_start": plan.plan_start.isoformat().replace("+00:00", "Z"),
        "policy": asdict(plan.policy),
        "kickoff_groups": [
            {
                "commence_time": group.commence_time.isoformat().replace("+00:00", "Z"),
                "game_ids": list(group.game_ids),
            }
            for group in plan.kickoff_groups
        ],
        "polls": [
            {
                "scheduled_at": poll.scheduled_at.isoformat().replace("+00:00", "Z"),
                "next_kickoff": poll.next_kickoff.isoformat().replace("+00:00", "Z"),
                "hours_to_next_kickoff": poll.hours_to_next_kickoff,
                "reason": poll.reason.value,
            }
            for poll in plan.polls
        ],
        "planned_poll_count": plan.planned_poll_count,
        "planned_credit_cost": plan.planned_credit_cost,
        "remaining_poll_capacity": plan.remaining_poll_capacity,
        "omitted_candidate_count": plan.omitted_candidate_count,
    }


def _from_payload(payload: dict[str, object]) -> WeeklyQuoteCollectionPlan:
    """Deserialize the exact plan schema."""
    required = {
        "schema_version",
        "status",
        "season",
        "week",
        "created_at",
        "plan_start",
        "policy",
        "kickoff_groups",
        "polls",
        "planned_poll_count",
        "planned_credit_cost",
        "remaining_poll_capacity",
        "omitted_candidate_count",
    }
    if set(payload) != required:
        raise ValueError("Collection plan JSON keys do not match the current schema.")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported collection plan schema_version.")
    policy_data = cast(dict[str, int], payload["policy"])
    groups_data = cast(list[dict[str, object]], payload["kickoff_groups"])
    polls_data = cast(list[dict[str, object]], payload["polls"])
    return WeeklyQuoteCollectionPlan(
        schema_version=int(cast(int, payload["schema_version"])),
        status=CollectionPlanStatus(str(payload["status"])),
        season=str(payload["season"]),
        week=int(cast(int, payload["week"])),
        created_at=_datetime(payload["created_at"]),
        plan_start=_datetime(payload["plan_start"]),
        policy=QuoteCollectionPolicy(**policy_data),
        kickoff_groups=tuple(
            KickoffGroup(
                commence_time=_datetime(item["commence_time"]),
                game_ids=tuple(str(value) for value in cast(list[object], item["game_ids"])),
            )
            for item in groups_data
        ),
        polls=tuple(
            PlannedQuoteCollection(
                scheduled_at=_datetime(item["scheduled_at"]),
                next_kickoff=_datetime(item["next_kickoff"]),
                hours_to_next_kickoff=float(cast(float, item["hours_to_next_kickoff"])),
                reason=CollectionReason(str(item["reason"])),
            )
            for item in polls_data
        ),
        planned_poll_count=int(cast(int, payload["planned_poll_count"])),
        planned_credit_cost=int(cast(int, payload["planned_credit_cost"])),
        remaining_poll_capacity=int(cast(int, payload["remaining_poll_capacity"])),
        omitted_candidate_count=int(cast(int, payload["omitted_candidate_count"])),
    )


def _validate_current_selection(selection: CurrentCollectionPlanSelection) -> None:
    if selection.schema_version != CURRENT_COLLECTION_PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported current collection-plan selection schema_version.")
    collection_plan_path(season=selection.season, week=selection.week, repo=Path("."))
    _require_utc(selection.selected_at, label="selected_at")


def _selection_payload(selection: CurrentCollectionPlanSelection) -> dict[str, object]:
    return {
        "schema_version": selection.schema_version,
        "season": selection.season,
        "week": selection.week,
        "selected_at": selection.selected_at.isoformat().replace("+00:00", "Z"),
    }


def _require_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value


def _datetime(value: object) -> datetime:
    """Parse one ISO timestamp."""
    if not isinstance(value, str):
        raise ValueError("Collection plan timestamps must be strings.")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

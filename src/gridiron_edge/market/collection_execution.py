"""Single-shot evaluation and execution of validated quote plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from pandas import DataFrame
import requests

from gridiron_edge.ingest.odds.the_odds_api import (
    OddsIngestError,
    OddsIngestPartialPersistenceError,
    OddsRequestError,
    ingest_the_odds_api_current,
)
from gridiron_edge.market.collection_plan import (
    CollectionPlanStatus,
    PlannedQuoteCollection,
    WeeklyQuoteCollectionPlan,
    validate_weekly_quote_collection_plan,
)
from gridiron_edge.market.collection_receipt_store import (
    RECEIPT_SCHEMA_VERSION,
    CollectionExecutionClaim,
    CollectionExecutionResult,
    CollectionExecutionStatus,
    QuotaPrecheckStatus,
    claim_path,
    latest_known_requests_remaining,
    load_results,
    result_path,
    write_claim,
    write_result,
)


class CollectionDueStatus(StrEnum):
    """Pure evaluation state for the earliest unresolved planned poll."""

    PLAN_UNAVAILABLE = "plan_unavailable"
    NOT_DUE = "not_due"
    DUE = "due"
    MISSED = "missed"
    CLAIMED = "claimed"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class CollectionDueResult:
    """Pure due-time evaluation for at most one poll."""

    status: CollectionDueStatus
    poll: PlannedQuoteCollection | None


def evaluate_collection_due(
    plan: WeeklyQuoteCollectionPlan,
    *,
    evaluated_at: datetime,
    grace_period: timedelta,
    repo: Path,
) -> CollectionDueResult:
    """Evaluate only the earliest poll without terminal state."""
    validate_weekly_quote_collection_plan(plan)
    evaluated_at = _require_utc(evaluated_at, label="evaluated_at")
    if grace_period < timedelta(0):
        raise ValueError("grace_period must not be negative.")
    if plan.status is CollectionPlanStatus.SCHEDULE_UNAVAILABLE:
        return CollectionDueResult(CollectionDueStatus.PLAN_UNAVAILABLE, None)

    for poll in plan.polls:
        result = result_path(
            season=plan.season, week=plan.week, scheduled_at=poll.scheduled_at, repo=repo
        )
        claim = claim_path(
            season=plan.season, week=plan.week, scheduled_at=poll.scheduled_at, repo=repo
        )
        if result.exists():
            continue
        if claim.exists():
            return CollectionDueResult(CollectionDueStatus.CLAIMED, poll)
        if evaluated_at < poll.scheduled_at:
            return CollectionDueResult(CollectionDueStatus.NOT_DUE, poll)
        if evaluated_at <= poll.scheduled_at + grace_period:
            return CollectionDueResult(CollectionDueStatus.DUE, poll)
        return CollectionDueResult(CollectionDueStatus.MISSED, poll)
    return CollectionDueResult(CollectionDueStatus.COMPLETED, None)


def execute_due_collection(
    plan: WeeklyQuoteCollectionPlan,
    *,
    schedule: DataFrame,
    api_key: str,
    evaluated_at: datetime,
    repo: Path,
    grace_period: timedelta = timedelta(minutes=15),
    minimum_credit_reserve: int = 30,
    timeout: float = 15.0,
    session: requests.Session | None = None,
) -> CollectionDueResult | CollectionExecutionResult:
    """Execute at most one due poll through the established ingest boundary."""
    if minimum_credit_reserve < 0:
        raise ValueError("minimum_credit_reserve must not be negative.")
    evaluated_at = _require_utc(evaluated_at, label="evaluated_at")
    due = evaluate_collection_due(
        plan, evaluated_at=evaluated_at, grace_period=grace_period, repo=repo
    )
    if due.status not in {CollectionDueStatus.DUE, CollectionDueStatus.MISSED}:
        return due
    assert due.poll is not None
    poll = due.poll
    prior = load_results(season=plan.season, week=plan.week, repo=repo)
    last_known = latest_known_requests_remaining(prior)
    precheck = _quota_precheck(
        last_known,
        request_cost=plan.policy.credit_cost_per_poll,
        reserve=minimum_credit_reserve,
    )
    if due.status is CollectionDueStatus.MISSED:
        return _write_terminal(
            plan,
            poll,
            status=CollectionExecutionStatus.MISSED,
            started_at=evaluated_at,
            completed_at=evaluated_at,
            quota_precheck=precheck,
            minimum_credit_reserve=minimum_credit_reserve,
            last_known=last_known,
            repo=repo,
        )
    if precheck is QuotaPrecheckStatus.BLOCKED:
        return _write_terminal(
            plan,
            poll,
            status=CollectionExecutionStatus.QUOTA_RESERVE_BLOCKED,
            started_at=evaluated_at,
            completed_at=evaluated_at,
            quota_precheck=precheck,
            minimum_credit_reserve=minimum_credit_reserve,
            last_known=last_known,
            repo=repo,
        )

    claim = CollectionExecutionClaim(
        schema_version=RECEIPT_SCHEMA_VERSION,
        season=plan.season,
        week=plan.week,
        scheduled_at=poll.scheduled_at,
        claimed_at=evaluated_at,
        next_kickoff=poll.next_kickoff,
        reason=poll.reason,
    )
    write_claim(claim, repo=repo)
    try:
        ingest = ingest_the_odds_api_current(
            api_key=api_key,
            schedule=schedule,
            season=plan.season,
            week=plan.week,
            repo=repo,
            session=session,
            timeout=timeout,
            fetched_at=evaluated_at,
        )
    except (
        OddsIngestPartialPersistenceError,
        OddsRequestError,
        OddsIngestError,
    ) as exc:
        return _failure_result(
            plan,
            poll,
            evaluated_at=evaluated_at,
            precheck=precheck,
            minimum_credit_reserve=minimum_credit_reserve,
            last_known_requests_remaining=last_known,
            status=_failure_status(exc),
            error=exc,
            repo=repo,
        )

    result = CollectionExecutionResult(
        schema_version=RECEIPT_SCHEMA_VERSION,
        status=CollectionExecutionStatus.COMPLETED,
        season=plan.season,
        week=plan.week,
        scheduled_at=poll.scheduled_at,
        started_at=evaluated_at,
        completed_at=evaluated_at,
        next_kickoff=poll.next_kickoff,
        reason=poll.reason,
        quota_precheck=precheck,
        minimum_credit_reserve=minimum_credit_reserve,
        last_known_requests_remaining=last_known,
        quote_count=ingest.quote_count,
        game_count=ingest.game_count,
        sportsbook_count=ingest.sportsbook_count,
        requests_remaining=ingest.usage.requests_remaining,
        requests_used=ingest.usage.requests_used,
        request_cost=ingest.usage.request_cost,
        ledger_path=str(ingest.ledger_path),
        snapshot_path=str(ingest.snapshot_path),
    )
    write_result(result, repo=repo)
    return result


def _quota_precheck(
    remaining: int | None, *, request_cost: int, reserve: int
) -> QuotaPrecheckStatus:
    if remaining is None:
        return QuotaPrecheckStatus.UNKNOWN
    if remaining < request_cost + reserve:
        return QuotaPrecheckStatus.BLOCKED
    return QuotaPrecheckStatus.SATISFIED


def _write_terminal(
    plan: WeeklyQuoteCollectionPlan,
    poll: PlannedQuoteCollection,
    *,
    status: CollectionExecutionStatus,
    started_at: datetime,
    completed_at: datetime,
    quota_precheck: QuotaPrecheckStatus,
    minimum_credit_reserve: int,
    last_known: int | None,
    repo: Path,
) -> CollectionExecutionResult:
    result = CollectionExecutionResult(
        schema_version=RECEIPT_SCHEMA_VERSION,
        status=status,
        season=plan.season,
        week=plan.week,
        scheduled_at=poll.scheduled_at,
        started_at=started_at,
        completed_at=completed_at,
        next_kickoff=poll.next_kickoff,
        reason=poll.reason,
        quota_precheck=quota_precheck,
        minimum_credit_reserve=minimum_credit_reserve,
        last_known_requests_remaining=last_known,
    )
    write_result(result, repo=repo)
    return result


OddsCollectionError = OddsIngestError | OddsIngestPartialPersistenceError | OddsRequestError


def _failure_status(
    error: OddsCollectionError,
) -> CollectionExecutionStatus:
    if isinstance(error, OddsIngestPartialPersistenceError):
        return CollectionExecutionStatus.PARTIAL_PERSISTENCE
    if isinstance(error, OddsRequestError):
        return CollectionExecutionStatus.REQUEST_FAILED
    return CollectionExecutionStatus.INGEST_FAILED


def _failure_result(
    plan: WeeklyQuoteCollectionPlan,
    poll: PlannedQuoteCollection,
    *,
    evaluated_at: datetime,
    precheck: QuotaPrecheckStatus,
    minimum_credit_reserve: int,
    last_known_requests_remaining: int | None,
    status: CollectionExecutionStatus,
    error: OddsCollectionError,
    repo: Path,
) -> CollectionExecutionResult:
    result = CollectionExecutionResult(
        schema_version=RECEIPT_SCHEMA_VERSION,
        status=status,
        season=plan.season,
        week=plan.week,
        scheduled_at=poll.scheduled_at,
        started_at=evaluated_at,
        completed_at=evaluated_at,
        next_kickoff=poll.next_kickoff,
        reason=poll.reason,
        quota_precheck=precheck,
        minimum_credit_reserve=minimum_credit_reserve,
        last_known_requests_remaining=last_known_requests_remaining,
        error_type=type(error).__name__,
        error_message=str(error),
    )
    write_result(result, repo=repo)
    return result


def _require_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value.astimezone(UTC)

"""Immutable claims and terminal results for planned quote collections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import json
from pathlib import Path
from typing import cast

from gridiron_edge.market.collection_plan import CollectionReason

RECEIPT_SCHEMA_VERSION = 1


class QuotaPrecheckStatus(StrEnum):
    """Knowledge available before a provider request."""

    UNKNOWN = "unknown"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"


class CollectionExecutionStatus(StrEnum):
    """Terminal outcome for one planned collection."""

    COMPLETED = "completed"
    MISSED = "missed"
    QUOTA_RESERVE_BLOCKED = "quota_reserve_blocked"
    REQUEST_FAILED = "request_failed"
    INGEST_FAILED = "ingest_failed"
    PARTIAL_PERSISTENCE = "partial_persistence"


@dataclass(frozen=True, slots=True)
class CollectionExecutionClaim:
    """Exclusive declaration that one planned poll began execution."""

    schema_version: int
    season: str
    week: int
    scheduled_at: datetime
    claimed_at: datetime
    next_kickoff: datetime
    reason: CollectionReason


@dataclass(frozen=True, slots=True)
class CollectionExecutionResult:
    """Immutable terminal outcome for one planned poll."""

    schema_version: int
    status: CollectionExecutionStatus
    season: str
    week: int
    scheduled_at: datetime
    started_at: datetime
    completed_at: datetime
    next_kickoff: datetime
    reason: CollectionReason
    quota_precheck: QuotaPrecheckStatus
    minimum_credit_reserve: int
    last_known_requests_remaining: int | None = None
    quote_count: int | None = None
    game_count: int | None = None
    sportsbook_count: int | None = None
    requests_remaining: int | None = None
    requests_used: int | None = None
    request_cost: int | None = None
    ledger_path: str | None = None
    snapshot_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None


def collection_run_directory(*, season: str, week: int, scheduled_at: datetime, repo: Path) -> Path:
    """Return the deterministic directory for one planned poll execution."""
    normalized = season.strip()
    if not normalized or not all(char.isalnum() or char in {"-", "_"} for char in normalized):
        raise ValueError("season must be a safe nonempty path component.")
    if week < 1 or week > 22:
        raise ValueError("week must be between 1 and 22.")
    timestamp = _require_utc(scheduled_at, label="scheduled_at")
    token = timestamp.strftime("%Y-%m-%dT%H%M%SZ")
    return (
        repo
        / "data"
        / "odds"
        / "collection_runs"
        / f"season={normalized}"
        / f"week={week:02d}"
        / f"scheduled_at={token}"
    )


def claim_path(*, season: str, week: int, scheduled_at: datetime, repo: Path) -> Path:
    """Return the immutable claim path for one planned poll."""
    return (
        collection_run_directory(season=season, week=week, scheduled_at=scheduled_at, repo=repo)
        / "claim.json"
    )


def result_path(*, season: str, week: int, scheduled_at: datetime, repo: Path) -> Path:
    """Return the immutable terminal-result path for one planned poll."""
    return (
        collection_run_directory(season=season, week=week, scheduled_at=scheduled_at, repo=repo)
        / "result.json"
    )


def write_claim(claim: CollectionExecutionClaim, *, repo: Path) -> Path:
    """Create one claim atomically without replacing prior content."""
    validate_claim(claim)
    path = claim_path(
        season=claim.season,
        week=claim.week,
        scheduled_at=claim.scheduled_at,
        repo=repo,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_claim_payload(claim), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def write_result(result: CollectionExecutionResult, *, repo: Path) -> Path:
    """Create one terminal result atomically without replacement."""
    validate_result(result)
    path = result_path(
        season=result.season,
        week=result.week,
        scheduled_at=result.scheduled_at,
        repo=repo,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_result_payload(result), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def read_claim(path: Path) -> CollectionExecutionClaim:
    """Read and validate one exact claim artifact."""
    payload = _object_payload(path)
    required = {
        "schema_version",
        "season",
        "week",
        "scheduled_at",
        "claimed_at",
        "next_kickoff",
        "reason",
    }
    if set(payload) != required:
        raise ValueError("Collection claim keys do not match the current schema.")
    claim = CollectionExecutionClaim(
        schema_version=int(cast(int, payload["schema_version"])),
        season=str(payload["season"]),
        week=int(cast(int, payload["week"])),
        scheduled_at=_datetime(payload["scheduled_at"]),
        claimed_at=_datetime(payload["claimed_at"]),
        next_kickoff=_datetime(payload["next_kickoff"]),
        reason=CollectionReason(str(payload["reason"])),
    )
    validate_claim(claim)
    return claim


def read_result(path: Path) -> CollectionExecutionResult:
    """Read and validate one exact terminal-result artifact."""
    payload = _object_payload(path)
    expected = set(asdict(_empty_result()).keys())
    if set(payload) != expected:
        raise ValueError("Collection result keys do not match the current schema.")
    result = CollectionExecutionResult(
        schema_version=int(cast(int, payload["schema_version"])),
        status=CollectionExecutionStatus(str(payload["status"])),
        season=str(payload["season"]),
        week=int(cast(int, payload["week"])),
        scheduled_at=_datetime(payload["scheduled_at"]),
        started_at=_datetime(payload["started_at"]),
        completed_at=_datetime(payload["completed_at"]),
        next_kickoff=_datetime(payload["next_kickoff"]),
        reason=CollectionReason(str(payload["reason"])),
        quota_precheck=QuotaPrecheckStatus(str(payload["quota_precheck"])),
        minimum_credit_reserve=int(cast(int, payload["minimum_credit_reserve"])),
        last_known_requests_remaining=_optional_int(payload["last_known_requests_remaining"]),
        quote_count=_optional_int(payload["quote_count"]),
        game_count=_optional_int(payload["game_count"]),
        sportsbook_count=_optional_int(payload["sportsbook_count"]),
        requests_remaining=_optional_int(payload["requests_remaining"]),
        requests_used=_optional_int(payload["requests_used"]),
        request_cost=_optional_int(payload["request_cost"]),
        ledger_path=_optional_text(payload["ledger_path"]),
        snapshot_path=_optional_text(payload["snapshot_path"]),
        error_type=_optional_text(payload["error_type"]),
        error_message=_optional_text(payload["error_message"]),
    )
    validate_result(result)
    return result


def load_results(*, season: str, week: int, repo: Path) -> tuple[CollectionExecutionResult, ...]:
    """Load terminal results for one weekly scope in scheduled order."""
    root = repo / "data" / "odds" / "collection_runs" / f"season={season}" / f"week={week:02d}"
    results = [read_result(path) for path in sorted(root.glob("scheduled_at=*/result.json"))]
    return tuple(sorted(results, key=lambda item: item.scheduled_at))


def validate_claim(claim: CollectionExecutionClaim) -> None:
    """Validate one execution claim."""
    if claim.schema_version != RECEIPT_SCHEMA_VERSION:
        raise ValueError("Unsupported collection claim schema_version.")
    _validate_scope(claim.season, claim.week)
    _require_utc(claim.scheduled_at, label="scheduled_at")
    _require_utc(claim.claimed_at, label="claimed_at")
    _require_utc(claim.next_kickoff, label="next_kickoff")
    if claim.claimed_at < claim.scheduled_at:
        raise ValueError("claimed_at must not precede scheduled_at.")
    if claim.scheduled_at >= claim.next_kickoff:
        raise ValueError("scheduled_at must precede next_kickoff.")


def validate_result(result: CollectionExecutionResult) -> None:
    """Validate one terminal execution result."""
    if result.schema_version != RECEIPT_SCHEMA_VERSION:
        raise ValueError("Unsupported collection result schema_version.")
    _validate_scope(result.season, result.week)
    for label, value in (
        ("scheduled_at", result.scheduled_at),
        ("started_at", result.started_at),
        ("completed_at", result.completed_at),
        ("next_kickoff", result.next_kickoff),
    ):
        _require_utc(value, label=label)
    if result.completed_at < result.started_at:
        raise ValueError("completed_at must not precede started_at.")
    if result.minimum_credit_reserve < 0:
        raise ValueError("minimum_credit_reserve must not be negative.")
    for value in (
        result.last_known_requests_remaining,
        result.quote_count,
        result.game_count,
        result.sportsbook_count,
        result.requests_remaining,
        result.requests_used,
        result.request_cost,
    ):
        if value is not None and value < 0:
            raise ValueError("Collection result counts must not be negative.")
    if result.status is CollectionExecutionStatus.COMPLETED:
        if result.quote_count is None or result.ledger_path is None or result.snapshot_path is None:
            raise ValueError("Completed results require quote and artifact metadata.")
        if result.error_type is not None or result.error_message is not None:
            raise ValueError("Completed results cannot contain error metadata.")


def latest_known_requests_remaining(
    results: tuple[CollectionExecutionResult, ...],
) -> int | None:
    """Return the latest completed authoritative remaining-credit value."""
    known = [
        result
        for result in results
        if result.status is CollectionExecutionStatus.COMPLETED
        and result.requests_remaining is not None
    ]
    if not known:
        return None
    return max(known, key=lambda item: item.completed_at).requests_remaining


def _claim_payload(claim: CollectionExecutionClaim) -> dict[str, object]:
    return {
        "schema_version": claim.schema_version,
        "season": claim.season,
        "week": claim.week,
        "scheduled_at": _iso(claim.scheduled_at),
        "claimed_at": _iso(claim.claimed_at),
        "next_kickoff": _iso(claim.next_kickoff),
        "reason": claim.reason.value,
    }


def _result_payload(result: CollectionExecutionResult) -> dict[str, object]:
    payload = asdict(result)
    payload["status"] = result.status.value
    payload["reason"] = result.reason.value
    payload["quota_precheck"] = result.quota_precheck.value
    for field in ("scheduled_at", "started_at", "completed_at", "next_kickoff"):
        payload[field] = _iso(cast(datetime, payload[field]))
    return payload


def _empty_result() -> CollectionExecutionResult:
    now = datetime(2000, 1, 1, tzinfo=UTC)
    return CollectionExecutionResult(
        schema_version=RECEIPT_SCHEMA_VERSION,
        status=CollectionExecutionStatus.MISSED,
        season="x",
        week=1,
        scheduled_at=now,
        started_at=now,
        completed_at=now,
        next_kickoff=now + timedelta(hours=1),
        reason=CollectionReason.BASELINE,
        quota_precheck=QuotaPrecheckStatus.UNKNOWN,
        minimum_credit_reserve=0,
    )


def _object_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Collection execution artifact must contain a JSON object.")
    return cast(dict[str, object], payload)


def _validate_scope(season: str, week: int) -> None:
    if not season.strip():
        raise ValueError("season must not be empty.")
    if week < 1 or week > 22:
        raise ValueError("week must be between 1 and 22.")


def _require_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC.")
    return value.astimezone(UTC)


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Collection execution timestamps must be strings.")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _optional_int(value: object) -> int | None:
    return None if value is None else int(cast(int, value))


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")

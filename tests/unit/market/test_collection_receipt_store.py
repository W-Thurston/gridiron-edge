"""Tests for immutable quote-collection execution artifacts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gridiron_edge.market.collection_plan import CollectionReason
from gridiron_edge.market.collection_receipt_store import (
    RECEIPT_SCHEMA_VERSION,
    CollectionExecutionClaim,
    CollectionExecutionResult,
    CollectionExecutionStatus,
    QuotaPrecheckStatus,
    claim_path,
    latest_known_requests_remaining,
    read_claim,
    read_result,
    result_path,
    write_claim,
    write_result,
)

SCHEDULED = datetime(2026, 9, 9, 12, tzinfo=UTC)
NEXT = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)


def _claim() -> CollectionExecutionClaim:
    return CollectionExecutionClaim(
        RECEIPT_SCHEMA_VERSION,
        "2026-2027",
        1,
        SCHEDULED,
        SCHEDULED + timedelta(minutes=1),
        NEXT,
        CollectionReason.APPROACH,
    )


def _result() -> CollectionExecutionResult:
    return CollectionExecutionResult(
        schema_version=RECEIPT_SCHEMA_VERSION,
        status=CollectionExecutionStatus.COMPLETED,
        season="2026-2027",
        week=1,
        scheduled_at=SCHEDULED,
        started_at=SCHEDULED + timedelta(minutes=1),
        completed_at=SCHEDULED + timedelta(minutes=2),
        next_kickoff=NEXT,
        reason=CollectionReason.APPROACH,
        quota_precheck=QuotaPrecheckStatus.SATISFIED,
        minimum_credit_reserve=30,
        last_known_requests_remaining=100,
        quote_count=816,
        game_count=16,
        sportsbook_count=9,
        requests_remaining=97,
        requests_used=3,
        request_cost=3,
        ledger_path="history.parquet",
        snapshot_path="current.parquet",
    )


def test_paths_are_deterministic(tmp_path: Path) -> None:
    assert (
        claim_path(season="2026-2027", week=1, scheduled_at=SCHEDULED, repo=tmp_path).name
        == "claim.json"
    )
    assert "scheduled_at=2026-09-09T120000Z" in str(
        result_path(season="2026-2027", week=1, scheduled_at=SCHEDULED, repo=tmp_path)
    )


def test_claim_and_result_round_trip(tmp_path: Path) -> None:
    claim = _claim()
    result = _result()
    assert read_claim(write_claim(claim, repo=tmp_path)) == claim
    assert read_result(write_result(result, repo=tmp_path)) == result


def test_claim_and_result_are_immutable(tmp_path: Path) -> None:
    write_claim(_claim(), repo=tmp_path)
    with pytest.raises(FileExistsError):
        write_claim(_claim(), repo=tmp_path)
    write_result(_result(), repo=tmp_path)
    with pytest.raises(FileExistsError):
        write_result(_result(), repo=tmp_path)


def test_latest_known_remaining_uses_latest_completed_result() -> None:
    first = _result()
    second = replace(
        first, completed_at=first.completed_at + timedelta(hours=1), requests_remaining=80
    )
    failed = replace(
        first,
        status=CollectionExecutionStatus.REQUEST_FAILED,
        requests_remaining=40,
        quote_count=None,
        ledger_path=None,
        snapshot_path=None,
        error_type="OddsRequestError",
        error_message="failed",
    )
    assert latest_known_requests_remaining((first, second, failed)) == 80

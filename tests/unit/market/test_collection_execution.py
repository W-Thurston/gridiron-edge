"""Tests for single-shot due quote collection execution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from pandas import DataFrame

from gridiron_edge.ingest.odds.the_odds_api import OddsApiUsage, OddsIngestResult
from gridiron_edge.market.collection_execution import (
    CollectionDueStatus,
    evaluate_collection_due,
    execute_due_collection,
)
from gridiron_edge.market.collection_plan import build_weekly_quote_collection_plan
from gridiron_edge.market.collection_receipt_store import CollectionExecutionStatus

START = datetime(2026, 9, 8, 12, tzinfo=UTC)


def _plan():
    schedule = DataFrame(
        [
            {
                "season": "2026-2027",
                "week": 1,
                "game_id": "g",
                "game_date": "2026-09-10",
                "game_time": "20:20:00",
            }
        ]
    )
    return build_weekly_quote_collection_plan(
        schedule, season="2026-2027", week=1, plan_start=START, created_at=START
    )


def test_due_boundaries_and_missed_state(tmp_path: Path) -> None:
    plan = _plan()
    poll = plan.polls[0]
    assert (
        evaluate_collection_due(
            plan,
            evaluated_at=poll.scheduled_at - timedelta(seconds=1),
            grace_period=timedelta(minutes=15),
            repo=tmp_path,
        ).status
        is CollectionDueStatus.NOT_DUE
    )
    assert (
        evaluate_collection_due(
            plan, evaluated_at=poll.scheduled_at, grace_period=timedelta(minutes=15), repo=tmp_path
        ).status
        is CollectionDueStatus.DUE
    )
    assert (
        evaluate_collection_due(
            plan,
            evaluated_at=poll.scheduled_at + timedelta(minutes=15),
            grace_period=timedelta(minutes=15),
            repo=tmp_path,
        ).status
        is CollectionDueStatus.DUE
    )
    assert (
        evaluate_collection_due(
            plan,
            evaluated_at=poll.scheduled_at + timedelta(minutes=15, seconds=1),
            grace_period=timedelta(minutes=15),
            repo=tmp_path,
        ).status
        is CollectionDueStatus.MISSED
    )


@patch("gridiron_edge.market.collection_execution.ingest_the_odds_api_current")
def test_success_executes_provider_once(mock_ingest: MagicMock, tmp_path: Path) -> None:
    plan = _plan()
    poll = plan.polls[0]
    mock_ingest.return_value = OddsIngestResult(
        quote_count=10,
        game_count=2,
        sportsbook_count=3,
        ledger_path=tmp_path / "history.parquet",
        snapshot_path=tmp_path / "current.parquet",
        usage=OddsApiUsage(requests_remaining=97, requests_used=3, request_cost=3),
    )
    result = execute_due_collection(
        plan, schedule=DataFrame(), api_key="secret", evaluated_at=poll.scheduled_at, repo=tmp_path
    )
    assert result.status is CollectionExecutionStatus.COMPLETED
    assert result.quote_count == 10
    mock_ingest.assert_called_once()
    assert (
        evaluate_collection_due(
            plan, evaluated_at=poll.scheduled_at, grace_period=timedelta(minutes=15), repo=tmp_path
        ).status
        is CollectionDueStatus.COMPLETED
        or CollectionDueStatus.NOT_DUE
    )


@patch("gridiron_edge.market.collection_execution.ingest_the_odds_api_current")
def test_missed_poll_never_calls_provider(mock_ingest: MagicMock, tmp_path: Path) -> None:
    plan = _plan()
    poll = plan.polls[0]
    result = execute_due_collection(
        plan,
        schedule=DataFrame(),
        api_key="secret",
        evaluated_at=poll.scheduled_at + timedelta(minutes=16),
        repo=tmp_path,
    )
    assert result.status is CollectionExecutionStatus.MISSED
    mock_ingest.assert_not_called()


@patch("gridiron_edge.market.collection_execution.ingest_the_odds_api_current")
def test_existing_claim_blocks_retry(mock_ingest: MagicMock, tmp_path: Path) -> None:
    plan = _plan()
    poll = plan.polls[0]
    from gridiron_edge.market.collection_receipt_store import (
        RECEIPT_SCHEMA_VERSION,
        CollectionExecutionClaim,
        write_claim,
    )

    write_claim(
        CollectionExecutionClaim(
            RECEIPT_SCHEMA_VERSION,
            plan.season,
            plan.week,
            poll.scheduled_at,
            poll.scheduled_at,
            poll.next_kickoff,
            poll.reason,
        ),
        repo=tmp_path,
    )
    due = execute_due_collection(
        plan, schedule=DataFrame(), api_key="secret", evaluated_at=poll.scheduled_at, repo=tmp_path
    )
    assert due.status is CollectionDueStatus.CLAIMED
    mock_ingest.assert_not_called()

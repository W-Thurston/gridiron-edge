"""Tests for write-safe The Odds API current-market ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.ingest.odds.the_odds_api import (
    OddsApiResponse,
    OddsApiUsage,
    OddsIngestError,
    ingest_the_odds_api_current,
)

SEASON = "2026-2027"
WEEK = 1
FETCHED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)


def _schedule() -> DataFrame:
    return DataFrame(
        [
            {
                "season": SEASON,
                "week": WEEK,
                "game_id": "2026_01_KC_LAC",
                "game_date": "2026-09-10",
                "away_team": "Kansas City Chiefs",
                "home_team": "Los Angeles Chargers",
            }
        ]
    )


def _payload() -> list[dict[str, object]]:
    return [
        {
            "id": "event-1",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2026-09-10T00:20:00Z",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "last_update": "2026-09-01T11:59:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-09-01T11:59:30Z",
                            "outcomes": [
                                {"name": "Kansas City Chiefs", "price": 125},
                                {"name": "Los Angeles Chargers", "price": -145},
                            ],
                        }
                    ],
                }
            ],
        }
    ]


def _response(payload: list[dict[str, object]] | None = None) -> OddsApiResponse:
    return OddsApiResponse(
        payload=cast(
            list[Mapping[str, object]],
            _payload() if payload is None else payload,
        ),
        usage=OddsApiUsage(requests_remaining=99, requests_used=1, request_cost=1),
    )


def _ingest(tmp_path: Path, **overrides: object):
    values: dict[str, object] = {
        "api_key": "key",
        "schedule": _schedule(),
        "season": SEASON,
        "week": WEEK,
        "repo": tmp_path,
        "fetched_at": FETCHED_AT,
    }
    values.update(overrides)
    return ingest_the_odds_api_current(**values)  # type: ignore[arg-type]


@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_success_writes_ledger_then_snapshot(mock_fetch: MagicMock, tmp_path: Path) -> None:
    mock_fetch.return_value = _response()
    result = _ingest(tmp_path)

    assert result.quote_count == 2
    assert result.game_count == 1
    assert result.sportsbook_count == 1
    assert result.usage.requests_remaining == 99
    assert result.ledger_path == (
        tmp_path
        / "data"
        / "odds"
        / "history"
        / "season=2026-2027"
        / "week=01"
        / "observations.parquet"
    )
    assert result.snapshot_path == tmp_path / "data" / "odds" / "odds_current.parquet"

    ledger = pd.read_parquet(result.ledger_path)
    snapshot = pd.read_parquet(result.snapshot_path)
    assert len(ledger) == 2
    assert len(snapshot) == 2
    assert set(snapshot["provider"]) == {"the_odds_api"}
    assert set(snapshot["sportsbook"]) == {"fanduel"}
    mock_fetch.assert_called_once_with(api_key="key", session=None, timeout=15.0)


@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_exact_successful_rerun_is_idempotent(mock_fetch: MagicMock, tmp_path: Path) -> None:
    mock_fetch.return_value = _response()
    _ingest(tmp_path)
    _ingest(tmp_path)
    ledger = pd.read_parquet(
        tmp_path
        / "data"
        / "odds"
        / "history"
        / "season=2026-2027"
        / "week=01"
        / "observations.parquet"
    )
    assert len(ledger) == 2


@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_empty_provider_payload_does_not_touch_existing_artifacts(
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    odds_dir = tmp_path / "data" / "odds"
    odds_dir.mkdir(parents=True)
    ledger = odds_dir / "odds_log.parquet"
    snapshot = odds_dir / "odds_current.parquet"
    ledger.write_bytes(b"ledger-sentinel")
    snapshot.write_bytes(b"snapshot-sentinel")
    mock_fetch.return_value = _response([])

    with pytest.raises(OddsIngestError, match="returned no events"):
        _ingest(tmp_path)

    assert ledger.read_bytes() == b"ledger-sentinel"
    assert snapshot.read_bytes() == b"snapshot-sentinel"


@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_zero_matched_rows_does_not_touch_existing_artifacts(
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    odds_dir = tmp_path / "data" / "odds"
    odds_dir.mkdir(parents=True)
    snapshot = odds_dir / "odds_current.parquet"
    snapshot.write_bytes(b"snapshot-sentinel")
    payload = _payload()
    payload[0]["away_team"] = "Buffalo Bills"
    payload[0]["home_team"] = "Miami Dolphins"
    mock_fetch.return_value = _response(payload)

    with pytest.raises(OddsIngestError, match="no usable matched pregame quotes"):
        _ingest(tmp_path)

    assert snapshot.read_bytes() == b"snapshot-sentinel"
    assert not (odds_dir / "odds_log.parquet").exists()


@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_parse_failure_does_not_touch_existing_artifacts(
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    odds_dir = tmp_path / "data" / "odds"
    odds_dir.mkdir(parents=True)
    snapshot = odds_dir / "odds_current.parquet"
    snapshot.write_bytes(b"snapshot-sentinel")
    payload = _payload()
    payload[0]["id"] = ""
    mock_fetch.return_value = _response(payload)

    with pytest.raises(ValueError, match="event id"):
        _ingest(tmp_path)

    assert snapshot.read_bytes() == b"snapshot-sentinel"
    assert not (odds_dir / "odds_log.parquet").exists()


@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_request_failure_does_not_touch_existing_artifacts(
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    odds_dir = tmp_path / "data" / "odds"
    odds_dir.mkdir(parents=True)
    snapshot = odds_dir / "odds_current.parquet"
    snapshot.write_bytes(b"snapshot-sentinel")
    mock_fetch.side_effect = RuntimeError("request failed")

    with pytest.raises(RuntimeError, match="request failed"):
        _ingest(tmp_path)

    assert snapshot.read_bytes() == b"snapshot-sentinel"
    assert not (odds_dir / "odds_log.parquet").exists()


@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_partial_schedule_coverage_is_written_truthfully(
    mock_fetch: MagicMock,
    tmp_path: Path,
) -> None:
    schedule = pd.concat(
        [
            _schedule(),
            DataFrame(
                [
                    {
                        "season": SEASON,
                        "week": WEEK,
                        "game_id": "2026_01_BAL_BUF",
                        "game_date": "2026-09-10",
                        "away_team": "Baltimore Ravens",
                        "home_team": "Buffalo Bills",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    mock_fetch.return_value = _response()

    result = _ingest(tmp_path, schedule=schedule)

    assert result.game_count == 1
    snapshot = pd.read_parquet(result.snapshot_path)
    assert snapshot["game_id"].unique().tolist() == ["2026_01_KC_LAC"]


@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_forwards_session_and_timeout(mock_fetch: MagicMock, tmp_path: Path) -> None:
    mock_fetch.return_value = _response()
    session = MagicMock()
    _ingest(tmp_path, session=session, timeout=7.5)
    mock_fetch.assert_called_once_with(api_key="key", session=session, timeout=7.5)


@patch("gridiron_edge.ingest.odds.the_odds_api.write_current_odds_snapshot")
@patch("gridiron_edge.ingest.odds.the_odds_api.fetch_the_odds_api_payload")
def test_snapshot_failure_retains_history_and_prior_snapshot(
    mock_fetch: MagicMock,
    mock_snapshot: MagicMock,
    tmp_path: Path,
) -> None:
    """History remains truthful when current snapshot replacement fails."""
    odds_dir = tmp_path / "data" / "odds"
    odds_dir.mkdir(parents=True)
    snapshot = odds_dir / "odds_current.parquet"
    snapshot.write_bytes(b"prior-snapshot")
    mock_fetch.return_value = _response()
    mock_snapshot.side_effect = OSError("snapshot failed")

    with pytest.raises(
        OddsIngestError,
        match=r"historical ledger.*current snapshot",
    ) as exc_info:
        _ingest(tmp_path)

    assert isinstance(exc_info.value.__cause__, OSError)
    ledger = pd.read_parquet(
        odds_dir / "history" / "season=2026-2027" / "week=01" / "observations.parquet"
    )
    assert len(ledger) == 2
    assert snapshot.read_bytes() == b"prior-snapshot"

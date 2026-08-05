"""Tests for The Odds API v4 HTTP client boundary."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from gridiron_edge.ingest.odds.the_odds_api import (
    THE_ODDS_API_BASE_URL,
    OddsApiResponse,
    OddsRequestError,
    fetch_the_odds_api_payload,
)


def _response(
    *,
    payload: object | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    response = MagicMock()
    response.json.return_value = [] if payload is None else payload
    response.headers = headers or {}
    return response


def test_builds_locked_nfl_featured_market_request() -> None:
    session = MagicMock()
    response = _response()
    session.get.return_value = response

    result = fetch_the_odds_api_payload(
        api_key="secret-key",
        session=session,
        timeout=12.5,
    )

    assert isinstance(result, OddsApiResponse)
    assert result.payload == []
    session.get.assert_called_once_with(
        f"{THE_ODDS_API_BASE_URL}/sports/americanfootball_nfl/odds",
        params={
            "apiKey": "secret-key",
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
        timeout=12.5,
    )
    response.raise_for_status.assert_called_once_with()


def test_retains_payload_and_quota_headers() -> None:
    session = MagicMock()
    payload = [{"id": "event-1"}]
    session.get.return_value = _response(
        payload=payload,
        headers={
            "x-requests-remaining": "487",
            "x-requests-used": "13",
            "x-requests-last": "3",
        },
    )

    result = fetch_the_odds_api_payload(api_key="key", session=session)

    assert result.payload == payload
    assert result.usage.requests_remaining == 487
    assert result.usage.requests_used == 13
    assert result.usage.request_cost == 3


def test_missing_quota_headers_are_explicitly_unknown() -> None:
    session = MagicMock()
    session.get.return_value = _response()
    usage = fetch_the_odds_api_payload(api_key="key", session=session).usage
    assert usage.requests_remaining is None
    assert usage.requests_used is None
    assert usage.request_cost is None


@pytest.mark.parametrize(
    ("api_key", "timeout", "message"),
    [
        ("", 15.0, "api_key must not be empty"),
        (" ", 15.0, "api_key must not be empty"),
        ("key", 0.0, "timeout must be positive"),
        ("key", -1.0, "timeout must be positive"),
    ],
)
def test_rejects_invalid_client_inputs(api_key: str, timeout: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fetch_the_odds_api_payload(api_key=api_key, timeout=timeout)


def test_wraps_request_failures_without_exposing_key() -> None:
    session = MagicMock()
    session.get.side_effect = requests.Timeout("secret-key")
    with pytest.raises(OddsRequestError, match="request failed") as exc_info:
        fetch_the_odds_api_payload(api_key="secret-key", session=session)
    assert "secret-key" not in str(exc_info.value)


def test_wraps_invalid_json() -> None:
    session = MagicMock()
    response = _response()
    response.json.side_effect = ValueError("bad json")
    session.get.return_value = response
    with pytest.raises(OddsRequestError, match="not valid JSON"):
        fetch_the_odds_api_payload(api_key="key", session=session)


def test_rejects_non_array_json() -> None:
    session = MagicMock()
    session.get.return_value = _response(payload={"message": "error"})
    with pytest.raises(ValueError, match="payload must be an array"):
        fetch_the_odds_api_payload(api_key="key", session=session)


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("x-requests-remaining", "not-a-number"),
        ("x-requests-used", "-1"),
        ("x-requests-last", "1.5"),
    ],
)
def test_rejects_invalid_quota_headers(header: str, value: str) -> None:
    session = MagicMock()
    session.get.return_value = _response(headers={header: value})
    with pytest.raises(OddsRequestError, match="Response header"):
        fetch_the_odds_api_payload(api_key="key", session=session)

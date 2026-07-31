# tests/unit/ingest/test_draftkings_legacy.py

"""Tests for legacy best-effort DraftKings payload handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from gridiron_edge.ingest.odds.draftkings import (
    DraftKingsUnavailableError,
    _load_legacy_payload,
    _validate_legacy_payload,
    fetch_dk_odds,
    fetch_dk_odds_wide,
)


def _response(
    *,
    payload: object | None = None,
    content_type: str = "application/json",
    text: str = "",
    json_error: ValueError | None = None,
) -> MagicMock:
    response = MagicMock()
    response.headers = {"content-type": content_type}
    response.text = text
    response.raise_for_status.return_value = None
    if json_error is not None:
        response.json.side_effect = json_error
    else:
        response.json.return_value = payload
    return response


def test_valid_empty_payload_is_available_but_has_no_rows() -> None:
    result = fetch_dk_odds_wide(payload_override={"events": [], "markets": [], "selections": []})
    assert result.empty


def test_non_json_response_fails_clearly() -> None:
    response = _response(json_error=ValueError("decode failed"))
    with pytest.raises(DraftKingsUnavailableError, match="not valid JSON"):
        _load_legacy_payload(response)


def test_html_human_verification_response_fails_clearly() -> None:
    response = _response(
        content_type="text/html",
        text="<!doctype html><title>Are you human?</title>",
    )
    with pytest.raises(
        DraftKingsUnavailableError,
        match="HTML or human-verification",
    ):
        _load_legacy_payload(response)
    response.json.assert_not_called()


def test_http_failure_is_wrapped_as_legacy_unavailability() -> None:
    response = _response(payload={})
    response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
    with pytest.raises(DraftKingsUnavailableError, match="HTTP request failed"):
        _load_legacy_payload(response)


def test_non_mapping_payload_is_rejected() -> None:
    with pytest.raises(DraftKingsUnavailableError, match="not an object"):
        _validate_legacy_payload([])


def test_missing_expected_collection_is_rejected() -> None:
    with pytest.raises(
        DraftKingsUnavailableError,
        match="'markets' is not a list",
    ):
        _validate_legacy_payload({"events": [], "selections": []})


def test_empty_valid_fetch_does_not_fabricate_storage_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gridiron_edge.ingest.odds.draftkings.fetch_dk_odds_wide",
        lambda: fetch_dk_odds_wide(
            payload_override={"events": [], "markets": [], "selections": []}
        ),
    )
    assert fetch_dk_odds(season="2026-2027", week=1) is None


def test_payload_loader_contains_no_bypass_or_retry_behavior() -> None:
    import inspect

    source = inspect.getsource(_load_legacy_payload).lower()
    forbidden = ("selenium", "playwright", "cloudscraper", "cookie", "retry")
    assert all(token not in source for token in forbidden)

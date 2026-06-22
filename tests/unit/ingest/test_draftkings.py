# tests/unit/ingest/test_draftkings.py
"""Tests for gridiron_edge.ingest.odds.draftkings - DK API parsing."""

from __future__ import annotations

from tests.fixtures.dk_payload_fixture import DK_PAYLOAD_FIXTURE


class TestDkPayloadFixture:
    """Verify the test fixture itself is well-formed."""

    def test_has_events(self) -> None:
        assert "events" in DK_PAYLOAD_FIXTURE
        assert len(DK_PAYLOAD_FIXTURE["events"]) > 0

    def test_events_have_participants(self) -> None:
        for event in DK_PAYLOAD_FIXTURE["events"]:
            assert "participants" in event
            assert len(event["participants"]) == 2

    def test_participants_have_venue_roles(self) -> None:
        for event in DK_PAYLOAD_FIXTURE["events"]:
            roles: set = {p["venueRole"] for p in event["participants"]}
            assert roles == {"Away", "Home"}

    def test_events_have_start_date(self) -> None:
        for event in DK_PAYLOAD_FIXTURE["events"]:
            assert "startEventDate" in event
            assert "id" in event

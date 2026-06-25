# tests/unit/api/test_meta.py
"""Unit tests for api/meta.py.

Covers:
- BlockedStatus serialization shape.
- "pending" literal serialization.
- ResponseMeta builder methods (with_pending, with_blocked).
- Frozen / extra=forbid behavior.
- Blocker registry: completeness and uniqueness.
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta


class TestBlockedStatus:
    def test_serializes_with_full_shape(self) -> None:
        bs = BlockedStatus(blocker="injury_data_source", roadmap="§5.3")
        assert bs.model_dump() == {
            "status": "blocked",
            "blocker": "injury_data_source",
            "roadmap": "§5.3",
        }

    def test_status_is_always_blocked(self) -> None:
        bs = BlockedStatus(blocker="multi_book_ingest", roadmap="W7")
        assert bs.status == "blocked"

    def test_status_field_rejects_other_values(self) -> None:
        with pytest.raises(ValidationError):
            # status is Literal["blocked"]; "pending" must not validate here
            BlockedStatus(status="pending", blocker="x", roadmap="y")  # type: ignore[arg-type]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            BlockedStatus(
                blocker="x",
                roadmap="y",
                unexpected_field="z",  # type: ignore[call-arg]
            )

    def test_is_frozen(self) -> None:
        bs = BlockedStatus(blocker="x", roadmap="y")
        with pytest.raises(ValidationError):
            bs.blocker = "other"  # type: ignore[misc]


class TestResponseMetaDefault:
    def test_default_field_status_is_empty(self) -> None:
        meta = ResponseMeta()
        assert meta.field_status == {}

    def test_serializes_empty_field_status(self) -> None:
        meta = ResponseMeta()
        assert meta.model_dump() == {"field_status": {}}


class TestResponseMetaWithPending:
    def test_marks_a_field_pending(self) -> None:
        meta = ResponseMeta().with_pending("network")
        assert meta.field_status == {"network": "pending"}

    def test_returns_new_instance(self) -> None:
        original = ResponseMeta()
        updated = original.with_pending("network")
        assert original.field_status == {}
        assert updated.field_status == {"network": "pending"}
        assert original is not updated

    def test_supports_dot_notation_paths(self) -> None:
        meta = ResponseMeta().with_pending("model.home_win_prob")
        assert "model.home_win_prob" in meta.field_status

    def test_chains(self) -> None:
        meta = ResponseMeta().with_pending("network").with_pending("storyline")
        assert meta.field_status == {
            "network": "pending",
            "storyline": "pending",
        }


class TestResponseMetaWithBlocked:
    def test_marks_a_field_blocked(self) -> None:
        meta = ResponseMeta().with_blocked(
            "injuries",
            "injury_data_source",
            "§5.3",
        )
        status = meta.field_status["injuries"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "injury_data_source"
        assert status.roadmap == "§5.3"

    def test_returns_new_instance(self) -> None:
        original = ResponseMeta()
        updated = original.with_blocked("injuries", "injury_data_source", "§5.3")
        assert original.field_status == {}
        assert "injuries" in updated.field_status

    def test_blocker_registry_splat(self) -> None:
        meta = ResponseMeta().with_blocked("injuries", *Blocker.INJURY_DATA)
        status = meta.field_status["injuries"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "injury_data_source"
        assert status.roadmap == "§5.3"


class TestResponseMetaMixed:
    def test_pending_and_blocked_coexist(self) -> None:
        meta = (
            ResponseMeta()
            .with_pending("network")
            .with_blocked("injuries", *Blocker.INJURY_DATA)
            .with_blocked("swing_factors", *Blocker.FEATURE_ATTRIBUTION)
        )
        assert meta.field_status["network"] == "pending"
        assert isinstance(meta.field_status["injuries"], BlockedStatus)
        assert isinstance(meta.field_status["swing_factors"], BlockedStatus)
        assert meta.field_status["injuries"].blocker == "injury_data_source"
        assert meta.field_status["swing_factors"].blocker == "feature_attribution"

    def test_later_writes_overwrite_earlier(self) -> None:
        meta = (
            ResponseMeta().with_pending("injuries").with_blocked("injuries", *Blocker.INJURY_DATA)
        )
        status = meta.field_status["injuries"]
        assert isinstance(status, BlockedStatus)


class TestResponseMetaSerialization:
    def test_serializes_with_mixed_statuses(self) -> None:
        meta = ResponseMeta().with_pending("network").with_blocked("injuries", *Blocker.INJURY_DATA)
        dumped = meta.model_dump()
        assert dumped == {
            "field_status": {
                "network": "pending",
                "injuries": {
                    "status": "blocked",
                    "blocker": "injury_data_source",
                    "roadmap": "§5.3",
                },
            },
        }

    def test_json_round_trips(self) -> None:
        meta = ResponseMeta().with_pending("network").with_blocked("injuries", *Blocker.INJURY_DATA)
        json_str = meta.model_dump_json()
        rebuilt = ResponseMeta.model_validate_json(json_str)
        assert rebuilt.field_status == meta.field_status


class TestResponseMetaIsFrozen:
    def test_field_status_assignment_rejected(self) -> None:
        meta = ResponseMeta()
        with pytest.raises(ValidationError):
            meta.field_status = {"x": "pending"}  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ResponseMeta(unexpected="x")  # type: ignore[call-arg]


class TestBlockerRegistry:
    def test_all_slugs_returns_frozenset(self) -> None:
        slugs = Blocker.all_slugs()
        assert isinstance(slugs, frozenset)

    def test_all_known_blockers_appear(self) -> None:
        expected = {
            "injury_data_source",
            "multi_book_ingest",
            "live_state_ingest",
            "scenario_engine",
            "feature_attribution",
            "comparables_retrieval",
            "historical_line_movement",
            "gameday_metadata",
            "news_ingest",
            "war_computation",
        }
        assert Blocker.all_slugs() == expected

    def test_no_duplicate_slugs(self) -> None:
        entries = [
            value
            for name, value in vars(Blocker).items()
            if not name.startswith("_") and isinstance(value, tuple) and len(value) == 2
        ]
        slugs = [slug for slug, _ in entries]
        assert len(slugs) == len(set(slugs))

    def test_each_entry_is_a_pair_of_strings(self) -> None:
        for name, value in vars(Blocker).items():
            if name.startswith("_") or not isinstance(value, tuple):
                continue
            if name == "all_slugs":
                continue
            assert len(value) == 2, f"{name} is not a 2-tuple"
            slug, roadmap = value
            assert isinstance(slug, str) and slug, f"{name} has invalid slug"
            assert isinstance(roadmap, str) and roadmap, f"{name} has invalid roadmap"

    def test_splat_works_in_with_blocked(self) -> None:
        meta = ResponseMeta().with_blocked("test_field", *Blocker.WAR)
        status = meta.field_status["test_field"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "war_computation"
        assert status.roadmap == "deferred"

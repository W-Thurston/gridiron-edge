"""Tests for shared enums."""

from __future__ import annotations

from gridiron_edge.core.enums import (
    COVERED_STADIUMS,
    DOME_LIKE_ROOFS,
    ConfidenceTier,
    Lean,
    RoofType,
)


class TestLean:
    def test_values_match_archived_strings(self) -> None:
        assert Lean.OVER.value == "Over"
        assert Lean.UNDER.value == "Under"
        assert Lean.NO_EDGE.value == "No Edge"

    def test_str_enum_comparison(self) -> None:
        """StrEnum members compare equal to their underlying strings."""
        assert Lean.OVER == "Over"
        assert Lean.NO_EDGE == "No Edge"


class TestConfidenceTier:
    def test_values_match_archived_strings(self) -> None:
        assert ConfidenceTier.HIGH.value == "High"
        assert ConfidenceTier.MODERATE.value == "Moderate"
        assert ConfidenceTier.LOW.value == "Low"

    def test_str_enum_comparison(self) -> None:
        assert ConfidenceTier.HIGH == "High"


class TestRoofType:
    def test_all_canonical_values_present(self) -> None:
        expected = {"dome", "outdoors", "open", "closed", "retractable"}
        assert {r.value for r in RoofType} == expected


class TestRoofGroupings:
    def test_covered_stadiums_includes_dome_and_retractable(self) -> None:
        assert RoofType.DOME in COVERED_STADIUMS
        assert RoofType.RETRACTABLE in COVERED_STADIUMS

    def test_covered_stadiums_excludes_closed_and_outdoors(self) -> None:
        assert RoofType.CLOSED not in COVERED_STADIUMS
        assert RoofType.OUTDOORS not in COVERED_STADIUMS
        assert RoofType.OPEN not in COVERED_STADIUMS

    def test_dome_like_roofs_includes_dome_and_closed(self) -> None:
        assert RoofType.DOME in DOME_LIKE_ROOFS
        assert RoofType.CLOSED in DOME_LIKE_ROOFS

    def test_dome_like_roofs_excludes_retractable(self) -> None:
        """A retractable stadium with open roof is not dome-like."""
        assert RoofType.RETRACTABLE not in DOME_LIKE_ROOFS

    def test_dome_like_and_covered_stadiums_differ(self) -> None:
        """The two groupings are genuinely different semantic categories."""
        assert COVERED_STADIUMS != DOME_LIKE_ROOFS

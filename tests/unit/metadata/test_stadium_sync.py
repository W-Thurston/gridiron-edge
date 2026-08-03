"""Tests for reviewed stadium metadata synchronization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.metadata.stadium_sync import (
    apply_approved_stadium_updates,
    audit_stadium_coverage,
    prepare_stadium_updates,
    validate_stadium_reference,
)


def _stadiums() -> DataFrame:
    return DataFrame(
        [
            {
                "HOME_TEAM": "Kansas City Chiefs",
                "YEAR": "2025-2026",
                "STADIUM": "Arrowhead Stadium",
                "LATITUDE": 39.0489,
                "LONGITUDE": -94.4839,
                "ROOF": "outdoors",
                "SURFACE": "grass",
                "ALTITUDE": 274.0,
            },
            {
                "HOME_TEAM": "Buffalo Bills",
                "YEAR": "2025-2026",
                "STADIUM": "New Era Field",
                "LATITUDE": 42.7738,
                "LONGITUDE": -78.7870,
                "ROOF": "outdoors",
                "SURFACE": "artificial",
                "ALTITUDE": 183.0,
            },
            {
                "HOME_TEAM": "International",
                "YEAR": "2025-2026",
                "STADIUM": "Wembley Stadium",
                "LATITUDE": 51.5556,
                "LONGITUDE": -0.2794,
                "ROOF": "outdoors",
                "SURFACE": "grass",
                "ALTITUDE": 60.0,
            },
        ]
    )


def _schedule() -> DataFrame:
    return DataFrame(
        {
            "season": ["2026-2027", "2026-2027"],
            "away_team": ["Buffalo Bills", "Kansas City Chiefs"],
            "home_team": ["Kansas City Chiefs", "Buffalo Bills"],
            "stadium": ["Arrowhead Stadium", "Highmark Stadium"],
        }
    )


def test_audit_reports_missing_origins_and_site() -> None:
    result = audit_stadium_coverage(_stadiums(), _schedule(), season="2026-2027")
    assert set(result["ISSUE"]) == {"missing_franchise_origin", "unresolved_game_site"}
    assert set(result.loc[result["ISSUE"] == "missing_franchise_origin", "HOME_TEAM"]) == {
        "Buffalo Bills",
        "Kansas City Chiefs",
    }
    assert result.loc[result["ISSUE"] == "unresolved_game_site", "STADIUM"].tolist() == [
        "Highmark Stadium"
    ]


def test_prepare_carries_forward_exact_home_site() -> None:
    result = prepare_stadium_updates(_stadiums(), _schedule(), season="2026-2027")
    row = result.loc[result["HOME_TEAM"] == "Kansas City Chiefs"].iloc[0]
    assert row["ACTION"] == "carry_forward"
    assert row["REVIEW_STATUS"] == "proposed"
    assert row["STADIUM"] == "Arrowhead Stadium"


def test_prepare_uses_explicit_alias_for_renamed_site() -> None:
    aliases = DataFrame(
        {
            "SOURCE_STADIUM": ["Highmark Stadium"],
            "CANONICAL_STADIUM": ["New Era Field"],
        }
    )
    result = prepare_stadium_updates(_stadiums(), _schedule(), season="2026-2027", aliases=aliases)
    row = result.loc[result["HOME_TEAM"] == "Buffalo Bills"].iloc[0]
    assert row["ACTION"] == "alias_existing"
    assert row["STADIUM"] == "Highmark Stadium"
    assert row["LATITUDE"] == pytest.approx(42.7738)


def test_prepare_does_not_guess_renamed_site() -> None:
    result = prepare_stadium_updates(_stadiums(), _schedule(), season="2026-2027")
    row = result.loc[result["HOME_TEAM"] == "Buffalo Bills"].iloc[0]
    assert row["ACTION"] == "unresolved"
    assert row["REVIEW_STATUS"] == "unresolved"


def test_non_nfl_home_team_must_use_special_convention(tmp_path: Path) -> None:
    updates = prepare_stadium_updates(_stadiums(), _schedule(), season="2026-2027")
    row = updates.loc[updates["STADIUM"] == "Highmark Stadium"].index[-1]
    columns = [
        "REVIEW_STATUS",
        "HOME_TEAM",
        "LATITUDE",
        "LONGITUDE",
        "ROOF",
        "SURFACE",
        "ALTITUDE",
    ]
    updates.loc[row, columns] = [
        "approved",
        "Visitor Site",
        42.7,
        -78.7,
        "outdoors",
        "grass",
        180.0,
    ]
    with pytest.raises(ValueError, match="must be NFL teams, Alternate, or International"):
        apply_approved_stadium_updates(_stadiums(), updates, path=tmp_path / "stadiums.csv")


def test_apply_is_atomic_and_preserves_history(tmp_path: Path) -> None:
    original = _stadiums()
    updates = prepare_stadium_updates(original, _schedule(), season="2026-2027")
    index = updates.index[updates["HOME_TEAM"] == "Kansas City Chiefs"][0]
    updates.loc[index, "REVIEW_STATUS"] = "approved"
    path = tmp_path / "NFL_stadium_reference.csv"

    result = apply_approved_stadium_updates(original, updates, path=path)

    pd.testing.assert_frame_equal(result.iloc[: len(original)].reset_index(drop=True), original)
    assert path.is_file()
    assert not path.with_name(f"{path.name}.tmp").exists()
    assert len(result) == len(original) + 1


def test_apply_rejects_incomplete_approved_metadata(tmp_path: Path) -> None:
    updates = prepare_stadium_updates(_stadiums(), _schedule(), season="2026-2027")
    index = updates.index[updates["HOME_TEAM"] == "Kansas City Chiefs"][0]
    updates.loc[index, "REVIEW_STATUS"] = "approved"
    updates.loc[index, "LATITUDE"] = pd.NA
    with pytest.raises(ValueError, match="require complete metadata"):
        apply_approved_stadium_updates(_stadiums(), updates, path=tmp_path / "stadiums.csv")


def test_identical_franchise_origin_alias_is_allowed() -> None:
    alias = _stadiums().iloc[[0]].copy()
    alias["STADIUM"] = "GEHA Field at Arrowhead Stadium"

    stadiums = pd.concat(
        [_stadiums(), alias],
        ignore_index=True,
    )

    validate_stadium_reference(stadiums)


def test_conflicting_franchise_origin_coordinates_are_rejected() -> None:
    conflict = _stadiums().iloc[[0]].copy()
    conflict["STADIUM"] = "Conflicting Stadium"
    conflict["LATITUDE"] = 40.0
    conflict["LONGITUDE"] = -95.0

    stadiums = pd.concat(
        [_stadiums(), conflict],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="conflicting franchise-season origin coordinates",
    ):
        validate_stadium_reference(stadiums)

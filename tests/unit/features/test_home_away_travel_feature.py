# tests/unit/features/test_home_away_travel_feature.py

"""Tests for canonical Away/Home travel features."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.travel import HomeAwayTravelFeature


def _datasets(
    *,
    games: DataFrame,
    upcoming: DataFrame,
    stadiums: DataFrame,
) -> MagicMock:
    """Return controlled game, schedule, and stadium datasets."""
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.games.return_value = games.copy()
    datasets.schedule_upcoming_rich.return_value = upcoming.copy()
    datasets.stadiums.return_value = stadiums.copy()
    return datasets


def _target(
    *,
    game_id: str = "2025_01_LV_KC",
    year: str = "2025-2026",
) -> DataFrame:
    """Return one canonical target game."""
    return DataFrame(
        {
            "GAME_ID": [game_id],
            "YEAR": [year],
            "AWAY_TEAM": ["Las Vegas Raiders"],
            "HOME_TEAM": ["Kansas City Chiefs"],
            "MARKER": ["preserved"],
        }
    )


def _historical_games(
    *,
    stadium: str = "Arrowhead Stadium",
) -> DataFrame:
    """Return a historical venue source for the target game."""
    return DataFrame(
        {
            "GAME_ID": ["2025_01_LV_KC"],
            "STADIUM": [stadium],
        }
    )


def _upcoming_games(
    *,
    game_id: str = "2025_01_LV_KC",
    stadium: str = "Arrowhead Stadium",
) -> DataFrame:
    """Return an upcoming venue source for the target game."""
    return DataFrame(
        {
            "game_id": [game_id],
            "stadium": [stadium],
        }
    )


def _stadiums() -> DataFrame:
    """Return franchise origins and actual venue coordinates."""
    return DataFrame(
        [
            {
                "HOME_TEAM": "Kansas City Chiefs",
                "YEAR": "2025-2026",
                "STADIUM": "Arrowhead Stadium",
                "LATITUDE": 39.0489,
                "LONGITUDE": -94.4839,
                "ALTITUDE": 274.0,
            },
            {
                "HOME_TEAM": "Las Vegas Raiders",
                "YEAR": "2025-2026",
                "STADIUM": "Allegiant Stadium",
                "LATITUDE": 36.0909,
                "LONGITUDE": -115.1833,
                "ALTITUDE": 628.0,
            },
            {
                "HOME_TEAM": "International",
                "YEAR": "2025-2026",
                "STADIUM": "Wembley Stadium",
                "LATITUDE": 51.5556,
                "LONGITUDE": -0.2794,
                "ALTITUDE": 60.0,
            },
        ]
    )


def _compute(
    *,
    target: DataFrame | None = None,
    games: DataFrame | None = None,
    upcoming: DataFrame | None = None,
    stadiums: DataFrame | None = None,
) -> DataFrame:
    """Run the canonical travel feature with controlled inputs."""
    return HomeAwayTravelFeature().compute(
        df=_target() if target is None else target,
        datasets=_datasets(
            games=_historical_games() if games is None else games,
            upcoming=DataFrame() if upcoming is None else upcoming,
            stadiums=_stadiums() if stadiums is None else stadiums,
        ),
    )


def test_registered_under_canonical_name() -> None:
    assert FeatureRegistry.get("home_away_travel") is HomeAwayTravelFeature


def test_spec_declares_canonical_outputs() -> None:
    assert HomeAwayTravelFeature.spec.name == "home_away_travel"
    assert HomeAwayTravelFeature.spec.produces == [
        "GAME_SITE_ALTITUDE",
        "AWAY_KM_TRAVELED",
        "HOME_KM_TRAVELED",
        "AWAY_TZ_SHIFT",
        "HOME_TZ_SHIFT",
    ]


def test_standard_home_game_uses_actual_site() -> None:
    row = _compute().iloc[0]

    assert row["GAME_SITE_ALTITUDE"] == pytest.approx(274.0)
    assert row["HOME_KM_TRAVELED"] == pytest.approx(0.0, abs=0.01)
    assert row["HOME_TZ_SHIFT"] == 0
    assert row["AWAY_KM_TRAVELED"] > 0
    assert pd.notna(row["AWAY_TZ_SHIFT"])


def test_upcoming_venue_resolves_without_historical_game() -> None:
    row = _compute(
        games=DataFrame(),
        upcoming=_upcoming_games(),
    ).iloc[0]

    assert row["GAME_SITE_ALTITUDE"] == pytest.approx(274.0)
    assert row["HOME_KM_TRAVELED"] == pytest.approx(0.0, abs=0.01)
    assert row["AWAY_KM_TRAVELED"] > 0


def test_neutral_venue_calculates_travel_for_both_teams() -> None:
    row = _compute(
        games=_historical_games(stadium="Wembley Stadium"),
    ).iloc[0]

    assert row["GAME_SITE_ALTITUDE"] == pytest.approx(60.0)
    assert row["AWAY_KM_TRAVELED"] > 0
    assert row["HOME_KM_TRAVELED"] > 0
    assert pd.notna(row["AWAY_TZ_SHIFT"])
    assert pd.notna(row["HOME_TZ_SHIFT"])


def test_missing_rich_upcoming_artifact_does_not_break_historical_target() -> None:
    datasets = _datasets(
        games=_historical_games(),
        upcoming=DataFrame(),
        stadiums=_stadiums(),
    )
    datasets.schedule_upcoming_rich.side_effect = FileNotFoundError

    result = HomeAwayTravelFeature().compute(
        df=_target(),
        datasets=datasets,
    )

    assert result.iloc[0]["GAME_SITE_ALTITUDE"] == pytest.approx(274.0)


def test_missing_venue_preserves_row_with_null_outputs() -> None:
    result = _compute(
        games=DataFrame(),
        upcoming=DataFrame(),
    )

    assert len(result) == 1
    assert (
        result[
            [
                "GAME_SITE_ALTITUDE",
                "AWAY_KM_TRAVELED",
                "HOME_KM_TRAVELED",
                "AWAY_TZ_SHIFT",
                "HOME_TZ_SHIFT",
            ]
        ]
        .isna()
        .all()
        .all()
    )


def test_missing_team_origin_only_nulls_affected_side() -> None:
    stadiums = _stadiums().loc[
        lambda frame: frame["HOME_TEAM"] != "Las Vegas Raiders",
        :,
    ]

    row = _compute(stadiums=stadiums).iloc[0]

    assert pd.isna(row["AWAY_KM_TRAVELED"])
    assert pd.isna(row["AWAY_TZ_SHIFT"])
    assert row["HOME_KM_TRAVELED"] == pytest.approx(0.0, abs=0.01)
    assert row["HOME_TZ_SHIFT"] == 0
    assert row["GAME_SITE_ALTITUDE"] == pytest.approx(274.0)


def test_identical_coordinate_aliases_are_allowed() -> None:
    alias = DataFrame(
        [
            {
                "HOME_TEAM": "Kansas City Chiefs",
                "YEAR": "2025-2026",
                "STADIUM": "GEHA Field at Arrowhead Stadium",
                "LATITUDE": 39.0489,
                "LONGITUDE": -94.4839,
                "ALTITUDE": 274.0,
            }
        ]
    )
    stadiums = pd.concat([_stadiums(), alias], ignore_index=True)

    row = _compute(stadiums=stadiums).iloc[0]

    assert row["HOME_KM_TRAVELED"] == pytest.approx(0.0, abs=0.01)


def test_conflicting_game_venue_sources_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="conflicting stadium identities",
    ):
        _compute(
            games=_historical_games(stadium="Arrowhead Stadium"),
            upcoming=_upcoming_games(stadium="Wembley Stadium"),
        )


def test_conflicting_franchise_coordinates_are_rejected() -> None:
    conflict = DataFrame(
        [
            {
                "HOME_TEAM": "Kansas City Chiefs",
                "YEAR": "2025-2026",
                "STADIUM": "Conflicting Stadium",
                "LATITUDE": 40.0,
                "LONGITUDE": -95.0,
                "ALTITUDE": 300.0,
            }
        ]
    )
    stadiums = pd.concat([_stadiums(), conflict], ignore_index=True)

    with pytest.raises(
        ValueError,
        match="Franchise-season stadium reference contains conflicting coordinate identities",
    ):
        _compute(stadiums=stadiums)


def test_conflicting_site_coordinates_are_rejected() -> None:
    conflict = DataFrame(
        [
            {
                "HOME_TEAM": "Alternate",
                "YEAR": "2025-2026",
                "STADIUM": "Arrowhead Stadium",
                "LATITUDE": 40.0,
                "LONGITUDE": -95.0,
                "ALTITUDE": 300.0,
            }
        ]
    )
    stadiums = pd.concat([_stadiums(), conflict], ignore_index=True)

    with pytest.raises(
        ValueError,
        match="Stadium reference contains conflicting coordinate identities",
    ):
        _compute(stadiums=stadiums)


def test_preserves_input_order_columns_and_immutability() -> None:
    target = pd.concat(
        [
            _target(game_id="2025_01_LV_KC"),
            _target(game_id="unknown-game"),
        ],
        ignore_index=True,
    )
    expected = target.copy(deep=True)

    result = _compute(target=target)

    pd.testing.assert_frame_equal(target, expected)
    assert result["GAME_ID"].tolist() == [
        "2025_01_LV_KC",
        "unknown-game",
    ]
    assert result["MARKER"].tolist() == ["preserved", "preserved"]


@pytest.mark.parametrize(
    "column",
    ["GAME_ID", "YEAR", "AWAY_TEAM", "HOME_TEAM"],
)
def test_missing_target_column_is_rejected(column: str) -> None:
    target = _target().drop(columns=[column])

    with pytest.raises(
        ValueError,
        match=f"Home/away game frame is missing required columns: {column}",
    ):
        _compute(target=target)


def test_canonical_class_has_no_retired_orientation_names() -> None:
    source = inspect.getsource(HomeAwayTravelFeature)

    assert "TEAM_A" not in source
    assert "TEAM_B" not in source
    assert "HOME_FIELD" not in source

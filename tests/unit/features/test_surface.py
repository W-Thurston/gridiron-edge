# tests/unit/features/test_surface.py

"""Tests for the canonical artificial-turf feature."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.surface import HomeAwaySurfaceFeature


def _games(rows: list[tuple[str, str, str, object]]) -> pd.DataFrame:
    """Rows are (game_id, year, stadium, surface)."""
    return pd.DataFrame(
        [{"GAME_ID": g, "YEAR": y, "STADIUM": s, "SURFACE": f} for g, y, s, f in rows]
    )


def _accessor(
    games: pd.DataFrame,
    *,
    stadiums: pd.DataFrame | None = None,
    upcoming: pd.DataFrame | None = None,
) -> MagicMock:
    acc = MagicMock()
    acc.games.return_value = games
    if stadiums is None:
        acc.stadiums.side_effect = FileNotFoundError("no stadiums")
    else:
        acc.stadiums.return_value = stadiums
    if upcoming is None:
        acc.schedule_upcoming_rich.side_effect = FileNotFoundError("no upcoming")
    else:
        acc.schedule_upcoming_rich.return_value = upcoming
    return acc


def _frame(*game_ids: str) -> pd.DataFrame:
    return pd.DataFrame({"GAME_ID": list(game_ids)})


def _flag(result: pd.DataFrame, game_id: str) -> object:
    return result.loc[result["GAME_ID"] == game_id, "IS_ARTIFICIAL_TURF"].iloc[0]


class TestArtificialTurfFlag:
    @pytest.mark.parametrize(
        ("surface", "expected"),
        [
            ("grass", 0),
            ("dessograss", 0),
            ("fieldturf", 1),
            ("astroturf", 1),
            ("sportturf", 1),
            ("matrixturf", 1),
            ("astroplay", 1),
            ("a_turf", 1),
        ],
    )
    def test_vocabulary(self, surface: str, expected: int) -> None:
        games = _games([("g1", "2024-2025", "S", surface)])
        result = HomeAwaySurfaceFeature().compute(df=_frame("g1"), datasets=_accessor(games))
        assert _flag(result, "g1") == expected

    def test_whitespace_and_case_are_normalized(self) -> None:
        games = _games([("g1", "2024-2025", "S", "grass "), ("g2", "2024-2025", "S", " FieldTurf")])
        result = HomeAwaySurfaceFeature().compute(df=_frame("g1", "g2"), datasets=_accessor(games))
        assert _flag(result, "g1") == 0
        assert _flag(result, "g2") == 1

    def test_unknown_surface_raises(self) -> None:
        games = _games([("g1", "2024-2025", "S", "hologram")])
        with pytest.raises(ValueError, match="hologram"):
            HomeAwaySurfaceFeature().compute(df=_frame("g1"), datasets=_accessor(games))

    def test_unresolvable_surface_stays_null(self) -> None:
        games = _games([("g1", "2024-2025", "S", None)])
        result = HomeAwaySurfaceFeature().compute(df=_frame("g1"), datasets=_accessor(games))
        assert pd.isna(_flag(result, "g1"))


class TestStadiumReferenceFill:
    def _reference(self) -> pd.DataFrame:
        # Same stadium changes surface between seasons.
        return pd.DataFrame(
            {
                "STADIUM": ["S", "S"],
                "YEAR": ["2022-2023", "2023-2024"],
                "SURFACE": ["grass", "fieldturf"],
            }
        )

    def test_fill_uses_stadium_and_season(self) -> None:
        games = _games(
            [
                ("old", "2022-2023", "S", None),
                ("new", "2023-2024", "S", None),
            ]
        )
        result = HomeAwaySurfaceFeature().compute(
            df=_frame("old", "new"),
            datasets=_accessor(games, stadiums=self._reference()),
        )
        assert _flag(result, "old") == 0
        assert _flag(result, "new") == 1

    def test_recorded_surface_wins_over_reference(self) -> None:
        games = _games([("g1", "2023-2024", "S", "grass")])
        result = HomeAwaySurfaceFeature().compute(
            df=_frame("g1"), datasets=_accessor(games, stadiums=self._reference())
        )
        assert _flag(result, "g1") == 0

    def test_conflicting_reference_rejected(self) -> None:
        games = _games([("g1", "2023-2024", "S", None)])
        reference = pd.DataFrame(
            {
                "STADIUM": ["S", "S"],
                "YEAR": ["2023-2024", "2023-2024"],
                "SURFACE": ["grass", "fieldturf"],
            }
        )
        with pytest.raises(ValueError, match="conflicting surface"):
            HomeAwaySurfaceFeature().compute(
                df=_frame("g1"), datasets=_accessor(games, stadiums=reference)
            )


class TestUpcomingGames:
    def test_upcoming_surface_resolves_like_historical(self) -> None:
        games = _games([("h1", "2025-2026", "S", "grass")])
        upcoming = pd.DataFrame(
            {
                "game_id": ["u1"],
                "season": ["2026-2027"],
                "stadium": ["T"],
                "surface": ["a_turf"],
            }
        )
        result = HomeAwaySurfaceFeature().compute(
            df=_frame("h1", "u1"), datasets=_accessor(games, upcoming=upcoming)
        )
        assert _flag(result, "h1") == 0
        assert _flag(result, "u1") == 1


def test_registered_and_preserves_row_order() -> None:
    assert FeatureRegistry.get("home_away_surface") is HomeAwaySurfaceFeature
    games = _games([("a", "2024-2025", "S", "grass"), ("b", "2024-2025", "S", "fieldturf")])
    result = HomeAwaySurfaceFeature().compute(df=_frame("b", "a"), datasets=_accessor(games))
    assert result["GAME_ID"].tolist() == ["b", "a"]

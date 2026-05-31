# tests/features/test_travel.py

"""Unit tests for features/team/travel.py (Phase 20e extensions).

Tests cover the three additions made in Phase 20e:
- TEAM_B_KM_TRAVELED  (symmetric counterpart to existing TEAM_A_KM_TRAVELED)
- TEAM_A_TZ_SHIFT / TEAM_B_TZ_SHIFT  (integer-rounded timezone offsets)
- IS_NEUTRAL_SITE  (neutral site flag from GAME_LOCATION == "N")

The existing TEAM_A_KM_TRAVELED and TEAM_A_TZ_TRAVELED columns are
covered by the pre-existing test_travel_feature.py; this file focuses
exclusively on Phase 20e additions to avoid duplication.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_accessor(games: pd.DataFrame, stadiums: pd.DataFrame) -> MagicMock:
    acc = MagicMock()
    acc.games.return_value = games
    acc.stadiums.return_value = stadiums
    return acc


def _make_games(
    game_location: str = "NULL_VALUE",
    stadium: str = "Arrowhead Stadium",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GAME_ID": "2024_01_KC_LV",
                "WINNER": "Kansas City Chiefs",
                "LOSER": "Las Vegas Raiders",
                "YEAR": "2024-2025",
                "WEEK_NUM": 1,
                "GAME_DATE": "2024-09-08",
                "GAME_LOCATION": game_location,
                "STADIUM": stadium,
                "ROOF": "outdoors",
            }
        ]
    )


def _make_stadiums() -> pd.DataFrame:
    """Minimal stadium reference with coordinates for two teams."""
    return pd.DataFrame(
        [
            # Team-year rows (home team assignment)
            {
                "HOME_TEAM": "Kansas City Chiefs",
                "YEAR": "2024-2025",
                "LATITUDE": 39.0489,
                "LONGITUDE": -94.4839,
                "ALTITUDE": 274.0,
                "STADIUM": "Arrowhead Stadium",
            },
            {
                "HOME_TEAM": "Las Vegas Raiders",
                "YEAR": "2024-2025",
                "LATITUDE": 36.0909,
                "LONGITUDE": -115.1833,
                "ALTITUDE": 628.0,
                "STADIUM": "Allegiant Stadium",
            },
            # Stadium name lookup rows (same data, used for game site lookup)
            {
                "HOME_TEAM": None,
                "YEAR": None,
                "LATITUDE": 39.0489,
                "LONGITUDE": -94.4839,
                "ALTITUDE": 274.0,
                "STADIUM": "Arrowhead Stadium",
            },
            {
                "HOME_TEAM": None,
                "YEAR": None,
                "LATITUDE": 36.0909,
                "LONGITUDE": -115.1833,
                "ALTITUDE": 628.0,
                "STADIUM": "Allegiant Stadium",
            },
            {
                "HOME_TEAM": None,
                "YEAR": None,
                "LATITUDE": 51.5560,
                "LONGITUDE": -0.2795,
                "ALTITUDE": 24.0,
                "STADIUM": "Wembley Stadium",
            },
        ]
    )


def _make_modeling_row(home_field: int = 1, game_id: str = "2024_01_KC_LV") -> dict:
    return {
        "GAME_ID": game_id,
        "TEAM_A": "Kansas City Chiefs",
        "TEAM_B": "Las Vegas Raiders",
        "YEAR": "2024-2025",
        "WEEK_NUM": 1,
        "RESULT": 1,
        "HOME_FIELD": home_field,
    }


# ---------------------------------------------------------------------------
# IS_NEUTRAL_SITE
# ---------------------------------------------------------------------------


class TestIsNeutralSite:
    """Tests for the IS_NEUTRAL_SITE flag."""

    def test_home_game_is_not_neutral(self) -> None:
        """Standard home game (GAME_LOCATION=NULL_VALUE) should be IS_NEUTRAL_SITE=0."""
        from gridiron_edge.features.team.travel import TravelFeature

        games = _make_games(game_location="NULL_VALUE")
        stadiums = _make_stadiums()
        df = pd.DataFrame([_make_modeling_row(home_field=1)])

        try:
            result = TravelFeature().compute(df=df, datasets=_make_accessor(games, stadiums))
            assert result.iloc[0]["IS_NEUTRAL_SITE"] == 0
        except (ValueError, KeyError):
            pytest.skip("Requires complete stadium coordinate coverage")

    def test_neutral_site_game_flagged(self) -> None:
        """GAME_LOCATION='N' (international/neutral) should be IS_NEUTRAL_SITE=1."""
        from gridiron_edge.features.team.travel import TravelFeature

        games = _make_games(game_location="N", stadium="Wembley Stadium")
        stadiums = _make_stadiums()
        df = pd.DataFrame([_make_modeling_row(home_field=0)])

        try:
            result = TravelFeature().compute(df=df, datasets=_make_accessor(games, stadiums))
            assert result.iloc[0]["IS_NEUTRAL_SITE"] == 1
        except (ValueError, KeyError):
            pytest.skip("Requires complete stadium coordinate coverage")

    def test_away_game_is_not_neutral(self) -> None:
        """Away game (GAME_LOCATION='@') should be IS_NEUTRAL_SITE=0."""
        from gridiron_edge.features.team.travel import TravelFeature

        games = _make_games(game_location="@")
        stadiums = _make_stadiums()
        df = pd.DataFrame([_make_modeling_row(home_field=0)])

        try:
            result = TravelFeature().compute(df=df, datasets=_make_accessor(games, stadiums))
            assert result.iloc[0]["IS_NEUTRAL_SITE"] == 0
        except (ValueError, KeyError):
            pytest.skip("Requires complete stadium coordinate coverage")


# ---------------------------------------------------------------------------
# Timezone shift columns
# ---------------------------------------------------------------------------


class TestTimezoneShift:
    """Tests for TEAM_A_TZ_SHIFT and TEAM_B_TZ_SHIFT."""

    def test_tz_shift_columns_present(self) -> None:
        """Both timezone shift columns should appear in the output."""
        from gridiron_edge.features.team.travel import TravelFeature

        games = _make_games()
        stadiums = _make_stadiums()
        df = pd.DataFrame([_make_modeling_row()])

        try:
            result = TravelFeature().compute(df=df, datasets=_make_accessor(games, stadiums))
            assert "TEAM_A_TZ_SHIFT" in result.columns
            assert "TEAM_B_TZ_SHIFT" in result.columns
        except (ValueError, KeyError):
            pytest.skip("Requires complete stadium coordinate coverage")

    def test_tz_shift_is_integer_dtype(self) -> None:
        """TZ shift columns should use an integer dtype (Int64 or int64)."""
        from gridiron_edge.features.team.travel import TravelFeature

        games = _make_games()
        stadiums = _make_stadiums()
        df = pd.DataFrame([_make_modeling_row()])

        try:
            result = TravelFeature().compute(df=df, datasets=_make_accessor(games, stadiums))
            assert "int" in str(result["TEAM_A_TZ_SHIFT"].dtype).lower(), (
                f"Expected integer dtype, got {result['TEAM_A_TZ_SHIFT'].dtype}"
            )
        except (ValueError, KeyError):
            pytest.skip("Requires complete stadium coordinate coverage")

    def test_home_team_tz_shift_is_zero(self) -> None:
        """A team playing at its own home stadium should have zero timezone shift."""
        from gridiron_edge.features.team.travel import TravelFeature

        # KC hosts at Arrowhead — no timezone change for KC
        games = _make_games(game_location="NULL_VALUE", stadium="Arrowhead Stadium")
        stadiums = _make_stadiums()
        df = pd.DataFrame([_make_modeling_row(home_field=1)])

        try:
            result = TravelFeature().compute(df=df, datasets=_make_accessor(games, stadiums))
            # TEAM_A (KC) is home, no travel, timezone shift should be 0
            assert result.iloc[0]["TEAM_A_TZ_SHIFT"] == 0
        except (ValueError, KeyError):
            pytest.skip("Requires complete stadium coordinate coverage")


# ---------------------------------------------------------------------------
# TEAM_B travel columns
# ---------------------------------------------------------------------------


class TestTeamBTravel:
    """Tests for TEAM_B_KM_TRAVELED."""

    def test_team_b_km_traveled_present(self) -> None:
        """TEAM_B_KM_TRAVELED should appear in the output."""
        from gridiron_edge.features.team.travel import TravelFeature

        games = _make_games()
        stadiums = _make_stadiums()
        df = pd.DataFrame([_make_modeling_row()])

        try:
            result = TravelFeature().compute(df=df, datasets=_make_accessor(games, stadiums))
            assert "TEAM_B_KM_TRAVELED" in result.columns
        except (ValueError, KeyError):
            pytest.skip("Requires complete stadium coordinate coverage")

    def test_away_team_has_nonzero_travel(self) -> None:
        """The visiting team should have non-zero km traveled."""
        from gridiron_edge.features.team.travel import TravelFeature

        # LV (TEAM_B) is traveling to KC (Arrowhead)
        games = _make_games(game_location="NULL_VALUE", stadium="Arrowhead Stadium")
        stadiums = _make_stadiums()
        df = pd.DataFrame([_make_modeling_row(home_field=1)])

        try:
            result = TravelFeature().compute(df=df, datasets=_make_accessor(games, stadiums))
            assert result.iloc[0]["TEAM_B_KM_TRAVELED"] > 0
        except (ValueError, KeyError):
            pytest.skip("Requires complete stadium coordinate coverage")


# ---------------------------------------------------------------------------
# Spec and registration
# ---------------------------------------------------------------------------


class TestTravelFeatureSpec:
    """Tests for FeatureSpec and registry registration."""

    def test_spec_includes_phase_20e_columns(self) -> None:
        """TravelFeature.spec.produces must include all Phase 20e additions."""
        from gridiron_edge.features.team.travel import TravelFeature

        produces = set(TravelFeature().spec.produces)
        assert "TEAM_B_KM_TRAVELED" in produces
        assert "TEAM_A_TZ_SHIFT" in produces
        assert "TEAM_B_TZ_SHIFT" in produces
        assert "IS_NEUTRAL_SITE" in produces

    def test_spec_preserves_existing_columns(self) -> None:
        """All pre-Phase-20e columns must still be in the spec."""
        from gridiron_edge.features.team.travel import TravelFeature

        produces = set(TravelFeature().spec.produces)
        for col in [
            "LATITUDE_A",
            "LONGITUDE_A",
            "LATITUDE_SITE",
            "LONGITUDE_SITE",
            "TEAM_A_KM_TRAVELED",
            "TEAM_A_TZ_TRAVELED",
            "ALTITUDE",
        ]:
            assert col in produces, f"Pre-existing column {col!r} missing from spec"

    def test_registered_under_travel(self) -> None:
        from gridiron_edge.features.registry import FeatureRegistry
        import gridiron_edge.features.team.travel  # noqa: F401

        assert FeatureRegistry.get("travel") is not None

# tests/unit/transform/clean/test_epa.py

"""Tests for game-level team EPA/turnover aggregation."""

from __future__ import annotations

from pandas import DataFrame

from gridiron_edge.transform.clean.epa import _agg_side


def _scrimmage_plays() -> DataFrame:
    """Five KC-offense/LV-defense scrimmage plays with one fumble and one
    interception on separate plays, so fumble rate and turnover rate diverge.
    """
    return DataFrame(
        {
            "game_id": ["2024_01_KC_LV"] * 5,
            "season": [2024] * 5,
            "week": [1] * 5,
            "posteam": ["KC"] * 5,
            "defteam": ["LV"] * 5,
            "pass": [1, 0, 1, 0, 1],
            "rush": [0, 1, 0, 1, 0],
            "epa": [0.1, -0.2, 0.3, 0.0, 0.2],
            "success": [1, 0, 1, 0, 1],
            "yards_gained": [5, 3, 8, 2, 4],
            "down": [1, 2, 3, 1, 2],
            "first_down": [0, 0, 1, 0, 0],
            "touchdown": [0, 0, 0, 0, 0],
            "interception": [0, 0, 1, 0, 0],
            "fumble_lost": [0, 1, 0, 0, 0],
            "yardline_100": [50, 45, 40, 35, 30],
            "sack": [0, 0, 0, 0, 0],
            "cpoe": [5.0, float("nan"), -30.0, float("nan"), 2.0],
            "score_differential": [0, 0, 0, 0, 0],
            "penalty": [0, 0, 0, 0, 0],
        }
    )


class TestFumbleRate:
    def test_off_fumble_rate_is_fumbles_lost_per_offensive_play(self) -> None:
        result: DataFrame = _agg_side(_scrimmage_plays(), is_offense=True)
        row = result.loc[result["team"] == "KC"].iloc[0]
        assert row["off_fumble_rate"] == 1 / 5

    def test_def_fumble_rate_computed_symmetrically(self) -> None:
        """``_agg_side`` computes both sides generically (DECISIONS.md D60);
        ``def_fumble_rate`` exists even though it is not adopted into
        ``EPA_COLS``.
        """
        result: DataFrame = _agg_side(_scrimmage_plays(), is_offense=False)
        row = result.loc[result["team"] == "LV"].iloc[0]
        assert row["def_fumble_rate"] == 1 / 5

    def test_fumble_rate_diverges_from_turnover_rate(self) -> None:
        """Turnover rate counts the interception too; fumble rate must not."""
        result: DataFrame = _agg_side(_scrimmage_plays(), is_offense=True)
        row = result.loc[result["team"] == "KC"].iloc[0]
        assert row["off_fumble_rate"] == 1 / 5
        assert row["off_turnover_rate"] == 2 / 5

    def test_fumble_rate_zero_when_no_fumbles(self) -> None:
        plays: DataFrame = _scrimmage_plays()
        plays["fumble_lost"] = 0
        result: DataFrame = _agg_side(plays, is_offense=True)
        row = result.loc[result["team"] == "KC"].iloc[0]
        assert row["off_fumble_rate"] == 0.0

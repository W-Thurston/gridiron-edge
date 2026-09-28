# tests/unit/transform/clean/test_epa.py

"""Tests for game-level team EPA/turnover aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.transform.clean.epa import _agg_side, _agg_special_teams, aggregate_epa


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


def _special_teams_plays() -> DataFrame:
    """One game of special-teams plays between KC and LV.

    Includes a kickoff and a punt with no EPA, both of which must be ignored.
    """
    plays = [
        ("punt", "KC", "LV", -0.5),
        ("punt", "LV", "KC", 0.3),
        ("field_goal", "KC", "LV", 1.0),
        ("extra_point", "LV", "KC", 0.05),
        ("kickoff", "KC", "LV", 0.25),
        ("punt", "KC", "LV", float("nan")),
    ]
    return DataFrame(
        {
            "game_id": ["2024_01_KC_LV"] * len(plays),
            "season": [2024] * len(plays),
            "week": [1] * len(plays),
            "play_type": [p[0] for p in plays],
            "posteam": [p[1] for p in plays],
            "defteam": [p[2] for p in plays],
            "epa": [p[3] for p in plays],
        }
    )


class TestSpecialTeamsAggregation:
    def _row(self, team: str) -> pd.Series:
        result = _agg_special_teams(_special_teams_plays())
        return result.loc[result["team"] == team].iloc[0]

    def test_punt_epa_is_net_of_punting_and_returning(self) -> None:
        # KC: own punt -0.5; returned LV's 0.3 punt -> -0.3.
        assert self._row("KC")["punt_epa_per_play"] == pytest.approx((-0.5 - 0.3) / 2)
        # LV: own punt 0.3; returned KC's -0.5 punt -> +0.5.
        assert self._row("LV")["punt_epa_per_play"] == pytest.approx((0.3 + 0.5) / 2)

    def test_special_teams_epa_adds_own_kicks_only(self) -> None:
        # KC: punt rows plus its own field goal (1.0); LV's extra point is not
        # counted against KC's defense.
        assert self._row("KC")["st_epa_per_play"] == pytest.approx((-0.5 - 0.3 + 1.0) / 3)
        assert self._row("LV")["st_epa_per_play"] == pytest.approx((0.3 + 0.5 + 0.05) / 3)

    def test_kickoffs_and_missing_epa_are_excluded(self) -> None:
        plays = _special_teams_plays()
        baseline = _agg_special_teams(plays)
        changed = _agg_special_teams(
            plays.assign(epa=plays["epa"].where(plays["play_type"] != "kickoff", 99.0))
        )
        pd.testing.assert_frame_equal(baseline, changed)

    def test_team_without_punt_has_null_punt_epa_but_kick_epa(self) -> None:
        plays = _special_teams_plays()
        plays = plays.loc[plays["play_type"] != "punt", :]
        row = _agg_special_teams(plays).set_index("team").loc["KC"]

        assert pd.isna(row["punt_epa_per_play"])
        assert row["st_epa_per_play"] == pytest.approx(1.0)

    def test_one_row_per_team_game(self) -> None:
        result = _agg_special_teams(_special_teams_plays())

        assert result[["game_id", "team"]].duplicated().sum() == 0
        assert sorted(result["team"]) == ["KC", "LV"]


def test_aggregate_epa_joins_special_teams_under_long_team_names(tmp_path: Path) -> None:
    """Special-teams columns survive the short-code -> long-name mapping (D64)."""
    scrimmage = _scrimmage_plays().assign(play_type="pass")
    specials = _special_teams_plays()
    pbp = pd.concat([scrimmage, specials], ignore_index=True)
    pbp["play_type"] = pbp["play_type"].fillna("pass")
    pbp_dir = tmp_path / "data" / "raw" / "pbp"
    pbp_dir.mkdir(parents=True)
    (tmp_path / "data" / "cleaned").mkdir(parents=True)
    pbp.to_parquet(pbp_dir / "play_by_play_2024.parquet", index=False)

    out = aggregate_epa(seasons=[2024], repo=tmp_path)
    result = pd.read_parquet(out)

    kc = result.loc[result["team"] == "Kansas City Chiefs"]
    assert len(kc) == 1
    assert kc.iloc[0]["punt_epa_per_play"] == pytest.approx((-0.5 - 0.3) / 2)
    assert kc.iloc[0]["st_epa_per_play"] == pytest.approx((-0.5 - 0.3 + 1.0) / 3)

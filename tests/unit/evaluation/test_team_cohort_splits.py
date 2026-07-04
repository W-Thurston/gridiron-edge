# tests/unit/evaluation/test_team_cohort_splits.py

"""Unit tests for evaluation/team_cohort_splits.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

LONG_TO_SHORT = {
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "Buffalo Bills": "BUF",
    "Baltimore Ravens": "BAL",
}


def _make_epa_data() -> pd.DataFrame:
    """4 teams playing 4 games in 2024.

    Structure:
    - Game 1 (week 1): KC vs LAC at LAC. KC away, LAC home.
    - Game 2 (week 2): KC vs BUF at KC. KC home, BUF away.
    - Game 3 (week 3): KC vs BAL at BAL. KC away, BAL home.
    - Game 4 (week 4): KC vs LAC at KC. KC home, LAC away.
    """
    return pd.DataFrame(
        [
            # Game 1: KC @ LAC
            {
                "game_id": "2024_01_KC_LAC",
                "season": 2024,
                "week": 1,
                "team": "Kansas City Chiefs",
                "off_epa_per_play": 0.15,
                "off_pass_epa": 0.20,
                "off_rush_epa": 0.05,
                "def_epa_per_play": -0.10,
                "def_rush_epa": -0.05,
                "off_third_down_pct": 0.45,
                "off_redzone_td_pct": 0.60,
                "off_turnover_rate": 0.02,
                "def_turnover_rate": 0.05,
            },
            {
                "game_id": "2024_01_KC_LAC",
                "season": 2024,
                "week": 1,
                "team": "Los Angeles Chargers",
                "off_epa_per_play": 0.10,
                "off_pass_epa": 0.15,
                "off_rush_epa": 0.00,
                "def_epa_per_play": -0.05,
                "def_rush_epa": 0.00,
                "off_third_down_pct": 0.40,
                "off_redzone_td_pct": 0.50,
                "off_turnover_rate": 0.05,
                "def_turnover_rate": 0.02,
            },
            # Game 2: BUF @ KC
            {
                "game_id": "2024_02_BUF_KC",
                "season": 2024,
                "week": 2,
                "team": "Kansas City Chiefs",
                "off_epa_per_play": 0.20,
                "off_pass_epa": 0.25,
                "off_rush_epa": 0.10,
                "def_epa_per_play": -0.15,
                "def_rush_epa": -0.10,
                "off_third_down_pct": 0.50,
                "off_redzone_td_pct": 0.70,
                "off_turnover_rate": 0.01,
                "def_turnover_rate": 0.06,
            },
            {
                "game_id": "2024_02_BUF_KC",
                "season": 2024,
                "week": 2,
                "team": "Buffalo Bills",
                "off_epa_per_play": 0.05,
                "off_pass_epa": 0.10,
                "off_rush_epa": -0.05,
                "def_epa_per_play": 0.00,
                "def_rush_epa": 0.05,
                "off_third_down_pct": 0.35,
                "off_redzone_td_pct": 0.45,
                "off_turnover_rate": 0.06,
                "def_turnover_rate": 0.01,
            },
            # Game 3: KC @ BAL
            {
                "game_id": "2024_03_KC_BAL",
                "season": 2024,
                "week": 3,
                "team": "Kansas City Chiefs",
                "off_epa_per_play": 0.10,
                "off_pass_epa": 0.12,
                "off_rush_epa": 0.05,
                "def_epa_per_play": -0.05,
                "def_rush_epa": 0.00,
                "off_third_down_pct": 0.42,
                "off_redzone_td_pct": 0.55,
                "off_turnover_rate": 0.03,
                "def_turnover_rate": 0.04,
            },
            {
                "game_id": "2024_03_KC_BAL",
                "season": 2024,
                "week": 3,
                "team": "Baltimore Ravens",
                "off_epa_per_play": 0.25,
                "off_pass_epa": 0.30,
                "off_rush_epa": 0.15,
                "def_epa_per_play": -0.20,
                "def_rush_epa": -0.15,
                "off_third_down_pct": 0.55,
                "off_redzone_td_pct": 0.75,
                "off_turnover_rate": 0.02,
                "def_turnover_rate": 0.07,
            },
            # Game 4: LAC @ KC
            {
                "game_id": "2024_04_LAC_KC",
                "season": 2024,
                "week": 4,
                "team": "Kansas City Chiefs",
                "off_epa_per_play": 0.18,
                "off_pass_epa": 0.22,
                "off_rush_epa": 0.08,
                "def_epa_per_play": -0.12,
                "def_rush_epa": -0.08,
                "off_third_down_pct": 0.48,
                "off_redzone_td_pct": 0.65,
                "off_turnover_rate": 0.02,
                "def_turnover_rate": 0.06,
            },
            {
                "game_id": "2024_04_LAC_KC",
                "season": 2024,
                "week": 4,
                "team": "Los Angeles Chargers",
                "off_epa_per_play": 0.12,
                "off_pass_epa": 0.18,
                "off_rush_epa": 0.02,
                "def_epa_per_play": -0.08,
                "def_rush_epa": -0.02,
                "off_third_down_pct": 0.42,
                "off_redzone_td_pct": 0.55,
                "off_turnover_rate": 0.04,
                "def_turnover_rate": 0.03,
            },
        ]
    )


class TestComputeTeamCohortSplits:
    def test_empty_input_returns_empty(self) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        result = compute_team_cohort_splits(pd.DataFrame(), LONG_TO_SHORT)
        assert result.empty

    def test_missing_columns_returns_empty(self) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        df = pd.DataFrame([{"team": "KC", "game_id": "2024_01_KC_LAC"}])
        result = compute_team_cohort_splits(df, LONG_TO_SHORT)
        assert result.empty

    def test_season_cohort_computes_full_sample(self) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        result = compute_team_cohort_splits(_make_epa_data(), LONG_TO_SHORT)

        # KC played 4 games in season cohort. Off EPA mean: (0.15 + 0.20 + 0.10 + 0.18) / 4 = 0.1575
        kc_season = result.loc[
            (result["team_abbr"] == "KC") & (result["cohort"] == "season"),
            :,
        ]
        assert len(kc_season) == 1
        assert kc_season.iloc[0]["off_epa_per_play"] == pytest.approx(0.1575)
        assert kc_season.iloc[0]["sample_size"] == 4

    def test_home_cohort_partitions_correctly(self) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        result = compute_team_cohort_splits(_make_epa_data(), LONG_TO_SHORT)

        # KC played home in games 2 and 4.
        # game_id parsing: "2024_02_BUF_KC" → HOME=KC (yes)
        # game_id parsing: "2024_04_LAC_KC" → HOME=KC (yes)
        kc_home = result.loc[
            (result["team_abbr"] == "KC") & (result["cohort"] == "home"),
            :,
        ]
        assert len(kc_home) == 1
        assert kc_home.iloc[0]["sample_size"] == 2
        # Off EPA mean: (0.20 + 0.18) / 2 = 0.19
        assert kc_home.iloc[0]["off_epa_per_play"] == pytest.approx(0.19)

    def test_away_cohort_partitions_correctly(self) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        result = compute_team_cohort_splits(_make_epa_data(), LONG_TO_SHORT)

        # KC played away in games 1 and 3.
        kc_away = result.loc[
            (result["team_abbr"] == "KC") & (result["cohort"] == "away"),
            :,
        ]
        assert kc_away.iloc[0]["sample_size"] == 2
        # Off EPA mean: (0.15 + 0.10) / 2 = 0.125
        assert kc_away.iloc[0]["off_epa_per_play"] == pytest.approx(0.125)

    def test_l4_returns_last_4_games(self) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        result = compute_team_cohort_splits(_make_epa_data(), LONG_TO_SHORT)

        # KC has exactly 4 games; l4 = season for this fixture.
        kc_l4 = result.loc[
            (result["team_abbr"] == "KC") & (result["cohort"] == "l4"),
            :,
        ]
        assert kc_l4.iloc[0]["sample_size"] == 4

    def test_turnover_diff_computed(self) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        result = compute_team_cohort_splits(_make_epa_data(), LONG_TO_SHORT)

        # KC turnover_diff (season):
        # Games: (0.02-0.05, 0.01-0.06, 0.03-0.04, 0.02-0.06)
        #      = (-0.03, -0.05, -0.01, -0.04)
        # Mean: -0.0325
        kc_season = result.loc[
            (result["team_abbr"] == "KC") & (result["cohort"] == "season"),
            :,
        ]
        assert kc_season.iloc[0]["turnover_diff"] == pytest.approx(-0.0325)

    def test_ranks_off_metrics_descending(self) -> None:
        """Off metrics: rank 1 = highest."""
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        result = compute_team_cohort_splits(_make_epa_data(), LONG_TO_SHORT)

        # In season cohort:
        # BAL off_epa: 0.25 (rank 1)
        # KC off_epa: 0.1575 (rank 2)
        # LAC off_epa: 0.11 avg (rank 3)
        # BUF off_epa: 0.05 (rank 4)
        season = result.loc[result["cohort"] == "season", :]
        by_team = dict(zip(season["team_abbr"], season["rank_off_epa_per_play"], strict=False))
        assert by_team["BAL"] == 1
        assert by_team["KC"] == 2

    def test_ranks_def_metrics_ascending(self) -> None:
        """Def metrics: rank 1 = lowest (stingiest)."""
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        result = compute_team_cohort_splits(_make_epa_data(), LONG_TO_SHORT)

        # In season cohort:
        # BAL def_epa: -0.20 (rank 1, stingiest)
        # KC def_epa: -0.105 avg (rank 2)
        # LAC def_epa: -0.065 avg (rank 3)
        # BUF def_epa: 0.00 (rank 4)
        season = result.loc[result["cohort"] == "season", :]
        by_team = dict(zip(season["team_abbr"], season["rank_def_epa_per_play"], strict=False))
        assert by_team["BAL"] == 1

    def test_only_latest_season_included(self) -> None:
        """Older seasons are ignored."""
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        old = _make_epa_data().copy()
        old["season"] = 2023
        old["off_epa_per_play"] = 0.999  # Very different value

        combined = pd.concat([old, _make_epa_data()], ignore_index=True)

        result = compute_team_cohort_splits(combined, LONG_TO_SHORT)

        # KC's off_epa_per_play (season) should reflect 2024 data only.
        kc_season = result.loc[
            (result["team_abbr"] == "KC") & (result["cohort"] == "season"),
            :,
        ]
        assert kc_season.iloc[0]["off_epa_per_play"] == pytest.approx(0.1575)

    def test_teams_without_long_to_short_mapping(self) -> None:
        """Teams not in long_to_short map fall back to their raw name."""
        from gridiron_edge.evaluation.team_cohort_splits import (
            compute_team_cohort_splits,
        )

        # Add a row with an unmapped team.
        epa = _make_epa_data().copy()
        new_row = epa.iloc[0].copy()
        new_row["team"] = "Mystery Team"
        new_row["game_id"] = "2024_01_MTS_LAC"
        epa = pd.concat([epa, new_row.to_frame().T], ignore_index=True)

        result = compute_team_cohort_splits(epa, LONG_TO_SHORT)

        # Mystery Team should appear as team_abbr = "Mystery Team".
        mystery = result.loc[result["team_abbr"] == "Mystery Team", :]
        assert len(mystery) > 0


class TestWriteTeamCohortSplits:
    def test_writes_to_expected_path(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            write_team_cohort_splits,
        )

        df = pd.DataFrame(
            [
                {
                    "team_abbr": "KC",
                    "cohort": "season",
                    "off_epa_per_play": 0.15,
                    "off_pass_epa": 0.20,
                    "off_rush_epa": 0.05,
                    "def_epa_per_play": -0.10,
                    "def_rush_epa": -0.05,
                    "off_third_down_pct": 0.45,
                    "off_redzone_td_pct": 0.60,
                    "turnover_diff": -0.03,
                    "sample_size": 4,
                    "rank_off_epa_per_play": 2,
                    "rank_off_pass_epa": 2,
                    "rank_off_rush_epa": 2,
                    "rank_def_epa_per_play": 2,
                    "rank_def_rush_epa": 2,
                    "rank_off_third_down_pct": 2,
                    "rank_off_redzone_td_pct": 2,
                    "rank_turnover_diff": 2,
                }
            ]
        )

        path = write_team_cohort_splits(df, tmp_path)
        assert path.exists()
        assert path.name == "team_cohort_splits.parquet"


class TestLoadTeamCohortSplits:
    def test_empty_when_missing(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.team_cohort_splits import (
            load_team_cohort_splits,
        )

        result = load_team_cohort_splits(tmp_path)
        assert result.empty

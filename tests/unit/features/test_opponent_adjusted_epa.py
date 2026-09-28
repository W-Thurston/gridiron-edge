# tests/unit/features/test_opponent_adjusted_epa.py

"""Tests for the ridge-regression opponent-adjustment core and feature."""

from __future__ import annotations

from itertools import permutations
from unittest.mock import MagicMock

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.datasets.accessor import DatasetAccessor
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team.opponent_adjusted_epa import (
    ADJUSTED_EPA_METRICS,
    RIDGE_ALPHA,
    HomeAwayOpponentAdjustedEpaFeature,
    _build_target_opponent_adjusted_epa,
    _fit_opponent_adjusted_epa,
)

# True generating strengths for a small synthetic league. Plays are
# constructed with no noise (epa = off_strength[offense] +
# def_strength[defense] exactly), so a correct regression should recover
# each team's relative standing within its own group.
_OFFENSE_STRENGTH: dict[str, float] = {"A": 0.30, "B": 0.10, "C": -0.10, "D": -0.30}
_DEFENSE_STRENGTH: dict[str, float] = {"A": -0.05, "B": 0.0, "C": 0.0, "D": 0.05}


def _round_robin_plays(plays_per_matchup: int = 50) -> DataFrame:
    """Every team plays every other team, both ways, with no noise."""
    rows: list[dict[str, object]] = []
    for offense, defense in permutations(_OFFENSE_STRENGTH, 2):
        epa = _OFFENSE_STRENGTH[offense] + _DEFENSE_STRENGTH[defense]
        rows.extend(
            {"posteam": offense, "defteam": defense, "epa": epa} for _ in range(plays_per_matchup)
        )
    return DataFrame(rows)


class TestFitOpponentAdjustedEpa:
    def test_recovers_relative_offense_strength(self) -> None:
        coefficients = _fit_opponent_adjusted_epa(_round_robin_plays(), alpha=0.01)

        off_a, _ = coefficients["A"]
        off_b, _ = coefficients["B"]
        off_c, _ = coefficients["C"]
        off_d, _ = coefficients["D"]

        assert off_a - off_b == pytest.approx(
            _OFFENSE_STRENGTH["A"] - _OFFENSE_STRENGTH["B"], abs=0.02
        )
        assert off_a - off_d == pytest.approx(
            _OFFENSE_STRENGTH["A"] - _OFFENSE_STRENGTH["D"], abs=0.02
        )
        assert off_a > off_b > off_c > off_d

    def test_recovers_relative_defense_strength(self) -> None:
        coefficients = _fit_opponent_adjusted_epa(_round_robin_plays(), alpha=0.01)

        _, def_a = coefficients["A"]
        _, def_d = coefficients["D"]

        assert def_a - def_d == pytest.approx(
            _DEFENSE_STRENGTH["A"] - _DEFENSE_STRENGTH["D"], abs=0.02
        )
        assert def_a < def_d

    def test_every_team_gets_a_coefficient(self) -> None:
        coefficients = _fit_opponent_adjusted_epa(_round_robin_plays())
        assert set(coefficients) == set(_OFFENSE_STRENGTH)

    def test_default_alpha_matches_module_constant(self) -> None:
        with_default = _fit_opponent_adjusted_epa(_round_robin_plays())
        with_explicit = _fit_opponent_adjusted_epa(_round_robin_plays(), alpha=RIDGE_ALPHA)
        assert with_default == with_explicit

    def test_team_appearing_on_only_one_side_still_gets_both_coefficients(self) -> None:
        plays = _round_robin_plays()
        one_sided = pd.concat(
            [
                plays,
                DataFrame([{"posteam": "E", "defteam": "A", "epa": 0.0}]),
            ],
            ignore_index=True,
        )

        coefficients = _fit_opponent_adjusted_epa(one_sided)

        assert "E" in coefficients
        off_e, def_e = coefficients["E"]
        assert pd.notna(off_e)
        assert pd.notna(def_e)

    def test_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            _fit_opponent_adjusted_epa(DataFrame({"posteam": ["A"], "defteam": ["B"]}))

    def test_empty_plays_raises(self) -> None:
        with pytest.raises(ValueError, match="empty play sample"):
            _fit_opponent_adjusted_epa(DataFrame(columns=["posteam", "defteam", "epa"]))


class TestShrinkage:
    def test_thin_sample_team_is_pulled_toward_average(self) -> None:
        """A team with the same true strength but far fewer plays should
        end up with a smaller-magnitude coefficient than a team with ample
        data at the same true strength - ridge's shrinkage in action."""
        plays = _round_robin_plays(plays_per_matchup=50)

        # Add a fifth team with the same offensive strength as "A" but
        # only one observed play, against a neutral-strength defense.
        thin_sample = pd.concat(
            [
                plays,
                DataFrame(
                    [
                        {
                            "posteam": "E",
                            "defteam": "B",
                            "epa": _OFFENSE_STRENGTH["A"] + _DEFENSE_STRENGTH["B"],
                        },
                        {"posteam": "C", "defteam": "E", "epa": 0.0},
                    ]
                ),
            ],
            ignore_index=True,
        )

        coefficients = _fit_opponent_adjusted_epa(thin_sample)

        off_a, _ = coefficients["A"]
        off_e, _ = coefficients["E"]

        assert abs(off_e) < abs(off_a)


def _play(
    season: int,
    week: int,
    posteam: str,
    defteam: str,
    epa: float,
    *,
    is_pass: bool,
) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "posteam": posteam,
        "defteam": defteam,
        "pass": 1 if is_pass else 0,
        "rush": 0 if is_pass else 1,
        "epa": epa,
    }


def _balanced_week(
    season: int,
    week: int,
    *,
    teams: tuple[str, ...] = ("A", "B", "C", "D"),
    epa: float = 0.1,
) -> list[dict[str, object]]:
    """One pass and one rush play for every ordered pair, all equal epa.

    Fully symmetric, so every team's fitted coefficient should land near
    zero - useful as an inert "known week" to build leakage/timing tests
    around without the fit's own noise complicating the assertion.
    """
    rows: list[dict[str, object]] = []
    for offense, defense in permutations(teams, 2):
        rows.append(_play(season, week, offense, defense, epa, is_pass=True))
        rows.append(_play(season, week, offense, defense, epa, is_pass=False))
    return rows


def _lopsided_week(
    season: int,
    week: int,
    *,
    teams: tuple[str, ...] = ("A", "B", "C", "D"),
    favored_team: str = "A",
    favored_epa: float = 50.0,
    other_epa: float = 0.0,
) -> list[dict[str, object]]:
    """Like ``_balanced_week``, but one team's offense is a clear outlier.

    A symmetric week can never move one team's coefficient relative to
    another's, no matter its magnitude - it only shifts the intercept.
    Tests that need "does including/excluding this week change the
    result" need an asymmetric week like this one.
    """
    rows: list[dict[str, object]] = []
    for offense, defense in permutations(teams, 2):
        epa = favored_epa if offense == favored_team else other_epa
        rows.append(_play(season, week, offense, defense, epa, is_pass=True))
        rows.append(_play(season, week, offense, defense, epa, is_pass=False))
    return rows


class TestBuildTargetOpponentAdjustedEpa:
    def test_week_one_has_no_data_and_is_nan(self) -> None:
        plays = DataFrame(_balanced_week(2024, 1))
        targets = DataFrame([{"season": 2024, "week": 1, "team": "A"}])

        result = _build_target_opponent_adjusted_epa(plays, targets)

        row = result.iloc[0]
        for metric in ADJUSTED_EPA_METRICS:
            assert pd.isna(row[metric])

    def test_no_cross_season_leakage(self) -> None:
        extreme_prior_season = [
            _play(2023, 18, "A", "B", 100.0, is_pass=True),
            _play(2023, 18, "B", "A", -100.0, is_pass=True),
        ]
        current_season_week1 = _balanced_week(2024, 1, epa=0.1)
        plays = DataFrame(extreme_prior_season + current_season_week1)
        targets = DataFrame([{"season": 2024, "week": 2, "team": "A"}])

        result = _build_target_opponent_adjusted_epa(plays, targets)

        expected = _fit_opponent_adjusted_epa(
            DataFrame([play for play in current_season_week1 if play["pass"] == 1])
        )
        assert result.iloc[0]["off_pass_epa_adj"] == pytest.approx(expected["A"][0])

    def test_no_lookahead_to_target_week_or_later(self) -> None:
        week1 = _balanced_week(2024, 1, epa=0.1)
        week2 = _balanced_week(2024, 2, epa=0.2)
        # Week 3 (the target week itself) has wildly different data that
        # must not influence week 3's own as-of value.
        week3 = _balanced_week(2024, 3, epa=999.0)
        plays = DataFrame(week1 + week2 + week3)
        targets = DataFrame([{"season": 2024, "week": 3, "team": "A"}])

        result = _build_target_opponent_adjusted_epa(plays, targets)

        expected = _fit_opponent_adjusted_epa(
            DataFrame([play for play in (week1 + week2) if play["pass"] == 1])
        )
        assert result.iloc[0]["off_pass_epa_adj"] == pytest.approx(expected["A"][0])

    def test_excludes_playoffs_by_default(self) -> None:
        regular_season = _balanced_week(2024, 1, epa=0.1)
        playoff_week = _lopsided_week(2024, 19, favored_team="A", favored_epa=999.0)
        plays = DataFrame(regular_season + playoff_week)
        targets = DataFrame([{"season": 2024, "week": 20, "team": "A"}])

        excluded = _build_target_opponent_adjusted_epa(plays, targets, exclude_playoffs=True)
        included = _build_target_opponent_adjusted_epa(plays, targets, exclude_playoffs=False)

        expected_excluded = _fit_opponent_adjusted_epa(
            DataFrame([play for play in regular_season if play["pass"] == 1])
        )
        assert excluded.iloc[0]["off_pass_epa_adj"] == pytest.approx(expected_excluded["A"][0])
        assert included.iloc[0]["off_pass_epa_adj"] != pytest.approx(
            excluded.iloc[0]["off_pass_epa_adj"]
        )

    def test_team_with_no_season_to_date_plays_is_nan(self) -> None:
        plays = DataFrame(_balanced_week(2024, 1, teams=("A", "B", "C", "D")))
        targets = DataFrame([{"season": 2024, "week": 2, "team": "Z"}])

        result = _build_target_opponent_adjusted_epa(plays, targets)

        row = result.iloc[0]
        for metric in ADJUSTED_EPA_METRICS:
            assert pd.isna(row[metric])

    def test_duplicate_target_identity_is_computed_once(self) -> None:
        plays = DataFrame(_balanced_week(2024, 1) + _balanced_week(2024, 2))
        targets = DataFrame(
            [
                {"season": 2024, "week": 3, "team": "A"},
                {"season": 2024, "week": 3, "team": "A"},
            ]
        )

        result = _build_target_opponent_adjusted_epa(plays, targets)

        assert len(result) == 1

    def test_result_has_expected_columns(self) -> None:
        plays = DataFrame(_balanced_week(2024, 1))
        targets = DataFrame([{"season": 2024, "week": 1, "team": "A"}])

        result = _build_target_opponent_adjusted_epa(plays, targets)

        assert list(result.columns) == ["season", "week", "team", *ADJUSTED_EPA_METRICS]

    def test_missing_play_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="Play-level PBP is missing"):
            _build_target_opponent_adjusted_epa(
                DataFrame({"season": [2024], "week": [1]}),
                DataFrame({"season": [2024], "week": [1], "team": ["A"]}),
            )

    def test_missing_target_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="targets are missing"):
            _build_target_opponent_adjusted_epa(
                DataFrame(_balanced_week(2024, 1)),
                DataFrame({"season": [2024], "week": [1]}),
            )


def _datasets(plays: DataFrame) -> MagicMock:
    """Return a controlled PBP dataset accessor."""
    datasets = MagicMock(spec=DatasetAccessor)
    datasets.pbp.return_value = plays.copy()
    return datasets


def _game_frame(
    *,
    year: str = "2025-2026",
    week: int = 3,
    away: str = "Philadelphia Eagles",
    home: str = "Green Bay Packers",
) -> DataFrame:
    """Return one canonical game row with an extra column to test preservation."""
    return DataFrame(
        {
            "GAME_ID": ["2025_03_PHI_GB"],
            "YEAR": [year],
            "WEEK_NUM": [week],
            "AWAY_TEAM": [away],
            "HOME_TEAM": [home],
            "MARKER": ["preserved"],
        }
    )


def _short_code_play(
    season: int,
    week: int,
    posteam: str,
    defteam: str,
    epa: float,
    *,
    is_pass: bool,
) -> dict[str, object]:
    """A play using nflverse short team codes, as real PBP does."""
    return {
        "season": season,
        "week": week,
        "play_type": "pass" if is_pass else "run",
        "posteam": posteam,
        "defteam": defteam,
        "pass": 1 if is_pass else 0,
        "rush": 0 if is_pass else 1,
        "epa": epa,
    }


def _short_code_week(season: int, week: int) -> list[dict[str, object]]:
    """One asymmetric round-robin week across four real short team codes."""
    teams = ("PHI", "GB", "KC", "SF")
    rows: list[dict[str, object]] = []
    for offense, defense in permutations(teams, 2):
        epa = 0.3 if offense == "PHI" else 0.0
        rows.append(_short_code_play(season, week, offense, defense, epa, is_pass=True))
        rows.append(_short_code_play(season, week, offense, defense, epa, is_pass=False))
    return rows


class TestHomeAwayOpponentAdjustedEpaFeature:
    def test_registered_as_home_away_opponent_adjusted_epa(self) -> None:
        assert (
            FeatureRegistry.get("home_away_opponent_adjusted_epa")
            is HomeAwayOpponentAdjustedEpaFeature
        )

    def test_spec_produces_all_eight_columns(self) -> None:
        spec = HomeAwayOpponentAdjustedEpaFeature.spec
        assert set(spec.produces) == {
            "AWAY_OFF_PASS_EPA_ADJ",
            "HOME_OFF_PASS_EPA_ADJ",
            "AWAY_OFF_RUSH_EPA_ADJ",
            "HOME_OFF_RUSH_EPA_ADJ",
            "AWAY_DEF_PASS_EPA_ADJ",
            "HOME_DEF_PASS_EPA_ADJ",
            "AWAY_DEF_RUSH_EPA_ADJ",
            "HOME_DEF_RUSH_EPA_ADJ",
        }

    def test_maps_short_team_codes_to_long_names(self) -> None:
        """If the short-to-long mapping were broken, the merge would
        silently match nothing and every output would be NaN - this proves
        it actually joins real values, not just that it runs."""
        plays = DataFrame(_short_code_week(2025, 1) + _short_code_week(2025, 2))
        game = _game_frame(week=3)

        result = HomeAwayOpponentAdjustedEpaFeature().compute(df=game, datasets=_datasets(plays))

        assert pd.notna(result["AWAY_OFF_PASS_EPA_ADJ"].iloc[0])
        assert pd.notna(result["HOME_OFF_PASS_EPA_ADJ"].iloc[0])

    def test_preserves_input_and_unrelated_columns(self) -> None:
        plays = DataFrame(_short_code_week(2025, 1) + _short_code_week(2025, 2))
        game = _game_frame(week=3)

        result = HomeAwayOpponentAdjustedEpaFeature().compute(df=game, datasets=_datasets(plays))

        assert result["MARKER"].iloc[0] == "preserved"
        assert result["GAME_ID"].iloc[0] == "2025_03_PHI_GB"
        assert len(result) == 1

    def test_week_one_produces_nan(self) -> None:
        plays = DataFrame(_short_code_week(2025, 1))
        game = _game_frame(week=1)

        result = HomeAwayOpponentAdjustedEpaFeature().compute(df=game, datasets=_datasets(plays))

        assert pd.isna(result["AWAY_OFF_PASS_EPA_ADJ"].iloc[0])
        assert pd.isna(result["HOME_OFF_PASS_EPA_ADJ"].iloc[0])

    def test_empty_pbp_source_adds_null_columns(self) -> None:
        game = _game_frame(week=3)

        result = HomeAwayOpponentAdjustedEpaFeature().compute(
            df=game, datasets=_datasets(DataFrame())
        )

        for column in HomeAwayOpponentAdjustedEpaFeature.spec.produces:
            assert pd.isna(result[column].iloc[0])
        assert result["MARKER"].iloc[0] == "preserved"

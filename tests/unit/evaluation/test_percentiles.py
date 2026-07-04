# tests/unit/evaluation/test_percentiles.py

"""Unit tests for evaluation/percentiles.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

LONG_TO_SHORT = {
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "Buffalo Bills": "BUF",
    "Miami Dolphins": "MIA",
}


def _make_elo_state() -> pd.DataFrame:
    """Elo state with 4 teams, 2 weeks in 2026-2027."""
    return pd.DataFrame(
        [
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1600.0,
            },
            {
                "NFL_TEAM": "Kansas City Chiefs",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 2,
                "ELO": 1620.0,
            },
            {
                "NFL_TEAM": "Los Angeles Chargers",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 1,
                "ELO": 1500.0,
            },
            {
                "NFL_TEAM": "Los Angeles Chargers",
                "NFL_YEAR": "2026-2027",
                "NFL_WEEK": 2,
                "ELO": 1520.0,
            },
            {"NFL_TEAM": "Buffalo Bills", "NFL_YEAR": "2026-2027", "NFL_WEEK": 1, "ELO": 1580.0},
            {"NFL_TEAM": "Buffalo Bills", "NFL_YEAR": "2026-2027", "NFL_WEEK": 2, "ELO": 1590.0},
            {"NFL_TEAM": "Miami Dolphins", "NFL_YEAR": "2026-2027", "NFL_WEEK": 1, "ELO": 1450.0},
            {"NFL_TEAM": "Miami Dolphins", "NFL_YEAR": "2026-2027", "NFL_WEEK": 2, "ELO": 1440.0},
        ]
    )


def _make_projections() -> pd.DataFrame:
    """Projections with 4 teams."""
    return pd.DataFrame(
        [
            {"TEAM": "KC", "AVG_WINS": 12.5, "P_MAKE_PLAYOFFS": 0.90, "P_WIN_SB": 0.20},
            {"TEAM": "LAC", "AVG_WINS": 8.0, "P_MAKE_PLAYOFFS": 0.45, "P_WIN_SB": 0.05},
            {"TEAM": "BUF", "AVG_WINS": 11.0, "P_MAKE_PLAYOFFS": 0.75, "P_WIN_SB": 0.15},
            {"TEAM": "MIA", "AVG_WINS": 6.5, "P_MAKE_PLAYOFFS": 0.20, "P_WIN_SB": 0.02},
        ]
    )


class TestComputeTeamPercentiles:
    def test_empty_inputs_returns_empty_schema(self) -> None:
        from gridiron_edge.evaluation.percentiles import compute_team_percentiles

        result = compute_team_percentiles(
            pd.DataFrame(),
            pd.DataFrame(),
            LONG_TO_SHORT,
        )
        assert result.empty
        assert set(result.columns) == {
            "team_abbr",
            "season",
            "week",
            "rating_pct",
            "avg_wins_pct",
            "make_playoffs_pct",
            "win_sb_pct",
        }

    def test_ranks_by_current_week(self) -> None:
        """Percentiles use latest (season, week) — not aggregate history."""
        from gridiron_edge.evaluation.percentiles import compute_team_percentiles

        result = compute_team_percentiles(
            _make_elo_state(),
            _make_projections(),
            LONG_TO_SHORT,
        )
        assert not result.empty
        assert (result["season"] == "2026-2027").all()
        assert (result["week"] == 2).all()

    def test_percentile_formula_descending(self) -> None:
        """Higher raw value → higher percentile. Best team = (count-1)/count."""
        from gridiron_edge.evaluation.percentiles import compute_team_percentiles

        result = compute_team_percentiles(
            _make_elo_state(),
            _make_projections(),
            LONG_TO_SHORT,
        )
        by_team = {row["team_abbr"]: row for _, row in result.iterrows()}

        # Ratings: KC 1620 (1st), BUF 1590 (2nd), LAC 1520 (3rd), MIA 1440 (4th)
        # Formula: (count - rank) / count with count=4
        assert by_team["KC"]["rating_pct"] == pytest.approx(0.75)  # (4-1)/4
        assert by_team["BUF"]["rating_pct"] == pytest.approx(0.50)  # (4-2)/4
        assert by_team["LAC"]["rating_pct"] == pytest.approx(0.25)  # (4-3)/4
        assert by_team["MIA"]["rating_pct"] == pytest.approx(0.0)  # (4-4)/4

    def test_all_four_stats_computed(self) -> None:
        from gridiron_edge.evaluation.percentiles import compute_team_percentiles

        result = compute_team_percentiles(
            _make_elo_state(),
            _make_projections(),
            LONG_TO_SHORT,
        )
        by_team = {row["team_abbr"]: row for _, row in result.iterrows()}

        # KC leads on all 4 stats → all percentiles at top
        assert by_team["KC"]["rating_pct"] == pytest.approx(0.75)
        assert by_team["KC"]["avg_wins_pct"] == pytest.approx(0.75)
        assert by_team["KC"]["make_playoffs_pct"] == pytest.approx(0.75)
        assert by_team["KC"]["win_sb_pct"] == pytest.approx(0.75)

    def test_ties_get_same_rank(self) -> None:
        """Method="min" — tied teams share the lower rank."""
        from gridiron_edge.evaluation.percentiles import compute_team_percentiles

        # KC and BUF tied at 1600 Elo in Week 2
        elo = _make_elo_state()
        elo.loc[
            (elo["NFL_TEAM"] == "Buffalo Bills") & (elo["NFL_WEEK"] == 2),
            "ELO",
        ] = 1620.0  # Match KC

        result = compute_team_percentiles(elo, _make_projections(), LONG_TO_SHORT)
        by_team = {row["team_abbr"]: row for _, row in result.iterrows()}

        # Both KC and BUF at rank 1 (tied); LAC now rank 3, MIA rank 4
        assert by_team["KC"]["rating_pct"] == pytest.approx(0.75)  # (4-1)/4
        assert by_team["BUF"]["rating_pct"] == pytest.approx(0.75)  # tied at 1

    def test_nan_stat_gets_nan_percentile(self) -> None:
        """Team with NaN raw value gets NaN percentile."""
        from gridiron_edge.evaluation.percentiles import compute_team_percentiles

        proj = _make_projections()
        proj.loc[proj["TEAM"] == "MIA", "AVG_WINS"] = float("nan")

        result = compute_team_percentiles(_make_elo_state(), proj, LONG_TO_SHORT)
        by_team = {row["team_abbr"]: row for _, row in result.iterrows()}

        assert pd.isna(by_team["MIA"]["avg_wins_pct"])
        # Other stats still populated for MIA
        assert not pd.isna(by_team["MIA"]["rating_pct"])

    def test_team_in_elo_not_in_projections(self) -> None:
        """Team present in Elo but not projections keeps rating_pct populated."""
        from gridiron_edge.evaluation.percentiles import compute_team_percentiles

        proj = _make_projections()
        proj = proj[proj["TEAM"] != "MIA"]  # Drop MIA from projections

        result = compute_team_percentiles(_make_elo_state(), proj, LONG_TO_SHORT)
        by_team = {row["team_abbr"]: row for _, row in result.iterrows()}

        assert "MIA" in by_team
        assert not pd.isna(by_team["MIA"]["rating_pct"])
        # Missing projections stats → NaN
        assert pd.isna(by_team["MIA"]["avg_wins_pct"])

    def test_projections_only_returns_empty(self) -> None:
        """No Elo state → no season/week resolvable → empty result."""
        from gridiron_edge.evaluation.percentiles import compute_team_percentiles

        result = compute_team_percentiles(
            pd.DataFrame(),
            _make_projections(),
            LONG_TO_SHORT,
        )
        assert result.empty


class TestWriteTeamPercentiles:
    def test_writes_to_expected_path(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.percentiles import write_team_percentiles

        df = pd.DataFrame(
            [
                {
                    "team_abbr": "KC",
                    "season": "2026-2027",
                    "week": 2,
                    "rating_pct": 0.75,
                    "avg_wins_pct": 0.75,
                    "make_playoffs_pct": 0.75,
                    "win_sb_pct": 0.75,
                },
            ]
        )

        path = write_team_percentiles(df, season="2026-2027", week=2, repo=tmp_path)

        assert path.exists()
        assert path.name == "percentiles_2026-2027_wk02.parquet"
        assert path.parent == tmp_path / "data" / "output" / "rankings" / "percentiles"

    def test_overwrites_same_week(self, tmp_path: Path) -> None:
        """Repeat write to same (season, week) overwrites — natural dedup."""
        from gridiron_edge.evaluation.percentiles import write_team_percentiles

        df1 = pd.DataFrame(
            [
                {
                    "team_abbr": "KC",
                    "season": "2026-2027",
                    "week": 2,
                    "rating_pct": 0.5,
                    "avg_wins_pct": 0.5,
                    "make_playoffs_pct": 0.5,
                    "win_sb_pct": 0.5,
                }
            ]
        )
        df2 = pd.DataFrame(
            [
                {
                    "team_abbr": "KC",
                    "season": "2026-2027",
                    "week": 2,
                    "rating_pct": 0.75,
                    "avg_wins_pct": 0.75,
                    "make_playoffs_pct": 0.75,
                    "win_sb_pct": 0.75,
                }
            ]
        )

        write_team_percentiles(df1, season="2026-2027", week=2, repo=tmp_path)
        path2 = write_team_percentiles(df2, season="2026-2027", week=2, repo=tmp_path)

        loaded = pd.read_parquet(path2)
        assert loaded.iloc[0]["rating_pct"] == 0.75


class TestLoadLatestTeamPercentiles:
    def test_empty_when_dir_missing(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.percentiles import load_latest_team_percentiles

        result = load_latest_team_percentiles(tmp_path)
        assert result.empty

    def test_empty_when_no_files(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.percentiles import load_latest_team_percentiles

        (tmp_path / "data" / "output" / "rankings" / "percentiles").mkdir(parents=True)
        result = load_latest_team_percentiles(tmp_path)
        assert result.empty

    def test_loads_latest_file(self, tmp_path: Path) -> None:
        """Loads the file with the highest (season, week) filename."""
        from gridiron_edge.evaluation.percentiles import (
            load_latest_team_percentiles,
            write_team_percentiles,
        )

        for wk, rating in [(1, 0.10), (2, 0.20), (3, 0.30)]:
            df = pd.DataFrame(
                [
                    {
                        "team_abbr": "KC",
                        "season": "2026-2027",
                        "week": wk,
                        "rating_pct": rating,
                        "avg_wins_pct": rating,
                        "make_playoffs_pct": rating,
                        "win_sb_pct": rating,
                    }
                ]
            )
            write_team_percentiles(df, season="2026-2027", week=wk, repo=tmp_path)

        result = load_latest_team_percentiles(tmp_path)
        assert len(result) == 1
        assert result.iloc[0]["week"] == 3
        assert result.iloc[0]["rating_pct"] == 0.30

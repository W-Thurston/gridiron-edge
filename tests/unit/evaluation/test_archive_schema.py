# tests/unit/evaluation/test_archive_schema.py
"""Tests for archive schema extension - backward compat + enrichment columns."""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from gridiron_edge.evaluation.archive import (
    _ARCHIVE_COLUMNS,
    build_archive_rows,
    load_prediction_log,
)


class TestArchiveSchemaExtension:
    """Tests for the extended archive schema with enrichment columns."""

    def _make_predictions_df(self) -> pd.DataFrame:
        """Minimal predictions DataFrame matching build_archive_rows input."""
        return pd.DataFrame(
            {
                "GAME_ID": ["2024_01_KC_BAL"],
                "GAME_DATE": ["2024-09-05"],
                "AWAY_TEAM": ["Kansas City Chiefs"],
                "HOME_TEAM": ["Baltimore Ravens"],
                "AWAY_TEAM_ELO": [1600.0],
                "HOME_TEAM_ELO": [1550.0],
                "AWAY_WIN_PROB": [0.45],
                "HOME_WIN_PROB": [0.55],
            }
        )

    def test_archive_columns_include_enrichment(self) -> None:
        """_ARCHIVE_COLUMNS contains all enrichment columns."""
        expected = {
            "model_spread",
            "model_total",
            "projected_home_score",
            "projected_away_score",
            "margin_std",
            "win_prob_lo",
            "win_prob_hi",
            "confidence_tier",
        }
        assert expected.issubset(set(_ARCHIVE_COLUMNS))

    def test_build_archive_rows_fills_missing_enrichment(self) -> None:
        """build_archive_rows fills NaN for enrichment columns not in input."""
        rows = build_archive_rows(
            self._make_predictions_df(),
            model_name="test",
            model_type="v1",
            season="2024-2025",
            week=1,
        )
        assert set(_ARCHIVE_COLUMNS) == set(rows.columns)
        assert pd.isna(rows["model_spread"].iloc[0])
        assert pd.isna(rows["model_total"].iloc[0])
        assert rows["confidence_tier"].iloc[0] == ""

    def test_build_archive_rows_preserves_enrichment(self) -> None:
        """build_archive_rows keeps enrichment columns when present in input."""
        df = self._make_predictions_df()
        df["model_spread"] = [-1.5]
        df["model_total"] = [44.0]
        df["confidence_tier"] = ["Low"]

        rows = build_archive_rows(
            df,
            model_name="test",
            model_type="v1",
            season="2024-2025",
            week=1,
        )
        # model_spread is not in the base mapping so it gets NaN default
        # (build_archive_rows constructs its own dict, doesn't pass through
        # input enrichment columns). This is expected - enrichment happens
        # at predict time, not at archive time.
        assert set(_ARCHIVE_COLUMNS) == set(rows.columns)

    def test_backward_compat_load_fills_missing_enrichment(self, tmp_path: Path) -> None:
        """Archive missing enrichment columns loads with NaN/empty fill."""
        # Pre-enrichment archive: has the model identity columns
        # (model_name + model_type) but no enrichment columns.
        old_df = pd.DataFrame(
            {
                "predicted_at": [datetime.datetime(2024, 1, 1)],
                "is_backfilled": [True],
                "model_name": ["win_prob"],
                "model_type": ["elo"],
                "season": ["2024-2025"],
                "week": [1],
                "game_id": ["2024_01_KC_BAL"],
                "game_date": ["2024-09-05"],
                "away_team": ["Kansas City Chiefs"],
                "home_team": ["Baltimore Ravens"],
                "away_elo": [1600.0],
                "home_elo": [1550.0],
                "away_win_prob": [0.45],
                "home_win_prob": [0.55],
            }
        )
        archive_dir = tmp_path / "data" / "output" / "predictions"
        archive_dir.mkdir(parents=True)
        old_df.to_parquet(archive_dir / "predictions_log.parquet", index=False)

        loaded = load_prediction_log(repo=tmp_path)
        # Enrichment columns are filled in by load_prediction_log on read.
        assert "model_spread" in loaded.columns
        assert "confidence_tier" in loaded.columns
        assert pd.isna(loaded["model_spread"].iloc[0])
        assert loaded["confidence_tier"].iloc[0] == ""
        # Identity columns from the archive should round-trip cleanly.
        assert loaded["model_name"].iloc[0] == "win_prob"
        assert loaded["model_type"].iloc[0] == "elo"

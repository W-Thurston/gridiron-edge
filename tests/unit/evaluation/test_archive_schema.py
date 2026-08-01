# tests/unit/evaluation/test_archive_schema.py
"""Tests for the current prediction archive schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.evaluation.archive import (
    _ARCHIVE_COLUMNS,
    build_archive_rows,
    load_prediction_log,
)


class TestPredictionArchiveSchema:
    """Tests for current archive columns and strict loading."""

    def _make_predictions_df(self) -> pd.DataFrame:
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

    def test_build_archive_rows_uses_explicit_null_enrichment(self) -> None:
        rows = build_archive_rows(
            self._make_predictions_df(),
            model_name="test",
            model_type="v1",
            season="2024-2025",
            week=1,
        )
        assert list(rows.columns) == _ARCHIVE_COLUMNS
        assert pd.isna(rows["model_spread"].iloc[0])
        assert pd.isna(rows["model_total"].iloc[0])
        assert rows["confidence_tier"].iloc[0] == ""

    def test_build_archive_rows_preserves_enrichment(self) -> None:
        predictions = self._make_predictions_df()
        predictions["model_spread"] = [-1.5]
        predictions["model_total"] = [44.0]
        predictions["projected_home_score"] = [22.75]
        predictions["projected_away_score"] = [21.25]
        predictions["margin_std"] = [11.0]
        predictions["win_prob_lo"] = [0.35]
        predictions["win_prob_hi"] = [0.75]
        predictions["confidence_tier"] = ["Low"]

        rows = build_archive_rows(
            predictions,
            model_name="win_prob",
            model_type="random_forest",
            season="2024-2025",
            week=1,
        )

        assert list(rows.columns) == _ARCHIVE_COLUMNS
        assert rows["model_spread"].iloc[0] == -1.5
        assert rows["model_total"].iloc[0] == 44.0
        assert rows["projected_home_score"].iloc[0] == 22.75
        assert rows["projected_away_score"].iloc[0] == 21.25
        assert rows["margin_std"].iloc[0] == 11.0
        assert rows["win_prob_lo"].iloc[0] == 0.35
        assert rows["win_prob_hi"].iloc[0] == 0.75
        assert rows["confidence_tier"].iloc[0] == "Low"

    def test_load_rejects_missing_current_columns(self, tmp_path: Path) -> None:
        rows = build_archive_rows(
            self._make_predictions_df(),
            model_name="win_prob",
            model_type="elo",
            season="2024-2025",
            week=1,
        ).drop(columns=["margin_std"])

        archive_dir = tmp_path / "data" / "output" / "predictions"
        archive_dir.mkdir(parents=True)
        rows.to_parquet(archive_dir / "predictions_log.parquet", index=False)

        with pytest.raises(ValueError, match="missing columns"):
            load_prediction_log(repo=tmp_path)

    def test_load_rejects_unexpected_columns(self, tmp_path: Path) -> None:
        rows = build_archive_rows(
            self._make_predictions_df(),
            model_name="win_prob",
            model_type="elo",
            season="2024-2025",
            week=1,
        )
        rows["obsolete_field"] = "old"

        archive_dir = tmp_path / "data" / "output" / "predictions"
        archive_dir.mkdir(parents=True)
        rows.to_parquet(archive_dir / "predictions_log.parquet", index=False)

        with pytest.raises(ValueError, match="unexpected columns"):
            load_prediction_log(repo=tmp_path)

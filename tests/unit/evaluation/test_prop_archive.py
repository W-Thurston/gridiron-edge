# tests/unit/evaluation/test_prop_archive.py
"""Tests for gridiron_edge.evaluation.prop_archive."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.prop_archive import (
    _ARCHIVE_COLUMNS,
    _DEDUP_KEYS,
    archive_prop_predictions,
    load_prop_archive,
)


def _make_predictions(n: int = 5) -> DataFrame:
    """Build minimal predictions DataFrame."""
    return DataFrame(
        {
            "game_id": [f"2024_{i:02d}_LV_KC" for i in range(1, n + 1)],
            "player_id": ["QB1_KC"] * n,
            "player_name": ["P.Mahomes"] * n,
            "position": ["QB"] * n,
            "team": ["KC"] * n,
            "season": [2024] * n,
            "week": list(range(1, n + 1)),
            "stat_type": ["qb_pass_yards"] * n,
            "predicted_mean": [250.0 + i * 10 for i in range(n)],
            "predicted_std": [70.0] * n,
            "lo_90": [135.0] * n,
            "hi_90": [365.0] * n,
            "p_over": [0.55] * n,
            "lean": ["Over"] * n,
            "confidence_tier": ["Low"] * n,
            "line": [260.0] * n,
        }
    )


class TestArchiveColumns:
    """Verify schema constants."""

    def test_dedup_keys_in_schema(self) -> None:
        for key in _DEDUP_KEYS:
            assert key in _ARCHIVE_COLUMNS

    def test_schema_has_metadata(self) -> None:
        assert "predicted_at" in _ARCHIVE_COLUMNS
        assert "is_backfilled" in _ARCHIVE_COLUMNS
        assert "model_name" in _ARCHIVE_COLUMNS
        assert "model_type" in _ARCHIVE_COLUMNS
        assert "model_version" not in _ARCHIVE_COLUMNS

    def test_schema_has_enrichment(self) -> None:
        for col in [
            "predicted_mean",
            "predicted_std",
            "lo_90",
            "hi_90",
            "p_over",
            "lean",
            "confidence_tier",
            "line",
        ]:
            assert col in _ARCHIVE_COLUMNS


class TestArchivePropPredictions:
    """Verify archive write behavior."""

    def test_creates_file(self, tmp_path: Path) -> None:
        df: DataFrame = _make_predictions()
        path: Path = archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        df: DataFrame = _make_predictions()
        path: Path = archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )
        assert path.parent.exists()

    def test_row_count(self, tmp_path: Path) -> None:
        df: DataFrame = _make_predictions(n=3)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )
        loaded: DataFrame = pd.read_parquet(
            tmp_path / "data" / "output" / "props" / "prop_predictions_log.parquet"
        )
        assert len(loaded) == 3

    def test_adds_predicted_at(self, tmp_path: Path) -> None:
        df: DataFrame = _make_predictions(n=1)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )
        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert loaded["predicted_at"].notna().all()

    def test_adds_is_backfilled(self, tmp_path: Path) -> None:
        df: DataFrame = _make_predictions(n=1)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
            is_backfilled=True,
        )
        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert loaded["is_backfilled"].iloc[0] == True  # noqa: E712

    def test_adds_model_identity(self, tmp_path: Path) -> None:
        df: DataFrame = _make_predictions(n=1)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="random_forest",
        )
        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert loaded["model_name"].iloc[0] == "qb_pass_yards"
        assert loaded["model_type"].iloc[0] == "random_forest"

    def test_dedup_last_wins(self, tmp_path: Path) -> None:
        """Writing same player-game-stat-model twice → second value kept."""
        df1: DataFrame = _make_predictions(n=1)
        df1["predicted_mean"] = 200.0
        archive_prop_predictions(
            df1,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        df2: DataFrame = _make_predictions(n=1)
        df2["predicted_mean"] = 300.0
        archive_prop_predictions(
            df2,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert len(loaded) == 1
        assert loaded["predicted_mean"].iloc[0] == 300.0

    def test_append_different_keys(self, tmp_path: Path) -> None:
        """Different game_ids → both kept."""
        df1: DataFrame = _make_predictions(n=1)
        df1["game_id"] = "2024_01_LV_KC"
        archive_prop_predictions(
            df1,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        df2: DataFrame = _make_predictions(n=1)
        df2["game_id"] = "2024_02_LV_KC"
        archive_prop_predictions(
            df2,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert len(loaded) == 2

    def test_missing_optional_cols_filled_na(self, tmp_path: Path) -> None:
        """Columns not in df but in schema → filled with NA."""
        df = DataFrame(
            {
                "game_id": ["2024_01_LV_KC"],
                "player_id": ["QB1"],
                "stat_type": ["qb_pass_yards"],
                "predicted_mean": [250.0],
            }
        )
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )
        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert pd.isna(loaded["line"].iloc[0])

    def test_raises_without_required_cols(self, tmp_path: Path) -> None:
        df = DataFrame({"game_id": ["x"], "player_id": ["y"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            archive_prop_predictions(
                df,
                repo=tmp_path,
                model_name="qb_pass_yards",
                model_type="elasticnet",
            )

    def test_different_model_types_do_not_dedup(self, tmp_path: Path) -> None:
        """Same player-game-stat with different algorithms should both persist.

        This guards the Unit 5b migration from model_version to
        (model_name, model_type). Previously all algorithms collapsed into
        model_version='v1'.
        """
        df1: DataFrame = _make_predictions(n=1)
        df1["predicted_mean"] = 200.0
        archive_prop_predictions(
            df1,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        df2: DataFrame = _make_predictions(n=1)
        df2["predicted_mean"] = 300.0
        archive_prop_predictions(
            df2,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="random_forest",
        )

        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert len(loaded) == 2
        assert set(loaded["model_type"]) == {"elasticnet", "random_forest"}

    def test_different_model_names_do_not_dedup(self, tmp_path: Path) -> None:
        """Different prop families should both persist."""
        df1: DataFrame = _make_predictions(n=1)
        df1["stat_type"] = "qb_pass_yards"
        archive_prop_predictions(
            df1,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        df2: DataFrame = _make_predictions(n=1)
        df2["stat_type"] = "qb_rush_yards"
        df2["predicted_mean"] = 45.0
        archive_prop_predictions(
            df2,
            repo=tmp_path,
            model_name="qb_rush_yards",
            model_type="elasticnet",
        )

        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert len(loaded) == 2
        assert set(loaded["model_name"]) == {"qb_pass_yards", "qb_rush_yards"}


class TestLoadPropArchive:
    """Verify archive read behavior."""

    def test_empty_when_no_file(self, tmp_path: Path) -> None:
        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert len(loaded) == 0
        assert list(loaded.columns) == _ARCHIVE_COLUMNS

    def test_filter_by_stat_type(self, tmp_path: Path) -> None:
        df1: DataFrame = _make_predictions(n=2)
        df1["stat_type"] = "qb_pass_yards"
        df2: DataFrame = _make_predictions(n=2)
        df2["stat_type"] = "rb_rush_yards"
        df2["game_id"] = ["2024_10_LV_KC", "2024_11_LV_KC"]
        combined: DataFrame = pd.concat([df1, df2], ignore_index=True)
        archive_prop_predictions(
            combined,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        loaded: DataFrame = load_prop_archive(repo=tmp_path, stat_type="qb_pass_yards")
        assert len(loaded) == 2
        assert (loaded["stat_type"] == "qb_pass_yards").all()

    def test_filter_by_season(self, tmp_path: Path) -> None:
        df: DataFrame = _make_predictions(n=2)
        df["season"] = [2023, 2024]
        df["game_id"] = ["2023_01_LV_KC", "2024_01_LV_KC"]
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        loaded: DataFrame = load_prop_archive(repo=tmp_path, season=2024)
        assert len(loaded) == 1
        assert loaded["season"].iloc[0] == 2024

    def test_round_trip(self, tmp_path: Path) -> None:
        """Write → read preserves data."""
        df: DataFrame = _make_predictions(n=3)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )
        loaded: DataFrame = load_prop_archive(repo=tmp_path)
        assert len(loaded) == 3
        assert loaded["predicted_mean"].tolist() == df["predicted_mean"].tolist()


# ---------------------------------------------------------------------------
# Helpers for build_prop_evaluation_df tests
# ---------------------------------------------------------------------------


def _make_actuals(n: int = 3) -> DataFrame:
    """Synthetic actuals DataFrame matching the canonical join contract."""
    return DataFrame(
        {
            "game_id": [f"2024_{i:02d}_LV_KC" for i in range(1, n + 1)],
            "player_id": ["QB1_KC"] * n,
            "passing_yards": [275.0 + i * 5 for i in range(n)],
        }
    )


def _ensure_prop_models_registered() -> None:
    """Import prop trainers so ModelRegistry knows about them in tests."""
    import gridiron_edge.models.prop_prediction.qb_pass_yards  # noqa: F401


# ---------------------------------------------------------------------------
# build_prop_evaluation_df
# ---------------------------------------------------------------------------


class TestBuildPropEvaluationDf:
    """Canonical archive → actuals join surface."""

    def test_returns_empty_when_archive_missing(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.prop_archive import build_prop_evaluation_df

        _ensure_prop_models_registered()

        result = build_prop_evaluation_df(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            repo=tmp_path,
        )
        assert result.empty
        assert "actual" in result.columns

    def test_returns_empty_when_no_predictions_match_identity(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.prop_archive import (
            archive_prop_predictions,
            build_prop_evaluation_df,
        )

        _ensure_prop_models_registered()

        df = _make_predictions(n=1)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="random_forest",
        )

        result = build_prop_evaluation_df(
            model_name="qb_pass_yards",
            model_type="elasticnet",  # different algorithm
            repo=tmp_path,
            actuals_df=_make_actuals(),
        )
        assert result.empty

    def test_inner_join_against_provided_actuals(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.prop_archive import (
            archive_prop_predictions,
            build_prop_evaluation_df,
        )

        _ensure_prop_models_registered()

        # Archive 3 predictions
        df = _make_predictions(n=3)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        # Provide actuals for 2 of those game_ids
        actuals = _make_actuals(n=2)
        result = build_prop_evaluation_df(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            repo=tmp_path,
            actuals_df=actuals,
        )

        assert len(result) == 2
        assert "actual" in result.columns
        # Identity must round-trip.
        assert (result["model_name"] == "qb_pass_yards").all()
        assert (result["model_type"] == "elasticnet").all()

    def test_actual_column_normalized_regardless_of_stat(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.prop_archive import (
            archive_prop_predictions,
            build_prop_evaluation_df,
        )

        _ensure_prop_models_registered()

        df = _make_predictions(n=1)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        result = build_prop_evaluation_df(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            repo=tmp_path,
            actuals_df=_make_actuals(n=1),
        )
        assert "actual" in result.columns
        # The raw stat column should be gone after normalization.
        assert "passing_yards" not in result.columns

    def test_filters_strictly_by_model_identity(self, tmp_path: Path) -> None:
        """Predictions from a different algorithm must be ignored."""
        from gridiron_edge.evaluation.prop_archive import (
            archive_prop_predictions,
            build_prop_evaluation_df,
        )

        _ensure_prop_models_registered()

        df_en = _make_predictions(n=1)
        archive_prop_predictions(
            df_en,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        df_rf = _make_predictions(n=1)
        archive_prop_predictions(
            df_rf,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="random_forest",
        )

        out_en = build_prop_evaluation_df(
            model_name="qb_pass_yards",
            model_type="elasticnet",
            repo=tmp_path,
            actuals_df=_make_actuals(n=1),
        )
        if not out_en.empty:
            assert (out_en["model_type"] == "elasticnet").all()

        out_rf = build_prop_evaluation_df(
            model_name="qb_pass_yards",
            model_type="random_forest",
            repo=tmp_path,
            actuals_df=_make_actuals(n=1),
        )
        if not out_rf.empty:
            assert (out_rf["model_type"] == "random_forest").all()

    def test_raises_when_actuals_missing_target_col(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.prop_archive import (
            archive_prop_predictions,
            build_prop_evaluation_df,
        )

        _ensure_prop_models_registered()

        df = _make_predictions(n=1)
        archive_prop_predictions(
            df,
            repo=tmp_path,
            model_name="qb_pass_yards",
            model_type="elasticnet",
        )

        bad_actuals = DataFrame(
            {
                "game_id": ["2024_01_LV_KC"],
                "player_id": ["QB1_KC"],
                # missing passing_yards
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            build_prop_evaluation_df(
                model_name="qb_pass_yards",
                model_type="elasticnet",
                repo=tmp_path,
                actuals_df=bad_actuals,
            )

    def test_unknown_model_raises_key_error(self, tmp_path: Path) -> None:
        from gridiron_edge.evaluation.prop_archive import build_prop_evaluation_df

        _ensure_prop_models_registered()

        with pytest.raises(KeyError):
            build_prop_evaluation_df(
                model_name="not_a_real_model",
                model_type="elasticnet",
                repo=tmp_path,
                actuals_df=_make_actuals(),
            )

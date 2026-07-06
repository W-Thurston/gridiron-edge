# tests/unit/datasets/test_registry.py
"""Tests for gridiron_edge.datasets.registry."""

from __future__ import annotations

from pathlib import Path

from gridiron_edge.datasets.registry import DATASETS, DatasetSpec, dataset_path


class TestDatasetRegistry:
    def test_datasets_not_empty(self) -> None:
        assert len(DATASETS) > 0

    def test_has_19_keys(self) -> None:
        assert len(DATASETS) == 19

    def test_all_values_are_dataset_spec(self) -> None:
        for key, spec in DATASETS.items():
            assert isinstance(spec, DatasetSpec), f"{key}: expected DatasetSpec, got {type(spec)}"

    def test_all_relpaths_are_nonempty_strings(self) -> None:
        for key, spec in DATASETS.items():
            assert isinstance(spec.relpath, str), f"{key}: relpath is not str"
            assert len(spec.relpath) > 0, f"{key}: relpath is empty"

    def test_expected_keys_present(self) -> None:
        expected: set[str] = {
            # Raw ingest
            "games_raw_nflverse",
            "schedule_upcoming_raw_nflverse",
            # Cleaned datasets
            "games",
            "schedule_upcoming",
            "weather_enriched",
            "elo_state",
            "stadiums",
            "moneylines",
            "team_metadata",
            "epa_by_game",
            "player_game_logs",
            # Derived modeling artifacts
            "modeling_base",
            "modeling_full",
            # Archive logs
            "prediction_log",
            "prop_prediction_log",
            "bet_ledger",
            "bankroll_txn",
            # Output directories
            "predictions_csv",
            "elo_rankings_csv",
        }
        assert set(DATASETS.keys()) == expected

    def test_raw_datasets_are_parquet(self) -> None:
        for key in ("games_raw_nflverse", "schedule_upcoming_raw_nflverse"):
            assert DATASETS[key].relpath.endswith(".parquet"), f"{key} should be parquet"

    def test_modeling_datasets_are_parquet(self) -> None:
        for key in ("modeling_base", "modeling_full"):
            assert DATASETS[key].relpath.endswith(".parquet"), f"{key} should be parquet"


class TestDatasetPath:
    def test_returns_path(self, tmp_path: Path) -> None:
        result: Path = dataset_path(tmp_path, "games")
        assert isinstance(result, Path)

    def test_is_absolute(self, tmp_path: Path) -> None:
        result: Path = dataset_path(tmp_path, "games")
        assert result.is_absolute()

    def test_combines_root_and_relpath(self, tmp_path: Path) -> None:
        result: Path = dataset_path(tmp_path, "games")
        expected: Path = tmp_path / DATASETS["games"].relpath
        assert result == expected

    def test_all_keys_resolve(self, tmp_path: Path) -> None:
        for key in DATASETS:
            result: Path = dataset_path(tmp_path, key)
            assert result.is_absolute(), f"{key}: path not absolute"


class TestArchiveLogKeys:
    """Verify archive log dataset keys are correctly registered (Pattern 11)."""

    def test_prediction_log_is_parquet(self) -> None:
        assert DATASETS["prediction_log"].relpath.endswith(".parquet")

    def test_prop_prediction_log_is_parquet(self) -> None:
        assert DATASETS["prop_prediction_log"].relpath.endswith(".parquet")

    def test_bet_ledger_is_parquet(self) -> None:
        assert DATASETS["bet_ledger"].relpath.endswith(".parquet")

    def test_bankroll_txn_is_parquet(self) -> None:
        assert DATASETS["bankroll_txn"].relpath.endswith(".parquet")

    def test_prediction_log_path(self, tmp_path: Path) -> None:
        result = dataset_path(tmp_path, "prediction_log")
        assert result == tmp_path / "data" / "output" / "predictions" / "predictions_log.parquet"

    def test_prop_prediction_log_path(self, tmp_path: Path) -> None:
        result = dataset_path(tmp_path, "prop_prediction_log")
        assert result == tmp_path / "data" / "output" / "props" / "prop_predictions_log.parquet"

    def test_bet_ledger_path(self, tmp_path: Path) -> None:
        result = dataset_path(tmp_path, "bet_ledger")
        assert result == tmp_path / "data" / "betting" / "bet_ledger.parquet"

    def test_bankroll_txn_path(self, tmp_path: Path) -> None:
        result = dataset_path(tmp_path, "bankroll_txn")
        assert result == tmp_path / "data" / "betting" / "bankroll_txn.parquet"

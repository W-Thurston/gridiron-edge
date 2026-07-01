# tests/unit/api/test_loaders.py

"""Unit tests for api/loaders.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gridiron_edge.api.loaders import (
    load_bankroll_history_df,
    load_bankroll_txns_df,
    load_bets_df,
    load_current_bankroll,
    resolve_current_week,
)
from gridiron_edge.core.settings import Settings


def _make_settings(root: Path) -> Settings:
    return Settings(
        repo_root=root,
        owm_api_key=None,
        data_raw=root / "data" / "raw",
        data_cleaned=root / "data" / "cleaned",
        data_modeling=root / "data" / "modeling",
        data_output=root / "data" / "output",
    )


class TestLoadBetsDf:
    def test_passes_settings_repo_root(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with patch("gridiron_edge.betting.ledger.load_bets") as mock:
            mock.return_value = pd.DataFrame()
            load_bets_df(settings)
        mock.assert_called_once_with(status=None, repo=tmp_path)

    def test_passes_status_filter(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with patch("gridiron_edge.betting.ledger.load_bets") as mock:
            mock.return_value = pd.DataFrame()
            load_bets_df(settings, status="open")
        mock.assert_called_once_with(status="open", repo=tmp_path)


class TestLoadBankrollTxns:
    def test_passes_settings_repo_root(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with patch("gridiron_edge.betting.bankroll.load_transactions") as mock:
            mock.return_value = pd.DataFrame()
            load_bankroll_txns_df(settings)
        mock.assert_called_once_with(txn_type=None, repo=tmp_path)


class TestLoadBankrollHistory:
    def test_passes_settings_repo_root(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with patch("gridiron_edge.betting.bankroll.balance_history") as mock:
            mock.return_value = pd.DataFrame()
            load_bankroll_history_df(settings)
        mock.assert_called_once_with(repo=tmp_path)


class TestLoadCurrentBankroll:
    def test_passes_settings_repo_root(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with patch("gridiron_edge.betting.bankroll.current_balance") as mock:
            mock.return_value = 1234.56
            result = load_current_bankroll(settings)
        mock.assert_called_once_with(repo=tmp_path)
        assert result == 1234.56


class TestResolveCurrentWeek:
    def test_falls_back_when_schedule_missing(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with patch(
            "gridiron_edge.datasets.loaders.load_schedule_upcoming",
            side_effect=FileNotFoundError,
        ):
            season, week, source = resolve_current_week(settings)
        assert isinstance(season, int)
        assert week == 1
        assert source == "fallback"

    def test_falls_back_when_schedule_empty(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        with patch(
            "gridiron_edge.datasets.loaders.load_schedule_upcoming",
            return_value=pd.DataFrame(columns=["season", "week"]),
        ):
            _season, week, source = resolve_current_week(settings)
        assert week == 1
        assert source == "fallback"

    def test_reads_first_upcoming_week(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        schedule = pd.DataFrame(
            {
                "season": [2025, 2025, 2025],
                "week": [10, 11, 12],
            },
        )
        with patch(
            "gridiron_edge.datasets.loaders.load_schedule_upcoming",
            return_value=schedule,
        ):
            season, week, source = resolve_current_week(settings)
        assert (season, week, source) == (2025, 10, "schedule")

    def test_falls_back_when_columns_missing(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        schedule = pd.DataFrame({"unexpected_col": [1, 2, 3]})
        with patch(
            "gridiron_edge.datasets.loaders.load_schedule_upcoming",
            return_value=schedule,
        ):
            _season, week, source = resolve_current_week(settings)
        assert week == 1
        assert source == "fallback"

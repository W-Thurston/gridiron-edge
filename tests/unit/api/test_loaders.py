# tests/unit/api/test_loaders.py

"""Unit tests for api/loaders.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

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


class TestLoadGamesForWeek:
    """Cover load_games_for_week (W8 Tier 2 Step 5a)."""

    def _fake_settings(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass
        class FakeSettings:
            repo_root: Path

        return FakeSettings(repo_root=tmp_path)

    def _write_manifest(self, tmp_path: Path, model_type: str) -> None:
        import json

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "updated_at": "2026-07-01T14:00:00+00:00",
            "models": {
                "win_prob": {
                    "model_type": model_type,
                    "promoted_at": "2026-07-01T14:00:00",
                    "source_run_id": "RUN_X",
                    "metrics": {"brier": 0.213},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(manifest))

    def test_returns_champion_filtered_predictions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_games_for_week

        self._write_manifest(tmp_path, "random_forest")
        settings = self._fake_settings(tmp_path)

        # Two archived predictions, only one from the champion model.
        archive = pd.DataFrame(
            [
                {
                    "game_id": "2026_01_KC_LAC",
                    "model_name": "win_prob",
                    "model_type": "random_forest",
                    "season": "2026-2027",
                    "week": 1,
                    "game_date": "2026-09-05",
                    "away_team": "Kansas City Chiefs",
                    "home_team": "Los Angeles Chargers",
                    "home_win_prob": 0.55,
                    "away_win_prob": 0.45,
                    "model_spread": -2.0,
                    "model_total": 47.5,
                    "projected_home_score": 25.0,
                    "projected_away_score": 23.0,
                    "confidence_tier": "Moderate",
                    "win_prob_lo": 0.42,
                    "win_prob_hi": 0.68,
                },
            ]
        )

        games_df = pd.DataFrame(
            [
                {
                    "GAME_ID": "2026_01_KC_LAC",
                    "YEAR": "2026-2027",
                    "WEEK_NUM": 1,
                    "GAME_DATE": "2026-09-05",
                },
            ]
        )

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )
        monkeypatch.setattr(
            "gridiron_edge.api.loaders.load_games_df",
            lambda s: games_df,
        )
        monkeypatch.setattr(
            "gridiron_edge.api.loaders.load_team_name_map",
            lambda s: {
                "Kansas City Chiefs": "KC",
                "Los Angeles Chargers": "LAC",
            },
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: archive,
        )

        result = load_games_for_week(settings, season="2026-2027", week=1)

        assert len(result) == 1
        assert result.iloc[0]["game_id"] == "2026_01_KC_LAC"
        assert result.iloc[0]["away_team"] == "KC"
        assert result.iloc[0]["home_team"] == "LAC"
        assert result.iloc[0]["home_win_prob"] == 0.55

    def test_empty_archive_returns_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_games_for_week

        self._write_manifest(tmp_path, "random_forest")
        settings = self._fake_settings(tmp_path)

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame(),
        )

        result = load_games_for_week(settings, season="2026-2027", week=1)
        assert result.empty

    def test_raises_champion_not_found_when_no_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_games_for_week
        from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError

        # No manifest written.
        settings = self._fake_settings(tmp_path)

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        with pytest.raises(ChampionNotFoundError):
            load_games_for_week(settings, season="2026-2027", week=1)


class TestLoadGame:
    """Cover load_game (W8 Tier 2 Step 5a)."""

    def _fake_settings(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass
        class FakeSettings:
            repo_root: Path

        return FakeSettings(repo_root=tmp_path)

    def _write_manifest(self, tmp_path: Path, model_type: str) -> None:
        import json

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "champions.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-07-01T14:00:00+00:00",
                    "models": {
                        "win_prob": {
                            "model_type": model_type,
                            "promoted_at": "2026-07-01T14:00:00",
                            "source_run_id": "RUN_X",
                            "metrics": {"brier": 0.213},
                        },
                    },
                }
            )
        )

    def test_returns_dict_for_matching_game(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_game

        self._write_manifest(tmp_path, "random_forest")
        settings = self._fake_settings(tmp_path)

        archive = pd.DataFrame(
            [
                {
                    "game_id": "2026_01_KC_LAC",
                    "model_name": "win_prob",
                    "model_type": "random_forest",
                    "season": "2026-2027",
                    "week": 1,
                    "game_date": "2026-09-05",
                    "away_team": "Kansas City Chiefs",
                    "home_team": "Los Angeles Chargers",
                    "home_win_prob": 0.55,
                    "away_win_prob": 0.45,
                    "model_spread": -2.0,
                    "model_total": 47.5,
                    "projected_home_score": 25.0,
                    "projected_away_score": 23.0,
                    "confidence_tier": "Moderate",
                    "win_prob_lo": 0.42,
                    "win_prob_hi": 0.68,
                },
            ]
        )

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )
        monkeypatch.setattr(
            "gridiron_edge.api.loaders.load_games_df",
            lambda s: pd.DataFrame(
                [
                    {
                        "GAME_ID": "2026_01_KC_LAC",
                        "YEAR": "2026-2027",
                        "WEEK_NUM": 1,
                        "GAME_DATE": "2026-09-05",
                    }
                ]
            ),
        )
        monkeypatch.setattr(
            "gridiron_edge.api.loaders.load_team_name_map",
            lambda s: {
                "Kansas City Chiefs": "KC",
                "Los Angeles Chargers": "LAC",
            },
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: archive,
        )

        result = load_game(settings, game_id="2026_01_KC_LAC")

        assert result is not None
        assert result["game_id"] == "2026_01_KC_LAC"
        assert result["away_team"] == "KC"
        assert result["home_team"] == "LAC"

    def test_returns_none_for_unknown_game_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_game

        self._write_manifest(tmp_path, "random_forest")
        settings = self._fake_settings(tmp_path)

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame(),
        )

        result = load_game(settings, game_id="bogus_game_id")
        assert result is None


class TestLoadEdgesForWeek:
    """Cover load_edges_for_week (W8 Tier 2 Step 6a)."""

    def _fake_settings(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass
        class FakeSettings:
            repo_root: Path

        return FakeSettings(repo_root=tmp_path)

    def _write_manifest(self, tmp_path: Path, model_type: str) -> None:
        import json

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "champions.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-07-01T14:00:00+00:00",
                    "models": {
                        "win_prob": {
                            "model_type": model_type,
                            "promoted_at": "2026-07-01T14:00:00",
                            "source_run_id": "RUN_X",
                            "metrics": {"brier": 0.213},
                        },
                    },
                }
            )
        )

    def _make_prediction_row(self) -> dict:
        return {
            "predicted_at": pd.Timestamp("2026-09-01"),
            "is_backfilled": False,
            "model_name": "win_prob",
            "model_type": "random_forest",
            "season": "2026-2027",
            "week": 1,
            "game_id": "2026_01_KC_LAC",
            "game_date": "2026-09-05",
            "away_team": "Kansas City Chiefs",
            "home_team": "Los Angeles Chargers",
            "away_elo": 1550.0,
            "home_elo": 1520.0,
            "away_win_prob": 0.30,
            "home_win_prob": 0.70,
            "model_spread": -7.0,
            "model_total": 50.0,
            "projected_home_score": 28.0,
            "projected_away_score": 22.0,
            "margin_std": 13.54,
            "win_prob_lo": 0.55,
            "win_prob_hi": 0.85,
            "confidence_tier": "High",
        }

    def _make_odds_rows(self) -> pd.DataFrame:
        ts = pd.Timestamp("2026-09-05 12:00:00")
        return pd.DataFrame(
            [
                {
                    "fetched_at": ts,
                    "sportsbook": "draftkings",
                    "season": "2026-2027",
                    "week": 1,
                    "game_id": "2026_01_KC_LAC",
                    "game_date": "2026-09-05",
                    "away_team": "Kansas City Chiefs",
                    "home_team": "Los Angeles Chargers",
                    "market": "moneyline",
                    "side": "home",
                    "odds": -200.0,
                    "line": float("nan"),
                },
                {
                    "fetched_at": ts,
                    "sportsbook": "draftkings",
                    "season": "2026-2027",
                    "week": 1,
                    "game_id": "2026_01_KC_LAC",
                    "game_date": "2026-09-05",
                    "away_team": "Kansas City Chiefs",
                    "home_team": "Los Angeles Chargers",
                    "market": "moneyline",
                    "side": "away",
                    "odds": 170.0,
                    "line": float("nan"),
                },
            ]
        )

    def test_returns_ranked_edges_with_short_codes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_edges_for_week

        self._write_manifest(tmp_path, "random_forest")
        settings = self._fake_settings(tmp_path)

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame([self._make_prediction_row()]),
        )
        monkeypatch.setattr(
            "gridiron_edge.ingest.odds.store.load_current_odds",
            lambda **kwargs: self._make_odds_rows(),
        )
        monkeypatch.setattr(
            "gridiron_edge.models.game_prediction.post_process.get_margin_std",
            lambda *args, **kwargs: 13.54,
        )
        monkeypatch.setattr(
            "gridiron_edge.models.game_prediction.post_process.get_total_std",
            lambda *args, **kwargs: 13.0,
        )
        monkeypatch.setattr(
            "gridiron_edge.api.loaders.load_team_name_map",
            lambda _settings: {
                "Kansas City Chiefs": "KC",
                "Los Angeles Chargers": "LAC",
            },
        )

        result = load_edges_for_week(settings, season="2026-2027", week=1)

        assert not result.empty
        assert "away_team" in result.columns
        assert "home_team" in result.columns
        assert set(result["home_team"].unique()) <= {"KC", "LAC"}
        assert set(result["away_team"].unique()) <= {"KC", "LAC"}
        assert "ev" in result.columns
        # Sorted descending by EV.
        assert result["ev"].is_monotonic_decreasing

    def test_empty_predictions_returns_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_edges_for_week

        self._write_manifest(tmp_path, "random_forest")
        settings = self._fake_settings(tmp_path)

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame(),
        )

        result = load_edges_for_week(settings, season="2026-2027", week=1)
        assert result.empty

    def test_missing_odds_raises_odds_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.exceptions import OddsUnavailableError
        from gridiron_edge.api.loaders import load_edges_for_week

        self._write_manifest(tmp_path, "random_forest")
        settings = self._fake_settings(tmp_path)

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame([self._make_prediction_row()]),
        )
        monkeypatch.setattr(
            "gridiron_edge.ingest.odds.store.load_current_odds",
            lambda **kwargs: None,
        )

        with pytest.raises(OddsUnavailableError):
            load_edges_for_week(settings, season="2026-2027", week=1)

    def test_empty_odds_raises_odds_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.exceptions import OddsUnavailableError
        from gridiron_edge.api.loaders import load_edges_for_week

        self._write_manifest(tmp_path, "random_forest")
        settings = self._fake_settings(tmp_path)

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame([self._make_prediction_row()]),
        )
        monkeypatch.setattr(
            "gridiron_edge.ingest.odds.store.load_current_odds",
            lambda **kwargs: pd.DataFrame(),
        )

        with pytest.raises(OddsUnavailableError):
            load_edges_for_week(settings, season="2026-2027", week=1)

    def test_missing_manifest_raises_champion_not_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_edges_for_week
        from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError

        # No manifest written.
        settings = self._fake_settings(tmp_path)

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        with pytest.raises(ChampionNotFoundError):
            load_edges_for_week(settings, season="2026-2027", week=1)

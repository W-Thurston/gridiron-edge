# tests/unit/api/test_loaders.py

"""Unit tests for api/loaders.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.api.loaders import (
    load_bankroll_history_df,
    load_bankroll_txns_df,
    load_bets_df,
    load_current_bankroll,
    load_projection_grid_data,
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
    """Cover weekly game loading."""

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
    """Cover individual game loading."""

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
    """Cover the API loader boundary around the weekly edge service."""

    def _fake_settings(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass
        class FakeSettings:
            repo_root: Path

        return FakeSettings(repo_root=tmp_path)

    def test_forwards_scope_sizing_and_repository(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_edges_for_week
        from gridiron_edge.market.edge_diagnostics import (
            EdgeDiagnosticBlocker,
            EdgeDiagnostics,
            EdgeResultState,
        )
        from gridiron_edge.market.recommendations import EdgeResult

        diagnostics = EdgeDiagnostics(
            season="2026-2027",
            week=1,
            prediction_game_count=0,
            market_game_count=0,
            matched_game_count=0,
            complete_moneyline_count=0,
            complete_spread_count=0,
            complete_total_count=0,
            eligible_market_count=0,
            calculated_edge_count=0,
            positive_edge_count=0,
            filtered_edge_count=0,
            state=EdgeResultState.BLOCKED,
            blockers=(EdgeDiagnosticBlocker.NO_PREDICTIONS,),
        )
        expected = EdgeResult(rows=pd.DataFrame(), diagnostics=diagnostics)
        calls: list[dict[str, object]] = []

        def fake_service(**kwargs):
            calls.append(kwargs)
            return expected

        monkeypatch.setattr(
            "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
            fake_service,
        )
        settings = self._fake_settings(tmp_path)
        result = load_edges_for_week(
            settings,
            season="2026-2027",
            week=1,
            min_ev=0.03,
            bankroll=2500.0,
            kelly_multiplier=0.10,
        )

        assert result is expected
        assert calls == [
            {
                "season": "2026-2027",
                "week": 1,
                "min_ev": 0.03,
                "bankroll": 2500.0,
                "kelly_multiplier": 0.10,
                "repo": tmp_path,
            }
        ]

    def test_normalizes_team_names_on_a_defensive_copy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_edges_for_week
        from gridiron_edge.market.edge_diagnostics import (
            EdgeDiagnostics,
            EdgeResultState,
        )
        from gridiron_edge.market.recommendations import EdgeResult

        rows = pd.DataFrame(
            [
                {
                    "away_team": "Kansas City Chiefs",
                    "home_team": "Los Angeles Chargers",
                    "ev": 0.08,
                }
            ]
        )
        diagnostics = EdgeDiagnostics(
            season="2026-2027",
            week=1,
            prediction_game_count=1,
            market_game_count=1,
            matched_game_count=1,
            complete_moneyline_count=1,
            complete_spread_count=0,
            complete_total_count=0,
            eligible_market_count=1,
            calculated_edge_count=1,
            positive_edge_count=1,
            filtered_edge_count=1,
            state=EdgeResultState.POSITIVE_EDGES,
        )
        service_result = EdgeResult(rows=rows, diagnostics=diagnostics)
        monkeypatch.setattr(
            "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
            lambda **kwargs: service_result,
        )
        monkeypatch.setattr(
            "gridiron_edge.api.loaders.load_team_name_map",
            lambda _settings: {
                "Kansas City Chiefs": "KC",
                "Los Angeles Chargers": "LAC",
            },
        )

        result = load_edges_for_week(
            self._fake_settings(tmp_path),
            season="2026-2027",
            week=1,
        )

        assert result.diagnostics is diagnostics
        assert result.rows.loc[0, "away_team"] == "KC"
        assert result.rows.loc[0, "home_team"] == "LAC"
        assert service_result.rows.loc[0, "away_team"] == "Kansas City Chiefs"
        assert service_result.rows.loc[0, "home_team"] == "Los Angeles Chargers"


class TestParseSeasonInt:
    """Cover season-label parsing."""

    def test_hyphenated_returns_leading_year(self) -> None:
        from gridiron_edge.api.loaders import _parse_season_int

        assert _parse_season_int("2026-2027") == 2026

    def test_single_year_returns_int(self) -> None:
        from gridiron_edge.api.loaders import _parse_season_int

        assert _parse_season_int("2026") == 2026

    def test_malformed_raises(self) -> None:
        from gridiron_edge.api.loaders import _parse_season_int

        with pytest.raises(ValueError, match="Cannot parse season"):
            _parse_season_int("not a season")


class TestLoadPropsForWeek:
    """Cover weekly prop loading."""

    def _fake_settings(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass
        class FakeSettings:
            repo_root: Path

        return FakeSettings(repo_root=tmp_path)

    def _write_manifest(
        self,
        tmp_path: Path,
        families: dict[str, str],
    ) -> None:
        """Write a manifest with the given family → model_type entries."""
        import json

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "champions.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-07-01T14:00:00+00:00",
                    "models": {
                        family: {
                            "model_type": model_type,
                            "promoted_at": "2026-07-01T14:00:00",
                            "source_run_id": "RUN_X",
                            "metrics": {"mae": 63.0},
                        }
                        for family, model_type in families.items()
                    },
                }
            )
        )

    def _make_archive_row(
        self,
        *,
        game_id: str = "2026_01_KC_LAC",
        player_id: str = "00-0033873",
        stat_type: str = "qb_pass_yards",
        model_type: str = "elasticnet",
        week: int = 1,
        season: int = 2026,
    ) -> dict:
        return {
            "predicted_at": "2026-08-01T00:00:00+00:00",
            "is_backfilled": True,
            "season": season,
            "week": week,
            "game_id": game_id,
            "player_id": player_id,
            "player_name": "P.Mahomes",
            "position": "QB",
            "team": "KC",
            "stat_type": stat_type,
            "model_name": stat_type,
            "model_type": model_type,
            "predicted_mean": 265.0,
            "predicted_std": 45.0,
            "lo_90": 190.0,
            "hi_90": 340.0,
            "line": None,
            "p_over": float("nan"),
            "lean": float("nan"),
            "confidence_tier": float("nan"),
        }

    def _write_archive(self, tmp_path: Path, rows: list[dict]) -> None:
        archive_dir = tmp_path / "data" / "output" / "props"
        archive_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(
            archive_dir / "prop_predictions_log.parquet",
            index=False,
        )

    def test_returns_champion_rows_for_one_family(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_props_for_week

        self._write_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        self._write_archive(tmp_path, [self._make_archive_row()])

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        result = load_props_for_week(
            settings,
            season="2026-2027",
            week=1,
        )

        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]["stat_type"] == "qb_pass_yards"
        assert result.iloc[0]["model_type"] == "elasticnet"

    def test_stat_type_filter_processes_only_one_family(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_props_for_week

        self._write_manifest(
            tmp_path,
            {
                "qb_pass_yards": "elasticnet",
                "rb_rush_yards": "random_forest",
            },
        )
        self._write_archive(
            tmp_path,
            [
                self._make_archive_row(stat_type="qb_pass_yards"),
                self._make_archive_row(
                    game_id="2026_01_BUF_MIA",
                    player_id="00-0035700",
                    stat_type="rb_rush_yards",
                    model_type="random_forest",
                ),
            ],
        )

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        result = load_props_for_week(
            settings,
            season="2026-2027",
            week=1,
            stat_type="qb_pass_yards",
        )

        assert len(result) == 1
        assert result.iloc[0]["stat_type"] == "qb_pass_yards"

    def test_position_filter_applied(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_props_for_week

        self._write_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        rows = [
            self._make_archive_row(player_id="qb1"),
            {**self._make_archive_row(player_id="wr1"), "position": "WR"},
        ]
        self._write_archive(tmp_path, rows)

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        result = load_props_for_week(
            settings,
            season="2026-2027",
            week=1,
            position="QB",
        )

        assert len(result) == 1
        assert result.iloc[0]["position"] == "QB"

    def test_missing_families_silently_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Manifest only has qb_pass_yards; others missing.
        from gridiron_edge.api.loaders import load_props_for_week

        self._write_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        self._write_archive(tmp_path, [self._make_archive_row()])

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        # Call without stat_type filter — should iterate all 5 families,
        # find champion for only qb_pass_yards, return its row.
        result = load_props_for_week(
            settings,
            season="2026-2027",
            week=1,
        )
        assert len(result) == 1
        assert result.iloc[0]["stat_type"] == "qb_pass_yards"

    def test_no_families_resolved_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No manifest written at all.
        from gridiron_edge.api.loaders import load_props_for_week
        from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        with pytest.raises(ChampionNotFoundError, match="No prop champions"):
            load_props_for_week(settings, season="2026-2027", week=1)

    def test_family_resolved_but_empty_archive_returns_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Manifest present, archive empty — legitimate "no data yet" state.
        from gridiron_edge.api.loaders import load_props_for_week

        self._write_manifest(tmp_path, {"qb_pass_yards": "elasticnet"})
        # No archive file at all.

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        result = load_props_for_week(
            settings,
            season="2026-2027",
            week=1,
        )
        # Empty, but no exception — legitimate state.
        assert result.empty


class TestLoadProp:
    """Cover individual prop loading."""

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
                        "qb_pass_yards": {
                            "model_type": model_type,
                            "promoted_at": "2026-07-01T14:00:00",
                            "source_run_id": "RUN_X",
                            "metrics": {"mae": 63.0},
                        },
                    },
                }
            )
        )

    def _write_archive_row(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "data" / "output" / "props"
        archive_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "predicted_at": "2026-08-01T00:00:00+00:00",
                    "is_backfilled": True,
                    "season": 2026,
                    "week": 1,
                    "game_id": "2026_01_KC_LAC",
                    "player_id": "00-0033873",
                    "player_name": "P.Mahomes",
                    "position": "QB",
                    "team": "KC",
                    "stat_type": "qb_pass_yards",
                    "model_name": "qb_pass_yards",
                    "model_type": "elasticnet",
                    "predicted_mean": 265.0,
                    "predicted_std": 45.0,
                    "lo_90": 190.0,
                    "hi_90": 340.0,
                    "line": None,
                    "p_over": float("nan"),
                    "lean": float("nan"),
                    "confidence_tier": float("nan"),
                }
            ]
        ).to_parquet(
            archive_dir / "prop_predictions_log.parquet",
            index=False,
        )

    def test_returns_dict_for_matching_prop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_prop

        self._write_manifest(tmp_path, "elasticnet")
        self._write_archive_row(tmp_path)

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        result = load_prop(
            settings,
            game_id="2026_01_KC_LAC",
            player_id="00-0033873",
            stat_type="qb_pass_yards",
        )

        assert result is not None
        assert result["stat_type"] == "qb_pass_yards"
        assert result["player_name"] == "P.Mahomes"
        assert result["predicted_mean"] == 265.0

    def test_returns_none_for_unknown_composite(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_prop

        self._write_manifest(tmp_path, "elasticnet")
        self._write_archive_row(tmp_path)

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        result = load_prop(
            settings,
            game_id="2026_01_BOGUS",
            player_id="00-0033873",
            stat_type="qb_pass_yards",
        )
        assert result is None

    def test_missing_manifest_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.api.loaders import load_prop
        from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError

        settings = self._fake_settings(tmp_path)
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            lambda: settings,
        )

        with pytest.raises(ChampionNotFoundError):
            load_prop(
                settings,
                game_id="2026_01_KC_LAC",
                player_id="00-0033873",
                stat_type="qb_pass_yards",
            )


class TestComputeEloDeltas:
    """Cover current-versus-prior-week Elo delta computation."""

    def _long_to_short(self) -> dict[str, str]:
        return {
            "Kansas City Chiefs": "KAN",
            "Los Angeles Chargers": "LAC",
            "Seattle Seahawks": "SEA",
        }

    def test_empty_elo_state_returns_empty(self) -> None:
        from gridiron_edge.api.loaders import compute_elo_deltas

        result: DataFrame = compute_elo_deltas(pd.DataFrame(), self._long_to_short())
        assert result.empty

    def test_computes_delta_for_latest_week_with_short_codes(self) -> None:
        from gridiron_edge.api.loaders import compute_elo_deltas

        elo = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1580.0,
                },
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 2,
                    "ELO": 1595.0,
                },
                {
                    "NFL_TEAM": "Los Angeles Chargers",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1520.0,
                },
                {
                    "NFL_TEAM": "Los Angeles Chargers",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 2,
                    "ELO": 1512.0,
                },
            ]
        )

        result: DataFrame = compute_elo_deltas(elo, self._long_to_short())

        assert len(result) == 2
        by_team: dict[Any, Any] = dict(zip(result["team_abbr"], result["elo_delta"], strict=False))
        assert by_team["KAN"] == 15.0
        assert by_team["LAC"] == -8.0

    def test_week_1_returns_null_deltas(self) -> None:
        from gridiron_edge.api.loaders import compute_elo_deltas

        elo = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1580.0,
                },
                {
                    "NFL_TEAM": "Los Angeles Chargers",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1520.0,
                },
            ]
        )

        result: DataFrame = compute_elo_deltas(elo, self._long_to_short())

        assert len(result) == 2
        assert result["elo_delta"].isnull().all()
        assert set(result["team_abbr"]) == {"KAN", "LAC"}

    def test_uses_latest_season(self) -> None:
        """Delta computed for latest NFL_YEAR only; prior seasons ignored."""
        from gridiron_edge.api.loaders import compute_elo_deltas

        elo = pd.DataFrame(
            [
                # Prior season — should be ignored for delta computation.
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2025-2026",
                    "NFL_WEEK": 22,
                    "ELO": 1600.0,
                },
                # Current season.
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1580.0,
                },
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 2,
                    "ELO": 1595.0,
                },
            ]
        )

        result: DataFrame = compute_elo_deltas(elo, self._long_to_short())

        assert len(result) == 1
        assert result.iloc[0]["team_abbr"] == "KAN"
        assert result.iloc[0]["elo_delta"] == 15.0  # 1595 - 1580, not 1595 - 1600

    def test_team_missing_from_prior_week_gets_null(self) -> None:
        """New team that wasn't in prior week (e.g., expansion, relocation)."""
        from gridiron_edge.api.loaders import compute_elo_deltas

        elo = pd.DataFrame(
            [
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 1,
                    "ELO": 1580.0,
                },
                {
                    "NFL_TEAM": "Kansas City Chiefs",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 2,
                    "ELO": 1595.0,
                },
                # LAC only appears in current week.
                {
                    "NFL_TEAM": "Los Angeles Chargers",
                    "NFL_YEAR": "2026-2027",
                    "NFL_WEEK": 2,
                    "ELO": 1512.0,
                },
            ]
        )

        result: DataFrame = compute_elo_deltas(elo, self._long_to_short())

        by_team: dict[Any, Any] = dict(zip(result["team_abbr"], result["elo_delta"], strict=False))
        assert by_team["KAN"] == 15.0
        assert pd.isna(by_team["LAC"])

    def test_unmapped_team_falls_back_to_long_name(self) -> None:
        """A team not in long_to_short map falls back to its long name."""
        from gridiron_edge.api.loaders import compute_elo_deltas

        elo = pd.DataFrame(
            [
                {"NFL_TEAM": "Mystery Team", "NFL_YEAR": "2026-2027", "NFL_WEEK": 1, "ELO": 1500.0},
                {"NFL_TEAM": "Mystery Team", "NFL_YEAR": "2026-2027", "NFL_WEEK": 2, "ELO": 1510.0},
            ]
        )

        result: DataFrame = compute_elo_deltas(elo, self._long_to_short())

        assert len(result) == 1
        assert result.iloc[0]["team_abbr"] == "Mystery Team"
        assert result.iloc[0]["elo_delta"] == 10.0


class TestLoadProjectionGridData:
    def _schedule(self) -> DataFrame:
        return pd.DataFrame(
            [
                {
                    "WEEK_NUM": 1,
                    "GAME_DAY_OF_WEEK": "Sunday",
                    "GAME_DATE": "2026-09-13",
                    "AWAY_TEAM": "Buffalo Bills",
                    "HOME_TEAM": "Seattle Seahawks",
                    "GAMETIME": "13:00:00",
                    "YEAR": "2026-2027",
                    "GAME_ID": "2026_01_BUF_SEA",
                },
                {
                    "WEEK_NUM": 19,
                    "GAME_DAY_OF_WEEK": "Sunday",
                    "GAME_DATE": "2027-01-17",
                    "AWAY_TEAM": "Buffalo Bills",
                    "HOME_TEAM": "Seattle Seahawks",
                    "GAMETIME": "13:00:00",
                    "YEAR": "2026-2027",
                    "GAME_ID": "2026_19_BUF_SEA",
                },
            ]
        )

    def _games(self) -> DataFrame:
        return pd.DataFrame(
            [
                {
                    "GAME_ID": "2025_18_BUF_SEA",
                    "YEAR": "2025-2026",
                    "WEEK_NUM": 18,
                    "WINNER": "Seattle Seahawks",
                    "LOSER": "Buffalo Bills",
                    "WIN_OR_TIE": 1.0,
                },
                {
                    "GAME_ID": "2026_01_BUF_SEA",
                    "YEAR": "2026-2027",
                    "WEEK_NUM": 1,
                    "WINNER": "Seattle Seahawks",
                    "LOSER": "Buffalo Bills",
                    "WIN_OR_TIE": 1.0,
                },
                {
                    "GAME_ID": "2026_19_BUF_SEA",
                    "YEAR": "2026-2027",
                    "WEEK_NUM": 19,
                    "WINNER": "Seattle Seahawks",
                    "LOSER": "Buffalo Bills",
                    "WIN_OR_TIE": 1.0,
                },
            ]
        )

    def _mapping(self) -> DataFrame:
        return pd.DataFrame(
            [
                {
                    "NFL_LONG_NAME": "Seattle Seahawks",
                    "NFL_SHORT_NAME": "SEA",
                },
                {
                    "NFL_LONG_NAME": "Buffalo Bills",
                    "NFL_SHORT_NAME": "BUF",
                },
            ]
        )

    def _write_probabilities(
        self,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "data" / "output" / "temp"
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "TEAM": "SEA",
                    "W01_WIN_P": 0.64,
                    "W02_WIN_P": 0.0,
                },
                {
                    "TEAM": "BUF",
                    "W01_WIN_P": 0.36,
                    "W02_WIN_P": 0.58,
                },
            ]
        ).to_csv(
            output_dir / "season_grid.csv",
            index=False,
        )

    def test_loads_preseason_sources(
        self,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        self._write_probabilities(tmp_path)

        with (
            patch(
                "gridiron_edge.datasets.loaders.load_schedule_upcoming",
                return_value=self._schedule(),
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_games",
                return_value=pd.DataFrame(),
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_teams_long_short",
                return_value=self._mapping(),
            ),
        ):
            result = load_projection_grid_data(settings)

        assert result.season == "2026-2027"
        assert result.completed_through_week == 0
        assert result.schedule_available is True
        assert len(result.probabilities) == 2

        # Week 19 is outside the regular-season grid.
        assert result.schedule["WEEK_NUM"].tolist() == [1]

        assert result.games.empty
        assert result.long_to_short == {
            "Seattle Seahawks": "SEA",
            "Buffalo Bills": "BUF",
        }

    def test_filters_games_to_grid_season_and_regular_season(
        self,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        self._write_probabilities(tmp_path)

        with (
            patch(
                "gridiron_edge.datasets.loaders.load_schedule_upcoming",
                return_value=self._schedule(),
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_games",
                return_value=self._games(),
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_teams_long_short",
                return_value=self._mapping(),
            ),
        ):
            result = load_projection_grid_data(settings)

        assert result.completed_through_week == 1
        assert result.games["YEAR"].unique().tolist() == ["2026-2027"]
        assert result.games["WEEK_NUM"].tolist() == [1]

    def test_missing_probability_artifact_returns_empty_frame(
        self,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)

        with (
            patch(
                "gridiron_edge.datasets.loaders.load_schedule_upcoming",
                return_value=self._schedule(),
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_games",
                return_value=pd.DataFrame(),
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_teams_long_short",
                return_value=self._mapping(),
            ),
        ):
            result = load_projection_grid_data(settings)

        assert result.probabilities.empty
        assert result.schedule_available is True
        assert result.season == "2026-2027"

    def test_missing_schedule_is_distinct_from_bye_source(
        self,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        self._write_probabilities(tmp_path)

        games = pd.DataFrame(
            [
                {
                    "GAME_ID": "2025_22_SEA_BUF",
                    "YEAR": "2025-2026",
                    "WEEK_NUM": 22,
                    "WINNER": "Seattle Seahawks",
                    "LOSER": "Buffalo Bills",
                    "WIN_OR_TIE": 1.0,
                },
            ]
        )

        with (
            patch(
                "gridiron_edge.datasets.loaders.load_schedule_upcoming",
                side_effect=FileNotFoundError,
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_games",
                return_value=games,
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_teams_long_short",
                return_value=self._mapping(),
            ),
        ):
            result = load_projection_grid_data(settings)

        assert result.schedule.empty
        assert result.schedule_available is False
        assert result.season == "2025-2026"
        assert result.completed_through_week == 0

    def test_missing_games_still_loads_projected_sources(
        self,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        self._write_probabilities(tmp_path)

        with (
            patch(
                "gridiron_edge.datasets.loaders.load_schedule_upcoming",
                return_value=self._schedule(),
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_games",
                side_effect=FileNotFoundError,
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_teams_long_short",
                return_value=self._mapping(),
            ),
        ):
            result = load_projection_grid_data(settings)

        assert result.season == "2026-2027"
        assert result.games.empty
        assert result.completed_through_week == 0
        assert result.schedule_available is True

    def test_selects_latest_schedule_season(
        self,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        self._write_probabilities(tmp_path)

        schedule = pd.concat(
            [
                self._schedule(),
                pd.DataFrame(
                    [
                        {
                            "WEEK_NUM": 1,
                            "GAME_DAY_OF_WEEK": "Sunday",
                            "GAME_DATE": "2025-09-07",
                            "AWAY_TEAM": "Seattle Seahawks",
                            "HOME_TEAM": "Buffalo Bills",
                            "GAMETIME": "13:00:00",
                            "YEAR": "2025-2026",
                            "GAME_ID": "2025_01_SEA_BUF",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

        with (
            patch(
                "gridiron_edge.datasets.loaders.load_schedule_upcoming",
                return_value=schedule,
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_games",
                return_value=pd.DataFrame(),
            ),
            patch(
                "gridiron_edge.datasets.loaders.load_teams_long_short",
                return_value=self._mapping(),
            ),
        ):
            result = load_projection_grid_data(settings)

        assert result.season == "2026-2027"
        assert result.schedule["YEAR"].unique().tolist() == ["2026-2027"]

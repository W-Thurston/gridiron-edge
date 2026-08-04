# tests/unit/cli/test_edges.py

"""Tests for edges CLI model resolution retained by historical CLV."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner


class TestClvModelTypeResolution:
    """Cover --model-type auto sentinel on `gridiron edges clv`."""

    def _fake_settings(self, tmp_path: Path):
        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def _stub_data_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "gridiron_edge.evaluation.archive.load_prediction_log",
            lambda **kwargs: pd.DataFrame(),
        )

    def test_explicit_model_type_passes_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(
            edges_app,
            [
                "clv",
                "--model-type",
                "xgboost",
            ],
        )

        assert "model=xgboost" in result.output
        assert "win_prob/xgboost" in result.output

    def test_auto_resolves_from_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "updated_at": "2026-07-01T14:00:00+00:00",
            "models": {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:00:00",
                    "source_run_id": "RUN_X",
                    "metrics": {"brier": 0.213},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(manifest))

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )
        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(edges_app, ["clv"])

        assert "model=random_forest" in result.output
        assert "win_prob/random_forest" in result.output

    def test_auto_fails_when_manifest_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion_resolver.get_settings",
            self._fake_settings(tmp_path),
        )
        self._stub_data_load(monkeypatch)

        runner = CliRunner()
        result = runner.invoke(edges_app, ["clv"])

        assert result.exit_code != 0
        assert "requires a champion manifest" in result.output


class TestWeeklyReport:
    """Cover standalone report semantics over the shared edge service."""

    @staticmethod
    def _result(*, state: str, blockers: tuple[str, ...] = (), rows: int = 0):
        from gridiron_edge.market.edge_diagnostics import (
            EdgeDiagnosticBlocker,
            EdgeDiagnostics,
            EdgeResultState,
        )
        from gridiron_edge.market.recommendations import EdgeResult

        calculated = 1 if state == EdgeResultState.NO_POSITIVE_EDGES.value else rows
        positive = rows
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
            calculated_edge_count=calculated,
            positive_edge_count=positive,
            filtered_edge_count=rows,
            state=EdgeResultState(state),
            blockers=tuple(EdgeDiagnosticBlocker(value) for value in blockers),
        )
        frame = pd.DataFrame()
        if rows:
            frame = pd.DataFrame(
                {
                    "game_id": ["2026_01_KC_LAC"],
                    "away_team": ["Kansas City Chiefs"],
                    "home_team": ["Los Angeles Chargers"],
                    "market_type": ["moneyline"],
                    "side": ["away"],
                    "ev": [0.05],
                    "edge_strength": ["moderate"],
                    "kelly_stake": [pd.NA],
                    "confidence_tier": ["moderate"],
                }
            )
        return EdgeResult(rows=frame, diagnostics=diagnostics)

    def test_positive_rows_render_table(self) -> None:
        from gridiron_edge.cli.edges import edges_app

        result_value = self._result(state="positive_edges", rows=1)
        with (
            patch(
                "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
                return_value=result_value,
            ) as service,
            patch("gridiron_edge.cli.edges._render_edge_table") as render,
            patch("gridiron_edge.cli.edges._remove_edge_csv") as remove,
        ):
            result = CliRunner().invoke(
                edges_app,
                ["report", "--season", "2026-2027", "--week", "1"],
            )

        assert result.exit_code == 0, result.output
        service.assert_called_once_with(
            season="2026-2027",
            week=1,
            bankroll=None,
            kelly_multiplier=0.25,
            min_ev=0.0,
        )
        remove.assert_called_once_with("2026-2027", 1)
        render.assert_called_once_with(result_value.rows)

    def test_positive_rows_write_csv(self, tmp_path: Path) -> None:
        from gridiron_edge.cli.edges import edges_app

        result_value = self._result(state="positive_edges", rows=1)
        path = tmp_path / "edges_2026-2027_wk01.csv"
        with (
            patch(
                "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
                return_value=result_value,
            ),
            patch("gridiron_edge.cli.edges._edge_csv_path", return_value=path),
        ):
            result = CliRunner().invoke(
                edges_app,
                [
                    "report",
                    "--season",
                    "2026-2027",
                    "--week",
                    "1",
                    "--format",
                    "csv",
                ],
            )

        assert result.exit_code == 0, result.output
        assert path.exists()
        written = pd.read_csv(path)
        assert written["ev"].tolist() == pytest.approx([0.05])

    @pytest.mark.parametrize(
        ("blocker", "message"),
        [
            ("no_predictions", "No current weekly product"),
            ("no_market_data", "No current market snapshot"),
            ("zero_matched_games", "no matching game IDs"),
            ("incomplete_markets", "market families are incomplete"),
        ],
    )
    def test_blocked_result_exits_nonzero_and_removes_stale_csv(
        self,
        blocker: str,
        message: str,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        result_value = self._result(state="blocked", blockers=(blocker,))
        with (
            patch(
                "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
                return_value=result_value,
            ),
            patch("gridiron_edge.cli.edges._remove_edge_csv") as remove,
        ):
            result = CliRunner().invoke(
                edges_app,
                ["report", "--season", "2026-2027", "--week", "1"],
            )

        assert result.exit_code != 0
        assert message in result.output
        remove.assert_called_once_with("2026-2027", 1)

    @pytest.mark.parametrize(
        ("state", "message"),
        [
            ("no_calculable_edges", "No calculable edges"),
            ("no_positive_edges", "no positive expected-value edges"),
        ],
    )
    def test_analytical_empty_result_is_success_and_removes_stale_csv(
        self,
        state: str,
        message: str,
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        result_value = self._result(state=state)
        with (
            patch(
                "gridiron_edge.market.weekly_edge_service.build_weekly_edge_result",
                return_value=result_value,
            ),
            patch("gridiron_edge.cli.edges._remove_edge_csv") as remove,
        ):
            result = CliRunner().invoke(
                edges_app,
                ["report", "--season", "2026-2027", "--week", "1"],
            )

        assert result.exit_code == 0, result.output
        assert message in result.output
        remove.assert_called_once_with("2026-2027", 1)

    @pytest.mark.parametrize(
        "args",
        [
            ["--format", "pdf"],
            ["--bankroll", "-1"],
            ["--kelly-multiplier", "-0.1"],
            ["--kelly-multiplier", "1.1"],
            ["--min-ev", "-0.1"],
        ],
    )
    @patch("gridiron_edge.market.weekly_edge_service.build_weekly_edge_result")
    def test_invalid_options_fail_before_service(
        self,
        service: MagicMock,
        args: list[str],
    ) -> None:
        from gridiron_edge.cli.edges import edges_app

        result = CliRunner().invoke(
            edges_app,
            ["report", "--season", "2026-2027", "--week", "1", *args],
        )

        assert result.exit_code != 0
        service.assert_not_called()

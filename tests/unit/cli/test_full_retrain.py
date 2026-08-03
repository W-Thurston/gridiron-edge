# test/unit/cli/test_full_retrain.py

"""Tests for the full-retrain composite command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gridiron_edge.cli._composites import StageResult
from gridiron_edge.cli.full_retrain import (
    _GAME_MODEL_PAIRS,
    _PROP_ALGORITHMS,
    _PROP_STAT_FAMILIES,
    ModelPair,
    _build_stages,
    _resolve_game_pairs,
    _resolve_prop_pairs,
)


class TestStageList:
    """Verify the stage list is well-formed."""

    def test_stages_have_expected_names(self) -> None:
        names = [s.name for s in _build_stages()]
        assert names == [
            "refresh-all-data",
            "backfill-game-models",
            "backfill-prop-models",
            "train-game-models",
            "train-prop-models",
            "refresh-calibrations",
            "promote-champions",
            "baseline-report",
        ]

    def test_backfill_game_depends_on_refresh(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert "refresh-all-data" in stages["backfill-game-models"].depends_on

    def test_calibrations_depend_on_game_backfill(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert "backfill-game-models" in stages["refresh-calibrations"].depends_on

    def test_no_stages_are_soft_fail(self) -> None:
        for stage in _build_stages():
            assert not stage.soft_fail, (
                f"Stage {stage.name!r} should not be soft-fail in full-retrain"
            )

    def test_promote_champions_depends_on_calibrations(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert set(stages["promote-champions"].depends_on) == {
            "refresh-calibrations",
            "train-game-models",
        }

    def test_baseline_report_depends_on_promote_champions(self) -> None:
        stages = {s.name: s for s in _build_stages()}
        assert set(stages["baseline-report"].depends_on) == {"promote-champions"}


class TestPairResolution:
    """Cover --game-models and --prop-models resolution."""

    def test_empty_game_models_returns_all_pairs(self) -> None:
        pairs = _resolve_game_pairs([])
        assert len(pairs) == len(_GAME_MODEL_PAIRS)

    def test_specific_game_model_returns_only_that_pair(self) -> None:
        pairs = _resolve_game_pairs(["win_prob_random_forest"])
        assert len(pairs) == 1
        assert pairs[0].model_name == "win_prob"
        assert pairs[0].model_type == "random_forest"

    def test_unknown_game_model_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter, match="Unknown game"):
            _resolve_game_pairs(["nonexistent_model"])

    def test_empty_prop_models_returns_all_15(self) -> None:
        pairs = _resolve_prop_pairs([])
        assert len(pairs) == len(_PROP_STAT_FAMILIES) * len(_PROP_ALGORITHMS)

    def test_specific_prop_model_returns_only_that_pair(self) -> None:
        pairs = _resolve_prop_pairs(["qb_pass_yards_elasticnet"])
        assert pairs == [("qb_pass_yards", "elasticnet")]

    def test_unknown_prop_model_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter, match="Unknown prop"):
            _resolve_prop_pairs(["nonexistent_pair"])


class TestModelPair:
    def test_composite_key(self) -> None:
        pair = ModelPair(model_name="win_prob", model_type="random_forest")
        assert pair.composite_key == "win_prob_random_forest"


class TestBackfillGameModelsStage:
    """Cover the backfill-game-models stage's main paths."""

    def test_returns_no_op_when_no_pairs(self) -> None:
        from gridiron_edge.cli.full_retrain import (
            _stage_backfill_game_models,
        )

        ctx = {"game_pairs": []}
        result = _stage_backfill_game_models(ctx)
        assert result.success
        assert "no game pairs requested" in result.detail

    @patch("gridiron_edge.cli.full_retrain.backfill_model")
    def test_iterates_over_pairs(self, mock_backfill: MagicMock) -> None:
        from gridiron_edge.cli.full_retrain import (
            _stage_backfill_game_models,
        )

        mock_backfill.return_value = 100
        ctx = {
            "game_pairs": [
                ModelPair(model_name="win_prob", model_type="elo"),
                ModelPair(model_name="win_prob", model_type="random_forest"),
            ]
        }

        result = _stage_backfill_game_models(ctx)
        assert result.success
        assert result.rows == 200  # 100 per pair x 2 pairs
        assert mock_backfill.call_count == 2
        for call in mock_backfill.call_args_list:
            assert "overwrite" not in call.kwargs


class TestBackfillPropModelsStage:
    """Cover the backfill-prop-models stage's no-op path."""

    def test_returns_no_op_when_no_pairs(self) -> None:
        from gridiron_edge.cli.full_retrain import (
            _stage_backfill_prop_models,
        )

        ctx = {"prop_pairs": []}
        result = _stage_backfill_prop_models(ctx)
        assert result.success
        assert "no prop pairs requested" in result.detail


class TestTrainGameModelsStage:
    """Cover final deployable game artifact training."""

    def test_elo_only_is_no_op(self) -> None:
        from gridiron_edge.cli.full_retrain import _stage_train_game_models

        result = _stage_train_game_models({"game_pairs": [ModelPair("win_prob", "elo")]})

        assert result.success
        assert result.rows is None
        assert "no trainable game pairs" in result.detail

    def test_trains_non_elo_pair(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dataclasses import dataclass

        import pandas as pd

        from gridiron_edge.cli.full_retrain import _stage_train_game_models

        trained: list[tuple[int, Path | None]] = []

        @dataclass
        class FakeSettings:
            repo_root: Path

        class FakeModel:
            def train(
                self,
                df: pd.DataFrame,
                *,
                repo: Path | None = None,
            ) -> object:
                trained.append((len(df), repo))
                return object()

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            lambda: FakeSettings(tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.loaders.load_modeling_file",
            lambda repo: pd.DataFrame({"GAME_ID": ["g1"]}),
        )
        monkeypatch.setattr(
            "gridiron_edge.models.registry.ModelRegistry.get",
            lambda key: FakeModel,
        )

        result = _stage_train_game_models({"game_pairs": [ModelPair("win_prob", "logistic")]})

        assert result.success
        assert result.rows == 1
        assert trained == [(1, tmp_path)]
        assert result.artifacts == [tmp_path / "data" / "models" / "win_prob" / "logistic"]


class TestTrainPropModelsStage:
    """Cover final deployable prop artifact training."""

    def test_returns_no_op_when_no_pairs(self) -> None:
        from gridiron_edge.cli.full_retrain import _stage_train_prop_models

        result = _stage_train_prop_models({"prop_pairs": []})

        assert result.success
        assert result.rows is None
        assert "no prop pairs requested" in result.detail


class TestBaselineReportStage:
    """Cover the baseline-report stage."""

    def test_returns_no_op_when_no_pairs(self) -> None:
        from gridiron_edge.cli.full_retrain import (
            _stage_baseline_report,
        )

        ctx = {"game_pairs": []}
        result = _stage_baseline_report(ctx)
        assert result.success
        assert "no pairs to report" in result.detail

    def test_writes_delta_section_when_previous_report_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dataclasses import dataclass

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_baseline_report,
        )

        reports_dir: Path = tmp_path / "data" / "output" / "reports"
        reports_dir.mkdir(parents=True)

        previous: Path = reports_dir / "full-retrain-2026-06-21-120000.md"
        previous.write_text(
            "\n".join(
                [
                    "| Pair | Brier | ECE | AUC | MAE | RMSE | R² |",
                    "|---|---|---|---|---|---|---|",
                    "| win_prob_logistic | 0.2215 | 0.0150 | 0.6800 | - | - | - |",
                ]
            )
        )

        @dataclass
        class FakeSettings:
            repo_root: Path

        @dataclass
        class FakeMeta:
            metrics: dict[str, float]

        class FakeArtifactStore:
            def __init__(self, repo: Path) -> None:
                self.repo = repo

            def is_trained(self, model_name: str, model_type: str) -> bool:
                return True

            def read_metadata(self, model_name: str, model_type: str) -> FakeMeta:
                return FakeMeta(metrics={"brier": 0.2200, "ece": 0.0140, "auc": 0.6900})

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            lambda: FakeSettings(repo_root=tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.ArtifactStore",
            FakeArtifactStore,
        )

        result: StageResult = _stage_baseline_report(
            {"game_pairs": [ModelPair("win_prob", "logistic")]}
        )

        assert result.success
        assert result.artifacts
        report_text: str = result.artifacts[0].read_text()

        assert "## Delta vs Previous Report" in report_text
        assert "| win_prob_logistic | -0.0015 | -0.0010 | +0.0100 |" in report_text

    def test_delta_parse_ignores_champions_block(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A previous report containing a Current Champions block should still
        have its Game Models metrics parsed correctly for the delta table."""
        from dataclasses import dataclass
        import json

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_baseline_report,
        )

        reports_dir = tmp_path / "data" / "output" / "reports"
        reports_dir.mkdir(parents=True)

        previous = reports_dir / "full-retrain-2026-06-30-120000.md"
        previous.write_text(
            "\n".join(
                [
                    "# Full Retrain Baseline Report",
                    "",
                    "## Current Champions",
                    "",
                    "Manifest updated: 2026-06-30T00:00:00+00:00",
                    "",
                    "- **win_prob** → 🏆 `win_prob_random_forest` (promoted 2026-06-30T00:00:00)",
                    "",
                    "## Game Models",
                    "",
                    "| Pair | Brier | ECE | AUC | MAE | RMSE | R² |",
                    "|---|---|---|---|---|---|---|",
                    "| win_prob_random_forest | 0.2215 | 0.0150 | 0.6800 | - | - | - |",
                ]
            )
        )

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "champions.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-07-01T14:00:00+00:00",
                    "models": {
                        "win_prob": {
                            "model_type": "random_forest",
                            "promoted_at": "2026-07-01T14:00:00",
                            "source_run_id": "RUN_B",
                            "metrics": {"brier": 0.2200, "ece": 0.0140, "auc": 0.6900},
                        },
                    },
                }
            )
        )

        @dataclass
        class FakeSettings:
            repo_root: Path

        @dataclass
        class FakeMeta:
            metrics: dict[str, float]

        class FakeArtifactStore:
            def __init__(self, repo: Path) -> None:
                self.repo = repo

            def is_trained(self, model_name: str, model_type: str) -> bool:
                return True

            def read_metadata(self, model_name: str, model_type: str) -> FakeMeta:
                return FakeMeta(metrics={"brier": 0.2200, "ece": 0.0140, "auc": 0.6900})

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            lambda: FakeSettings(repo_root=tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.ArtifactStore",
            FakeArtifactStore,
        )

        result = _stage_baseline_report({"game_pairs": [ModelPair("win_prob", "random_forest")]})

        assert result.success
        report_text = result.artifacts[0].read_text()

        # Delta table should still be present with correct numbers
        assert "## Delta vs Previous Report" in report_text
        assert "| win_prob_random_forest | -0.0015 | -0.0010 | +0.0100 |" in report_text

    def test_refresh_calibrations_persists_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """refresh-calibrations should save sigma + margin_std to disk."""
        from dataclasses import dataclass

        import pandas as pd

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_refresh_calibrations,
        )
        from gridiron_edge.models.game_prediction.post_process import (
            load_model_calibrations,
        )

        @dataclass
        class FakeSettings:
            repo_root: Path

        archive = pd.DataFrame(
            {
                "game_id": ["g1", "g2", "g3"],
                "home_team": ["A", "B", "C"],
                "home_win_prob": [0.60, 0.55, 0.70],
            }
        )

        modeling = pd.DataFrame(
            {
                "GAME_ID": [
                    "g1",
                    "g2",
                    "g3",
                ],
                "ACTUAL_MARGIN": [
                    7.0,
                    -1.0,
                    16.0,
                ],
            }
        )

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            lambda: FakeSettings(repo_root=tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.load_prediction_log",
            lambda **_: archive,
        )
        monkeypatch.setattr(
            "gridiron_edge.datasets.loaders.load_modeling_file",
            lambda repo: modeling,
        )
        calibration_calls: dict[str, object] = {}

        def fake_calibrate_spread_sigma(
            *,
            home_win_probs: pd.Series,
            actual_margins: pd.Series,
        ) -> float:
            calibration_calls["sigma_probs"] = home_win_probs.tolist()
            calibration_calls["sigma_margins"] = actual_margins.tolist()
            return 11.25

        def fake_compute_margin_std(
            *,
            home_win_probs: pd.Series,
            actual_margins: pd.Series,
            sigma: float,
        ) -> float:
            calibration_calls["std_probs"] = home_win_probs.tolist()
            calibration_calls["std_margins"] = actual_margins.tolist()
            calibration_calls["sigma"] = sigma
            return 13.75

        monkeypatch.setattr(
            "gridiron_edge.models.game_prediction.post_process.calibrate_spread_sigma",
            fake_calibrate_spread_sigma,
        )
        monkeypatch.setattr(
            "gridiron_edge.models.game_prediction.post_process.compute_margin_std",
            fake_compute_margin_std,
        )

        result = _stage_refresh_calibrations(
            {"game_pairs": [ModelPair("win_prob", "random_forest")]}
        )

        assert result.success

        saved = load_model_calibrations(tmp_path)
        payload = saved["win_prob_random_forest"]

        assert payload["sigma"] == pytest.approx(11.25)
        assert payload["margin_std"] == pytest.approx(13.75)
        assert calibration_calls["sigma_probs"] == pytest.approx([0.60, 0.55, 0.70])
        assert calibration_calls["sigma_margins"] == pytest.approx([7.0, -1.0, 16.0])
        assert calibration_calls["std_probs"] == pytest.approx([0.60, 0.55, 0.70])
        assert calibration_calls["std_margins"] == pytest.approx([7.0, -1.0, 16.0])
        assert calibration_calls["sigma"] == (pytest.approx(11.25))
        assert "updated_at" in payload

    def test_refresh_calibrations_rejects_missing_margin(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calibration requires canonical Actual Margin."""
        from dataclasses import dataclass

        import pandas as pd

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_refresh_calibrations,
        )

        @dataclass
        class FakeSettings:
            repo_root: Path

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            lambda: FakeSettings(repo_root=tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.datasets.loaders.load_modeling_file",
            lambda repo: pd.DataFrame(
                {
                    "GAME_ID": ["g1"],
                }
            ),
        )

        result = _stage_refresh_calibrations(
            {
                "game_pairs": [
                    ModelPair(
                        "win_prob",
                        "random_forest",
                    )
                ]
            }
        )

        assert result.success is False
        assert "ACTUAL_MARGIN" in result.detail

    def test_refresh_calibrations_rejects_duplicate_games(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Canonical actuals must contain one row per game."""
        from dataclasses import dataclass

        import pandas as pd

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_refresh_calibrations,
        )

        @dataclass
        class FakeSettings:
            repo_root: Path

        modeling = pd.DataFrame(
            {
                "GAME_ID": ["g1", "g1"],
                "ACTUAL_MARGIN": [7.0, 7.0],
            }
        )

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            lambda: FakeSettings(repo_root=tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.datasets.loaders.load_modeling_file",
            lambda repo: modeling,
        )

        result = _stage_refresh_calibrations(
            {
                "game_pairs": [
                    ModelPair(
                        "win_prob",
                        "random_forest",
                    )
                ]
            }
        )

        assert result.success is False
        assert "duplicate game IDs" in result.detail

    def test_appends_champions_block_when_manifest_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dataclasses import dataclass
        import json

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_baseline_report,
        )

        # Pre-populate manifest
        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "updated_at": "2026-07-01T14:00:00+00:00",
            "models": {
                "win_prob": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-07-01T14:00:00",
                    "source_run_id": "RUN_A",
                    "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
                },
                "qb_pass_yards": {
                    "model_type": "elasticnet",
                    "promoted_at": "2026-07-01T14:10:00",
                    "source_run_id": "RUN_A",
                    "metrics": {"mae": 63.4, "rmse": 80.6, "r2": 0.05, "coverage": 0.938},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(manifest))

        @dataclass
        class FakeSettings:
            repo_root: Path

        @dataclass
        class FakeMeta:
            metrics: dict[str, float]

        class FakeArtifactStore:
            def __init__(self, repo: Path) -> None:
                self.repo = repo

            def is_trained(self, model_name: str, model_type: str) -> bool:
                return True

            def read_metadata(self, model_name: str, model_type: str) -> FakeMeta:
                return FakeMeta(metrics={"brier": 0.213, "ece": 0.041, "auc": 0.721})

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            lambda: FakeSettings(repo_root=tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.ArtifactStore",
            FakeArtifactStore,
        )

        result = _stage_baseline_report({"game_pairs": [ModelPair("win_prob", "random_forest")]})

        assert result.success
        report_text = result.artifacts[0].read_text()

        assert "## Current Champions" in report_text
        assert "🏆 `win_prob_random_forest`" in report_text
        assert "🏆 `qb_pass_yards_elasticnet`" in report_text
        assert "2026-07-01T14:00:00" in report_text

    def test_omits_champions_block_when_no_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dataclasses import dataclass

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_baseline_report,
        )

        @dataclass
        class FakeSettings:
            repo_root: Path

        @dataclass
        class FakeMeta:
            metrics: dict[str, float]

        class FakeArtifactStore:
            def __init__(self, repo: Path) -> None:
                self.repo = repo

            def is_trained(self, model_name: str, model_type: str) -> bool:
                return True

            def read_metadata(self, model_name: str, model_type: str) -> FakeMeta:
                return FakeMeta(metrics={"brier": 0.213})

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            lambda: FakeSettings(repo_root=tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.ArtifactStore",
            FakeArtifactStore,
        )

        # No manifest written.
        result = _stage_baseline_report({"game_pairs": [ModelPair("win_prob", "random_forest")]})

        assert result.success
        report_text = result.artifacts[0].read_text()
        assert "## Current Champions" not in report_text
        assert "## Game Models" in report_text  # normal report still generated


class TestStagePromoteChampions:
    """Cover the promote-champions stage."""

    def _fake_settings(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass
        class FakeSettings:
            repo_root: Path

        return lambda: FakeSettings(repo_root=tmp_path)

    def test_writes_manifest_with_fresh_champions(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_promote_champions,
        )

        classification_result = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:00:00",
                "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
            },
        }
        regression_result = {
            "total": {
                "model_type": "xgboost",
                "promoted_at": "2026-07-01T14:05:00",
                "metrics": {"mae": 10.24, "rmse": 12.87, "r2": 0.18},
            },
        }
        prop_result = {
            "qb_pass_yards": {
                "model_type": "elasticnet",
                "promoted_at": "2026-07-01T14:10:00",
                "metrics": {"mae": 63.4, "rmse": 80.6, "r2": 0.05, "coverage": 0.938},
            },
        }

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            self._fake_settings(tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: classification_result,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: regression_result,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: prop_result,
        )

        ctx = {
            "game_pairs": [
                ModelPair("win_prob", "random_forest"),
                ModelPair("total", "xgboost"),
            ],
            "prop_pairs": [("qb_pass_yards", "elasticnet")],
            "upcoming_season_int": None,
        }

        result = _stage_promote_champions(ctx)

        assert result.success
        assert "3 fresh champion(s)" in result.detail
        assert result.rows == 3
        assert len(result.artifacts) == 1

        manifest_path = tmp_path / "data" / "output" / "champions" / "champions.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert set(manifest["models"].keys()) == {"win_prob", "total", "qb_pass_yards"}

    def test_preserves_existing_entries_outside_subset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_promote_champions,
        )

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        prior_manifest = {
            "schema_version": 1,
            "updated_at": "2026-06-01T00:00:00+00:00",
            "models": {
                "rb_rush_yards": {
                    "model_type": "random_forest",
                    "promoted_at": "2026-06-01T00:00:00",
                    "source_run_id": "OLD_RUN",
                    "metrics": {"mae": 25.0, "rmse": 32.0, "r2": 0.17, "coverage": 0.91},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(prior_manifest))

        classification_result = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:00:00",
                "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
            },
        }

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            self._fake_settings(tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: classification_result,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: {},
        )

        ctx = {
            "game_pairs": [ModelPair("win_prob", "random_forest")],
            "prop_pairs": [],
            "upcoming_season_int": None,
        }

        result = _stage_promote_champions(ctx)

        assert result.success
        assert "1 preserved" in result.detail

        manifest = json.loads((manifest_dir / "champions.json").read_text())
        assert set(manifest["models"].keys()) == {"win_prob", "rb_rush_yards"}
        assert manifest["models"]["rb_rush_yards"]["source_run_id"] == "OLD_RUN"
        assert manifest["models"]["win_prob"]["source_run_id"] != "OLD_RUN"

    def test_overwrites_existing_entry_when_in_subset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_promote_champions,
        )

        manifest_dir = tmp_path / "data" / "output" / "champions"
        manifest_dir.mkdir(parents=True)
        prior_manifest = {
            "schema_version": 1,
            "updated_at": "2026-06-01T00:00:00+00:00",
            "models": {
                "win_prob": {
                    "model_type": "xgboost",
                    "promoted_at": "2026-06-01T00:00:00",
                    "source_run_id": "OLD_RUN",
                    "metrics": {"brier": 0.25, "ece": 0.05, "auc": 0.70},
                },
            },
        }
        (manifest_dir / "champions.json").write_text(json.dumps(prior_manifest))

        classification_result = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:00:00",
                "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
            },
        }

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            self._fake_settings(tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: classification_result,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: {},
        )

        ctx = {
            "game_pairs": [
                ModelPair("win_prob", "random_forest"),
                ModelPair("win_prob", "xgboost"),
            ],
            "prop_pairs": [],
            "upcoming_season_int": None,
        }

        result = _stage_promote_champions(ctx)

        assert result.success

        manifest = json.loads((manifest_dir / "champions.json").read_text())
        assert manifest["models"]["win_prob"]["model_type"] == "random_forest"
        assert manifest["models"]["win_prob"]["source_run_id"] != "OLD_RUN"

    def test_cold_start_writes_only_fresh_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_promote_champions,
        )

        classification_result = {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:00:00",
                "metrics": {"brier": 0.213, "ece": 0.041, "auc": 0.721},
            },
        }

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            self._fake_settings(tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: classification_result,
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: {},
        )

        ctx = {
            "game_pairs": [ModelPair("win_prob", "random_forest")],
            "prop_pairs": [],
            "upcoming_season_int": None,
        }

        result = _stage_promote_champions(ctx)

        assert result.success
        assert "0 preserved" in result.detail

        manifest_path = tmp_path / "data" / "output" / "champions" / "champions.json"
        manifest = json.loads(manifest_path.read_text())
        assert set(manifest["models"].keys()) == {"win_prob"}

    def test_emits_warning_when_no_game_champions_selected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from gridiron_edge.cli.full_retrain import (
            ModelPair,
            _stage_promote_champions,
        )

        monkeypatch.setattr(
            "gridiron_edge.cli.full_retrain.get_settings",
            self._fake_settings(tmp_path),
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_classification_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_game_regression_champions",
            lambda pairs, *, repo: {},
        )
        monkeypatch.setattr(
            "gridiron_edge.evaluation.champion.select_prop_champions_all_families",
            lambda families, *, repo: {},
        )

        ctx = {
            "game_pairs": [ModelPair("win_prob", "random_forest")],
            "prop_pairs": [],
            "upcoming_season_int": None,
        }

        result = _stage_promote_champions(ctx)

        assert result.success
        assert any("no game champions selected" in w for w in result.warnings)


class TestCommandInvocation:
    """End-to-end test of the composite via CliRunner."""

    @patch("gridiron_edge.cli.full_retrain._stage_refresh_all_data")
    @patch("gridiron_edge.cli.full_retrain._stage_backfill_game_models")
    @patch("gridiron_edge.cli.full_retrain._stage_backfill_prop_models")
    @patch("gridiron_edge.cli.full_retrain._stage_train_game_models")
    @patch("gridiron_edge.cli.full_retrain._stage_train_prop_models")
    @patch("gridiron_edge.cli.full_retrain._stage_refresh_calibrations")
    @patch("gridiron_edge.cli.full_retrain._stage_promote_champions")
    @patch("gridiron_edge.cli.full_retrain._stage_baseline_report")
    def test_runs_all_stages_when_all_succeed(
        self,
        mock_report: MagicMock,
        mock_promote: MagicMock,
        mock_calib: MagicMock,
        mock_train_props: MagicMock,
        mock_train_games: MagicMock,
        mock_props: MagicMock,
        mock_games: MagicMock,
        mock_refresh: MagicMock,
    ) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.full_retrain import full_retrain_cmd

        for m in [
            mock_refresh,
            mock_games,
            mock_props,
            mock_train_games,
            mock_train_props,
            mock_calib,
            mock_promote,
            mock_report,
        ]:
            m.return_value = StageResult(success=True, detail="ok")

        app = typer.Typer()
        app.command()(full_retrain_cmd)

        runner = CliRunner()
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output

    def test_skip_prop_backfill_shorthand(self) -> None:
        import typer
        from typer.testing import CliRunner

        from gridiron_edge.cli.full_retrain import full_retrain_cmd

        with (
            patch("gridiron_edge.cli.full_retrain._stage_refresh_all_data") as mock_refresh,
            patch("gridiron_edge.cli.full_retrain._stage_backfill_game_models") as mock_games,
            patch("gridiron_edge.cli.full_retrain._stage_backfill_prop_models") as mock_props,
            patch("gridiron_edge.cli.full_retrain._stage_train_game_models") as mock_train_games,
            patch("gridiron_edge.cli.full_retrain._stage_train_prop_models") as mock_train_props,
            patch("gridiron_edge.cli.full_retrain._stage_refresh_calibrations") as mock_calib,
            patch("gridiron_edge.cli.full_retrain._stage_promote_champions") as mock_promote,
            patch("gridiron_edge.cli.full_retrain._stage_baseline_report") as mock_report,
        ):
            for m in [
                mock_refresh,
                mock_games,
                mock_props,
                mock_train_games,
                mock_train_props,
                mock_calib,
                mock_promote,
                mock_report,
            ]:
                m.return_value = StageResult(success=True, detail="ok")

            app = typer.Typer()
            app.command()(full_retrain_cmd)

            runner = CliRunner()
            result = runner.invoke(app, ["--skip-prop-backfill"])
            assert result.exit_code == 0, result.output
            mock_props.assert_not_called()
            mock_train_props.assert_not_called()
            mock_train_games.assert_called_once()
            promote_context = mock_promote.call_args.args[0]
            assert promote_context["prop_pairs"] == []


class TestBaselineReportDiffHelpers:
    """Tests for full-retrain baseline report parsing and delta formatting."""

    def test_parse_baseline_report_reads_metric_rows(self, tmp_path: Path) -> None:
        from gridiron_edge.cli.full_retrain import _parse_baseline_report

        report = tmp_path / "full-retrain-2026-06-21-120000.md"
        report.write_text(
            "\n".join(
                [
                    "# Full Retrain Baseline Report",
                    "",
                    "| Pair | Brier | ECE | AUC | MAE | RMSE | R² |",
                    "|---|---|---|---|---|---|---|",
                    "| win_prob_logistic | 0.2215 | 0.0153 | 0.6822 | - | - | - |",
                    "| total_random_forest | - | - | - | 10.24 | 13.12 | 0.056 |",
                ]
            )
        )

        parsed = _parse_baseline_report(report)

        assert parsed["win_prob_logistic"]["brier"] == 0.2215
        assert parsed["win_prob_logistic"]["ece"] == 0.0153
        assert parsed["win_prob_logistic"]["mae"] is None
        assert parsed["total_random_forest"]["mae"] == 10.24
        assert parsed["total_random_forest"]["r2"] == 0.056

    def test_parse_baseline_report_keeps_no_artifact_rows(
        self,
        tmp_path: Path,
    ) -> None:
        from gridiron_edge.cli.full_retrain import _parse_baseline_report

        report = tmp_path / "full-retrain-2026-06-21-120000.md"
        report.write_text(
            "\n".join(
                [
                    "| Pair | Brier | ECE | AUC | MAE | RMSE | R² |",
                    "|---|---|---|---|---|---|---|",
                    "| win_prob_elo | - no artifact - |",
                ]
            )
        )

        parsed = _parse_baseline_report(report)

        assert "win_prob_elo" in parsed
        assert all(value is None for value in parsed["win_prob_elo"].values())

    def test_format_metric_delta_requires_both_values(self) -> None:
        from gridiron_edge.cli.full_retrain import _format_metric_delta

        assert _format_metric_delta(current=0.2200, previous=0.2210, decimals=4) == "-0.0010"
        assert _format_metric_delta(current=None, previous=0.2210, decimals=4) == "-"
        assert _format_metric_delta(current=0.2200, previous=None, decimals=4) == "-"

    def test_find_previous_baseline_report_returns_latest(
        self,
        tmp_path: Path,
    ) -> None:
        from gridiron_edge.cli.full_retrain import _find_previous_baseline_report

        older: Path = tmp_path / "full-retrain-2026-06-21-120000.md"
        newer: Path = tmp_path / "full-retrain-2026-06-21-130000.md"
        older.write_text("older")
        newer.write_text("newer")

        assert _find_previous_baseline_report(tmp_path) == newer

# tests/unit/cli/test_models.py
"""Unit tests for cli/models.py - promotion decision and path handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gridiron_edge.cli.models import (
    _apply_promotion_decision,
    _metric_block_for,
    _primary_metric_for,
    _split_composite_key,
)
from gridiron_edge.models.game_prediction.base import GameModelMetadata


class TestSplitCompositeKey:
    """Verify composite key parsing against known model_name prefixes."""

    def test_splits_win_prob_random_forest(self) -> None:
        result = _split_composite_key("win_prob_random_forest")
        assert result == ("win_prob", "random_forest")

    def test_splits_win_prob_xgboost(self) -> None:
        result = _split_composite_key("win_prob_xgboost")
        assert result == ("win_prob", "xgboost")

    def test_splits_total_random_forest(self) -> None:
        result = _split_composite_key("total_random_forest")
        assert result == ("total", "random_forest")

    def test_splits_win_prob_logistic(self) -> None:
        result = _split_composite_key("win_prob_logistic")
        assert result == ("win_prob", "logistic")

    def test_splits_win_prob_elo(self) -> None:
        result = _split_composite_key("win_prob_elo")
        assert result == ("win_prob", "elo")

    def test_returns_none_for_unknown_key(self) -> None:
        result = _split_composite_key("fake_model_type")
        assert result is None


class TestApplyPromotionDecisionNoChampion:
    """When no champion exists, candidate is moved to champion_dir."""

    def test_promotes_candidate_when_no_champion(self, tmp_path: Path) -> None:
        candidate_dir = tmp_path / "rf__candidate"
        candidate_dir.mkdir()
        (candidate_dir / "model.joblib").touch()

        champion_dir = tmp_path / "rf"

        challenger_meta = GameModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-22T12:00:00",
            metrics={"brier": 0.225},
        )

        _apply_promotion_decision(
            champion_meta=None,
            challenger_meta=challenger_meta,
            champion_dir=champion_dir,
            candidate_dir=candidate_dir,
            force=False,
            no_promote=False,
        )

        assert champion_dir.exists()
        assert (champion_dir / "model.joblib").exists()
        assert not candidate_dir.exists()


class TestApplyPromotionDecisionWithChampion:
    """Comparison logic when both champion and candidate exist."""

    def _setup_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        champion_dir = tmp_path / "rf"
        candidate_dir = tmp_path / "rf__candidate"
        champion_dir.mkdir()
        candidate_dir.mkdir()
        (champion_dir / "champion_marker").touch()
        (candidate_dir / "candidate_marker").touch()
        return champion_dir, candidate_dir

    def test_reject_keeps_champion_deletes_candidate(self, tmp_path: Path) -> None:
        """When gates fail, champion is untouched and candidate is deleted."""
        champion_dir, candidate_dir = self._setup_dirs(tmp_path)
        champion_meta = MagicMock()
        challenger_meta = MagicMock()

        # Patch compare_classification_models to return should_promote=False
        from gridiron_edge.evaluation import champion as champion_module

        original_compare = champion_module.compare_classification_models
        result = MagicMock()
        result.should_promote = False
        champion_module.compare_classification_models = MagicMock(return_value=result)
        champion_module.format_classification_comparison = MagicMock(return_value="REJECT")

        try:
            _apply_promotion_decision(
                champion_meta=champion_meta,
                challenger_meta=challenger_meta,
                champion_dir=champion_dir,
                candidate_dir=candidate_dir,
                force=False,
                no_promote=False,
            )
        finally:
            champion_module.compare_classification_models = original_compare

        # Champion marker should still exist
        assert (champion_dir / "champion_marker").exists()
        # Candidate should be deleted
        assert not candidate_dir.exists()

    def test_force_promote_replaces_champion(self, tmp_path: Path) -> None:
        """With --force, candidate replaces champion even if gates fail."""
        champion_dir, candidate_dir = self._setup_dirs(tmp_path)
        champion_meta = MagicMock()
        challenger_meta = MagicMock()

        from gridiron_edge.evaluation import champion as champion_module

        original_compare = champion_module.compare_classification_models
        result = MagicMock()
        result.should_promote = False  # gates fail
        champion_module.compare_classification_models = MagicMock(return_value=result)
        champion_module.format_classification_comparison = MagicMock(return_value="REJECT")

        try:
            _apply_promotion_decision(
                champion_meta=champion_meta,
                challenger_meta=challenger_meta,
                champion_dir=champion_dir,
                candidate_dir=candidate_dir,
                force=True,
                no_promote=False,
            )
        finally:
            champion_module.compare_classification_models = original_compare

        # Candidate marker should now be in champion location
        assert (champion_dir / "candidate_marker").exists()
        # Old champion marker should be gone
        assert not (champion_dir / "champion_marker").exists()
        # Candidate dir should be gone (moved to champion)
        assert not candidate_dir.exists()

    def test_no_promote_keeps_champion(self, tmp_path: Path) -> None:
        """With --no-promote, candidate is deleted regardless of gate result."""
        champion_dir, candidate_dir = self._setup_dirs(tmp_path)
        champion_meta = MagicMock()
        challenger_meta = MagicMock()

        from gridiron_edge.evaluation import champion as champion_module

        original_compare = champion_module.compare_classification_models
        result = MagicMock()
        result.should_promote = True  # gates pass
        champion_module.compare_classification_models = MagicMock(return_value=result)
        champion_module.format_classification_comparison = MagicMock(return_value="PROMOTE")

        try:
            _apply_promotion_decision(
                champion_meta=champion_meta,
                challenger_meta=challenger_meta,
                champion_dir=champion_dir,
                candidate_dir=candidate_dir,
                force=False,
                no_promote=True,
            )
        finally:
            champion_module.compare_classification_models = original_compare

        # Champion marker untouched
        assert (champion_dir / "champion_marker").exists()
        # Candidate deleted
        assert not candidate_dir.exists()

    def test_promote_replaces_champion_with_candidate(self, tmp_path: Path) -> None:
        """Successful promotion: candidate becomes champion."""
        champion_dir, candidate_dir = self._setup_dirs(tmp_path)
        champion_meta = MagicMock()
        challenger_meta = MagicMock()

        from gridiron_edge.evaluation import champion as champion_module

        original_compare = champion_module.compare_classification_models
        result = MagicMock()
        result.should_promote = True
        champion_module.compare_classification_models = MagicMock(return_value=result)
        champion_module.format_classification_comparison = MagicMock(return_value="PROMOTE")

        try:
            _apply_promotion_decision(
                champion_meta=champion_meta,
                challenger_meta=challenger_meta,
                champion_dir=champion_dir,
                candidate_dir=candidate_dir,
                force=False,
                no_promote=False,
            )
        finally:
            champion_module.compare_classification_models = original_compare

        # Candidate marker is now at champion location
        assert (champion_dir / "candidate_marker").exists()
        # Old champion marker is gone
        assert not (champion_dir / "champion_marker").exists()
        # Candidate dir is gone
        assert not candidate_dir.exists()


class TestApplyPromotionDecisionAtomicity:
    """Verify the champion directory is never in an inconsistent state."""

    def test_reject_leaves_no_orphan_dirs(self, tmp_path: Path) -> None:
        """After reject, only champion_dir should exist; no leftover dirs."""
        champion_dir = tmp_path / "rf"
        candidate_dir = tmp_path / "rf__candidate"
        champion_dir.mkdir()
        candidate_dir.mkdir()
        (champion_dir / "model.joblib").touch()
        (candidate_dir / "model.joblib").touch()

        champion_meta = MagicMock()
        challenger_meta = MagicMock()

        from gridiron_edge.evaluation import champion as champion_module

        original_compare = champion_module.compare_classification_models
        result = MagicMock()
        result.should_promote = False
        champion_module.compare_classification_models = MagicMock(return_value=result)
        champion_module.format_classification_comparison = MagicMock(return_value="REJECT")

        try:
            _apply_promotion_decision(
                champion_meta=champion_meta,
                challenger_meta=challenger_meta,
                champion_dir=champion_dir,
                candidate_dir=candidate_dir,
                force=False,
                no_promote=False,
            )
        finally:
            champion_module.compare_classification_models = original_compare

        # Only champion_dir should remain
        siblings = list(tmp_path.iterdir())
        assert len(siblings) == 1
        assert siblings[0] == champion_dir

    def test_promote_leaves_no_orphan_dirs(self, tmp_path: Path) -> None:
        """After promote, only champion_dir should exist; no leftover dirs."""
        champion_dir = tmp_path / "rf"
        candidate_dir = tmp_path / "rf__candidate"
        champion_dir.mkdir()
        candidate_dir.mkdir()
        (champion_dir / "model.joblib").touch()
        (candidate_dir / "model.joblib").touch()

        champion_meta = MagicMock()
        challenger_meta = MagicMock()

        from gridiron_edge.evaluation import champion as champion_module

        original_compare = champion_module.compare_classification_models
        result = MagicMock()
        result.should_promote = True
        champion_module.compare_classification_models = MagicMock(return_value=result)
        champion_module.format_classification_comparison = MagicMock(return_value="PROMOTE")

        try:
            _apply_promotion_decision(
                champion_meta=champion_meta,
                challenger_meta=challenger_meta,
                champion_dir=champion_dir,
                candidate_dir=candidate_dir,
                force=False,
                no_promote=False,
            )
        finally:
            champion_module.compare_classification_models = original_compare

        # Only champion_dir should remain
        siblings = list(tmp_path.iterdir())
        assert len(siblings) == 1
        assert siblings[0] == champion_dir


class TestTrainChallengerIntoCandidate:
    """Verify candidate-pattern semantics during training."""

    def test_restores_champion_on_training_failure(self, tmp_path: Path) -> None:
        """If model.train() raises, champion is restored from holding."""
        from gridiron_edge.cli.models import _train_challenger_into_candidate

        champion_dir = tmp_path / "rf"
        candidate_dir = tmp_path / "rf__candidate"
        champion_dir.mkdir()
        (champion_dir / "champion_marker").touch()

        # Mock model whose train() raises
        model = MagicMock()
        model.train.side_effect = RuntimeError("training crashed")

        df = MagicMock()

        with pytest.raises(RuntimeError, match="training crashed"):
            _train_challenger_into_candidate(
                model=model,
                df=df,
                repo=tmp_path,
                champion_dir=champion_dir,
                candidate_dir=candidate_dir,
                model_type="rf",
            )

        # Champion should still be there with original marker
        assert champion_dir.exists()
        assert (champion_dir / "champion_marker").exists()
        # Candidate should not exist
        assert not candidate_dir.exists()
        # Holding should be cleaned up too
        holding = tmp_path / "rf__holding"
        assert not holding.exists()

    def test_no_champion_no_holding(self, tmp_path: Path) -> None:
        """When no champion exists, no holding directory is created."""
        from gridiron_edge.cli.models import _train_challenger_into_candidate

        champion_dir = tmp_path / "rf"
        candidate_dir = tmp_path / "rf__candidate"

        # Model that "trains" by creating files in champion_dir
        def fake_train(_df: object, *, repo: Path) -> object:
            champion_dir.mkdir()
            (champion_dir / "trained_marker").touch()
            meta = GameModelMetadata(
                model_name="win_prob",
                model_type="random_forest",
                task="classification",
                trained_at="2026-06-22T12:00:00",
                metrics={"brier": 0.225},
            )

            return meta

        model = MagicMock()
        model.train.side_effect = fake_train

        df = MagicMock()

        result = _train_challenger_into_candidate(
            model=model,
            df=df,
            repo=tmp_path,
            champion_dir=champion_dir,
            candidate_dir=candidate_dir,
            model_type="rf",
        )

        # Candidate exists with the trained marker
        assert candidate_dir.exists()
        assert (candidate_dir / "trained_marker").exists()
        # Champion does not exist (no original champion)
        assert not champion_dir.exists()
        # Holding never created
        holding = tmp_path / "rf__holding"
        assert not holding.exists()
        # Returned meta comes from the fake model
        assert result.metrics["brier"] == 0.225


class TestPrimaryMetricFor:
    """`_primary_metric_for` must read from meta.metrics, not deprecated
    top-level attribute-style fields.

    Regression test for the `.holdout_brier` bug that the # type: ignore
    comments in cli/models.py were hiding.
    """

    def test_classification_reads_brier_from_metrics_dict(self) -> None:
        meta = GameModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-22T12:00:00",
            metrics={"brier": 0.2215, "ece": 0.015, "auc": 0.68},
        )
        label, value = _primary_metric_for(meta)
        assert label == "Holdout Brier"
        assert value == "0.22150"

    def test_regression_reads_mae_from_metrics_dict(self) -> None:
        meta = GameModelMetadata(
            model_name="total",
            model_type="random_forest",
            task="regression",
            trained_at="2026-06-22T12:00:00",
            metrics={"mae": 10.5, "rmse": 13.2, "r2": 0.15},
        )
        label, value = _primary_metric_for(meta)
        assert label == "Holdout MAE"
        assert value == "10.50000"

    def test_missing_metric_returns_not_recorded(self) -> None:
        meta = GameModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-22T12:00:00",
            metrics={},
        )
        label, value = _primary_metric_for(meta)
        assert label == "Holdout Brier"
        assert value == "(not recorded)"

    def test_unknown_task_returns_dashes(self) -> None:
        meta = GameModelMetadata(
            model_name="mystery",
            model_type="random_forest",
            task="clustering",
            trained_at="2026-06-22T12:00:00",
            metrics={"score": 0.5},
        )
        label, value = _primary_metric_for(meta)
        assert label == "-"
        assert value == "-"


class TestMetricBlockFor:
    """`_metric_block_for` returns task-appropriate metric rows."""

    def test_classification_returns_five_rows(self) -> None:
        meta = GameModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-22T12:00:00",
            metrics={
                "brier": 0.2215,
                "ece": 0.015,
                "auc": 0.68,
                "log_loss": 0.62,
                "accuracy": 0.62,
            },
        )
        rows = _metric_block_for(meta)
        labels = [r[0] for r in rows]
        assert labels == [
            "Holdout Brier",
            "ECE",
            "AUC",
            "Log Loss",
            "Accuracy",
        ]

    def test_regression_returns_three_rows(self) -> None:
        meta = GameModelMetadata(
            model_name="total",
            model_type="random_forest",
            task="regression",
            trained_at="2026-06-22T12:00:00",
            metrics={"mae": 10.5, "rmse": 13.2, "r2": 0.15},
        )
        rows = _metric_block_for(meta)
        labels = [r[0] for r in rows]
        assert labels == ["Holdout MAE", "Holdout RMSE", "Holdout R²"]

    def test_missing_metric_shows_not_recorded(self) -> None:
        meta = GameModelMetadata(
            model_name="total",
            model_type="random_forest",
            task="regression",
            trained_at="2026-06-22T12:00:00",
            metrics={"mae": 10.5},  # rmse and r2 missing
        )
        rows = _metric_block_for(meta)
        assert rows[0] == ("Holdout MAE", "10.50000")
        assert rows[1] == ("Holdout RMSE", "(not recorded)")
        assert rows[2] == ("Holdout R²", "(not recorded)")


class TestColdStartMetricOutput:
    """Cold-start promotion path must report the correct task-appropriate
    metric label and value, drawn from meta.metrics rather than a
    deprecated attribute-style field.

    Direct regression test for the `.holdout_brier` # type: ignore bug.
    """

    def test_cold_start_reports_brier_for_classification(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        (candidate / "artifact.pkl").write_bytes(b"stub")

        champion = tmp_path / "champion"

        challenger_meta = GameModelMetadata(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            trained_at="2026-06-22T12:00:00",
            metrics={"brier": 0.2215},
        )

        _apply_promotion_decision(
            champion_meta=None,
            challenger_meta=challenger_meta,
            champion_dir=champion,
            candidate_dir=candidate,
            force=False,
            no_promote=False,
        )

        out = capsys.readouterr().out
        assert "Holdout Brier" in out
        assert "0.22150" in out

    def test_cold_start_reports_mae_for_regression(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        (candidate / "artifact.pkl").write_bytes(b"stub")

        champion = tmp_path / "champion"

        challenger_meta = GameModelMetadata(
            model_name="total",
            model_type="random_forest",
            task="regression",
            trained_at="2026-06-22T12:00:00",
            metrics={"mae": 10.5},
        )

        _apply_promotion_decision(
            champion_meta=None,
            challenger_meta=challenger_meta,
            champion_dir=champion,
            candidate_dir=candidate,
            force=False,
            no_promote=False,
        )

        out = capsys.readouterr().out
        assert "Holdout MAE" in out
        assert "10.50000" in out
        # The regression cold-start MUST NOT hardcode "Brier"
        assert "Brier" not in out


class TestNoDeprecatedAttributes:
    """The metadata refactor moved holdout_brier / holdout_mae etc from
    top-level fields into meta.metrics. Assert cli/models.py doesn't
    reach for them via attribute access anywhere.

    This is the guardrail against the exact pattern that was hiding
    behind `# type: ignore [attr-defined]` comments for weeks.
    """

    def test_source_has_no_holdout_brier_access(self) -> None:
        import inspect

        import gridiron_edge.cli.models as mod

        source = inspect.getsource(mod)
        assert ".holdout_brier" not in source, (
            "cli/models.py must not access .holdout_brier as an attribute; "
            "use meta.metrics['brier'] via _primary_metric_for instead."
        )

    def test_source_has_no_holdout_mae_access(self) -> None:
        import inspect

        import gridiron_edge.cli.models as mod

        source = inspect.getsource(mod)
        assert ".holdout_mae" not in source, (
            "cli/models.py must not access .holdout_mae as an attribute; "
            "use meta.metrics['mae'] via _primary_metric_for instead."
        )

    def test_source_has_no_type_ignore_attr_defined(self) -> None:
        """The `# type: ignore [attr-defined]` comments were hiding the
        bug. Assert they don't come back."""
        import inspect

        import gridiron_edge.cli.models as mod

        source = inspect.getsource(mod)
        assert "attr-defined" not in source, (
            "cli/models.py must not use `# type: ignore [attr-defined]` "
            "for metadata access; type-check the code properly instead."
        )

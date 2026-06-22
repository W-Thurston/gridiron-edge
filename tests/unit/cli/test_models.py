# tests/unit/cli/test_models.py
"""Unit tests for cli/models.py - promotion decision and path handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gridiron_edge.cli.models import _apply_promotion_decision, _split_composite_key


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

        challenger_meta = MagicMock()
        challenger_meta.holdout_brier = 0.225

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
        """If predictor.train() raises, champion is restored from holding."""
        from gridiron_edge.cli.models import _train_challenger_into_candidate

        champion_dir = tmp_path / "rf"
        candidate_dir = tmp_path / "rf__candidate"
        champion_dir.mkdir()
        (champion_dir / "champion_marker").touch()

        # Mock predictor whose train() raises
        predictor = MagicMock()
        predictor.train.side_effect = RuntimeError("training crashed")

        df = MagicMock()

        with pytest.raises(RuntimeError, match="training crashed"):
            _train_challenger_into_candidate(
                predictor=predictor,
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

        # Predictor that "trains" by creating files in champion_dir
        def fake_train(_df: object, *, repo: Path) -> object:
            champion_dir.mkdir()
            (champion_dir / "trained_marker").touch()
            meta = MagicMock()
            meta.holdout_brier = 0.225
            return meta

        predictor = MagicMock()
        predictor.train.side_effect = fake_train

        df = MagicMock()

        result = _train_challenger_into_candidate(
            predictor=predictor,
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
        # Returned meta
        assert result.holdout_brier == 0.225

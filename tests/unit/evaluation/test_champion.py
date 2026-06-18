# tests/unit/evaluation/test_champion.py
"""Tests for gridiron_edge.evaluation.champion — promotion gate logic."""

from __future__ import annotations

from math import isnan

import pytest

from gridiron_edge.evaluation.champion import (
    ClassificationComparisonResult,
    ClassificationPromotionGates,
    RegressionComparisonResult,
    RegressionModelResult,
    RegressionPromotionGates,
    compare_classification_models,
    compare_regression_models,
    extract_classification_metrics,
    format_classification_comparison,
    format_regression_comparison,
    select_prop_champion,
)
from gridiron_edge.models.game_prediction.base import GameModelMetadata


def _make_meta(
    brier: float,
    ece: float | None = None,
    auc: float | None = None,
    log_loss: float | None = None,
    accuracy: float | None = None,
    model_name: str = "win_prob",
    model_type: str = "test_model",
) -> GameModelMetadata:
    """Build a GameModelMetadata with classification metrics as first-class fields.

    All metrics are set as first-class fields on :class:`GameModelMetadata`
    (post-Workstream 2 D3). Optional kwargs default to NaN so callers can
    omit individual metrics to test the missing-data code paths.
    """
    return GameModelMetadata(
        model_name=model_name,
        model_type=model_type,
        task="classification",
        trained_at="2026-06-03T00:00:00",
        training_seasons=["2020-2021"],
        holdout_seasons=["2023-2024"],
        parameters={},
        feature_columns=["f1", "f2"],
        n_train_rows=10,
        n_holdout_rows=2,
        holdout_brier=brier,
        holdout_ece=ece if ece is not None else float("nan"),
        holdout_auc=auc if auc is not None else float("nan"),
        holdout_log_loss=log_loss if log_loss is not None else float("nan"),
        holdout_accuracy=accuracy if accuracy is not None else float("nan"),
    )


def _make_regression(
    model_type: str = "elasticnet",
    mae: float = 58.0,
    rmse: float = 72.6,
    r2: float = 0.071,
    coverage: float = 0.938,
) -> RegressionModelResult:
    """Build a RegressionModelResult for testing."""
    return RegressionModelResult(
        model_type=model_type,
        mae=mae,
        rmse=rmse,
        r2=r2,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# extract_classification_metrics
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    def test_all_metrics_present(self) -> None:
        meta: GameModelMetadata = _make_meta(
            brier=0.195, ece=0.036, auc=0.774, log_loss=0.58, accuracy=0.708
        )
        m: dict[str, float] = extract_classification_metrics(meta)
        assert m["brier"] == 0.195
        assert m["ece"] == 0.036
        assert m["auc"] == 0.774
        assert m["log_loss"] == 0.58
        assert m["accuracy"] == 0.708

    def test_missing_metrics_are_nan(self) -> None:
        meta: GameModelMetadata = _make_meta(brier=0.200)
        m: dict[str, float] = extract_classification_metrics(meta)
        assert m["brier"] == 0.200
        assert isnan(m["ece"])
        assert isnan(m["auc"])
        assert isnan(m["log_loss"])
        assert isnan(m["accuracy"])

    def test_partial_metrics(self) -> None:
        meta: GameModelMetadata = _make_meta(brier=0.195, ece=0.036)
        m: dict[str, float] = extract_classification_metrics(meta)
        assert m["brier"] == 0.195
        assert m["ece"] == 0.036
        assert isnan(m["auc"])


# ---------------------------------------------------------------------------
# compare_classification_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_clear_improvement_promotes(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.040, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.195, ece=0.035, auc=0.775)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        assert result.should_promote is True
        assert result.gates["brier_improved"] is True
        assert result.gates["ece_acceptable"] is True
        assert result.gates["auc_acceptable"] is True

    def test_brier_below_threshold_rejects(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.040, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.199, ece=0.035, auc=0.775)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        assert result.should_promote is False
        assert result.gates["brier_improved"] is False

    def test_ece_degradation_rejects(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.195, ece=0.050, auc=0.775)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        assert result.should_promote is False
        assert result.gates["brier_improved"] is True
        assert result.gates["ece_acceptable"] is False

    def test_auc_degradation_rejects(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.780)
        challenger: GameModelMetadata = _make_meta(brier=0.195, ece=0.025, auc=0.760)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        assert result.should_promote is False
        assert result.gates["brier_improved"] is True
        assert result.gates["auc_acceptable"] is False

    def test_missing_ece_skips_gate(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.195, auc=0.775)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        assert result.should_promote is True
        assert result.gates["ece_acceptable"] is True

    def test_missing_auc_skips_gate(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.030)
        challenger: GameModelMetadata = _make_meta(brier=0.195, ece=0.025)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        assert result.should_promote is True
        assert result.gates["auc_acceptable"] is True

    def test_custom_criteria(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.199, ece=0.035, auc=0.755)
        strict = ClassificationPromotionGates(
            min_brier_improvement=0.005,
            max_ece_degradation=0.002,
            max_auc_degradation=0.002,
        )
        result: ClassificationComparisonResult = compare_classification_models(
            champion, challenger, criteria=strict
        )
        assert result.should_promote is False
        assert result.gates["brier_improved"] is False

    def test_exact_threshold_boundary(self) -> None:
        """Brier improvement exactly at threshold should promote."""
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.198, ece=0.030, auc=0.760)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        assert result.should_promote is True
        assert result.gates["brier_improved"] is True


# ---------------------------------------------------------------------------
# format_classification_comparison
# ---------------------------------------------------------------------------


class TestFormatComparison:
    def test_promote_contains_verdict(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.040, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.195, ece=0.035, auc=0.775)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        text: str = format_classification_comparison(result)
        assert "PROMOTE" in text
        assert "VERDICT" in text

    def test_reject_contains_verdict(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.199, ece=0.050, auc=0.740)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        text: str = format_classification_comparison(result)
        assert "REJECT" in text
        assert "VERDICT" in text

    def test_contains_all_metric_labels(self) -> None:
        champion: GameModelMetadata = _make_meta(
            brier=0.200, ece=0.040, auc=0.760, log_loss=0.58, accuracy=0.70
        )
        challenger: GameModelMetadata = _make_meta(
            brier=0.195, ece=0.035, auc=0.775, log_loss=0.57, accuracy=0.71
        )
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        text: str = format_classification_comparison(result)
        for label in ("Brier", "ECE", "AUC", "Log Loss", "Accuracy"):
            assert label in text

    def test_missing_metrics_show_na(self) -> None:
        champion: GameModelMetadata = _make_meta(brier=0.200)
        challenger: GameModelMetadata = _make_meta(brier=0.195)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        text: str = format_classification_comparison(result)
        assert "n/a" in text


# ---------------------------------------------------------------------------
# RegressionPromotionGates
# ---------------------------------------------------------------------------
class TestRegressionPromotionGatesDefaults:
    def test_default_max_mae_tolerance(self) -> None:
        g = RegressionPromotionGates()
        assert g.max_mae_tolerance == 0.0

    def test_default_min_r2(self) -> None:
        g = RegressionPromotionGates()
        assert g.min_r2 == 0.0

    def test_default_coverage_range(self) -> None:
        g = RegressionPromotionGates()
        assert g.min_coverage == 0.85
        assert g.max_coverage == 0.97


# ---------------------------------------------------------------------------
# compare_regression_models
# ---------------------------------------------------------------------------
class TestCompareRegressionModels:
    def test_clear_improvement_promotes(self) -> None:
        champ: RegressionModelResult = _make_regression(
            model_type="elasticnet", mae=58.0, r2=0.071, coverage=0.938
        )
        chall: RegressionModelResult = _make_regression(
            model_type="random_forest", mae=52.0, r2=0.15, coverage=0.92
        )
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        assert result.should_promote is True
        assert all(result.gates.values())

    def test_r2_below_zero_rejects(self) -> None:
        champ: RegressionModelResult = _make_regression(mae=58.0, r2=0.071)
        chall: RegressionModelResult = _make_regression(model_type="xgboost", mae=50.0, r2=-0.01)
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        assert result.should_promote is False
        assert result.gates["r2_above_zero"] is False

    def test_coverage_too_low_rejects(self) -> None:
        champ: RegressionModelResult = _make_regression(mae=58.0, r2=0.071, coverage=0.93)
        chall: RegressionModelResult = _make_regression(
            model_type="xgboost", mae=50.0, r2=0.15, coverage=0.80
        )
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        assert result.should_promote is False
        assert result.gates["coverage_in_range_low"] is False

    def test_coverage_too_high_rejects(self) -> None:
        champ: RegressionModelResult = _make_regression(mae=58.0, r2=0.071, coverage=0.93)
        chall: RegressionModelResult = _make_regression(
            model_type="xgboost", mae=50.0, r2=0.15, coverage=0.99
        )
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        assert result.should_promote is False
        assert result.gates["coverage_in_range_high"] is False

    def test_mae_not_improved_rejects(self) -> None:
        champ: RegressionModelResult = _make_regression(mae=50.0, r2=0.15)
        chall: RegressionModelResult = _make_regression(model_type="xgboost", mae=55.0, r2=0.20)
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        assert result.should_promote is False
        assert result.gates["mae_improved"] is False

    def test_nan_coverage_skips_gates(self) -> None:
        champ: RegressionModelResult = _make_regression(mae=58.0, r2=0.071, coverage=float("nan"))
        chall: RegressionModelResult = _make_regression(
            model_type="rf", mae=50.0, r2=0.15, coverage=float("nan")
        )
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        assert result.gates["coverage_in_range_low"] is True
        assert result.gates["coverage_in_range_high"] is True
        assert result.should_promote is True

    def test_custom_gates(self) -> None:
        champ: RegressionModelResult = _make_regression(mae=58.0, r2=0.071, coverage=0.93)
        chall: RegressionModelResult = _make_regression(
            model_type="rf", mae=55.0, r2=0.001, coverage=0.90
        )
        strict = RegressionPromotionGates(min_r2=0.05)
        result: RegressionComparisonResult = compare_regression_models(champ, chall, gates=strict)
        assert result.should_promote is False
        assert result.gates["r2_above_zero"] is False


# ---------------------------------------------------------------------------
# select_prop_champion
# ---------------------------------------------------------------------------
class TestSelectPropChampion:
    def test_selects_lowest_mae(self) -> None:
        results: list[RegressionModelResult] = [
            _make_regression(model_type="elasticnet", mae=58.0, r2=0.07, coverage=0.93),
            _make_regression(model_type="random_forest", mae=52.0, r2=0.15, coverage=0.91),
            _make_regression(model_type="xgboost", mae=54.0, r2=0.13, coverage=0.92),
        ]
        champ, summary = select_prop_champion(results)
        assert champ.model_type == "random_forest"
        assert "Champion" in summary

    def test_r2_guardrail_excludes(self) -> None:
        results: list[RegressionModelResult] = [
            _make_regression(model_type="elasticnet", mae=58.0, r2=0.07, coverage=0.93),
            _make_regression(model_type="random_forest", mae=45.0, r2=-0.02, coverage=0.91),
        ]
        champ, _ = select_prop_champion(results)
        assert champ.model_type == "elasticnet"

    def test_coverage_guardrail_excludes(self) -> None:
        results: list[RegressionModelResult] = [
            _make_regression(model_type="elasticnet", mae=58.0, r2=0.07, coverage=0.93),
            _make_regression(model_type="xgboost", mae=45.0, r2=0.20, coverage=0.80),
        ]
        champ, _ = select_prop_champion(results)
        assert champ.model_type == "elasticnet"

    def test_fallback_to_elasticnet(self) -> None:
        results: list[RegressionModelResult] = [
            _make_regression(model_type="elasticnet", mae=58.0, r2=-0.01, coverage=0.93),
            _make_regression(model_type="random_forest", mae=45.0, r2=-0.05, coverage=0.80),
            _make_regression(model_type="xgboost", mae=50.0, r2=-0.02, coverage=0.75),
        ]
        champ, summary = select_prop_champion(results)
        assert champ.model_type == "elasticnet"
        assert "Fallback" in summary

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="empty results"):
            select_prop_champion([])

    def test_summary_contains_all_models(self) -> None:
        results: list[RegressionModelResult] = [
            _make_regression(model_type="elasticnet", mae=58.0, r2=0.07, coverage=0.93),
            _make_regression(model_type="random_forest", mae=52.0, r2=0.15, coverage=0.91),
        ]
        _, summary = select_prop_champion(results)
        assert "elasticnet" in summary
        assert "random_forest" in summary


# ---------------------------------------------------------------------------
# format_regression_comparison
# ---------------------------------------------------------------------------
class TestFormatRegressionComparison:
    def test_contains_metric_labels(self) -> None:
        champ: RegressionModelResult = _make_regression(
            model_type="elasticnet", mae=58.0, r2=0.07, coverage=0.93
        )
        chall: RegressionModelResult = _make_regression(
            model_type="rf", mae=52.0, r2=0.15, coverage=0.91
        )
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        text = format_regression_comparison(result)
        for label in ("MAE", "RMSE", "R²", "Coverage"):
            assert label in text

    def test_promote_verdict(self) -> None:
        champ: RegressionModelResult = _make_regression(
            model_type="elasticnet", mae=58.0, r2=0.07, coverage=0.93
        )
        chall: RegressionModelResult = _make_regression(
            model_type="rf", mae=52.0, r2=0.15, coverage=0.91
        )
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        text: str = format_regression_comparison(result)
        assert "PROMOTE" in text

    def test_reject_verdict(self) -> None:
        champ: RegressionModelResult = _make_regression(mae=50.0, r2=0.15)
        chall: RegressionModelResult = _make_regression(model_type="xgb", mae=55.0, r2=-0.01)
        result: RegressionComparisonResult = compare_regression_models(champ, chall)
        text: str = format_regression_comparison(result)
        assert "REJECT" in text

    def test_classification_mode_unchanged(self) -> None:
        """Existing classification comparison still works after additions."""
        champion: GameModelMetadata = _make_meta(brier=0.200, ece=0.040, auc=0.760)
        challenger: GameModelMetadata = _make_meta(brier=0.195, ece=0.035, auc=0.775)
        result: ClassificationComparisonResult = compare_classification_models(champion, challenger)
        assert result.should_promote is True
        text: str = format_classification_comparison(result)
        assert "PROMOTE" in text

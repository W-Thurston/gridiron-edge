# tests/unit/evaluation/test_champion.py

"""Tests for gridiron_edge.evaluation.champion — promotion gate logic."""

from __future__ import annotations

from math import isnan

from gridiron_edge.evaluation.champion import (
    ComparisonResult,
    PromotionCriteria,
    compare_models,
    extract_metrics,
    format_comparison,
)
from gridiron_edge.models.artifact import ModelMetadata


def _make_meta(
    brier: float,
    ece: float | None = None,
    auc: float | None = None,
    log_loss: float | None = None,
    accuracy: float | None = None,
    model_version: str = "test_model",
) -> ModelMetadata:
    """Build a ModelMetadata with metrics in the parameters dict."""
    params: dict[str, object] = {}
    if ece is not None:
        params["holdout_ece"] = ece
    if auc is not None:
        params["holdout_auc"] = auc
    if log_loss is not None:
        params["holdout_log_loss"] = log_loss
    if accuracy is not None:
        params["holdout_accuracy"] = accuracy
    return ModelMetadata(
        model_version=model_version,
        trained_at="2026-06-03T00:00:00",
        schema_version=4,
        training_seasons=["2020-2021"],
        holdout_seasons=["2023-2024"],
        holdout_brier=brier,
        parameters=params,
        feature_columns=["f1", "f2"],
    )


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    def test_all_metrics_present(self) -> None:
        meta: ModelMetadata = _make_meta(
            brier=0.195, ece=0.036, auc=0.774, log_loss=0.58, accuracy=0.708
        )
        m: dict[str, float] = extract_metrics(meta)
        assert m["brier"] == 0.195
        assert m["ece"] == 0.036
        assert m["auc"] == 0.774
        assert m["log_loss"] == 0.58
        assert m["accuracy"] == 0.708

    def test_missing_metrics_are_nan(self) -> None:
        meta: ModelMetadata = _make_meta(brier=0.200)
        m: dict[str, float] = extract_metrics(meta)
        assert m["brier"] == 0.200
        assert isnan(m["ece"])
        assert isnan(m["auc"])
        assert isnan(m["log_loss"])
        assert isnan(m["accuracy"])

    def test_partial_metrics(self) -> None:
        meta: ModelMetadata = _make_meta(brier=0.195, ece=0.036)
        m: dict[str, float] = extract_metrics(meta)
        assert m["brier"] == 0.195
        assert m["ece"] == 0.036
        assert isnan(m["auc"])


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_clear_improvement_promotes(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.040, auc=0.760)
        challenger: ModelMetadata = _make_meta(brier=0.195, ece=0.035, auc=0.775)
        result: ComparisonResult = compare_models(champion, challenger)
        assert result.should_promote is True
        assert result.gates["brier_improved"] is True
        assert result.gates["ece_acceptable"] is True
        assert result.gates["auc_acceptable"] is True

    def test_brier_below_threshold_rejects(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.040, auc=0.760)
        challenger: ModelMetadata = _make_meta(brier=0.199, ece=0.035, auc=0.775)
        result: ComparisonResult = compare_models(champion, challenger)
        assert result.should_promote is False
        assert result.gates["brier_improved"] is False

    def test_ece_degradation_rejects(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.760)
        challenger: ModelMetadata = _make_meta(brier=0.195, ece=0.050, auc=0.775)
        result: ComparisonResult = compare_models(champion, challenger)
        assert result.should_promote is False
        assert result.gates["brier_improved"] is True
        assert result.gates["ece_acceptable"] is False

    def test_auc_degradation_rejects(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.780)
        challenger: ModelMetadata = _make_meta(brier=0.195, ece=0.025, auc=0.760)
        result: ComparisonResult = compare_models(champion, challenger)
        assert result.should_promote is False
        assert result.gates["brier_improved"] is True
        assert result.gates["auc_acceptable"] is False

    def test_missing_ece_skips_gate(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, auc=0.760)
        challenger: ModelMetadata = _make_meta(brier=0.195, auc=0.775)
        result: ComparisonResult = compare_models(champion, challenger)
        assert result.should_promote is True
        assert result.gates["ece_acceptable"] is True

    def test_missing_auc_skips_gate(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.030)
        challenger: ModelMetadata = _make_meta(brier=0.195, ece=0.025)
        result: ComparisonResult = compare_models(champion, challenger)
        assert result.should_promote is True
        assert result.gates["auc_acceptable"] is True

    def test_custom_criteria(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.760)
        challenger: ModelMetadata = _make_meta(brier=0.199, ece=0.035, auc=0.755)
        strict = PromotionCriteria(
            min_brier_improvement=0.005,
            max_ece_degradation=0.002,
            max_auc_degradation=0.002,
        )
        result: ComparisonResult = compare_models(champion, challenger, criteria=strict)
        assert result.should_promote is False
        assert result.gates["brier_improved"] is False

    def test_exact_threshold_boundary(self) -> None:
        """Brier improvement exactly at threshold should promote."""
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.760)
        challenger: ModelMetadata = _make_meta(brier=0.198, ece=0.030, auc=0.760)
        result: ComparisonResult = compare_models(champion, challenger)
        assert result.should_promote is True
        assert result.gates["brier_improved"] is True


# ---------------------------------------------------------------------------
# format_comparison
# ---------------------------------------------------------------------------


class TestFormatComparison:
    def test_promote_contains_verdict(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.040, auc=0.760)
        challenger: ModelMetadata = _make_meta(brier=0.195, ece=0.035, auc=0.775)
        result: ComparisonResult = compare_models(champion, challenger)
        text: str = format_comparison(result)
        assert "PROMOTE" in text
        assert "VERDICT" in text

    def test_reject_contains_verdict(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200, ece=0.030, auc=0.760)
        challenger: ModelMetadata = _make_meta(brier=0.199, ece=0.050, auc=0.740)
        result: ComparisonResult = compare_models(champion, challenger)
        text: str = format_comparison(result)
        assert "REJECT" in text
        assert "VERDICT" in text

    def test_contains_all_metric_labels(self) -> None:
        champion: ModelMetadata = _make_meta(
            brier=0.200, ece=0.040, auc=0.760, log_loss=0.58, accuracy=0.70
        )
        challenger: ModelMetadata = _make_meta(
            brier=0.195, ece=0.035, auc=0.775, log_loss=0.57, accuracy=0.71
        )
        result: ComparisonResult = compare_models(champion, challenger)
        text: str = format_comparison(result)
        for label in ("Brier", "ECE", "AUC", "Log Loss", "Accuracy"):
            assert label in text

    def test_missing_metrics_show_na(self) -> None:
        champion: ModelMetadata = _make_meta(brier=0.200)
        challenger: ModelMetadata = _make_meta(brier=0.195)
        result: ComparisonResult = compare_models(champion, challenger)
        text: str = format_comparison(result)
        assert "n/a" in text

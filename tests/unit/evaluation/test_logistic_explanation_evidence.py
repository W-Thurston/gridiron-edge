"""Tests for immutable Logistic explanation evidence contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from gridiron_edge.evaluation.logistic_explanation_evidence import (
    LOGISTIC_EXPLANATION_EVIDENCE_SCHEMA_VERSION,
    LOGISTIC_EXPLANATION_RECONCILIATION_TOLERANCE,
    LogisticExplanationEvent,
    LogisticFeatureContribution,
    create_logistic_explanation_batch,
    logistic_explanation_batch_payload,
    sigmoid,
    validate_logistic_explanation_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema

GENERATED_AT = datetime(2026, 9, 26, 12, tzinfo=UTC)
RUN_ID = "run-123"
EVIDENCE_ID = "e" * 64
MODEL_DIGEST = "a" * 64
SCALER_DIGEST = "b" * 64
FEATURE_NAMES = ("ELO_DIFF", "OFF_EPA_PER_PLAY_DIFF")


def _feature_schema():
    return create_prediction_feature_schema(
        model_name="win_prob",
        model_type="logistic",
        task="classification",
        modeling_schema_version=5,
        epa_window=6,
        feature_set_name="combined_111",
        ordered_columns=FEATURE_NAMES,
    )


def _event(
    *,
    event_id: str = "event-1",
    game_id: str = "game-1",
    intercept: float = 0.1,
    transformed_values: tuple[float, ...] = (0.5, -0.25),
    coefficients: tuple[float, ...] = (0.8, -0.4),
    raw_estimator_output: float | None = None,
    tolerance: float = LOGISTIC_EXPLANATION_RECONCILIATION_TOLERANCE,
) -> LogisticExplanationEvent:
    contributions = tuple(
        LogisticFeatureContribution(
            feature_name=name,
            transformed_value=value,
            coefficient=coefficient,
            contribution=coefficient * value,
        )
        for name, value, coefficient in zip(
            FEATURE_NAMES, transformed_values, coefficients, strict=True
        )
    )
    reconstructed_log_odds = intercept + sum(item.contribution for item in contributions)
    reconstructed_probability = sigmoid(reconstructed_log_odds)
    resolved_raw = (
        reconstructed_probability if raw_estimator_output is None else raw_estimator_output
    )
    return LogisticExplanationEvent(
        event_id=event_id,
        game_id=game_id,
        intercept=intercept,
        contributions=contributions,
        reconstructed_log_odds=reconstructed_log_odds,
        reconstructed_probability=reconstructed_probability,
        raw_estimator_output=resolved_raw,
        tolerance=tolerance,
    )


def _batch(events=None):
    return create_logistic_explanation_batch(
        run_id=RUN_ID,
        evidence_id=EVIDENCE_ID,
        model_content_digest=MODEL_DIGEST,
        scaler_content_digest=SCALER_DIGEST,
        feature_schema=_feature_schema(),
        generated_at=GENERATED_AT,
        events=events if events is not None else (_event(),),
    )


class TestCreateLogisticExplanationBatch:
    def test_creates_valid_batch(self) -> None:
        batch = _batch()
        assert batch.schema_version == LOGISTIC_EXPLANATION_EVIDENCE_SCHEMA_VERSION
        assert batch.run_id == RUN_ID
        assert batch.evidence_id == EVIDENCE_ID
        assert len(batch.batch_id) == 64
        validate_logistic_explanation_batch(batch)

    def test_deterministic_identity(self) -> None:
        first = _batch()
        second = _batch()
        assert first.batch_id == second.batch_id

    def test_identity_changes_with_content(self) -> None:
        first = _batch()
        second = _batch(events=(_event(event_id="event-2"),))
        assert first.batch_id != second.batch_id

    def test_payload_is_json_stable(self) -> None:
        batch = _batch()
        payload = logistic_explanation_batch_payload(batch)
        assert payload["batch_id"] == batch.batch_id
        assert payload["events"][0]["event_id"] == "event-1"


class TestValidateLogisticExplanationBatch:
    def test_rejects_unsupported_schema_version(self) -> None:
        batch = replace(_batch(), schema_version=99)
        with pytest.raises(ValueError, match="schema_version"):
            validate_logistic_explanation_batch(batch)

    def test_rejects_tampered_batch_id(self) -> None:
        batch = replace(_batch(), batch_id="f" * 64)
        with pytest.raises(ValueError, match="batch_id"):
            validate_logistic_explanation_batch(batch)

    def test_rejects_tampered_run_id(self) -> None:
        batch = replace(_batch(), run_id="tampered")
        with pytest.raises(ValueError, match="batch_id"):
            validate_logistic_explanation_batch(batch)

    def test_rejects_empty_events(self) -> None:
        with pytest.raises(ValueError, match="events must not be empty"):
            _batch(events=())

    def test_rejects_duplicate_event_ids(self) -> None:
        with pytest.raises(ValueError, match="duplicate event IDs"):
            _batch(events=(_event(), _event(game_id="game-2")))

    def test_rejects_out_of_order_events(self) -> None:
        with pytest.raises(ValueError, match="ordered"):
            _batch(
                events=(
                    _event(event_id="event-2", game_id="game-2"),
                    _event(event_id="event-1", game_id="game-1"),
                )
            )

    def test_rejects_feature_schema_for_wrong_model_type(self) -> None:
        wrong_schema = create_prediction_feature_schema(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            modeling_schema_version=5,
            epa_window=8,
            feature_set_name="combined_111",
            ordered_columns=FEATURE_NAMES,
        )
        with pytest.raises(ValueError, match="win_prob/logistic"):
            create_logistic_explanation_batch(
                run_id=RUN_ID,
                evidence_id=EVIDENCE_ID,
                model_content_digest=MODEL_DIGEST,
                scaler_content_digest=SCALER_DIGEST,
                feature_schema=wrong_schema,
                generated_at=GENERATED_AT,
                events=(_event(),),
            )

    def test_rejects_contribution_feature_mismatch(self) -> None:
        bad_event = replace(
            _event(),
            contributions=(
                LogisticFeatureContribution(
                    feature_name="UNKNOWN_FEATURE",
                    transformed_value=0.5,
                    coefficient=0.8,
                    contribution=0.4,
                ),
                LogisticFeatureContribution(
                    feature_name="OFF_EPA_PER_PLAY_DIFF",
                    transformed_value=-0.25,
                    coefficient=-0.4,
                    contribution=0.1,
                ),
            ),
        )
        with pytest.raises(ValueError, match="ordered_columns"):
            _batch(events=(bad_event,))

    def test_rejects_contribution_arithmetic_mismatch(self) -> None:
        event = _event()
        tampered_contribution = replace(event.contributions[0], contribution=999.0)
        bad_event = replace(
            event,
            contributions=(tampered_contribution, event.contributions[1]),
        )
        with pytest.raises(ValueError, match="coefficient \\* transformed_value"):
            _batch(events=(bad_event,))

    def test_rejects_reconstructed_log_odds_mismatch(self) -> None:
        bad_event = replace(_event(), reconstructed_log_odds=999.0)
        with pytest.raises(ValueError, match="reconstructed_log_odds"):
            _batch(events=(bad_event,))

    def test_rejects_reconstructed_probability_mismatch(self) -> None:
        bad_event = replace(_event(), reconstructed_probability=0.999999)
        with pytest.raises(ValueError, match="reconstructed_probability"):
            _batch(events=(bad_event,))

    def test_rejects_reconciliation_violation(self) -> None:
        bad_event = _event(raw_estimator_output=0.01)
        with pytest.raises(ValueError, match="reconcile"):
            _batch(events=(bad_event,))

    def test_accepts_reconciliation_within_tolerance(self) -> None:
        event = _event()
        nudged = replace(
            event,
            raw_estimator_output=event.reconstructed_probability + 5e-10,
        )
        batch = _batch(events=(nudged,))
        validate_logistic_explanation_batch(batch)

    def test_rejects_nonpositive_tolerance(self) -> None:
        bad_event = replace(_event(), tolerance=0.0)
        with pytest.raises(ValueError, match="tolerance"):
            _batch(events=(bad_event,))


class TestSigmoid:
    def test_matches_naive_formula_for_moderate_values(self) -> None:
        import math

        for x in (-5.0, -1.0, 0.0, 1.0, 5.0):
            assert sigmoid(x) == pytest.approx(1.0 / (1.0 + math.exp(-x)))

    def test_stable_for_large_magnitude_values(self) -> None:
        assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)
        assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-12)

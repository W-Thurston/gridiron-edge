"""Tests for immutable Logistic explanation evidence persistence."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from gridiron_edge.evaluation.logistic_explanation_evidence import (
    LOGISTIC_EXPLANATION_RECONCILIATION_TOLERANCE,
    LogisticExplanationEvent,
    LogisticFeatureContribution,
    create_logistic_explanation_batch,
    sigmoid,
)
from gridiron_edge.evaluation.logistic_explanation_evidence_store import (
    find_logistic_explanation_by_event,
    list_logistic_explanation_batches,
    logistic_explanation_batch_path,
    read_logistic_explanation_batch,
    write_logistic_explanation_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema

GENERATED_AT = datetime(2026, 9, 26, 12, tzinfo=UTC)
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


def _event(*, event_id: str = "event-1", game_id: str = "game-1") -> LogisticExplanationEvent:
    intercept = 0.1
    transformed_values = (0.5, -0.25)
    coefficients = (0.8, -0.4)
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
    return LogisticExplanationEvent(
        event_id=event_id,
        game_id=game_id,
        intercept=intercept,
        contributions=contributions,
        reconstructed_log_odds=reconstructed_log_odds,
        reconstructed_probability=reconstructed_probability,
        raw_estimator_output=reconstructed_probability,
        tolerance=LOGISTIC_EXPLANATION_RECONCILIATION_TOLERANCE,
    )


def _batch(*, run_id: str = "run-123", events=None):
    return create_logistic_explanation_batch(
        run_id=run_id,
        evidence_id=EVIDENCE_ID,
        model_content_digest=MODEL_DIGEST,
        scaler_content_digest=SCALER_DIGEST,
        feature_schema=_feature_schema(),
        generated_at=GENERATED_AT,
        events=events if events is not None else (_event(),),
    )


def _temporary_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.name.startswith(".") and path.is_file()]


class TestLogisticExplanationStore:
    def test_write_read_round_trip_and_exact_replay(self, tmp_path: Path) -> None:
        batch = _batch()

        first = write_logistic_explanation_batch(batch, repo=tmp_path)
        second = write_logistic_explanation_batch(batch, repo=tmp_path)

        assert first == second
        assert read_logistic_explanation_batch(first) == batch
        assert _temporary_files(tmp_path) == []

    def test_existing_conflicting_content_is_rejected_without_overwrite(
        self, tmp_path: Path
    ) -> None:
        batch = _batch()
        path = logistic_explanation_batch_path(batch.batch_id, repo=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("conflict", encoding="utf-8")

        with pytest.raises(ValueError, match="cannot be reused"):
            write_logistic_explanation_batch(batch, repo=tmp_path)

        assert path.read_text(encoding="utf-8") == "conflict"
        assert _temporary_files(tmp_path) == []

    def test_rejects_missing_and_unexpected_artifact_keys(self, tmp_path: Path) -> None:
        batch = _batch()
        path = write_logistic_explanation_batch(batch, repo=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="keys do not match"):
            read_logistic_explanation_batch(path)

    def test_rejects_unsupported_store_schema(self, tmp_path: Path) -> None:
        batch = _batch()
        path = write_logistic_explanation_batch(batch, repo=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["store_schema_version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=r"Unsupported.*store schema"):
            read_logistic_explanation_batch(path)

    def test_rejects_tampered_embedded_batch(self, tmp_path: Path) -> None:
        batch = _batch()
        path = write_logistic_explanation_batch(batch, repo=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["batch"]["run_id"] = "tampered-run"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="does not match canonical"):
            read_logistic_explanation_batch(path)

    def test_rejects_noncanonical_path(self, tmp_path: Path) -> None:
        batch = _batch()
        canonical = write_logistic_explanation_batch(batch, repo=tmp_path)
        moved = canonical.with_name(f"{MODEL_DIGEST}.json")
        moved.write_bytes(canonical.read_bytes())

        with pytest.raises(ValueError, match="path and embedded identity disagree"):
            read_logistic_explanation_batch(moved)

    def test_malformed_json_fails_explicitly(self, tmp_path: Path) -> None:
        path = logistic_explanation_batch_path(MODEL_DIGEST, repo=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{malformed", encoding="utf-8")

        with pytest.raises(ValueError, match="malformed JSON"):
            read_logistic_explanation_batch(path)


class TestLogisticExplanationLookup:
    def test_listing_is_deterministic(self, tmp_path: Path) -> None:
        first = _batch(run_id="run-a")
        second = _batch(
            run_id="run-b",
            events=(_event(event_id="event-2", game_id="game-2"),),
        )
        write_logistic_explanation_batch(first, repo=tmp_path)
        write_logistic_explanation_batch(second, repo=tmp_path)

        batches = list_logistic_explanation_batches(repo=tmp_path)

        assert [batch.batch_id for batch in batches] == sorted([first.batch_id, second.batch_id])

    def test_event_lookup_returns_exact_event_or_none(self, tmp_path: Path) -> None:
        batch = _batch()
        write_logistic_explanation_batch(batch, repo=tmp_path)

        found = find_logistic_explanation_by_event("event-1", repo=tmp_path)
        missing = find_logistic_explanation_by_event("nonexistent", repo=tmp_path)

        assert found == batch.events[0]
        assert missing is None

    def test_event_lookup_rejects_ambiguous_claims(self, tmp_path: Path) -> None:
        first = _batch(run_id="run-a")
        second = _batch(run_id="run-b")
        write_logistic_explanation_batch(first, repo=tmp_path)
        write_logistic_explanation_batch(second, repo=tmp_path)

        with pytest.raises(ValueError, match="Multiple logistic explanation batches"):
            find_logistic_explanation_by_event("event-1", repo=tmp_path)

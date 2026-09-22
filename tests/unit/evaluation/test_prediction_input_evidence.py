# tests/unit/evaluation/test_prediction_input_evidence.py
"""Tests for immutable live prediction-input evidence contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from gridiron_edge.evaluation.forecast_store import FORECAST_EVENT_COLUMNS
from gridiron_edge.evaluation.prediction_input_evidence import (
    PREDICTION_INPUT_EVIDENCE_SCHEMA_VERSION,
    BinaryArtifactReference,
    CalibrationResolutionSource,
    EloPredictionEventEvidence,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionExecutionKind,
    PredictionInputEvidence,
    PredictionPostProcessingEvidence,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    StatisticalPredictionEventEvidence,
    authenticate_prediction_input_evidence,
    create_elo_prediction_input_evidence,
    create_prediction_feature_schema,
    create_statistical_prediction_input_evidence,
    prediction_feature_schema_id,
    prediction_input_evidence_payload,
    require_same_source_artifacts,
    validate_prediction_input_evidence,
)

GENERATED_AT = datetime(2026, 9, 17, 12, tzinfo=UTC)
REGISTRY_UPDATED_AT = datetime(2026, 9, 16, 20, tzinfo=UTC)
SEASON = "2026-2027"
WEEK = 2
COMMIT = "a" * 40
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _revision() -> SourceRevision:
    return SourceRevision(commit=COMMIT, tracked_worktree_clean=True)


def _source_artifacts() -> tuple[SourceArtifactReference, ...]:
    return (
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_Team_Elo.csv",
            state=PredictionSourceState.PRESENT,
            content_digest=DIGEST_A,
            size_bytes=100,
        ),
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_upcoming_schedule_rich.parquet",
            state=PredictionSourceState.PRESENT,
            content_digest=DIGEST_B,
            size_bytes=200,
        ),
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_wk_by_wk_w_weather.csv",
            state=PredictionSourceState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
        SourceArtifactReference(
            relative_path="data/output/calibration/game_model_calibration.json",
            state=PredictionSourceState.PRESENT,
            content_digest=DIGEST_C,
            size_bytes=300,
        ),
    )


def _statistical_artifacts(
    *, scaler_state: PredictionArtifactState = PredictionArtifactState.ABSENT
) -> tuple[BinaryArtifactReference, ...]:
    scaler_digest = DIGEST_D if scaler_state is PredictionArtifactState.PRESENT else None
    scaler_size = 40 if scaler_state is PredictionArtifactState.PRESENT else None
    return (
        BinaryArtifactReference(
            kind=PredictionArtifactKind.EXTERNAL_CALIBRATOR,
            source_relative_path="data/models/win_prob/random_forest/calibrator.joblib",
            state=PredictionArtifactState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.MODEL,
            source_relative_path="data/models/win_prob/random_forest/model.joblib",
            state=PredictionArtifactState.PRESENT,
            content_digest=DIGEST_A,
            size_bytes=10,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.MODEL_METADATA,
            source_relative_path="data/models/win_prob/random_forest/metadata.json",
            state=PredictionArtifactState.PRESENT,
            content_digest=DIGEST_B,
            size_bytes=20,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.SCALER,
            source_relative_path="data/models/win_prob/random_forest/scaler.joblib",
            state=scaler_state,
            content_digest=scaler_digest,
            size_bytes=scaler_size,
        ),
    )


def _elo_artifacts() -> tuple[BinaryArtifactReference, ...]:
    return (
        BinaryArtifactReference(
            kind=PredictionArtifactKind.ELO_LINEAGE,
            source_relative_path="data/cleaned/NFL_Team_Elo.metadata.json",
            state=PredictionArtifactState.PRESENT,
            content_digest=DIGEST_A,
            size_bytes=100,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.ELO_STATE,
            source_relative_path="data/cleaned/NFL_Team_Elo.csv",
            state=PredictionArtifactState.PRESENT,
            content_digest=DIGEST_B,
            size_bytes=200,
        ),
    )


def _feature_schema():
    return create_prediction_feature_schema(
        model_name="win_prob",
        model_type="random_forest",
        task="classification",
        modeling_schema_version=5,
        feature_set_name="expanded_current",
        ordered_columns=("ELO_DIFF", "HOME_ELO"),
    )


def _post_processing() -> PredictionPostProcessingEvidence:
    return PredictionPostProcessingEvidence(
        registry_reference=SourceArtifactReference(
            relative_path="data/output/calibration/game_model_calibration.json",
            state=PredictionSourceState.PRESENT,
            content_digest=DIGEST_C,
            size_bytes=300,
        ),
        registry_entry_updated_at=REGISTRY_UPDATED_AT,
        sigma=11.446,
        sigma_source=CalibrationResolutionSource.PERSISTED_REGISTRY,
        margin_std=13.7244,
        margin_std_source=CalibrationResolutionSource.PERSISTED_REGISTRY,
        external_calibrator_state=PredictionArtifactState.ABSENT,
        embedded_estimator_calibration=True,
    )


def _statistical_events() -> tuple[StatisticalPredictionEventEvidence, ...]:
    return (
        StatisticalPredictionEventEvidence(
            event_id="event-1",
            game_id="2026_02_A_B",
            raw_feature_values=(10.0, 1510.0),
            transformed_feature_values=(10.0, 1510.0),
            raw_estimator_output=0.55,
            post_estimator_output=0.55,
            final_outputs=(
                ("away_win_prob", 0.45),
                ("home_win_prob", 0.55),
                ("model_spread", -1.44),
            ),
        ),
        StatisticalPredictionEventEvidence(
            event_id="event-2",
            game_id="2026_02_C_D",
            raw_feature_values=(-20.0, 1490.0),
            transformed_feature_values=(-20.0, 1490.0),
            raw_estimator_output=0.40,
            post_estimator_output=0.40,
            final_outputs=(
                ("away_win_prob", 0.60),
                ("home_win_prob", 0.40),
                ("model_spread", 2.89),
            ),
        ),
    )


def _elo_events() -> tuple[EloPredictionEventEvidence, ...]:
    return (
        EloPredictionEventEvidence(
            event_id="elo-event-1",
            game_id="2026_02_A_B",
            season=SEASON,
            week=WEEK,
            away_team="Away A",
            home_team="Home B",
            away_elo=1500.0,
            home_elo=1510.0,
            formula_id="elo_win_probability_v1",
            divisor=480.0,
            away_win_probability=0.4880099837,
            home_win_probability=0.5119900163,
            final_outputs=(
                ("away_elo", 1500.0),
                ("away_win_prob", 0.4880099837),
                ("home_elo", 1510.0),
                ("home_win_prob", 0.5119900163),
            ),
        ),
        EloPredictionEventEvidence(
            event_id="elo-event-2",
            game_id="2026_02_C_D",
            season=SEASON,
            week=WEEK,
            away_team="Away C",
            home_team="Home D",
            away_elo=1520.0,
            home_elo=1480.0,
            formula_id="elo_win_probability_v1",
            divisor=480.0,
            away_win_probability=0.5478354290,
            home_win_probability=0.4521645710,
            final_outputs=(
                ("away_elo", 1520.0),
                ("away_win_prob", 0.5478354290),
                ("home_elo", 1480.0),
                ("home_win_prob", 0.4521645710),
            ),
        ),
    )


def _statistical_evidence() -> PredictionInputEvidence:
    return create_statistical_prediction_input_evidence(
        run_id="run-1",
        season=SEASON,
        week=WEEK,
        generated_at=GENERATED_AT,
        model_name="win_prob",
        model_type="random_forest",
        source_revision=_revision(),
        source_artifacts=_source_artifacts(),
        binary_artifacts=_statistical_artifacts(),
        feature_schema=_feature_schema(),
        post_processing=_post_processing(),
        events=_statistical_events(),
    )


def _elo_evidence() -> PredictionInputEvidence:
    return create_elo_prediction_input_evidence(
        run_id="run-elo",
        season=SEASON,
        week=WEEK,
        generated_at=GENERATED_AT,
        source_revision=_revision(),
        source_artifacts=_source_artifacts(),
        binary_artifacts=_elo_artifacts(),
        events=_elo_events(),
    )


def _forecast_events(evidence: PredictionInputEvidence) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_events = evidence.statistical_events or evidence.elo_events
    for event in source_events:
        values: dict[str, object] = {
            "event_id": event.event_id,
            "run_id": evidence.run_id,
            "role": "live",
            "generated_at": evidence.generated_at,
            "season": evidence.season,
            "week": evidence.week,
            "game_id": event.game_id,
            "model_name": evidence.model_name,
            "model_type": evidence.model_type,
            "game_date": "2026-09-17",
            "away_team": "Away",
            "home_team": "Home",
            "away_elo": None,
            "home_elo": None,
            "away_win_prob": None,
            "home_win_prob": None,
            "model_spread": None,
            "model_total": None,
            "projected_home_score": None,
            "projected_away_score": None,
            "margin_std": None,
            "win_prob_lo": None,
            "win_prob_hi": None,
            "confidence_tier": None,
        }
        for name, value in event.final_outputs:
            values[name] = value
        rows.append({column: values[column] for column in FORECAST_EVENT_COLUMNS})
    return pd.DataFrame(rows, columns=FORECAST_EVENT_COLUMNS)


class TestFeatureSchema:
    def test_identity_preserves_semantic_column_order(self) -> None:
        first = _feature_schema()
        reversed_id = prediction_feature_schema_id(
            model_name=first.model_name,
            model_type=first.model_type,
            task=first.task,
            modeling_schema_version=first.modeling_schema_version,
            feature_set_name=first.feature_set_name,
            ordered_columns=tuple(reversed(first.ordered_columns)),
        )

        assert first.schema_id != reversed_id

    def test_duplicate_columns_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="nonempty and unique"):
            create_prediction_feature_schema(
                model_name="win_prob",
                model_type="random_forest",
                task="classification",
                modeling_schema_version=5,
                feature_set_name="features",
                ordered_columns=("ELO_DIFF", "ELO_DIFF"),
            )


class TestStatisticalEvidence:
    def test_creates_valid_content_addressed_family(self) -> None:
        evidence = _statistical_evidence()

        assert evidence.schema_version == PREDICTION_INPUT_EVIDENCE_SCHEMA_VERSION
        assert len(evidence.evidence_id) == 64
        assert evidence.execution_kind is PredictionExecutionKind.PERSISTED_ESTIMATOR
        assert evidence.feature_schema == _feature_schema()
        assert len(evidence.statistical_events) == 2
        validate_prediction_input_evidence(evidence)

    def test_identity_changes_when_event_output_changes(self) -> None:
        original = _statistical_evidence()
        changed_event = replace(
            original.statistical_events[0],
            final_outputs=(
                ("away_win_prob", 0.44),
                ("home_win_prob", 0.56),
                ("model_spread", -1.44),
            ),
        )
        changed = create_statistical_prediction_input_evidence(
            run_id=original.run_id,
            season=original.season,
            week=original.week,
            generated_at=original.generated_at,
            model_name=original.model_name,
            model_type=original.model_type,
            source_revision=original.source_revision,
            source_artifacts=original.source_artifacts,
            binary_artifacts=original.binary_artifacts,
            feature_schema=original.feature_schema,
            post_processing=original.post_processing,
            events=(changed_event, original.statistical_events[1]),
        )

        assert changed.evidence_id != original.evidence_id

    def test_no_scaler_requires_equal_raw_and_transformed_values(self) -> None:
        changed_event = replace(
            _statistical_events()[0],
            transformed_feature_values=(0.1, 0.2),
        )

        with pytest.raises(ValueError, match="scaler is absent"):
            create_statistical_prediction_input_evidence(
                run_id="run-1",
                season=SEASON,
                week=WEEK,
                generated_at=GENERATED_AT,
                model_name="win_prob",
                model_type="random_forest",
                source_revision=_revision(),
                source_artifacts=_source_artifacts(),
                binary_artifacts=_statistical_artifacts(),
                feature_schema=_feature_schema(),
                post_processing=_post_processing(),
                events=(changed_event, _statistical_events()[1]),
            )

    def test_present_scaler_allows_transformed_values(self) -> None:
        schema = create_prediction_feature_schema(
            model_name="win_prob",
            model_type="random_forest",
            task="classification",
            modeling_schema_version=5,
            feature_set_name="features",
            ordered_columns=("ELO_DIFF", "HOME_ELO"),
        )
        transformed = tuple(
            replace(event, transformed_feature_values=(0.1, 0.2)) for event in _statistical_events()
        )

        evidence = create_statistical_prediction_input_evidence(
            run_id="run-1",
            season=SEASON,
            week=WEEK,
            generated_at=GENERATED_AT,
            model_name="win_prob",
            model_type="random_forest",
            source_revision=_revision(),
            source_artifacts=_source_artifacts(),
            binary_artifacts=_statistical_artifacts(scaler_state=PredictionArtifactState.PRESENT),
            feature_schema=schema,
            post_processing=_post_processing(),
            events=transformed,
        )

        validate_prediction_input_evidence(evidence)

    def test_missing_required_model_artifact_is_rejected(self) -> None:
        artifacts = tuple(
            replace(
                reference,
                state=PredictionArtifactState.ABSENT,
                content_digest=None,
                size_bytes=None,
            )
            if reference.kind is PredictionArtifactKind.MODEL
            else reference
            for reference in _statistical_artifacts()
        )

        with pytest.raises(ValueError, match="model artifact must be present"):
            create_statistical_prediction_input_evidence(
                run_id="run-1",
                season=SEASON,
                week=WEEK,
                generated_at=GENERATED_AT,
                model_name="win_prob",
                model_type="random_forest",
                source_revision=_revision(),
                source_artifacts=_source_artifacts(),
                binary_artifacts=artifacts,
                feature_schema=_feature_schema(),
                post_processing=_post_processing(),
                events=_statistical_events(),
            )

    def test_nonfinite_feature_value_is_rejected(self) -> None:
        event = replace(
            _statistical_events()[0],
            raw_feature_values=(float("nan"), 1510.0),
            transformed_feature_values=(float("nan"), 1510.0),
        )

        with pytest.raises(ValueError, match="must be finite"):
            create_statistical_prediction_input_evidence(
                run_id="run-1",
                season=SEASON,
                week=WEEK,
                generated_at=GENERATED_AT,
                model_name="win_prob",
                model_type="random_forest",
                source_revision=_revision(),
                source_artifacts=_source_artifacts(),
                binary_artifacts=_statistical_artifacts(),
                feature_schema=_feature_schema(),
                post_processing=_post_processing(),
                events=(event, _statistical_events()[1]),
            )


class TestEloEvidence:
    def test_creates_formula_evidence_without_estimator_contract(self) -> None:
        evidence = _elo_evidence()

        assert evidence.execution_kind is PredictionExecutionKind.ELO_FORMULA
        assert evidence.model_name == "win_prob"
        assert evidence.model_type == "elo"
        assert evidence.feature_schema is None
        assert evidence.post_processing is None
        assert evidence.statistical_events == ()
        validate_prediction_input_evidence(evidence)

    def test_noncomplementary_probabilities_are_rejected(self) -> None:
        event = replace(_elo_events()[0], home_win_probability=0.50)

        with pytest.raises(ValueError, match="must be complementary"):
            create_elo_prediction_input_evidence(
                run_id="run-elo",
                season=SEASON,
                week=WEEK,
                generated_at=GENERATED_AT,
                source_revision=_revision(),
                source_artifacts=_source_artifacts(),
                binary_artifacts=_elo_artifacts(),
                events=(event, _elo_events()[1]),
            )

    def test_mixed_formula_identity_is_rejected(self) -> None:
        event = replace(_elo_events()[1], formula_id="different_formula")

        with pytest.raises(ValueError, match="one formula identity"):
            create_elo_prediction_input_evidence(
                run_id="run-elo",
                season=SEASON,
                week=WEEK,
                generated_at=GENERATED_AT,
                source_revision=_revision(),
                source_artifacts=_source_artifacts(),
                binary_artifacts=_elo_artifacts(),
                events=(_elo_events()[0], event),
            )

    def test_statistical_artifact_is_rejected_for_elo(self) -> None:
        artifacts = tuple(
            sorted(
                (*_elo_artifacts(), _statistical_artifacts()[1]),
                key=lambda reference: reference.kind.value,
            )
        )

        with pytest.raises(ValueError, match="exactly elo_lineage and elo_state"):
            create_elo_prediction_input_evidence(
                run_id="run-elo",
                season=SEASON,
                week=WEEK,
                generated_at=GENERATED_AT,
                source_revision=_revision(),
                source_artifacts=_source_artifacts(),
                binary_artifacts=artifacts,
                events=_elo_events(),
            )


class TestSharedValidation:
    def test_dirty_source_revision_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="clean tracked worktree"):
            create_elo_prediction_input_evidence(
                run_id="run-elo",
                season=SEASON,
                week=WEEK,
                generated_at=GENERATED_AT,
                source_revision=SourceRevision(
                    commit=COMMIT,
                    tracked_worktree_clean=False,
                ),
                source_artifacts=_source_artifacts(),
                binary_artifacts=_elo_artifacts(),
                events=_elo_events(),
            )

    def test_unsafe_source_path_is_rejected(self) -> None:
        sources = (
            SourceArtifactReference(
                relative_path="../outside.csv",
                state=PredictionSourceState.PRESENT,
                content_digest=DIGEST_A,
                size_bytes=10,
            ),
        )

        with pytest.raises(ValueError, match="safe repository-relative path"):
            create_elo_prediction_input_evidence(
                run_id="run-elo",
                season=SEASON,
                week=WEEK,
                generated_at=GENERATED_AT,
                source_revision=_revision(),
                source_artifacts=sources,
                binary_artifacts=_elo_artifacts(),
                events=_elo_events(),
            )

    def test_absent_source_cannot_claim_digest(self) -> None:
        source = SourceArtifactReference(
            relative_path="data/cleaned/optional.csv",
            state=PredictionSourceState.ABSENT,
            content_digest=DIGEST_A,
            size_bytes=None,
        )

        with pytest.raises(ValueError, match="must not contain digest or size"):
            require_same_source_artifacts((source,), (source,))

    def test_source_artifact_drift_is_rejected(self) -> None:
        before = _source_artifacts()
        changed = replace(before[0], content_digest=DIGEST_D)
        after = (changed, *before[1:])

        with pytest.raises(ValueError, match="changed during execution"):
            require_same_source_artifacts(before, after)

    def test_evidence_id_tamper_is_rejected(self) -> None:
        evidence = replace(_elo_evidence(), evidence_id=DIGEST_D)

        with pytest.raises(ValueError, match="does not match canonical"):
            validate_prediction_input_evidence(evidence)

    def test_payload_is_stable_and_json_compatible(self) -> None:
        evidence = _statistical_evidence()

        first = prediction_input_evidence_payload(evidence)
        second = prediction_input_evidence_payload(evidence)

        assert first == second
        assert first["evidence_id"] == evidence.evidence_id
        assert first["execution_kind"] == "persisted_estimator"


class TestEventAuthentication:
    @pytest.mark.parametrize("factory", [_statistical_evidence, _elo_evidence])
    def test_authenticates_complete_exact_live_family(self, factory) -> None:
        evidence = factory()

        authenticate_prediction_input_evidence(
            evidence,
            forecast_events=_forecast_events(evidence),
        )

    def test_missing_event_is_rejected(self) -> None:
        evidence = _statistical_evidence()
        events = _forecast_events(evidence).iloc[:1].copy()

        with pytest.raises(ValueError, match="coverage does not match"):
            authenticate_prediction_input_evidence(evidence, forecast_events=events)

    def test_extra_family_event_is_rejected(self) -> None:
        evidence = _statistical_evidence()
        events = _forecast_events(evidence)
        extra = events.iloc[[0]].copy()
        extra["event_id"] = "event-extra"
        extra["game_id"] = "2026_02_E_F"
        combined = pd.concat([events, extra], ignore_index=True)

        with pytest.raises(ValueError, match="coverage does not match"):
            authenticate_prediction_input_evidence(evidence, forecast_events=combined)

    def test_output_tamper_is_rejected(self) -> None:
        evidence = _statistical_evidence()
        events = _forecast_events(evidence)
        events.loc[events["event_id"] == "event-1", "home_win_prob"] = 0.56

        with pytest.raises(ValueError, match="output does not match"):
            authenticate_prediction_input_evidence(evidence, forecast_events=events)

    def test_backfilled_role_does_not_authenticate_as_live(self) -> None:
        evidence = _elo_evidence()
        events = _forecast_events(evidence)
        events["role"] = "backfilled"

        with pytest.raises(ValueError, match="coverage does not match"):
            authenticate_prediction_input_evidence(evidence, forecast_events=events)

    def test_authentication_compares_event_identity_independent_of_game_order(
        self,
    ) -> None:
        evidence = _elo_evidence()
        events = _forecast_events(evidence).iloc[::-1].reset_index(drop=True)

        authenticate_prediction_input_evidence(
            evidence,
            forecast_events=events,
        )

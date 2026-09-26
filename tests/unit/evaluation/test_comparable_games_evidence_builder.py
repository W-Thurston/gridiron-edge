"""Tests for building and persisting comparable-games retrieval evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from gridiron_edge.evaluation.comparable_games_corpus import (
    ComparableGamesFrameReference,
    create_comparable_games_corpus,
    frame_content_digest,
)
from gridiron_edge.evaluation.comparable_games_corpus_store import (
    write_comparable_games_corpus,
)
from gridiron_edge.evaluation.comparable_games_evidence_builder import (
    build_and_write_comparable_games_batches,
)
from gridiron_edge.evaluation.comparable_games_evidence_store import (
    find_comparable_games_by_event,
)
from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    CalibrationResolutionSource,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionPostProcessingEvidence,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    StatisticalPredictionEventEvidence,
    create_prediction_feature_schema,
    create_statistical_prediction_input_evidence,
)
from gridiron_edge.evaluation.prediction_input_evidence_store import (
    write_prediction_input_evidence,
)

GENERATED_AT = datetime(2026, 9, 26, 12, tzinfo=UTC)
SEASON = "2026-2027"
WEEK = 2
COMMIT = "a" * 40
FEATURE_NAMES = ("ELO_DIFF", "OFF_EPA_PER_PLAY_DIFF")
MODEL_DIGEST = "a" * 64
SCALER_DIGEST = "b" * 64
COLUMNS = (
    "game_id",
    "season",
    "week",
    "game_date",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "favorite_team",
    "spread_magnitude",
    *FEATURE_NAMES,
)


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


def _corpus_rows() -> list[tuple]:
    return [
        # Query is at (0.0, 0.0). This row is very close (distance 0.1).
        (
            "2020_01_X_Y",
            "2020-2021",
            1,
            "2020-09-10",
            "Away One",
            "Home One",
            17,
            24,
            "Home One",
            3.0,
            0.1,
            0.0,
        ),
        # Moderately close (distance ~2.0), no line recorded (pick'em/missing).
        (
            "2021_02_X_Y",
            "2021-2022",
            2,
            "2021-09-17",
            "Away Two",
            "Home Two",
            20,
            17,
            None,
            None,
            2.0,
            0.0,
        ),
        # Far beyond any reasonable threshold.
        (
            "2022_03_X_Y",
            "2022-2023",
            3,
            "2022-09-24",
            "Away Three",
            "Home Three",
            10,
            30,
            "Away Three",
            7.0,
            100.0,
            100.0,
        ),
        # Same game_id as the query event -- must be excluded even though its
        # feature vector is closest of all (distance 0 to itself).
        (
            "2026_02_A_B",
            "2019-2020",
            1,
            "2019-09-08",
            "Away Four",
            "Home Four",
            14,
            21,
            "Home Four",
            2.5,
            0.0,
            0.0,
        ),
    ]


def _corpus(tmp_path: Path, *, model_digest: str = MODEL_DIGEST):
    rows = _corpus_rows()
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    reference = ComparableGamesFrameReference(
        artifact="schema=1/frames/{corpus_id}.parquet",
        row_count=len(rows),
        columns=COLUMNS,
        content_digest=frame_content_digest(COLUMNS, rows),
    )
    corpus = create_comparable_games_corpus(
        model_name="win_prob",
        model_type="logistic",
        model_content_digest=model_digest,
        scaler_content_digest=SCALER_DIGEST,
        feature_schema=_feature_schema(),
        generated_at=GENERATED_AT,
        frame=reference,
        distance_threshold=5.0,
        leave_one_out_median_distance=1.0,
        leave_one_out_percentile=90.0,
    )
    write_comparable_games_corpus(corpus, frame=frame, repo=tmp_path)
    return corpus


def _binary_artifacts(*, model_digest: str = MODEL_DIGEST) -> tuple[BinaryArtifactReference, ...]:
    return (
        BinaryArtifactReference(
            kind=PredictionArtifactKind.EXTERNAL_CALIBRATOR,
            source_relative_path="data/models/win_prob/logistic/calibrator.joblib",
            state=PredictionArtifactState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.MODEL,
            source_relative_path="data/models/win_prob/logistic/model.joblib",
            state=PredictionArtifactState.PRESENT,
            content_digest=model_digest,
            size_bytes=10,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.MODEL_METADATA,
            source_relative_path="data/models/win_prob/logistic/metadata.json",
            state=PredictionArtifactState.PRESENT,
            content_digest="c" * 64,
            size_bytes=20,
        ),
        BinaryArtifactReference(
            kind=PredictionArtifactKind.SCALER,
            source_relative_path="data/models/win_prob/logistic/scaler.joblib",
            state=PredictionArtifactState.PRESENT,
            content_digest=SCALER_DIGEST,
            size_bytes=10,
        ),
    )


def _post_processing() -> PredictionPostProcessingEvidence:
    return PredictionPostProcessingEvidence(
        registry_reference=SourceArtifactReference(
            relative_path="data/output/calibration/game_model_calibration.json",
            state=PredictionSourceState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
        registry_entry_updated_at=None,
        sigma=None,
        sigma_source=CalibrationResolutionSource.NOT_USED,
        margin_std=None,
        margin_std_source=CalibrationResolutionSource.NOT_USED,
        external_calibrator_state=PredictionArtifactState.ABSENT,
        embedded_estimator_calibration=False,
    )


def _write_evidence(tmp_path: Path, *, run_id: str = "run-1", model_digest: str = MODEL_DIGEST):
    event = StatisticalPredictionEventEvidence(
        event_id="event-1",
        game_id="2026_02_A_B",
        raw_feature_values=(0.0, 0.0),
        transformed_feature_values=(0.0, 0.0),
        raw_estimator_output=0.5,
        post_estimator_output=0.5,
        final_outputs=(("home_win_prob", 0.5),),
    )
    evidence = create_statistical_prediction_input_evidence(
        run_id=run_id,
        season=SEASON,
        week=WEEK,
        generated_at=GENERATED_AT,
        model_name="win_prob",
        model_type="logistic",
        source_revision=SourceRevision(commit=COMMIT, tracked_worktree_clean=True),
        source_artifacts=(
            SourceArtifactReference(
                relative_path="data/cleaned/NFL_Team_Elo.csv",
                state=PredictionSourceState.PRESENT,
                content_digest="a" * 64,
                size_bytes=100,
            ),
        ),
        binary_artifacts=_binary_artifacts(model_digest=model_digest),
        feature_schema=_feature_schema(),
        post_processing=_post_processing(),
        events=(event,),
    )
    write_prediction_input_evidence(evidence, repo=tmp_path)
    return evidence


class TestBuildAndWriteComparableGamesBatches:
    def test_ranks_by_distance_excludes_self_and_respects_threshold(self, tmp_path: Path) -> None:
        _corpus(tmp_path)
        _write_evidence(tmp_path)

        result = build_and_write_comparable_games_batches(
            run_id="run-1", generated_at=GENERATED_AT, repo=tmp_path, k=5
        )

        assert result.event_count == 1
        batch = result.batches[0]
        assert [match.game_id for match in batch.matches] == ["2020_01_X_Y", "2021_02_X_Y"]
        assert [match.rank for match in batch.matches] == [1, 2]
        assert batch.matches[0].distance < batch.matches[1].distance
        assert batch.sample_size == 2

    def test_persists_and_is_findable_by_event(self, tmp_path: Path) -> None:
        _corpus(tmp_path)
        _write_evidence(tmp_path)

        build_and_write_comparable_games_batches(
            run_id="run-1", generated_at=GENERATED_AT, repo=tmp_path, k=5
        )

        found = find_comparable_games_by_event("event-1", repo=tmp_path)
        assert found is not None
        assert found.game_id == "2026_02_A_B"

    def test_outcome_fields_reflect_real_recorded_results(self, tmp_path: Path) -> None:
        _corpus(tmp_path)
        _write_evidence(tmp_path)

        result = build_and_write_comparable_games_batches(
            run_id="run-1", generated_at=GENERATED_AT, repo=tmp_path, k=5
        )
        batch = result.batches[0]
        closest = batch.matches[0]
        assert closest.favorite_team == "Home One"
        assert closest.favorite_won is True  # Home One won 24-17, margin 7 > 0
        assert closest.favorite_covered is True  # margin 7 > spread 3.0

        second = batch.matches[1]
        assert second.favorite_team is None
        assert second.favorite_won is None
        assert second.favorite_covered is None

        assert batch.favorite_win_rate == 1.0
        assert batch.favorite_cover_rate == 1.0

    def test_idempotent_rerun_does_not_raise(self, tmp_path: Path) -> None:
        _corpus(tmp_path)
        _write_evidence(tmp_path)

        build_and_write_comparable_games_batches(
            run_id="run-1", generated_at=GENERATED_AT, repo=tmp_path, k=5
        )
        build_and_write_comparable_games_batches(
            run_id="run-1", generated_at=datetime(2027, 1, 1, tzinfo=UTC), repo=tmp_path, k=5
        )

    def test_rejects_stale_corpus_bound_to_different_model_snapshot(self, tmp_path: Path) -> None:
        _corpus(tmp_path, model_digest="c" * 64)
        _write_evidence(tmp_path, model_digest=MODEL_DIGEST)

        with pytest.raises(ValueError, match="different model/scaler snapshot"):
            build_and_write_comparable_games_batches(
                run_id="run-1", generated_at=GENERATED_AT, repo=tmp_path, k=5
            )

    def test_raises_when_no_corpus_built_yet(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)

        with pytest.raises(ValueError, match="No comparable-games corpus"):
            build_and_write_comparable_games_batches(
                run_id="run-1", generated_at=GENERATED_AT, repo=tmp_path, k=5
            )

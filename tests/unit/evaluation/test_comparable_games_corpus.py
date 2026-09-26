"""Tests for immutable comparable-games corpus contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from gridiron_edge.evaluation.comparable_games_corpus import (
    COMPARABLE_GAMES_CORPUS_SCHEMA_VERSION,
    ComparableGamesFrameReference,
    comparable_games_corpus_id,
    create_comparable_games_corpus,
    frame_content_digest,
    validate_comparable_games_corpus,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema

GENERATED_AT = datetime(2026, 9, 26, 12, tzinfo=UTC)
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


def _frame_reference() -> ComparableGamesFrameReference:
    columns = ("game_id", *FEATURE_NAMES)
    rows = [("game-1", 0.1, 0.2), ("game-2", 0.3, -0.4)]
    return ComparableGamesFrameReference(
        artifact="schema=1/frames/{corpus_id}.parquet",
        row_count=len(rows),
        columns=columns,
        content_digest=frame_content_digest(columns, rows),
    )


def _corpus(*, generated_at: datetime = GENERATED_AT):
    return create_comparable_games_corpus(
        model_name="win_prob",
        model_type="logistic",
        model_content_digest=MODEL_DIGEST,
        scaler_content_digest=SCALER_DIGEST,
        feature_schema=_feature_schema(),
        generated_at=generated_at,
        frame=_frame_reference(),
        distance_threshold=9.5,
        leave_one_out_median_distance=8.1,
        leave_one_out_percentile=90.0,
    )


class TestComparableGamesCorpus:
    def test_create_round_trips_through_validation(self) -> None:
        corpus = _corpus()
        validate_comparable_games_corpus(corpus)
        assert corpus.schema_version == COMPARABLE_GAMES_CORPUS_SCHEMA_VERSION

    def test_identity_excludes_generated_at(self) -> None:
        first = _corpus(generated_at=GENERATED_AT)
        second = _corpus(generated_at=datetime(2027, 1, 1, tzinfo=UTC))

        assert first.corpus_id == second.corpus_id

    def test_identity_changes_with_frame_content(self) -> None:
        base_id = _corpus().corpus_id
        different_frame = replace(
            _frame_reference(),
            content_digest=frame_content_digest(
                ("game_id", *FEATURE_NAMES), [("game-1", 9.9, 9.9)]
            ),
        )
        changed_id = comparable_games_corpus_id(
            model_name="win_prob",
            model_type="logistic",
            model_content_digest=MODEL_DIGEST,
            scaler_content_digest=SCALER_DIGEST,
            feature_schema=_feature_schema(),
            frame=different_frame,
        )

        assert changed_id != base_id

    def test_frame_content_digest_ignores_nan_vs_none(self) -> None:
        columns = ("a",)
        digest_with_none = frame_content_digest(columns, [(None,)])
        digest_with_nan = frame_content_digest(columns, [(float("nan"),)])

        assert digest_with_none == digest_with_nan

    def test_rejects_tampered_corpus_id(self) -> None:
        corpus = replace(_corpus(), corpus_id="c" * 64)

        with pytest.raises(ValueError, match="does not match its own content"):
            validate_comparable_games_corpus(corpus)

    def test_rejects_fewer_than_two_rows(self) -> None:
        single_row_frame = replace(_frame_reference(), row_count=1)
        corpus = replace(
            _corpus(),
            frame=single_row_frame,
        )
        # corpus_id no longer matches after mutating frame in place; rebuild it.
        corpus = replace(
            corpus,
            corpus_id=comparable_games_corpus_id(
                model_name=corpus.model_name,
                model_type=corpus.model_type,
                model_content_digest=corpus.model_content_digest,
                scaler_content_digest=corpus.scaler_content_digest,
                feature_schema=corpus.feature_schema,
                frame=single_row_frame,
            ),
        )

        with pytest.raises(ValueError, match="at least two rows"):
            validate_comparable_games_corpus(corpus)

    def test_rejects_negative_distance_threshold(self) -> None:
        corpus = replace(_corpus(), distance_threshold=-1.0)

        with pytest.raises(ValueError, match="non-negative"):
            validate_comparable_games_corpus(corpus)

    def test_rejects_out_of_range_percentile(self) -> None:
        corpus = replace(_corpus(), leave_one_out_percentile=150.0)

        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            validate_comparable_games_corpus(corpus)

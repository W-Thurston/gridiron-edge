"""Tests for immutable comparable-games corpus persistence."""

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
    comparable_games_corpus_manifest_path,
    find_latest_comparable_games_corpus,
    read_comparable_games_corpus,
    write_comparable_games_corpus,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema

GENERATED_AT = datetime(2026, 9, 26, 12, tzinfo=UTC)
MODEL_DIGEST = "a" * 64
SCALER_DIGEST = "b" * 64
FEATURE_NAMES = ("ELO_DIFF", "OFF_EPA_PER_PLAY_DIFF")
COLUMNS = ("game_id", *FEATURE_NAMES)
ROWS = [("game-1", 0.1, 0.2), ("game-2", 0.3, -0.4)]


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


def _frame() -> pd.DataFrame:
    return pd.DataFrame(ROWS, columns=list(COLUMNS))


def _corpus(*, generated_at: datetime = GENERATED_AT):
    reference = ComparableGamesFrameReference(
        artifact="schema=1/frames/{corpus_id}.parquet",
        row_count=len(ROWS),
        columns=COLUMNS,
        content_digest=frame_content_digest(COLUMNS, ROWS),
    )
    return create_comparable_games_corpus(
        model_name="win_prob",
        model_type="logistic",
        model_content_digest=MODEL_DIGEST,
        scaler_content_digest=SCALER_DIGEST,
        feature_schema=_feature_schema(),
        generated_at=generated_at,
        frame=reference,
        distance_threshold=9.5,
        leave_one_out_median_distance=8.1,
        leave_one_out_percentile=90.0,
    )


class TestComparableGamesCorpusStore:
    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        corpus = _corpus()

        path = write_comparable_games_corpus(corpus, frame=_frame(), repo=tmp_path)
        read_back, frame = read_comparable_games_corpus(corpus.corpus_id, repo=tmp_path)

        assert path == comparable_games_corpus_manifest_path(corpus.corpus_id, repo=tmp_path)
        assert read_back == corpus
        assert frame.equals(_frame())

    def test_rebuild_with_different_generated_at_is_idempotent(self, tmp_path: Path) -> None:
        first = _corpus(generated_at=GENERATED_AT)
        second = _corpus(generated_at=datetime(2027, 1, 1, tzinfo=UTC))
        assert first.corpus_id == second.corpus_id

        write_comparable_games_corpus(first, frame=_frame(), repo=tmp_path)
        path = write_comparable_games_corpus(second, frame=_frame(), repo=tmp_path)

        read_back, _ = read_comparable_games_corpus(first.corpus_id, repo=tmp_path)
        assert read_back.generated_at == GENERATED_AT
        assert path.exists()

    def test_existing_conflicting_manifest_is_rejected_without_overwrite(
        self, tmp_path: Path
    ) -> None:
        corpus = _corpus()
        path = comparable_games_corpus_manifest_path(corpus.corpus_id, repo=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("not json", encoding="utf-8")

        with pytest.raises(ValueError, match="cannot be reused"):
            write_comparable_games_corpus(corpus, frame=_frame(), repo=tmp_path)

        assert path.read_text(encoding="utf-8") == "not json"

    def test_frame_mismatch_is_rejected(self, tmp_path: Path) -> None:
        corpus = _corpus()

        with pytest.raises(ValueError, match="row count"):
            write_comparable_games_corpus(
                corpus, frame=pd.DataFrame(ROWS[:1], columns=list(COLUMNS)), repo=tmp_path
            )

    def test_read_rejects_frame_tampered_after_write(self, tmp_path: Path) -> None:
        corpus = _corpus()
        write_comparable_games_corpus(corpus, frame=_frame(), repo=tmp_path)
        from gridiron_edge.evaluation.comparable_games_corpus_store import (
            comparable_games_corpus_frame_path,
        )

        frame_path = comparable_games_corpus_frame_path(corpus.corpus_id, repo=tmp_path)
        tampered_rows = [("game-1", 9.9, 9.9), ("game-2", 0.3, -0.4)]
        pd.DataFrame(tampered_rows, columns=list(COLUMNS)).to_parquet(frame_path, index=False)

        with pytest.raises(ValueError, match="content digest"):
            read_comparable_games_corpus(corpus.corpus_id, repo=tmp_path)

    def test_find_latest_returns_most_recently_generated(self, tmp_path: Path) -> None:
        older = _corpus(generated_at=GENERATED_AT)
        # A distinct model digest forces a distinct corpus_id so both persist.
        newer_reference = ComparableGamesFrameReference(
            artifact="schema=1/frames/{corpus_id}.parquet",
            row_count=len(ROWS),
            columns=COLUMNS,
            content_digest=frame_content_digest(COLUMNS, ROWS),
        )
        newer = create_comparable_games_corpus(
            model_name="win_prob",
            model_type="logistic",
            model_content_digest="c" * 64,
            scaler_content_digest=SCALER_DIGEST,
            feature_schema=_feature_schema(),
            generated_at=datetime(2027, 1, 1, tzinfo=UTC),
            frame=newer_reference,
            distance_threshold=9.5,
            leave_one_out_median_distance=8.1,
            leave_one_out_percentile=90.0,
        )
        write_comparable_games_corpus(older, frame=_frame(), repo=tmp_path)
        write_comparable_games_corpus(newer, frame=_frame(), repo=tmp_path)

        found = find_latest_comparable_games_corpus(
            model_name="win_prob", model_type="logistic", repo=tmp_path
        )

        assert found is not None
        assert found.corpus_id == newer.corpus_id

    def test_find_latest_returns_none_when_absent(self, tmp_path: Path) -> None:
        found = find_latest_comparable_games_corpus(
            model_name="win_prob", model_type="logistic", repo=tmp_path
        )

        assert found is None

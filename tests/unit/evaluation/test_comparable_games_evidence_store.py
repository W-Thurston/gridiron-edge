"""Tests for immutable comparable-games retrieval evidence persistence."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from gridiron_edge.evaluation.comparable_games_evidence import (
    EUCLIDEAN_STANDARDIZED_METRIC,
    EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
    ComparableFeatureContribution,
    ComparableGameMatch,
    create_comparable_games_batch,
)
from gridiron_edge.evaluation.comparable_games_evidence_store import (
    AmbiguousComparableGamesError,
    comparable_games_batch_path,
    find_comparable_games_by_event,
    read_comparable_games_batch,
    write_comparable_games_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema

GENERATED_AT = datetime(2026, 9, 26, 12, tzinfo=UTC)
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


def _match() -> ComparableGameMatch:
    return ComparableGameMatch(
        game_id="game-1",
        rank=1,
        distance=1.0,
        season="2024-2025",
        week=5,
        game_date="2024-10-01",
        away_team="Away Team",
        home_team="Home Team",
        away_score=17,
        home_score=24,
        favorite_team="Home Team",
        spread_magnitude=3.0,
        favorite_won=True,
        favorite_covered=True,
        top_contributing_features=(
            ComparableFeatureContribution(
                feature_name="ELO_DIFF",
                query_value=0.1,
                candidate_value=0.2,
                squared_difference=0.01,
            ),
        ),
    )


def _batch(
    *,
    event_id: str = "event-1",
    corpus_id: str = "d" * 64,
    generated_at: datetime = GENERATED_AT,
):
    return create_comparable_games_batch(
        event_id=event_id,
        game_id="game-0",
        corpus_id=corpus_id,
        model_name="win_prob",
        model_type="logistic",
        feature_schema=_feature_schema(),
        metric=EUCLIDEAN_STANDARDIZED_METRIC,
        metric_version=EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
        k_requested=20,
        distance_threshold=5.0,
        generated_at=generated_at,
        matches=(_match(),),
        sample_size=1,
        favorite_win_rate=1.0,
        favorite_cover_rate=1.0,
    )


class TestComparableGamesEvidenceStore:
    def test_write_read_round_trip_and_idempotent_replay(self, tmp_path: Path) -> None:
        batch = _batch()

        first = write_comparable_games_batch(batch, repo=tmp_path)
        second = write_comparable_games_batch(batch, repo=tmp_path)

        assert first == second
        assert read_comparable_games_batch(first) == batch

    def test_rerun_with_different_generated_at_is_idempotent(self, tmp_path: Path) -> None:
        first = _batch(generated_at=GENERATED_AT)
        second = _batch(generated_at=datetime(2027, 1, 1, tzinfo=UTC))
        assert first.batch_id == second.batch_id

        write_comparable_games_batch(first, repo=tmp_path)
        path = write_comparable_games_batch(second, repo=tmp_path)

        read_back = read_comparable_games_batch(path)
        assert read_back.generated_at == GENERATED_AT

    def test_existing_conflicting_content_is_rejected_without_overwrite(
        self, tmp_path: Path
    ) -> None:
        batch = _batch()
        path = comparable_games_batch_path(batch.batch_id, repo=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("not json", encoding="utf-8")

        with pytest.raises(ValueError, match="cannot be reused"):
            write_comparable_games_batch(batch, repo=tmp_path)

        assert path.read_text(encoding="utf-8") == "not json"

    def test_find_by_event_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert find_comparable_games_by_event("missing-event", repo=tmp_path) is None

    def test_find_by_event_returns_the_one_match(self, tmp_path: Path) -> None:
        batch = _batch()
        write_comparable_games_batch(batch, repo=tmp_path)

        found = find_comparable_games_by_event("event-1", repo=tmp_path)

        assert found == batch

    def test_find_by_event_raises_when_two_corpora_both_claim_it(self, tmp_path: Path) -> None:
        older = _batch(corpus_id="d" * 64)
        newer = _batch(corpus_id="e" * 64)
        write_comparable_games_batch(older, repo=tmp_path)
        write_comparable_games_batch(newer, repo=tmp_path)

        with pytest.raises(AmbiguousComparableGamesError):
            find_comparable_games_by_event("event-1", repo=tmp_path)

    def test_rejects_unexpected_artifact_keys(self, tmp_path: Path) -> None:
        batch = _batch()
        path = write_comparable_games_batch(batch, repo=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="keys do not match"):
            read_comparable_games_batch(path)

    def test_rejects_tampered_embedded_batch(self, tmp_path: Path) -> None:
        batch = _batch()
        path = write_comparable_games_batch(batch, repo=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["batch"]["event_id"] = "tampered-event"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="does not match its own content"):
            read_comparable_games_batch(path)

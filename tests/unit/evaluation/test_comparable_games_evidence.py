"""Tests for immutable comparable-games retrieval evidence contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from gridiron_edge.evaluation.comparable_games_evidence import (
    COMPARABLE_GAMES_EVIDENCE_SCHEMA_VERSION,
    EUCLIDEAN_STANDARDIZED_METRIC,
    EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
    ComparableFeatureContribution,
    ComparableGameMatch,
    create_comparable_games_batch,
    validate_comparable_games_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema

GENERATED_AT = datetime(2026, 9, 26, 12, tzinfo=UTC)
CORPUS_ID = "d" * 64
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


def _match(*, rank: int = 1, distance: float = 1.0, game_id: str = "game-1") -> ComparableGameMatch:
    return ComparableGameMatch(
        game_id=game_id,
        rank=rank,
        distance=distance,
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


def _batch(*, matches=None, distance_threshold: float = 5.0, generated_at: datetime = GENERATED_AT):
    resolved_matches = matches if matches is not None else (_match(),)
    return create_comparable_games_batch(
        event_id="event-1",
        game_id="game-0",
        corpus_id=CORPUS_ID,
        model_name="win_prob",
        model_type="logistic",
        feature_schema=_feature_schema(),
        metric=EUCLIDEAN_STANDARDIZED_METRIC,
        metric_version=EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
        k_requested=20,
        distance_threshold=distance_threshold,
        generated_at=generated_at,
        matches=resolved_matches,
        sample_size=len(resolved_matches),
        favorite_win_rate=1.0 if resolved_matches else None,
        favorite_cover_rate=1.0 if resolved_matches else None,
    )


class TestComparableGamesBatch:
    def test_create_round_trips_through_validation(self) -> None:
        batch = _batch()
        validate_comparable_games_batch(batch)
        assert batch.schema_version == COMPARABLE_GAMES_EVIDENCE_SCHEMA_VERSION

    def test_identity_excludes_generated_at(self) -> None:
        first = _batch(generated_at=GENERATED_AT)
        second = _batch(generated_at=datetime(2027, 1, 1, tzinfo=UTC))

        assert first.batch_id == second.batch_id

    def test_identity_changes_with_matches(self) -> None:
        first = _batch(matches=(_match(),))
        second = _batch(matches=(_match(distance=2.0),))

        assert first.batch_id != second.batch_id

    def test_empty_matches_is_a_valid_honest_result(self) -> None:
        batch = _batch(matches=())
        validate_comparable_games_batch(batch)
        assert batch.sample_size == 0
        assert batch.favorite_win_rate is None

    def test_rejects_sample_size_mismatch(self) -> None:
        batch = replace(_batch(), sample_size=0)

        with pytest.raises(ValueError, match="sample_size must equal"):
            validate_comparable_games_batch(batch)

    def test_rejects_matches_exceeding_k_requested(self) -> None:
        matches = tuple(_match(rank=i + 1, distance=float(i)) for i in range(3))
        batch = replace(_batch(matches=matches, distance_threshold=10.0), k_requested=2)

        with pytest.raises(ValueError, match="cannot exceed k_requested"):
            validate_comparable_games_batch(batch)

    def test_rejects_non_contiguous_ranks(self) -> None:
        matches = (_match(rank=1, distance=1.0), _match(rank=3, distance=2.0, game_id="game-2"))
        batch = replace(_batch(), matches=matches, sample_size=2)

        with pytest.raises(ValueError, match="contiguously ranked"):
            validate_comparable_games_batch(batch)

    def test_rejects_out_of_order_distances(self) -> None:
        matches = (
            _match(rank=1, distance=5.0),
            _match(rank=2, distance=1.0, game_id="game-2"),
        )
        batch = replace(_batch(), matches=matches, sample_size=2)

        with pytest.raises(ValueError, match="non-decreasing distance"):
            validate_comparable_games_batch(batch)

    def test_rejects_match_beyond_distance_threshold(self) -> None:
        batch = replace(_batch(), matches=(_match(distance=100.0),))

        with pytest.raises(ValueError, match="within distance_threshold"):
            validate_comparable_games_batch(batch)

    def test_rejects_rate_outside_unit_interval(self) -> None:
        batch = replace(_batch(), favorite_win_rate=1.5)

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_comparable_games_batch(batch)

    def test_rejects_tampered_batch_id(self) -> None:
        batch = replace(_batch(), batch_id="f" * 64)

        with pytest.raises(ValueError, match="does not match its own content"):
            validate_comparable_games_batch(batch)

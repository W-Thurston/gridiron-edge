# tests/unit/api/test_serializers_comparables.py

"""Tests for /games/{game_id}/comparables serializer."""

from __future__ import annotations

from datetime import UTC, datetime

from gridiron_edge.api.serializers.comparables import serialize_game_comparables
from gridiron_edge.evaluation.comparable_games_evidence import (
    EUCLIDEAN_STANDARDIZED_METRIC,
    EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
    ComparableFeatureContribution,
    ComparableGameMatch,
    ComparableGamesBatch,
    create_comparable_games_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema

CORPUS_ID = "d" * 64


def _row(
    *,
    win_status: str = "available",
    win_model_type: str | None = "logistic",
    win_event_id: str = "event-1",
) -> dict:
    return {
        "game_id": "2026_02_CAR_ATL",
        "win_status": win_status,
        "win_model_type": win_model_type,
        "win_event_id": win_event_id,
    }


def _feature_schema():
    return create_prediction_feature_schema(
        model_name="win_prob",
        model_type="logistic",
        task="classification",
        modeling_schema_version=5,
        epa_window=6,
        feature_set_name="combined_111",
        ordered_columns=("ELO_DIFF",),
    )


def _match() -> ComparableGameMatch:
    return ComparableGameMatch(
        game_id="2006_19_DAL_SEA",
        rank=1,
        distance=8.59,
        season="2006-2007",
        week=19,
        game_date="2007-01-06",
        away_team="Dallas Cowboys",
        home_team="Seattle Seahawks",
        away_score=20,
        home_score=21,
        favorite_team="Seattle Seahawks",
        spread_magnitude=1.0,
        favorite_won=True,
        favorite_covered=False,
        top_contributing_features=(
            ComparableFeatureContribution(
                feature_name="ELO_DIFF",
                query_value=-2.3,
                candidate_value=0.15,
                squared_difference=6.13,
            ),
        ),
    )


def _batch(
    *, event_id: str = "event-1", matches: tuple[ComparableGameMatch, ...] | None = None
) -> ComparableGamesBatch:
    resolved = matches if matches is not None else (_match(),)
    return create_comparable_games_batch(
        event_id=event_id,
        game_id="2026_02_CAR_ATL",
        corpus_id=CORPUS_ID,
        model_name="win_prob",
        model_type="logistic",
        feature_schema=_feature_schema(),
        metric=EUCLIDEAN_STANDARDIZED_METRIC,
        metric_version=EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
        k_requested=20,
        distance_threshold=9.28,
        generated_at=datetime(2026, 9, 26, tzinfo=UTC),
        matches=resolved,
        sample_size=len(resolved),
        favorite_win_rate=1.0 if resolved else None,
        favorite_cover_rate=0.0 if resolved else None,
    )


class TestWinUnavailable:
    def test_all_fields_blocked_no_win_forecast(self) -> None:
        result = serialize_game_comparables(_row(win_status="forecast_missing"), None)

        assert result.comparables is None
        assert result.response_meta is not None
        status = result.response_meta.field_status["comparables"]
        assert status.blocker == "no_win_forecast"  # type: ignore[union-attr]
        assert result.response_meta.field_status["sample_size"].blocker == "no_win_forecast"  # type: ignore[union-attr]


class TestNonLogisticChampion:
    def test_all_fields_blocked_comparables_retrieval(self) -> None:
        result = serialize_game_comparables(_row(win_model_type="random_forest"), None)

        assert result.comparables is None
        status = result.response_meta.field_status["comparables"]  # type: ignore[union-attr]
        assert status.blocker == "comparables_retrieval"  # type: ignore[union-attr]


class TestAmbiguousEvidence:
    def test_all_fields_blocked_ambiguous(self) -> None:
        result = serialize_game_comparables(_row(), None, ambiguous_evidence=True)

        status = result.response_meta.field_status["comparables"]  # type: ignore[union-attr]
        assert status.blocker == "ambiguous_comparable_evidence"  # type: ignore[union-attr]


class TestNoEvidenceYet:
    def test_all_fields_blocked_no_comparable_evidence(self) -> None:
        result = serialize_game_comparables(_row(), None)

        status = result.response_meta.field_status["comparables"]  # type: ignore[union-attr]
        assert status.blocker == "no_comparable_evidence"  # type: ignore[union-attr]


class TestPopulated:
    def test_serializes_real_batch(self) -> None:
        result = serialize_game_comparables(_row(), _batch())

        assert result.comparables is not None
        assert len(result.comparables) == 1
        first = result.comparables[0]
        assert first.game_id == "2006_19_DAL_SEA"
        assert first.favorite_team == "Seattle Seahawks"
        assert first.favorite_won is True
        assert first.favorite_covered is False
        assert first.top_contributing_features[0].feature_name == "ELO_DIFF"
        assert result.sample_size == 1
        assert result.favorite_win_rate == 1.0
        assert result.favorite_cover_rate == 0.0
        field_status = result.response_meta.field_status if result.response_meta else {}
        assert "comparables" not in field_status

    def test_empty_matches_is_honestly_zero(self) -> None:
        result = serialize_game_comparables(_row(), _batch(matches=()))

        assert result.comparables == []
        assert result.sample_size == 0
        assert result.favorite_win_rate is None

# src/gridiron_edge/evaluation/comparable_games_evidence_builder.py
"""Build and persist comparable-games retrieval evidence for one run.

For every ``win_prob``/``logistic`` statistical event already persisted in
one run's prediction-input evidence (D50), finds that game's nearest
neighbors in the comparable-games historical corpus by Euclidean distance
in the champion's own standardized feature space, and persists one
immutable batch per event. Performs no feature construction, scaling, or
model inference of its own: the query vector comes directly from the
already-persisted evidence, exactly as ``logistic_explanation_evidence_builder``
reads it, never recomputed.

Fails closed if the corpus was built from a different model or scaler
snapshot than the one that produced this run's evidence (a stale corpus
after a champion retrain), rather than silently comparing incompatible
scaled spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from pandas import DataFrame

from gridiron_edge.evaluation.comparable_games_corpus import ComparableGamesCorpus
from gridiron_edge.evaluation.comparable_games_corpus_store import (
    find_latest_comparable_games_corpus,
    read_comparable_games_corpus,
)
from gridiron_edge.evaluation.comparable_games_evidence import (
    EUCLIDEAN_STANDARDIZED_METRIC,
    EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
    ComparableFeatureContribution,
    ComparableGameMatch,
    ComparableGamesBatch,
    create_comparable_games_batch,
)
from gridiron_edge.evaluation.comparable_games_evidence_store import (
    write_comparable_games_batch,
)
from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionExecutionKind,
    PredictionInputEvidence,
    StatisticalPredictionEventEvidence,
)
from gridiron_edge.evaluation.prediction_input_evidence_store import (
    list_prediction_input_evidence_by_run,
)

_MODEL_NAME = "win_prob"
_MODEL_TYPE = "logistic"
_DEFAULT_K = 20
_TOP_CONTRIBUTING_FEATURES = 5


@dataclass(frozen=True, slots=True)
class ComparableGamesBuildResult:
    """Accounting for one run's persisted comparable-games batches."""

    batches: tuple[ComparableGamesBatch, ...]
    event_count: int


def build_and_write_comparable_games_batches(
    *,
    run_id: str,
    generated_at: datetime,
    repo: Path,
    k: int = _DEFAULT_K,
) -> ComparableGamesBuildResult:
    """Persist one comparable-games batch per event in a run's Win evidence.

    Args:
        run_id: Exact run identity whose ``win_prob``/``logistic``
            prediction-input evidence should be matched against the corpus.
        generated_at: Timezone-aware UTC generation timestamp (informational
            only — batch identity does not depend on it).
        repo: Repository root.
        k: Maximum number of comparables requested per event. The persisted
            result may hold fewer if the corpus's empirically-derived
            distance threshold excludes some candidates.

    Raises:
        ValueError: If the run has no (or more than one) win_prob/logistic
            persisted-estimator evidence, if no comparable-games corpus has
            been built yet, or if the corpus's bound model/scaler snapshot
            does not match the one that produced this run's evidence.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    normalized_run_id = _require_text(run_id, "run_id")
    evidence_candidates = [
        evidence
        for evidence in list_prediction_input_evidence_by_run(normalized_run_id, repo=repo)
        if evidence.model_name == _MODEL_NAME
        and evidence.model_type == _MODEL_TYPE
        and evidence.execution_kind is PredictionExecutionKind.PERSISTED_ESTIMATOR
    ]
    if not evidence_candidates:
        raise ValueError(
            f"No {_MODEL_NAME}/{_MODEL_TYPE} persisted-estimator evidence found for "
            f"run_id={normalized_run_id!r}."
        )
    if len(evidence_candidates) > 1:
        raise ValueError(
            f"Multiple {_MODEL_NAME}/{_MODEL_TYPE} persisted-estimator evidence artifacts "
            f"found for run_id={normalized_run_id!r}."
        )
    evidence = evidence_candidates[0]
    if evidence.feature_schema is None:
        raise ValueError(f"{_MODEL_NAME}/{_MODEL_TYPE} evidence is missing its feature_schema.")

    corpus = find_latest_comparable_games_corpus(
        model_name=_MODEL_NAME, model_type=_MODEL_TYPE, repo=repo
    )
    if corpus is None:
        raise ValueError(
            f"No comparable-games corpus has been built for {_MODEL_NAME}/{_MODEL_TYPE} yet. "
            "Run `gridiron evaluate build-comparable-corpus` first."
        )

    query_model_digest, query_scaler_digest = _bound_digests(evidence)
    if (
        query_model_digest != corpus.model_content_digest
        or query_scaler_digest != corpus.scaler_content_digest
    ):
        raise ValueError(
            "Comparable-games corpus is bound to a different model/scaler snapshot than this "
            "run's evidence. Rebuild the corpus with `gridiron evaluate build-comparable-corpus`."
        )
    if evidence.feature_schema.ordered_columns != corpus.feature_schema.ordered_columns:
        raise ValueError(
            "Comparable-games corpus feature order does not match this run's evidence."
        )

    _, frame = read_comparable_games_corpus(corpus.corpus_id, repo=repo)
    feature_columns = list(corpus.feature_schema.ordered_columns)
    candidate_matrix = frame[feature_columns].to_numpy()

    batches: list[ComparableGamesBatch] = []
    for statistical_event in evidence.statistical_events:
        batch = _build_one_batch(
            statistical_event,
            corpus=corpus,
            frame=frame,
            candidate_matrix=candidate_matrix,
            feature_columns=feature_columns,
            k=k,
            generated_at=generated_at,
        )
        write_comparable_games_batch(batch, repo=repo)
        batches.append(batch)

    return ComparableGamesBuildResult(batches=tuple(batches), event_count=len(batches))


def _build_one_batch(
    statistical_event: StatisticalPredictionEventEvidence,
    *,
    corpus: ComparableGamesCorpus,
    frame: DataFrame,
    candidate_matrix: np.ndarray,
    feature_columns: list[str],
    k: int,
    generated_at: datetime,
) -> ComparableGamesBatch:
    query_vector = np.asarray(statistical_event.transformed_feature_values, dtype=float)
    exclude = (frame["game_id"] == statistical_event.game_id).to_numpy()

    diffs = candidate_matrix - query_vector
    squared = diffs**2
    distances = np.sqrt(squared.sum(axis=1))
    distances = np.where(exclude, np.inf, distances)

    order = np.argsort(distances, kind="stable")
    matches: list[ComparableGameMatch] = []
    for position in order:
        distance = float(distances[position])
        if not np.isfinite(distance) or distance > corpus.distance_threshold:
            break
        if len(matches) >= k:
            break
        row = frame.iloc[position]
        top_indices = np.argsort(-squared[position])[:_TOP_CONTRIBUTING_FEATURES]
        contributions = tuple(
            ComparableFeatureContribution(
                feature_name=feature_columns[index],
                query_value=float(query_vector[index]),
                candidate_value=float(candidate_matrix[position, index]),
                squared_difference=float(squared[position, index]),
            )
            for index in top_indices
        )
        favorite_team = _normalize_optional_str(row["favorite_team"])
        spread_magnitude = _normalize_optional_float(row["spread_magnitude"])
        away_team = str(row["away_team"])
        home_team = str(row["home_team"])
        away_score = int(row["away_score"])
        home_score = int(row["home_score"])
        favorite_won, favorite_covered = _outcome(
            favorite_team=favorite_team,
            spread_magnitude=spread_magnitude,
            away_team=away_team,
            home_team=home_team,
            away_score=away_score,
            home_score=home_score,
        )
        matches.append(
            ComparableGameMatch(
                game_id=str(row["game_id"]),
                rank=len(matches) + 1,
                distance=distance,
                season=str(row["season"]),
                week=int(row["week"]),
                game_date=str(row["game_date"]),
                away_team=away_team,
                home_team=home_team,
                away_score=away_score,
                home_score=home_score,
                favorite_team=favorite_team,
                spread_magnitude=spread_magnitude,
                favorite_won=favorite_won,
                favorite_covered=favorite_covered,
                top_contributing_features=contributions,
            )
        )

    known_favorite = [match for match in matches if match.favorite_won is not None]
    favorite_win_rate = (
        sum(1 for match in known_favorite if match.favorite_won) / len(known_favorite)
        if known_favorite
        else None
    )
    known_cover = [match for match in known_favorite if match.favorite_covered is not None]
    favorite_cover_rate = (
        sum(1 for match in known_cover if match.favorite_covered) / len(known_cover)
        if known_cover
        else None
    )

    return create_comparable_games_batch(
        event_id=statistical_event.event_id,
        game_id=statistical_event.game_id,
        corpus_id=corpus.corpus_id,
        model_name=corpus.model_name,
        model_type=corpus.model_type,
        feature_schema=corpus.feature_schema,
        metric=EUCLIDEAN_STANDARDIZED_METRIC,
        metric_version=EUCLIDEAN_STANDARDIZED_METRIC_VERSION,
        k_requested=k,
        distance_threshold=corpus.distance_threshold,
        generated_at=generated_at,
        matches=tuple(matches),
        sample_size=len(matches),
        favorite_win_rate=favorite_win_rate,
        favorite_cover_rate=favorite_cover_rate,
    )


def _normalize_optional_str(value: object) -> str | None:
    """Return ``None`` for a missing value (Python ``None`` or a float NaN)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def _normalize_optional_float(value: object) -> float | None:
    """Return ``None`` for a missing value (Python ``None`` or a float NaN)."""
    if value is None or not isinstance(value, int | float) or np.isnan(value):
        return None
    return float(value)


def _outcome(
    *,
    favorite_team: str | None,
    spread_magnitude: float | None,
    away_team: str,
    home_team: str,
    away_score: int,
    home_score: int,
) -> tuple[bool | None, bool | None]:
    if favorite_team is None:
        return None, None
    if favorite_team not in (away_team, home_team):
        return None, None
    favorite_score = home_score if favorite_team == home_team else away_score
    underdog_score = away_score if favorite_team == home_team else home_score
    margin = favorite_score - underdog_score
    favorite_won = margin > 0

    if spread_magnitude is None:
        return favorite_won, None
    favorite_covered = margin > spread_magnitude
    return favorite_won, favorite_covered


def _bound_digests(evidence: PredictionInputEvidence) -> tuple[str, str]:
    from gridiron_edge.evaluation.prediction_input_evidence import (
        PredictionArtifactKind,
        PredictionArtifactState,
    )

    by_kind = {reference.kind: reference for reference in evidence.binary_artifacts}
    model_reference = by_kind.get(PredictionArtifactKind.MODEL)
    scaler_reference = by_kind.get(PredictionArtifactKind.SCALER)
    if model_reference is None or model_reference.state is not PredictionArtifactState.PRESENT:
        raise ValueError(f"{_MODEL_NAME}/{_MODEL_TYPE} evidence requires a present model artifact.")
    if scaler_reference is None or scaler_reference.state is not PredictionArtifactState.PRESENT:
        raise ValueError(
            f"{_MODEL_NAME}/{_MODEL_TYPE} evidence requires a present scaler artifact."
        )
    assert model_reference.content_digest is not None
    assert scaler_reference.content_digest is not None
    return model_reference.content_digest, scaler_reference.content_digest


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()

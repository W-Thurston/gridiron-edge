# src/gridiron_edge/cli/_prediction_publication.py

"""Shared immutable publication helpers for role-aware weekly prediction.

Used by both the live ``weekly-predict`` composite and the development-role
forecast generation command: persisting and authenticating binary snapshots
and prediction-input evidence is identical regardless of forecast role.
"""

from __future__ import annotations

from pathlib import Path

from pandas import DataFrame

from gridiron_edge.evaluation.prediction_input_evidence import (
    PredictionArtifactState,
    PredictionInputEvidence,
    SourceArtifactReference,
    SourceRevision,
    authenticate_prediction_input_evidence,
)
from gridiron_edge.evaluation.prediction_input_evidence_store import (
    authenticate_binary_snapshot,
    read_prediction_input_evidence,
    write_binary_snapshot,
    write_prediction_input_evidence,
)


def _require_execution_provenance(
    evidence: tuple[PredictionInputEvidence, ...],
    *,
    source_revision: SourceRevision,
    source_artifacts: tuple[SourceArtifactReference, ...],
) -> None:
    """Require every selected family to preserve the captured provenance."""
    if not evidence:
        raise ValueError("Weekly prediction execution returned no input evidence.")
    for family in evidence:
        if family.source_revision != source_revision:
            raise ValueError("Prediction-input evidence source revision does not match execution.")
        if family.source_artifacts != source_artifacts:
            raise ValueError("Prediction-input evidence source artifacts do not match execution.")


def _publish_binary_snapshots(
    evidence: tuple[PredictionInputEvidence, ...],
    *,
    repo: Path,
) -> tuple[Path, ...]:
    """Publish and authenticate every distinct present binary artifact."""
    published: dict[tuple[str, int], Path] = {}
    for family in evidence:
        for reference in family.binary_artifacts:
            if reference.state is PredictionArtifactState.ABSENT:
                continue
            if reference.content_digest is None or reference.size_bytes is None:
                raise ValueError("Present binary artifact requires digest and size.")
            key = (reference.content_digest, reference.size_bytes)
            if key not in published:
                source = repo / reference.source_relative_path
                published[key] = write_binary_snapshot(
                    source,
                    expected_digest=reference.content_digest,
                    expected_size_bytes=reference.size_bytes,
                    repo=repo,
                )
            authenticated = authenticate_binary_snapshot(reference, repo=repo)
            if authenticated != published[key]:
                raise ValueError(
                    "Published binary snapshot path does not match authenticated snapshot."
                )
    return tuple(published[key] for key in sorted(published))


def _publish_and_reload_evidence(
    evidence: tuple[PredictionInputEvidence, ...],
    *,
    forecast_events: DataFrame,
    repo: Path,
) -> tuple[Path, ...]:
    """Publish, strictly reload, and authenticate every family evidence artifact."""
    paths: list[Path] = []
    for family in evidence:
        path = write_prediction_input_evidence(family, repo=repo)
        reloaded = read_prediction_input_evidence(path)
        if reloaded != family:
            raise ValueError("Reloaded prediction-input evidence does not match publication.")
        authenticate_prediction_input_evidence(
            reloaded,
            forecast_events=forecast_events,
        )
        paths.append(path)
    return tuple(paths)

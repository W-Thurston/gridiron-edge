# tests/unit/evaluation/test_prediction_input_evidence_store.py
"""Tests for immutable prediction-input evidence persistence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    EloPredictionEventEvidence,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    create_elo_prediction_input_evidence,
)
from gridiron_edge.evaluation.prediction_input_evidence_store import (
    authenticate_binary_snapshot,
    binary_snapshot_path,
    find_prediction_input_evidence_by_event,
    list_prediction_input_evidence_by_run,
    prediction_input_evidence_path,
    prediction_input_evidence_root,
    read_prediction_input_evidence,
    write_binary_snapshot,
    write_prediction_input_evidence,
)

GENERATED_AT = datetime(2026, 9, 17, 12, tzinfo=UTC)
SEASON = "2026-2027"
WEEK = 2
COMMIT = "a" * 40
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _digest(content: bytes) -> str:
    return sha256(content).hexdigest()


def _sources() -> tuple[SourceArtifactReference, ...]:
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
    )


def _artifacts() -> tuple[BinaryArtifactReference, ...]:
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


def _events(
    *,
    event_prefix: str = "event",
    game_prefix: str = "game",
) -> tuple[EloPredictionEventEvidence, ...]:
    return (
        EloPredictionEventEvidence(
            event_id=f"{event_prefix}-1",
            game_id=f"{game_prefix}-1",
            season=SEASON,
            week=WEEK,
            away_team="Away A",
            home_team="Home A",
            away_elo=1500.0,
            home_elo=1510.0,
            formula_id="elo_win_probability_v1",
            divisor=480.0,
            away_win_probability=0.48,
            home_win_probability=0.52,
            final_outputs=(
                ("away_win_prob", 0.48),
                ("home_win_prob", 0.52),
            ),
        ),
        EloPredictionEventEvidence(
            event_id=f"{event_prefix}-2",
            game_id=f"{game_prefix}-2",
            season=SEASON,
            week=WEEK,
            away_team="Away B",
            home_team="Home B",
            away_elo=1520.0,
            home_elo=1480.0,
            formula_id="elo_win_probability_v1",
            divisor=480.0,
            away_win_probability=0.55,
            home_win_probability=0.45,
            final_outputs=(
                ("away_win_prob", 0.55),
                ("home_win_prob", 0.45),
            ),
        ),
    )


def _evidence(
    *,
    run_id: str = "run-1",
    event_prefix: str = "event",
    game_prefix: str = "game",
):
    return create_elo_prediction_input_evidence(
        run_id=run_id,
        season=SEASON,
        week=WEEK,
        generated_at=GENERATED_AT,
        source_revision=SourceRevision(
            commit=COMMIT,
            tracked_worktree_clean=True,
        ),
        source_artifacts=_sources(),
        binary_artifacts=_artifacts(),
        events=_events(event_prefix=event_prefix, game_prefix=game_prefix),
    )


def _temporary_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.tmp") if path.is_file())


class TestPaths:
    def test_root_is_schema_owner_under_ignored_output(self, tmp_path: Path) -> None:
        assert prediction_input_evidence_root(tmp_path) == (
            tmp_path / "data/output/prediction_input_evidence"
        )

    def test_binary_snapshot_path_is_digest_addressed(self, tmp_path: Path) -> None:
        assert binary_snapshot_path(DIGEST_A, repo=tmp_path) == (
            tmp_path
            / "data/output/prediction_input_evidence/schema=1/artifacts"
            / f"{DIGEST_A}.bin"
        )

    def test_evidence_path_is_identity_addressed(self, tmp_path: Path) -> None:
        evidence = _evidence()

        assert prediction_input_evidence_path(evidence.evidence_id, repo=tmp_path) == (
            tmp_path
            / "data/output/prediction_input_evidence/schema=1/evidence"
            / f"{evidence.evidence_id}.json"
        )


class TestBinarySnapshots:
    def test_copies_exact_bytes_to_content_addressed_store(self, tmp_path: Path) -> None:
        content = b"exact model bytes"
        source = tmp_path / "mutable" / "model.joblib"
        source.parent.mkdir(parents=True)
        source.write_bytes(content)
        digest = _digest(content)

        path = write_binary_snapshot(
            source,
            expected_digest=digest,
            expected_size_bytes=len(content),
            repo=tmp_path,
        )

        assert path == binary_snapshot_path(digest, repo=tmp_path)
        assert path.read_bytes() == content
        assert path.stat().st_ino != source.stat().st_ino
        assert _temporary_files(tmp_path) == []

    def test_exact_replay_is_idempotent(self, tmp_path: Path) -> None:
        content = b"immutable artifact"
        source = tmp_path / "source.bin"
        source.write_bytes(content)
        digest = _digest(content)

        first = write_binary_snapshot(
            source,
            expected_digest=digest,
            expected_size_bytes=len(content),
            repo=tmp_path,
        )
        second = write_binary_snapshot(
            source,
            expected_digest=digest,
            expected_size_bytes=len(content),
            repo=tmp_path,
        )

        assert first == second
        assert second.read_bytes() == content
        assert _temporary_files(tmp_path) == []

    def test_concurrent_exact_replay_succeeds(self, tmp_path: Path) -> None:
        content = b"shared concurrent artifact"
        source = tmp_path / "source.bin"
        source.write_bytes(content)
        digest = _digest(content)

        def publish() -> Path:
            return write_binary_snapshot(
                source,
                expected_digest=digest,
                expected_size_bytes=len(content),
                repo=tmp_path,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            paths = tuple(executor.map(lambda _: publish(), range(16)))

        assert len(set(paths)) == 1
        assert paths[0].read_bytes() == content
        assert _temporary_files(tmp_path) == []

    def test_source_identity_mismatch_is_rejected_before_copy(self, tmp_path: Path) -> None:
        source = tmp_path / "source.bin"
        source.write_bytes(b"real bytes")

        with pytest.raises(ValueError, match="source identity"):
            write_binary_snapshot(
                source,
                expected_digest=DIGEST_A,
                expected_size_bytes=10,
                repo=tmp_path,
            )

        assert not binary_snapshot_path(DIGEST_A, repo=tmp_path).exists()
        assert _temporary_files(tmp_path) == []

    def test_tampered_existing_snapshot_is_rejected(self, tmp_path: Path) -> None:
        content = b"original bytes"
        source = tmp_path / "source.bin"
        source.write_bytes(content)
        digest = _digest(content)
        path = write_binary_snapshot(
            source,
            expected_digest=digest,
            expected_size_bytes=len(content),
            repo=tmp_path,
        )
        path.write_bytes(b"tampered bytes")

        with pytest.raises(ValueError, match="does not match stored bytes"):
            write_binary_snapshot(
                source,
                expected_digest=digest,
                expected_size_bytes=len(content),
                repo=tmp_path,
            )

        assert path.read_bytes() == b"tampered bytes"
        assert _temporary_files(tmp_path) == []

    def test_authenticate_present_snapshot_round_trips(self, tmp_path: Path) -> None:
        content = b"elo state bytes"
        source = tmp_path / "elo.csv"
        source.write_bytes(content)
        digest = _digest(content)
        expected = write_binary_snapshot(
            source,
            expected_digest=digest,
            expected_size_bytes=len(content),
            repo=tmp_path,
        )
        reference = BinaryArtifactReference(
            kind=PredictionArtifactKind.ELO_STATE,
            source_relative_path="data/cleaned/NFL_Team_Elo.csv",
            state=PredictionArtifactState.PRESENT,
            content_digest=digest,
            size_bytes=len(content),
        )

        assert authenticate_binary_snapshot(reference, repo=tmp_path) == expected

    def test_authenticate_absent_snapshot_returns_none(self, tmp_path: Path) -> None:
        reference = BinaryArtifactReference(
            kind=PredictionArtifactKind.SCALER,
            source_relative_path="data/models/win_prob/random_forest/scaler.joblib",
            state=PredictionArtifactState.ABSENT,
            content_digest=None,
            size_bytes=None,
        )

        assert authenticate_binary_snapshot(reference, repo=tmp_path) is None
        assert not prediction_input_evidence_root(tmp_path).exists()

    def test_temporary_file_removed_when_link_fails(self, tmp_path: Path) -> None:
        content = b"link failure bytes"
        source = tmp_path / "source.bin"
        source.write_bytes(content)
        digest = _digest(content)

        with (
            patch(
                "gridiron_edge.evaluation.prediction_input_evidence_store.os.link",
                side_effect=OSError("link failed"),
            ),
            pytest.raises(OSError, match="link failed"),
        ):
            write_binary_snapshot(
                source,
                expected_digest=digest,
                expected_size_bytes=len(content),
                repo=tmp_path,
            )

        assert not binary_snapshot_path(digest, repo=tmp_path).exists()
        assert _temporary_files(tmp_path) == []


class TestEvidenceStore:
    def test_write_read_round_trip_and_exact_replay(self, tmp_path: Path) -> None:
        evidence = _evidence()

        first = write_prediction_input_evidence(evidence, repo=tmp_path)
        second = write_prediction_input_evidence(evidence, repo=tmp_path)

        assert first == second
        assert read_prediction_input_evidence(first) == evidence
        assert _temporary_files(tmp_path) == []

    def test_concurrent_exact_replay_succeeds(self, tmp_path: Path) -> None:
        evidence = _evidence()

        def publish() -> Path:
            return write_prediction_input_evidence(evidence, repo=tmp_path)

        with ThreadPoolExecutor(max_workers=8) as executor:
            paths = tuple(executor.map(lambda _: publish(), range(16)))

        assert len(set(paths)) == 1
        assert read_prediction_input_evidence(paths[0]) == evidence
        assert _temporary_files(tmp_path) == []

    def test_existing_conflicting_content_is_rejected_without_overwrite(
        self,
        tmp_path: Path,
    ) -> None:
        evidence = _evidence()
        path = prediction_input_evidence_path(evidence.evidence_id, repo=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("conflict", encoding="utf-8")

        with pytest.raises(ValueError, match="cannot be reused"):
            write_prediction_input_evidence(evidence, repo=tmp_path)

        assert path.read_text(encoding="utf-8") == "conflict"
        assert _temporary_files(tmp_path) == []

    def test_rejects_missing_and_unexpected_artifact_keys(self, tmp_path: Path) -> None:
        evidence = _evidence()
        path = write_prediction_input_evidence(evidence, repo=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="keys do not match"):
            read_prediction_input_evidence(path)

    def test_rejects_unsupported_store_schema(self, tmp_path: Path) -> None:
        evidence = _evidence()
        path = write_prediction_input_evidence(evidence, repo=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["store_schema_version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match=r"Unsupported.*store schema"):
            read_prediction_input_evidence(path)

    def test_rejects_tampered_embedded_evidence(self, tmp_path: Path) -> None:
        evidence = _evidence()
        path = write_prediction_input_evidence(evidence, repo=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evidence"]["run_id"] = "tampered-run"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="does not match canonical"):
            read_prediction_input_evidence(path)

    def test_rejects_noncanonical_path(self, tmp_path: Path) -> None:
        evidence = _evidence()
        canonical = write_prediction_input_evidence(evidence, repo=tmp_path)
        moved = canonical.with_name(f"{DIGEST_A}.json")
        moved.write_bytes(canonical.read_bytes())

        with pytest.raises(ValueError, match="path and embedded identity disagree"):
            read_prediction_input_evidence(moved)

    def test_malformed_json_fails_explicitly(self, tmp_path: Path) -> None:
        path = prediction_input_evidence_path(DIGEST_A, repo=tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{malformed", encoding="utf-8")

        with pytest.raises(ValueError, match="malformed JSON"):
            read_prediction_input_evidence(path)

    def test_temporary_file_removed_when_link_fails(self, tmp_path: Path) -> None:
        evidence = _evidence()

        with (
            patch(
                "gridiron_edge.evaluation.prediction_input_evidence_store.os.link",
                side_effect=OSError("link failed"),
            ),
            pytest.raises(OSError, match="link failed"),
        ):
            write_prediction_input_evidence(evidence, repo=tmp_path)

        assert not prediction_input_evidence_path(
            evidence.evidence_id,
            repo=tmp_path,
        ).exists()
        assert _temporary_files(tmp_path) == []


class TestEvidenceLookup:
    def test_listing_by_run_is_deterministic(self, tmp_path: Path) -> None:
        second = _evidence(
            run_id="shared-run",
            event_prefix="z-event",
            game_prefix="z-game",
        )
        first = _evidence(
            run_id="shared-run",
            event_prefix="a-event",
            game_prefix="a-game",
        )
        unrelated = _evidence(
            run_id="other-run",
            event_prefix="other-event",
            game_prefix="other-game",
        )
        for evidence in (second, unrelated, first):
            write_prediction_input_evidence(evidence, repo=tmp_path)

        results = list_prediction_input_evidence_by_run("shared-run", repo=tmp_path)

        assert tuple(value.evidence_id for value in results) == tuple(
            sorted((first.evidence_id, second.evidence_id))
        )

    def test_event_lookup_returns_exact_evidence_or_none(self, tmp_path: Path) -> None:
        evidence = _evidence()
        write_prediction_input_evidence(evidence, repo=tmp_path)

        assert (
            find_prediction_input_evidence_by_event(
                "event-1",
                repo=tmp_path,
            )
            == evidence
        )
        assert (
            find_prediction_input_evidence_by_event(
                "unknown-event",
                repo=tmp_path,
            )
            is None
        )

    def test_event_lookup_rejects_ambiguous_claims(self, tmp_path: Path) -> None:
        first = _evidence(run_id="run-1")
        second = _evidence(run_id="run-2")
        assert first.evidence_id != second.evidence_id
        write_prediction_input_evidence(first, repo=tmp_path)
        write_prediction_input_evidence(second, repo=tmp_path)

        with pytest.raises(ValueError, match=r"Multiple.*claim event"):
            find_prediction_input_evidence_by_event("event-1", repo=tmp_path)

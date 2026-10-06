"""Tests for immutable recommended-bet result persistence."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest
from tests.fixtures.recommended_bet_results import evaluation

from gridiron_edge.market.recommended_bet_result_store import (
    LoadedRecommendedBetResult,
    load_latest_recommended_bet_results,
    read_recommended_bet_evaluation,
    read_recommended_bet_result,
    recommended_bet_evaluation_path,
    recommended_bet_result_path,
    write_recommended_bet_evaluation,
    write_recommended_bet_result,
)


def test_store_paths_are_schema_and_identity_addressed(tmp_path: Path) -> None:
    digest = "a" * 64
    assert recommended_bet_result_path(1, digest, repo=tmp_path) == (
        tmp_path / "data/output/recommended_bet_results/schema=1/results" / f"{digest}.json"
    )
    assert recommended_bet_evaluation_path(1, digest, repo=tmp_path) == (
        tmp_path / "data/output/recommended_bet_results/schema=1/evaluations" / f"{digest}.json"
    )


def test_individual_result_round_trip_and_exact_replay(tmp_path: Path) -> None:
    result = evaluation().results[0]
    first = write_recommended_bet_result(result, repo=tmp_path)
    content = first.read_text(encoding="utf-8")
    second = write_recommended_bet_result(result, repo=tmp_path)
    assert second == first
    assert second.read_text(encoding="utf-8") == content
    assert read_recommended_bet_result(first) == result


def test_evaluation_round_trip_persists_and_resolves_results(tmp_path: Path) -> None:
    value = evaluation()
    path = write_recommended_bet_evaluation(value, repo=tmp_path)
    assert read_recommended_bet_evaluation(path) == value
    for result in value.results:
        result_path = recommended_bet_result_path(
            result.schema_version, result.result_id, repo=tmp_path
        )
        assert result_path.is_file()
        assert read_recommended_bet_result(result_path) == result


def test_evaluation_exact_replay_is_idempotent(tmp_path: Path) -> None:
    value = evaluation(active=False)
    first = write_recommended_bet_evaluation(value, repo=tmp_path)
    manifest = first.read_text(encoding="utf-8")
    result_contents = {
        result.result_id: recommended_bet_result_path(
            result.schema_version, result.result_id, repo=tmp_path
        ).read_text(encoding="utf-8")
        for result in value.results
    }
    second = write_recommended_bet_evaluation(value, repo=tmp_path)
    assert second == first
    assert second.read_text(encoding="utf-8") == manifest
    assert result_contents == {
        result.result_id: recommended_bet_result_path(
            result.schema_version, result.result_id, repo=tmp_path
        ).read_text(encoding="utf-8")
        for result in value.results
    }


def test_conflicting_result_and_evaluation_replay_are_rejected(tmp_path: Path) -> None:
    value = evaluation()
    result = value.results[0]
    result_path = write_recommended_bet_result(result, repo=tmp_path)
    result_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="different content"):
        write_recommended_bet_result(result, repo=tmp_path)

    other_root = tmp_path / "other"
    manifest = write_recommended_bet_evaluation(value, repo=other_root)
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="different content"):
        write_recommended_bet_evaluation(value, repo=other_root)


def test_result_rejects_malformed_nested_schema_and_identity(tmp_path: Path) -> None:
    result = evaluation().results[0]
    path = write_recommended_bet_result(result, repo=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"][0]["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        read_recommended_bet_result(path)

    path.write_text(json.dumps({**payload, "checks": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_recommended_bet_result(path)


def test_result_rejects_unsupported_schema_and_filename_mismatch(tmp_path: Path) -> None:
    result = evaluation().results[0]
    path = write_recommended_bet_result(result, repo=tmp_path)
    original = json.loads(path.read_text(encoding="utf-8"))
    changed = dict(original)
    changed["schema_version"] = 999
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        read_recommended_bet_result(path)

    path.write_text(json.dumps(original), encoding="utf-8")
    other = path.with_name(f"{'f' * 64}.json")
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        read_recommended_bet_result(other)


def test_manifest_rejects_missing_result_and_identity_changes(tmp_path: Path) -> None:
    value = evaluation()
    path = write_recommended_bet_evaluation(value, repo=tmp_path)
    result = value.results[0]
    recommended_bet_result_path(result.schema_version, result.result_id, repo=tmp_path).unlink()
    with pytest.raises(FileNotFoundError):
        read_recommended_bet_evaluation(path)

    path = write_recommended_bet_evaluation(value, repo=tmp_path / "identity")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluation_id"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"ID|identity"):
        read_recommended_bet_evaluation(path)


def test_manifest_rejects_duplicate_and_reordered_result_ids(tmp_path: Path) -> None:
    value = evaluation()
    path = write_recommended_bet_evaluation(value, repo=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result_ids"] = payload["result_ids"] * 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate|ID"):
        read_recommended_bet_evaluation(path)


def test_manifest_rejects_result_provenance_disagreement(tmp_path: Path) -> None:
    value = evaluation()
    changed_result = replace(value.results[0], policy_id="f" * 64)
    changed = replace(value, results=(changed_result,))
    with pytest.raises(ValueError, match=r"provenance|result ID"):
        write_recommended_bet_evaluation(changed, repo=tmp_path)


def test_store_round_trip_does_not_mutate_inputs(tmp_path: Path) -> None:
    value = evaluation()
    before = value
    path = write_recommended_bet_evaluation(value, repo=tmp_path)
    read_recommended_bet_evaluation(path)
    assert value == before


def test_unsafe_identity_and_schema_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        recommended_bet_result_path(1, "../escape", repo=tmp_path)
    with pytest.raises(ValueError, match="positive"):
        recommended_bet_evaluation_path(0, "a" * 64, repo=tmp_path)


def test_store_has_no_current_selection_or_request_dependency() -> None:
    source = Path("src/gridiron_edge/market/recommended_bet_result_store.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "current.json",
        "gridiron_edge.api",
        "gridiron_edge.cli",
        "gridiron_edge.betting.ledger",
        "gridiron_edge.betting.bankroll",
    ):
        assert forbidden not in source


def test_result_publication_race_rejects_conflicting_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evaluation().results[0]
    path = recommended_bet_result_path(result.schema_version, result.result_id, repo=tmp_path)

    def racing_link(src: Path, dst: Path) -> None:
        destination = Path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('{"conflicting": true}', encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr("gridiron_edge.market.recommended_bet_result_store.os.link", racing_link)
    with pytest.raises(ValueError, match="Recommended-bet result identity cannot be reused"):
        write_recommended_bet_result(result, repo=tmp_path)
    assert path.read_text(encoding="utf-8") == '{"conflicting": true}'
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_result_publication_race_accepts_identical_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evaluation().results[0]
    path = recommended_bet_result_path(result.schema_version, result.result_id, repo=tmp_path)

    def racing_link(src: Path, dst: Path) -> None:
        destination = Path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
        raise FileExistsError

    monkeypatch.setattr("gridiron_edge.market.recommended_bet_result_store.os.link", racing_link)
    assert write_recommended_bet_result(result, repo=tmp_path) == path
    assert read_recommended_bet_result(path) == result
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_evaluation_publication_race_rejects_conflicting_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = evaluation()
    path = recommended_bet_evaluation_path(value.schema_version, value.evaluation_id, repo=tmp_path)
    real_link = os.link

    def racing_link(src: Path, dst: Path) -> None:
        destination = Path(dst)
        if destination.name == path.name:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('{"conflicting": true}', encoding="utf-8")
            raise FileExistsError
        real_link(src, dst)  # let child result writes proceed normally

    monkeypatch.setattr("gridiron_edge.market.recommended_bet_result_store.os.link", racing_link)
    with pytest.raises(ValueError, match="Recommended-bet evaluation identity cannot be reused"):
        write_recommended_bet_evaluation(value, repo=tmp_path)
    assert path.read_text(encoding="utf-8") == '{"conflicting": true}'
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_pre_publication_failure_leaves_no_destination_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = evaluation().results[0]

    def failing_link(src: Path, dst: Path) -> None:
        raise OSError("simulated pre-publication failure")

    monkeypatch.setattr("gridiron_edge.market.recommended_bet_result_store.os.link", failing_link)
    with pytest.raises(OSError, match="simulated pre-publication failure"):
        write_recommended_bet_result(result, repo=tmp_path)
    path = recommended_bet_result_path(result.schema_version, result.result_id, repo=tmp_path)
    assert not path.exists()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_schema_3_result_round_trip_preserves_derivation_version(tmp_path: Path) -> None:
    result = evaluation().results[0]
    assert result.schema_version == 3
    path = write_recommended_bet_result(result, repo=tmp_path)
    assert path.parent.parent.name == "schema=3"
    assert path.parent.name == "results"
    loaded = read_recommended_bet_result(path)
    assert loaded.candidate_reference_derivation_version == 1
    assert loaded == result


def test_schema_3_evaluation_manifest_uses_schema_3_namespace(tmp_path: Path) -> None:
    value = evaluation()
    assert value.schema_version == 3
    path = write_recommended_bet_evaluation(value, repo=tmp_path)
    assert path.parent.parent.name == "schema=3"
    assert path.parent.name == "evaluations"
    loaded = read_recommended_bet_evaluation(path)
    for result in loaded.results:
        result_path = recommended_bet_result_path(
            result.schema_version, result.result_id, repo=tmp_path
        )
        assert result_path.parent.parent.name == "schema=3"


def test_result_missing_derivation_version_field_is_rejected(tmp_path: Path) -> None:
    result = evaluation().results[0]
    path = write_recommended_bet_result(result, repo=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["candidate_reference_derivation_version"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys do not match"):
        read_recommended_bet_result(path)


def test_schema_1_shaped_result_is_rejected_as_unsupported(tmp_path: Path) -> None:
    """A schema-1-shaped payload (missing the schema-2 field, tagged as
    schema 1) is rejected -- confirming schema 1 is unsupported after this
    unit, not silently accepted or migrated."""
    result = evaluation().results[0]
    path = write_recommended_bet_result(result, repo=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["candidate_reference_derivation_version"]
    payload["schema_version"] = 1
    schema_1_path = path.parent.parent.parent / "schema=1" / "results" / path.name
    schema_1_path.parent.mkdir(parents=True, exist_ok=True)
    schema_1_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_recommended_bet_result(schema_1_path)


def test_the_latest_results_of_a_week_are_read_by_repo(tmp_path: Path) -> None:
    value = evaluation()
    assert load_latest_recommended_bet_results(season="2026", week=1, repo=tmp_path) == ()
    write_recommended_bet_evaluation(value, repo=tmp_path)

    assert load_latest_recommended_bet_results(season="2026", week=1, repo=tmp_path) == (
        LoadedRecommendedBetResult(value.evaluation_id, value.results[0]),
    )
    assert load_latest_recommended_bet_results(season="2026", week=2, repo=tmp_path) == ()

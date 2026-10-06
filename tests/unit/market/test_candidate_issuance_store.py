"""Tests for immutable candidate issuance persistence."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path

import pytest

from gridiron_edge.market.candidate_issuance import (
    CANDIDATE_ISSUANCE_SCHEMA_VERSION,
    CandidateIssuance,
    CandidateIssuanceReason,
    CandidateIssuanceRow,
    CandidateIssuanceState,
    candidate_issuance_id,
)
from gridiron_edge.market.candidate_issuance_store import (
    candidate_issuance_path,
    candidate_issuances_for_product,
    read_candidate_issuance,
    write_candidate_issuance,
)

EVALUATED = datetime(2026, 9, 1, 12, tzinfo=UTC)


def _issuance() -> CandidateIssuance:
    issuance_id = candidate_issuance_id(
        product_id="product-1",
        product_run_id="run-1",
        season="2026-2027",
        week=1,
        evaluated_at=EVALUATED,
    )
    row = CandidateIssuanceRow(
        game_id="2026_01_KC_LAC",
        market="moneyline",
        side="home",
        provider="the_odds_api",
        provider_event_id="event-1",
        sportsbook="draftkings",
        line=None,
        american_price=-110,
        fetched_at=EVALUATED,
        sportsbook_updated_at=EVALUATED,
        kickoff=datetime(2026, 9, 10, 0, 20, tzinfo=UTC),
        is_live=False,
        forecast_event_id="forecast-1",
        forecast_run_id="run-1",
        forecast_role="live",
        forecast_generated_at=EVALUATED,
        model_name="win_prob",
        model_type="elo",
        model_probability=0.60,
        expected_value=0.145,
        state=CandidateIssuanceState.CANDIDATE,
        reason=CandidateIssuanceReason.POSITIVE_EXPECTED_VALUE,
    )
    return CandidateIssuance(
        CANDIDATE_ISSUANCE_SCHEMA_VERSION,
        issuance_id,
        "product-1",
        "run-1",
        EVALUATED,
        "2026-2027",
        1,
        EVALUATED,
        (row,),
    )


def test_issuances_are_found_by_exact_product_run_and_week(tmp_path: Path) -> None:
    issuance = _issuance()
    write_candidate_issuance(issuance, repo=tmp_path)
    (tmp_path / "data/output/candidate_issuance/issuances/broken.json").write_text("{")
    query = {"season": "2026-2027", "week": 1, "product_id": "product-1", "repo": tmp_path}

    assert candidate_issuances_for_product(product_run_id="run-1", **query) == (issuance,)
    assert candidate_issuances_for_product(product_run_id="run-2", **query) == ()
    assert candidate_issuances_for_product(product_run_id="run-1", **(query | {"week": 2})) == ()
    assert (
        candidate_issuances_for_product(
            product_run_id="run-1", **(query | {"repo": tmp_path / "x"})
        )
        == ()
    )


def test_round_trip(tmp_path: Path) -> None:
    issuance = _issuance()
    path = write_candidate_issuance(issuance, repo=tmp_path)
    assert path == candidate_issuance_path(issuance.issuance_id, repo=tmp_path)
    assert read_candidate_issuance(path) == issuance


def test_exact_replay_is_idempotent(tmp_path: Path) -> None:
    issuance = _issuance()
    first = write_candidate_issuance(issuance, repo=tmp_path)
    first_content = first.read_text()
    second = write_candidate_issuance(issuance, repo=tmp_path)
    assert second == first
    assert second.read_text() == first_content


def test_conflicting_replay_is_rejected(tmp_path: Path) -> None:
    issuance = _issuance()
    write_candidate_issuance(issuance, repo=tmp_path)
    changed = replace(issuance, rows=(replace(issuance.rows[0], expected_value=0.20),))
    with pytest.raises(ValueError, match="different content"):
        write_candidate_issuance(changed, repo=tmp_path)


def test_malformed_artifact_is_rejected(tmp_path: Path) -> None:
    issuance = _issuance()
    path = candidate_issuance_path(issuance.issuance_id, repo=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="keys"):
        read_candidate_issuance(path)


def test_unsafe_identity_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        candidate_issuance_path("../escape", repo=tmp_path)


def _stored_payload(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Write one valid artifact and return its parsed mutable payload."""
    issuance = _issuance()
    path = write_candidate_issuance(issuance, repo=tmp_path)
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    return path, payload


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    """Stored artifacts must use the current exact schema version."""
    path, payload = _stored_payload(tmp_path)
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema version"):
        read_candidate_issuance(path)


def test_unexpected_row_keys_are_rejected(tmp_path: Path) -> None:
    """Stored rows cannot silently acquire unknown evidence fields."""
    path, payload = _stored_payload(tmp_path)
    rows = payload["rows"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    row["unexpected"] = "value"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="row keys"):
        read_candidate_issuance(path)


def test_embedded_issuance_id_mismatch_is_rejected(tmp_path: Path) -> None:
    """The embedded deterministic identity must match its payload."""
    path, payload = _stored_payload(tmp_path)
    payload["product_id"] = "different-product"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity payload"):
        read_candidate_issuance(path)


def test_filename_and_issuance_id_mismatch_is_rejected(tmp_path: Path) -> None:
    """A valid artifact cannot be moved under another issuance filename."""
    path, payload = _stored_payload(tmp_path)
    other = path.with_name(f"{'0' * 64}.json")
    other.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="filename"):
        read_candidate_issuance(other)


def test_nondeterministic_row_order_is_rejected(tmp_path: Path) -> None:
    """Stored row order is part of the canonical serialized contract."""
    issuance = _issuance()
    later = replace(
        issuance.rows[0],
        sportsbook="fanduel",
        fetched_at=datetime(2026, 9, 1, 13, tzinfo=UTC),
    )
    unordered = replace(issuance, rows=(later, issuance.rows[0]))
    with pytest.raises(ValueError, match="deterministically ordered"):
        write_candidate_issuance(unordered, repo=tmp_path)


def test_duplicate_stored_row_identity_is_rejected(tmp_path: Path) -> None:
    """One immutable quote identity cannot appear twice in an issuance."""
    issuance = _issuance()
    duplicated = replace(issuance, rows=(issuance.rows[0], issuance.rows[0]))
    with pytest.raises(ValueError, match="duplicate row identities"):
        write_candidate_issuance(duplicated, repo=tmp_path)


def test_publication_race_rejects_valid_conflicting_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuance = _issuance()
    conflicting = replace(issuance, rows=(replace(issuance.rows[0], expected_value=0.20),))
    competing_root = tmp_path / "competing"
    competing_path = write_candidate_issuance(conflicting, repo=competing_root)
    competing_bytes = competing_path.read_bytes()
    path = candidate_issuance_path(issuance.issuance_id, repo=tmp_path)

    def racing_link(src: Path, dst: Path) -> None:
        destination = Path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(competing_bytes)
        raise FileExistsError

    monkeypatch.setattr("gridiron_edge.market.candidate_issuance_store.os.link", racing_link)
    with pytest.raises(ValueError, match="different content"):
        write_candidate_issuance(issuance, repo=tmp_path)
    assert path.read_bytes() == competing_bytes
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_publication_race_accepts_identical_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuance = _issuance()
    path = candidate_issuance_path(issuance.issuance_id, repo=tmp_path)

    def racing_link(src: Path, dst: Path) -> None:
        destination = Path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(src).read_bytes())
        raise FileExistsError

    monkeypatch.setattr("gridiron_edge.market.candidate_issuance_store.os.link", racing_link)
    result_path = write_candidate_issuance(issuance, repo=tmp_path)
    assert result_path == path
    assert read_candidate_issuance(path) == issuance
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_pre_publication_failure_leaves_no_destination_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issuance = _issuance()
    path = candidate_issuance_path(issuance.issuance_id, repo=tmp_path)

    def failing_link(src: Path, dst: Path) -> None:
        raise OSError("simulated pre-publication failure")

    monkeypatch.setattr("gridiron_edge.market.candidate_issuance_store.os.link", failing_link)
    with pytest.raises(OSError, match="simulated pre-publication failure"):
        write_candidate_issuance(issuance, repo=tmp_path)
    assert not path.exists()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_temporary_serialization_failure_leaves_no_destination_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct from pre-publication failure: this fails DURING serialization,
    before the temporary file is even complete, and before os.link is ever called."""
    issuance = _issuance()
    path = candidate_issuance_path(issuance.issuance_id, repo=tmp_path)
    link_called = False
    real_link = os.link

    def spying_link(src: Path, dst: Path) -> None:
        nonlocal link_called
        link_called = True
        real_link(src, dst)

    def failing_dump(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr("gridiron_edge.market.candidate_issuance_store.os.link", spying_link)
    monkeypatch.setattr("gridiron_edge.market.candidate_issuance_store.json.dumps", failing_dump)
    with pytest.raises(RuntimeError, match="simulated serialization failure"):
        write_candidate_issuance(issuance, repo=tmp_path)
    assert not link_called
    assert not path.exists()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []

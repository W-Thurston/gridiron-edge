# tests/unit/evaluation/test_forecast_evidence_disposition_store.py
"""Tests for immutable forecast-evidence disposition storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gridiron_edge.evaluation.forecast_evidence_disposition import (
    ForecastEvidenceDisposition,
    create_forecast_evidence_disposition,
)
from gridiron_edge.evaluation.forecast_evidence_disposition_store import (
    FORECAST_EVIDENCE_DISPOSITION_STORE_SCHEMA_VERSION,
    forecast_evidence_disposition_path,
    forecast_evidence_disposition_root,
    list_forecast_evidence_dispositions,
    read_forecast_evidence_disposition,
    write_forecast_evidence_disposition,
)

RECORDED_AT = datetime(2026, 9, 18, 18, tzinfo=UTC)


def _disposition(
    *,
    recorded_at: datetime = RECORDED_AT,
    season: str = "2026-2027",
    week: int = 2,
    products: tuple[str, ...] = ("product-a", "product-b"),
    selected_product: str = "product-b",
) -> ForecastEvidenceDisposition:
    return create_forecast_evidence_disposition(
        recorded_at=recorded_at,
        season=season,
        week=week,
        affected_run_ids=("run-a", "run-b"),
        affected_event_ids=tuple(f"event-{index:02d}" for index in range(32)),
        affected_product_ids=products,
        selected_affected_product_id=selected_product,
        decision_references=("D38", "D39", "D40"),
        evidence_summary="Both Week 2 Win runs used reset Elo state.",
    )


def _payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_path_is_schema_versioned_and_identity_addressed(tmp_path: Path) -> None:
    disposition = _disposition()
    path = forecast_evidence_disposition_path(
        disposition.disposition_id,
        repo=tmp_path,
    )
    assert path == (
        tmp_path
        / "data/output/forecast_evidence_dispositions"
        / "schema=1"
        / "dispositions"
        / f"{disposition.disposition_id}.json"
    )


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    actual = read_forecast_evidence_disposition(path)
    assert actual == disposition
    payload = _payload(path)
    assert payload["store_schema_version"] == (FORECAST_EVIDENCE_DISPOSITION_STORE_SCHEMA_VERSION)
    assert payload["disposition_id"] == disposition.disposition_id


def test_exact_replay_is_idempotent(tmp_path: Path) -> None:
    disposition = _disposition()
    first = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    before = first.read_bytes()
    second = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    assert second == first
    assert second.read_bytes() == before


def test_conflicting_identity_reuse_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be reused"):
        write_forecast_evidence_disposition(disposition, repo=tmp_path)


def test_concurrent_exact_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    disposition = _disposition()
    expected_path = forecast_evidence_disposition_path(
        disposition.disposition_id,
        repo=tmp_path,
    )

    def publish_exact_replay(
        temporary: str | Path,
        destination: str | Path,
    ) -> None:
        source_path = Path(temporary)
        destination_path = Path(destination)

        destination_path.write_bytes(source_path.read_bytes())
        raise FileExistsError

    with patch(
        "gridiron_edge.evaluation.forecast_evidence_disposition_store.os.link",
        side_effect=publish_exact_replay,
    ) as mock_link:
        actual_path = write_forecast_evidence_disposition(
            disposition,
            repo=tmp_path,
        )

    assert actual_path == expected_path
    assert read_forecast_evidence_disposition(actual_path) == disposition
    mock_link.assert_called_once()

    temporary_files = tuple(expected_path.parent.glob(f".{expected_path.name}.*.tmp"))
    assert temporary_files == ()


def test_concurrent_conflicting_publication_is_rejected(
    tmp_path: Path,
) -> None:
    disposition = _disposition()
    expected_path = forecast_evidence_disposition_path(
        disposition.disposition_id,
        repo=tmp_path,
    )
    conflicting_content = b"{}\n"

    def publish_conflicting_content(
        _temporary: str | Path,
        destination: str | Path,
    ) -> None:
        destination_path = Path(destination)

        destination_path.write_bytes(conflicting_content)
        raise FileExistsError

    with (
        patch(
            "gridiron_edge.evaluation.forecast_evidence_disposition_store.os.link",
            side_effect=publish_conflicting_content,
        ) as mock_link,
        pytest.raises(
            ValueError,
            match="cannot be reused with different content",
        ),
    ):
        write_forecast_evidence_disposition(
            disposition,
            repo=tmp_path,
        )

    assert expected_path.read_bytes() == conflicting_content
    mock_link.assert_called_once()

    temporary_files = tuple(expected_path.parent.glob(f".{expected_path.name}.*.tmp"))
    assert temporary_files == ()


def test_invalid_disposition_id_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        forecast_evidence_disposition_path("bad", repo=tmp_path)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = forecast_evidence_disposition_path(
        disposition.disposition_id,
        repo=tmp_path,
    )
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        read_forecast_evidence_disposition(path)


@pytest.mark.parametrize(
    ("target", "key"),
    [
        ("root", "unexpected"),
        ("disposition", "unexpected"),
    ],
)
def test_unexpected_keys_are_rejected(
    tmp_path: Path,
    target: str,
    key: str,
) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    if target == "root":
        payload[key] = "value"
    else:
        body = payload["disposition"]
        assert isinstance(body, dict)
        body[key] = "value"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys do not match"):
        read_forecast_evidence_disposition(path)


def test_missing_keys_are_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    payload.pop("disposition_id")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="keys do not match"):
        read_forecast_evidence_disposition(path)


def test_unsupported_store_schema_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    payload["store_schema_version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"Unsupported.*store schema",
    ):
        read_forecast_evidence_disposition(path)


def test_unsupported_domain_schema_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    body = payload["disposition"]
    assert isinstance(body, dict)
    body["schema_version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"Unsupported.*schema_version",
    ):
        read_forecast_evidence_disposition(path)


def test_invalid_enum_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    body = payload["disposition"]
    assert isinstance(body, dict)
    body["status"] = "not-real"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported value"):
        read_forecast_evidence_disposition(path)


def test_invalid_timestamp_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    body = payload["disposition"]
    assert isinstance(body, dict)
    body["recorded_at"] = "not-a-timestamp"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ISO timestamp"):
        read_forecast_evidence_disposition(path)


def test_naive_timestamp_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    body = payload["disposition"]
    assert isinstance(body, dict)
    body["recorded_at"] = "2026-09-18T18:00:00"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        read_forecast_evidence_disposition(path)


def test_boolean_integer_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    body = payload["disposition"]
    assert isinstance(body, dict)
    body["week"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="week must be an integer"):
        read_forecast_evidence_disposition(path)


def test_embedded_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    payload["disposition_id"] = "a" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        read_forecast_evidence_disposition(path)


def test_content_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    path = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    payload = _payload(path)
    body = payload["disposition"]
    assert isinstance(body, dict)
    body["evidence_summary"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match canonical"):
        read_forecast_evidence_disposition(path)


def test_path_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    original = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    wrong = original.with_name(f"{'a' * 64}.json")
    wrong.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="path and embedded identity disagree"):
        read_forecast_evidence_disposition(wrong)


def test_path_outside_canonical_store_is_rejected(tmp_path: Path) -> None:
    disposition = _disposition()
    stored = write_forecast_evidence_disposition(disposition, repo=tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes(stored.read_bytes())
    with pytest.raises(ValueError, match="outside the canonical store"):
        read_forecast_evidence_disposition(outside)


def test_empty_store_lists_no_dispositions(tmp_path: Path) -> None:
    assert list_forecast_evidence_dispositions(repo=tmp_path) == ()


def test_listing_is_sorted_by_identity(tmp_path: Path) -> None:
    first = _disposition()
    second = _disposition(recorded_at=RECORDED_AT + timedelta(seconds=1))
    write_forecast_evidence_disposition(second, repo=tmp_path)
    write_forecast_evidence_disposition(first, repo=tmp_path)
    actual = list_forecast_evidence_dispositions(repo=tmp_path)
    assert tuple(value.disposition_id for value in actual) == tuple(
        sorted((first.disposition_id, second.disposition_id))
    )


def test_listing_filters_by_scope_and_product(tmp_path: Path) -> None:
    week_two = _disposition()
    week_three = _disposition(
        recorded_at=RECORDED_AT + timedelta(seconds=1),
        week=3,
        products=("product-c",),
        selected_product="product-c",
    )
    write_forecast_evidence_disposition(week_three, repo=tmp_path)
    write_forecast_evidence_disposition(week_two, repo=tmp_path)

    assert list_forecast_evidence_dispositions(season="2026-2027", week=2, repo=tmp_path) == (
        week_two,
    )
    assert list_forecast_evidence_dispositions(product_id="product-c", repo=tmp_path) == (
        week_three,
    )
    assert list_forecast_evidence_dispositions(season="2025-2026", repo=tmp_path) == ()


def test_listing_rejects_invalid_filters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="season must not be empty"):
        list_forecast_evidence_dispositions(season=" ", repo=tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        list_forecast_evidence_dispositions(week=True, repo=tmp_path)
    with pytest.raises(ValueError, match="product_id must be a nonempty string"):
        list_forecast_evidence_dispositions(product_id=" ", repo=tmp_path)


def test_listing_surfaces_malformed_artifact(tmp_path: Path) -> None:
    root = forecast_evidence_disposition_root(tmp_path) / "schema=1" / "dispositions"
    root.mkdir(parents=True)
    (root / f"{'a' * 64}.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        list_forecast_evidence_dispositions(repo=tmp_path)

# tests/unit/evaluation/test_forecast_evidence_disposition.py
"""Tests for immutable forecast-evidence disposition contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256
import json

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.forecast_evidence_disposition import (
    FORECAST_EVIDENCE_DISPOSITION_SCHEMA_VERSION,
    AffectedPredictionComponent,
    ForecastEvidenceDefectReason,
    ForecastEvidenceDisposition,
    ForecastEvidenceDispositionStatus,
    ForecastEvidenceNotOperationalError,
    ForecastEvidencePublicationEffect,
    ForecastEvidenceReplacementPolicy,
    authenticate_forecast_evidence_disposition,
    create_forecast_evidence_disposition,
    disposition_applies_to_product,
    require_operational_weekly_product,
    validate_forecast_evidence_disposition,
)
from gridiron_edge.evaluation.forecast_store import FORECAST_EVENT_COLUMNS

RECORDED_AT = datetime(2026, 9, 18, 18, tzinfo=UTC)
SEASON = "2026-2027"
WEEK = 2
RUN_IDS = ("run-a", "run-b")
PRODUCT_IDS = ("product-a", "product-b")


def _event_id(run_id: str, index: int) -> str:
    return f"event-{run_id}-{index:02d}"


def _event_ids() -> tuple[str, ...]:
    return tuple(sorted(_event_id(run_id, index) for run_id in RUN_IDS for index in range(16)))


def _disposition(**overrides: object) -> ForecastEvidenceDisposition:
    values: dict[str, object] = {
        "recorded_at": RECORDED_AT,
        "season": SEASON,
        "week": WEEK,
        "affected_run_ids": RUN_IDS,
        "affected_event_ids": _event_ids(),
        "affected_product_ids": PRODUCT_IDS,
        "selected_affected_product_id": "product-b",
        "decision_references": ("D38", "D39", "D40"),
        "evidence_summary": "Both Week 2 Win runs used reset Elo state.",
    }
    values.update(overrides)
    return create_forecast_evidence_disposition(**values)  # type: ignore[arg-type]


def _events() -> DataFrame:
    rows: list[dict[str, object]] = []
    for run_index, run_id in enumerate(RUN_IDS):
        generated_at = datetime(2026, 9, 15, 13 + run_index, tzinfo=UTC)
        for index in range(16):
            rows.append(
                {
                    "event_id": _event_id(run_id, index),
                    "run_id": run_id,
                    "role": "live",
                    "generated_at": generated_at,
                    "season": SEASON,
                    "week": WEEK,
                    "game_id": f"game-{index:02d}",
                    "model_name": "win_prob",
                    "model_type": "logistic",
                    "game_date": "2026-09-20",
                    "away_team": f"Away {index}",
                    "home_team": f"Home {index}",
                    "away_elo": 1490.0,
                    "home_elo": 1510.0,
                    "away_win_prob": 0.45,
                    "home_win_prob": 0.55,
                    "model_spread": None,
                    "model_total": None,
                    "projected_home_score": None,
                    "projected_away_score": None,
                    "margin_std": None,
                    "win_prob_lo": None,
                    "win_prob_hi": None,
                    "confidence_tier": None,
                }
            )
    return DataFrame(rows, columns=FORECAST_EVENT_COLUMNS)


def _product(product_id: str, run_id: str) -> DataFrame:
    return DataFrame(
        {
            "product_id": [product_id] * 16,
            "product_run_id": [run_id] * 16,
            "season": [SEASON] * 16,
            "week": [WEEK] * 16,
            "game_id": [f"game-{index:02d}" for index in range(16)],
            "win_event_id": [_event_id(run_id, index) for index in range(16)],
            "win_run_id": [run_id] * 16,
            "spread_status": ["available"] * 16,
            "spread_source_event_id": [_event_id(run_id, index) for index in range(16)],
            "total_event_id": [None] * 16,
            "total_run_id": [None] * 16,
            "total_model_name": [None] * 16,
            "total_model_type": [None] * 16,
            "total_role": [None] * 16,
        }
    )


def _products() -> dict[str, DataFrame]:
    return {
        "product-a": _product("product-a", "run-a"),
        "product-b": _product("product-b", "run-b"),
    }


def test_create_builds_schema_one_known_defect() -> None:
    result = _disposition()
    assert result.schema_version == FORECAST_EVIDENCE_DISPOSITION_SCHEMA_VERSION
    assert result.status is ForecastEvidenceDispositionStatus.KNOWN_DEFECT
    assert result.reason is ForecastEvidenceDefectReason.INCOMPLETE_ELO_SOURCE_HISTORY
    assert result.publication_effect is ForecastEvidencePublicationEffect.NOT_PREDICTION_READY
    assert result.replacement_policy is (
        ForecastEvidenceReplacementPolicy.PRESERVE_ORIGINAL_NO_AUTOMATIC_RESELECTION
    )
    assert result.affected_components == (
        AffectedPredictionComponent.DERIVED_SPREAD,
        AffectedPredictionComponent.WIN_PROBABILITY,
    )


def test_identity_is_digest_of_complete_canonical_payload() -> None:
    result = _disposition()
    payload = {
        "schema_version": 1,
        "recorded_at": RECORDED_AT.isoformat(),
        "season": SEASON,
        "week": WEEK,
        "status": "known_defect",
        "reason": "incomplete_elo_source_history",
        "publication_effect": "not_prediction_ready",
        "replacement_policy": "preserve_original_no_automatic_reselection",
        "affected_components": ["derived_spread", "win_probability"],
        "affected_run_ids": list(RUN_IDS),
        "affected_event_ids": list(_event_ids()),
        "affected_product_ids": list(PRODUCT_IDS),
        "selected_affected_product_id": "product-b",
        "decision_references": ["D38", "D39", "D40"],
        "evidence_summary": "Both Week 2 Win runs used reset Elo state.",
    }
    expected = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert result.disposition_id == expected


def test_same_inputs_create_same_identity() -> None:
    assert _disposition().disposition_id == _disposition().disposition_id


def test_recorded_at_changes_identity() -> None:
    assert (
        _disposition().disposition_id
        != _disposition(recorded_at=RECORDED_AT + timedelta(seconds=1)).disposition_id
    )


def test_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        validate_forecast_evidence_disposition(replace(_disposition(), schema_version=999))


def test_rejects_invalid_digest() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_forecast_evidence_disposition(replace(_disposition(), disposition_id="bad"))


def test_rejects_naive_recorded_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _disposition(recorded_at=datetime(2026, 9, 18, 18))


def test_rejects_non_utc_recorded_at() -> None:
    with pytest.raises(ValueError, match="must use UTC"):
        _disposition(recorded_at=datetime(2026, 9, 18, 12, tzinfo=timezone(timedelta(hours=-6))))


@pytest.mark.parametrize("season", ["2026", "2026-27", "2026-2028", "bad"])
def test_rejects_invalid_season(season: str) -> None:
    with pytest.raises(ValueError, match="season"):
        _disposition(season=season)


def test_rejects_invalid_week() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _disposition(week=0)


def test_rejects_boolean_week() -> None:
    with pytest.raises(ValueError, match="integer"):
        _disposition(week=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("affected_run_ids", ("run-a", "run-a")),
        ("affected_event_ids", ("event-b", "event-a")),
        ("affected_product_ids", ("product-b", "product-a")),
        ("decision_references", ("D40", "")),
    ],
)
def test_rejects_invalid_identity_collections(field: str, value: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match=r"sorted unique|nonempty"):
        _disposition(**{field: value})


def test_rejects_selected_product_outside_affected_products() -> None:
    with pytest.raises(ValueError, match="must belong"):
        _disposition(selected_affected_product_id="product-c")


def test_rejects_missing_required_component() -> None:
    result = _disposition()
    changed = replace(
        result,
        affected_components=(AffectedPredictionComponent.WIN_PROBABILITY,),
    )
    with pytest.raises(ValueError, match="exactly"):
        validate_forecast_evidence_disposition(changed)


def test_rejects_unexpected_component() -> None:
    result = _disposition()
    changed = replace(
        result,
        affected_components=(
            AffectedPredictionComponent.WIN_PROBABILITY,
            AffectedPredictionComponent.DERIVED_SPREAD,
        ),
    )
    with pytest.raises(ValueError, match="exactly"):
        validate_forecast_evidence_disposition(changed)


def test_rejects_empty_evidence_summary() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _disposition(evidence_summary=" ")


def test_rejects_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        validate_forecast_evidence_disposition(
            replace(_disposition(), evidence_summary="Changed summary")
        )


def test_authenticates_complete_affected_evidence() -> None:
    authenticate_forecast_evidence_disposition(
        _disposition(),
        forecast_events=_events(),
        weekly_products=_products(),
        selected_product_id="product-b",
    )


def test_rejects_missing_affected_event() -> None:
    with pytest.raises(ValueError, match="missing"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events().iloc[:-1].copy(),
            weekly_products=_products(),
            selected_product_id="product-b",
        )


def test_rejects_unlisted_affected_scope_event() -> None:
    events = _events()
    extra = events.iloc[[0]].copy()
    extra["event_id"] = "unexpected-event"
    events = pd.concat([events, extra], ignore_index=True)
    with pytest.raises(ValueError, match="every live logistic Win event"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=events,
            weekly_products=_products(),
            selected_product_id="product-b",
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("season", "2025-2026", r"every live logistic Win event|season"),
        ("role", "backfilled", "every live logistic Win event|live role"),
        ("model_name", "total", "every live logistic Win event|model_name"),
        ("model_type", "xgboost", "every live logistic Win event|model_type"),
    ],
)
def test_rejects_wrong_event_identity(column: str, value: object, message: str) -> None:
    events = _events()
    events.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=events,
            weekly_products=_products(),
            selected_product_id="product-b",
        )


def test_rejects_run_with_wrong_event_count() -> None:
    disposition = _disposition(affected_event_ids=_event_ids()[:-1])
    events = _events().iloc[:-1].copy()
    with pytest.raises(ValueError, match="exactly 16 events"):
        authenticate_forecast_evidence_disposition(
            disposition,
            forecast_events=events,
            weekly_products=_products(),
            selected_product_id="product-b",
        )


def test_rejects_run_with_duplicate_games() -> None:
    events = _events()
    events.loc[1, "game_id"] = events.loc[0, "game_id"]
    with pytest.raises(ValueError, match="unique game IDs"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=events,
            weekly_products=_products(),
            selected_product_id="product-b",
        )


def test_rejects_missing_product() -> None:
    with pytest.raises(ValueError, match="missing"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events(),
            weekly_products={"product-a": _products()["product-a"]},
            selected_product_id="product-b",
        )


def test_rejects_product_row_count_mismatch() -> None:
    products = _products()
    products["product-a"] = products["product-a"].iloc[:-1].copy()
    with pytest.raises(ValueError, match="exactly 16 rows"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events(),
            weekly_products=products,
            selected_product_id="product-b",
        )


def test_rejects_product_identity_mismatch() -> None:
    products = _products()
    products["product-a"]["product_id"] = "wrong"
    with pytest.raises(ValueError, match="mapping key"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events(),
            weekly_products=products,
            selected_product_id="product-b",
        )


def test_rejects_product_run_mismatch() -> None:
    products = _products()
    products["product-a"]["win_run_id"] = "run-b"
    with pytest.raises(ValueError, match="Win run IDs"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events(),
            weekly_products=products,
            selected_product_id="product-b",
        )


def test_rejects_product_event_set_mismatch() -> None:
    products = _products()
    products["product-a"].loc[0, "win_event_id"] = "wrong-event"
    with pytest.raises(ValueError, match="Win event IDs"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events(),
            weekly_products=products,
            selected_product_id="product-b",
        )


def test_rejects_spread_source_mismatch() -> None:
    products = _products()
    products["product-a"].loc[0, "spread_source_event_id"] = "wrong-event"
    with pytest.raises(ValueError, match="Derived spread source"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events(),
            weekly_products=products,
            selected_product_id="product-b",
        )


def test_rejects_claimed_total_evidence() -> None:
    products = _products()
    products["product-a"].loc[0, "total_event_id"] = "total-event"
    with pytest.raises(ValueError, match="must not claim Total"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events(),
            weekly_products=products,
            selected_product_id="product-b",
        )


def test_rejects_selected_product_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=_events(),
            weekly_products=_products(),
            selected_product_id="product-a",
        )


def test_disposition_applies_to_affected_product() -> None:
    assert disposition_applies_to_product(_disposition(), _product("product-a", "run-a"))


def test_disposition_does_not_apply_to_unrelated_product() -> None:
    assert not disposition_applies_to_product(_disposition(), _product("product-c", "run-a"))


def test_undisposed_product_is_operational() -> None:
    require_operational_weekly_product(_product("product-c", "run-a"), [_disposition()])


def test_one_known_defect_blocks_operational_use() -> None:
    disposition = _disposition()
    with pytest.raises(
        ForecastEvidenceNotOperationalError,
        match=f"product-a.*{disposition.disposition_id}.*known_defect.*incomplete_elo",
    ):
        require_operational_weekly_product(_product("product-a", "run-a"), [disposition])


def test_multiple_applicable_dispositions_are_ambiguous() -> None:
    first = _disposition()
    second = _disposition(recorded_at=RECORDED_AT + timedelta(seconds=1))
    with pytest.raises(ValueError, match="Multiple"):
        require_operational_weekly_product(
            _product("product-a", "run-a"),
            [first, second],
        )


def test_development_events_do_not_authenticate_live_defect_scope() -> None:
    events = _events()
    events["role"] = ForecastRole.DEVELOPMENT.value

    with pytest.raises(
        ValueError,
        match=r"every live logistic Win event|live role",
    ):
        authenticate_forecast_evidence_disposition(
            _disposition(),
            forecast_events=events,
            weekly_products=_products(),
            selected_product_id="product-b",
        )

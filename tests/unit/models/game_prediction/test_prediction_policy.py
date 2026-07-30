# tests/unit/models/game_prediction/test_prediction_policy.py

"""Tests for availability-aware game prediction policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from gridiron_edge.models.game_prediction.prediction_policy import (
    ModelProvenance,
    PredictionAvailability,
    PredictionModelSource,
    PredictionModelStatus,
    PredictionPolicyRationale,
    resolve_prediction_policy,
)


def _availability(
    *,
    week: int = 1,
    elo_available: bool = True,
    full_features_available: bool = False,
    total_features_available: bool = False,
) -> PredictionAvailability:
    """Create explicit weekly availability facts."""
    return PredictionAvailability(
        season="2026-2027",
        week=week,
        elo_available=elo_available,
        full_features_available=(full_features_available),
        total_features_available=(total_features_available),
    )


def _champion(
    model_name: str,
    model_type: str,
) -> ModelProvenance:
    """Create representative champion provenance."""
    return ModelProvenance(
        model_name=model_name,
        model_type=model_type,
        source=PredictionModelSource.CHAMPION,
        promoted_at="2026-07-01T14:20:00Z",
        source_run_id="20260701_142000",
        metrics=(
            (
                "score",
                0.21,
            ),
        ),
    )


def test_week_one_selects_only_eligible_elo_policy() -> None:
    policy = resolve_prediction_policy(
        _availability(),
        win_champion=_champion(
            "win_prob",
            "random_forest",
        ),
        total_champion=_champion(
            "total",
            "xgboost",
        ),
    )

    assert policy.win.status is (PredictionModelStatus.SELECTED)
    assert policy.win.model_type == "elo"
    assert policy.win.rationale is (PredictionPolicyRationale.ELO_ONLY_AVAILABLE)

    assert policy.total.status is (PredictionModelStatus.UNAVAILABLE)
    assert policy.total.model_type is None
    assert policy.total.rationale is (PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE)


def test_full_feature_champions_require_available_inputs() -> None:
    policy = resolve_prediction_policy(
        _availability(
            week=8,
            full_features_available=True,
            total_features_available=True,
        ),
        win_champion=_champion(
            "win_prob",
            "random_forest",
        ),
        total_champion=_champion(
            "total",
            "xgboost",
        ),
    )

    assert policy.win.model_type == "random_forest"
    assert policy.win.rationale is (PredictionPolicyRationale.CHAMPION_ELIGIBLE)

    assert policy.total.model_type == "xgboost"
    assert policy.total.rationale is (PredictionPolicyRationale.CHAMPION_ELIGIBLE)


def test_unavailable_total_remains_explicit() -> None:
    policy = resolve_prediction_policy(
        _availability(
            total_features_available=False,
        ),
        win_champion=None,
        total_champion=_champion(
            "total",
            "xgboost",
        ),
    )

    assert policy.total.status is (PredictionModelStatus.UNAVAILABLE)
    assert policy.total.model_type is None
    assert policy.total.provenance is None
    assert policy.total.rationale is (PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE)


def test_win_override_does_not_change_total_decision() -> None:
    policy = resolve_prediction_policy(
        _availability(
            week=8,
            full_features_available=True,
            total_features_available=True,
        ),
        win_champion=_champion(
            "win_prob",
            "random_forest",
        ),
        total_champion=_champion(
            "total",
            "xgboost",
        ),
        win_override="logistic",
    )

    assert policy.win.model_type == "logistic"
    assert policy.win.provenance is not None
    assert policy.win.provenance.source is (PredictionModelSource.OVERRIDE)

    assert policy.total.model_type == "xgboost"
    assert policy.total.provenance is not None
    assert policy.total.provenance.source is (PredictionModelSource.CHAMPION)


def test_total_override_does_not_change_win_decision() -> None:
    policy = resolve_prediction_policy(
        _availability(
            week=8,
            full_features_available=True,
            total_features_available=True,
        ),
        win_champion=_champion(
            "win_prob",
            "random_forest",
        ),
        total_champion=_champion(
            "total",
            "xgboost",
        ),
        total_override="random_forest",
    )

    assert policy.win.model_type == "random_forest"
    assert policy.win.provenance is not None
    assert policy.win.provenance.source is (PredictionModelSource.CHAMPION)

    assert policy.total.model_type == "random_forest"
    assert policy.total.provenance is not None
    assert policy.total.provenance.source is (PredictionModelSource.OVERRIDE)


def test_ineligible_override_does_not_fall_back() -> None:
    policy = resolve_prediction_policy(
        _availability(
            full_features_available=False,
        ),
        win_champion=_champion(
            "win_prob",
            "random_forest",
        ),
        total_champion=None,
        win_override="xgboost",
    )

    assert policy.win.status is (PredictionModelStatus.UNAVAILABLE)
    assert policy.win.model_type is None
    assert policy.win.rationale is (PredictionPolicyRationale.OVERRIDE_INELIGIBLE)


def test_explicit_elo_override_requires_elo_state() -> None:
    policy = resolve_prediction_policy(
        _availability(
            elo_available=False,
        ),
        win_champion=None,
        total_champion=None,
        win_override="elo",
    )

    assert policy.win.status is (PredictionModelStatus.UNAVAILABLE)
    assert policy.win.rationale is (PredictionPolicyRationale.OVERRIDE_INELIGIBLE)


def test_policy_records_champion_provenance() -> None:
    win_champion = _champion(
        "win_prob",
        "random_forest",
    )

    policy = resolve_prediction_policy(
        _availability(
            week=8,
            full_features_available=True,
        ),
        win_champion=win_champion,
        total_champion=None,
    )

    assert policy.win.provenance == win_champion
    assert policy.win.provenance is not None
    assert policy.win.provenance.promoted_at == "2026-07-01T14:20:00Z"
    assert policy.win.provenance.source_run_id == "20260701_142000"


def test_policy_is_json_serializable() -> None:
    policy = resolve_prediction_policy(
        _availability(),
        win_champion=_champion(
            "win_prob",
            "random_forest",
        ),
        total_champion=None,
    )

    serialized = policy.to_dict()
    encoded = json.dumps(
        serialized,
        sort_keys=True,
    )

    assert '"model_type": "elo"' in encoded
    assert '"status": "selected"' in encoded
    assert '"rationale": "elo_only_available"' in encoded


def test_same_inputs_produce_equal_policy() -> None:
    availability = _availability()
    win_champion = _champion(
        "win_prob",
        "random_forest",
    )

    first = resolve_prediction_policy(
        availability,
        win_champion=win_champion,
        total_champion=None,
    )
    second = resolve_prediction_policy(
        availability,
        win_champion=win_champion,
        total_champion=None,
    )

    assert first == second


def test_policy_contracts_are_immutable() -> None:
    availability = _availability()

    with pytest.raises(FrozenInstanceError):
        availability.week = 2  # type: ignore[misc]


def test_policy_module_does_not_import_api_or_pandas() -> None:
    from gridiron_edge.models.game_prediction import (
        prediction_policy,
    )

    source = Path(prediction_policy.__file__).read_text()

    assert "gridiron_edge.api" not in source
    assert "import pandas" not in source

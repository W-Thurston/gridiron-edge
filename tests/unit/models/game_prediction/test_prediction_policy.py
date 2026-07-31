"""Tests for availability-aware game prediction policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from gridiron_edge.evaluation.champion_resolver import ChampionNotFoundError
from gridiron_edge.models.game_prediction.prediction_policy import (
    ModelProvenance,
    PredictionAvailability,
    PredictionModelSource,
    PredictionModelStatus,
    PredictionPolicy,
    PredictionPolicyRationale,
    load_prediction_policy,
    resolve_prediction_policy,
)


def _availability(
    *,
    season: str = "2026-2027",
    week: int = 1,
    elo_available: bool = True,
    win_logistic_features_available: bool = False,
    win_random_forest_features_available: bool = False,
    win_xgboost_features_available: bool = False,
    total_random_forest_features_available: bool = False,
    total_xgboost_features_available: bool = False,
) -> PredictionAvailability:
    """Create explicit weekly availability facts."""
    return PredictionAvailability(
        season=season,
        week=week,
        elo_available=elo_available,
        win_logistic_features_available=win_logistic_features_available,
        win_random_forest_features_available=(win_random_forest_features_available),
        win_xgboost_features_available=win_xgboost_features_available,
        total_random_forest_features_available=(total_random_forest_features_available),
        total_xgboost_features_available=(total_xgboost_features_available),
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
        metrics=(("score", 0.21),),
    )


def test_week_one_selects_only_eligible_elo_policy() -> None:
    policy: PredictionPolicy = resolve_prediction_policy(
        _availability(),
        win_champion=_champion("win_prob", "random_forest"),
        total_champion=_champion("total", "xgboost"),
    )

    assert policy.win.status is PredictionModelStatus.SELECTED
    assert policy.win.model_type == "elo"
    assert policy.win.rationale is PredictionPolicyRationale.ELO_ONLY_AVAILABLE
    assert policy.total.status is PredictionModelStatus.UNAVAILABLE
    assert policy.total.model_type is None
    assert policy.total.rationale is (PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE)


def test_full_feature_champions_require_available_inputs() -> None:
    policy: PredictionPolicy = resolve_prediction_policy(
        _availability(
            week=8,
            win_random_forest_features_available=True,
            total_xgboost_features_available=True,
        ),
        win_champion=_champion("win_prob", "random_forest"),
        total_champion=_champion("total", "xgboost"),
    )

    assert policy.win.model_type == "random_forest"
    assert policy.win.rationale is PredictionPolicyRationale.CHAMPION_ELIGIBLE
    assert policy.total.model_type == "xgboost"
    assert policy.total.rationale is PredictionPolicyRationale.CHAMPION_ELIGIBLE


def test_unavailable_total_remains_explicit() -> None:
    policy: PredictionPolicy = resolve_prediction_policy(
        _availability(),
        win_champion=None,
        total_champion=_champion("total", "xgboost"),
    )

    assert policy.total.status is PredictionModelStatus.UNAVAILABLE
    assert policy.total.model_type is None
    assert policy.total.provenance is None
    assert policy.total.rationale is (PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE)


def test_win_override_does_not_change_total_decision() -> None:
    policy: PredictionPolicy = resolve_prediction_policy(
        _availability(
            week=8,
            win_logistic_features_available=True,
            total_xgboost_features_available=True,
        ),
        win_champion=_champion("win_prob", "random_forest"),
        total_champion=_champion("total", "xgboost"),
        win_override="logistic",
    )

    assert policy.win.model_type == "logistic"
    assert policy.win.provenance is not None
    assert policy.win.provenance.source is PredictionModelSource.OVERRIDE
    assert policy.total.model_type == "xgboost"
    assert policy.total.provenance is not None
    assert policy.total.provenance.source is PredictionModelSource.CHAMPION


def test_total_override_does_not_change_win_decision() -> None:
    policy: PredictionPolicy = resolve_prediction_policy(
        _availability(
            week=8,
            win_random_forest_features_available=True,
            total_random_forest_features_available=True,
        ),
        win_champion=_champion("win_prob", "random_forest"),
        total_champion=_champion("total", "xgboost"),
        total_override="random_forest",
    )

    assert policy.win.model_type == "random_forest"
    assert policy.win.provenance is not None
    assert policy.win.provenance.source is PredictionModelSource.CHAMPION
    assert policy.total.model_type == "random_forest"
    assert policy.total.provenance is not None
    assert policy.total.provenance.source is PredictionModelSource.OVERRIDE


def test_ineligible_override_does_not_fall_back() -> None:
    policy: PredictionPolicy = resolve_prediction_policy(
        _availability(),
        win_champion=_champion("win_prob", "random_forest"),
        total_champion=None,
        win_override="xgboost",
    )

    assert policy.win.status is PredictionModelStatus.UNAVAILABLE
    assert policy.win.model_type is None
    assert policy.win.rationale is PredictionPolicyRationale.OVERRIDE_INELIGIBLE


def test_explicit_elo_override_requires_elo_state() -> None:
    policy: PredictionPolicy = resolve_prediction_policy(
        _availability(elo_available=False),
        win_champion=None,
        total_champion=None,
        win_override="elo",
    )

    assert policy.win.status is PredictionModelStatus.UNAVAILABLE
    assert policy.win.rationale is PredictionPolicyRationale.OVERRIDE_INELIGIBLE


def test_policy_records_champion_provenance() -> None:
    win_champion: ModelProvenance = _champion("win_prob", "random_forest")

    policy: PredictionPolicy = resolve_prediction_policy(
        _availability(
            week=8,
            win_random_forest_features_available=True,
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
        win_champion=_champion("win_prob", "random_forest"),
        total_champion=None,
    )

    encoded = json.dumps(policy.to_dict(), sort_keys=True)

    assert '"model_type": "elo"' in encoded
    assert '"status": "selected"' in encoded
    assert '"rationale": "elo_only_available"' in encoded


def test_same_inputs_produce_equal_policy() -> None:
    availability = _availability()
    win_champion = _champion("win_prob", "random_forest")

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


def test_champion_entry_becomes_stable_provenance() -> None:
    provenance = ModelProvenance.from_champion_entry(
        model_name="win_prob",
        entry={
            "model_type": "random_forest",
            "promoted_at": "2026-07-01T14:20:00Z",
            "source_run_id": "run-1",
            "metrics": {
                "ece": 0.04,
                "brier": 0.21,
            },
        },
    )

    assert provenance.source is PredictionModelSource.CHAMPION
    assert provenance.metrics == (
        ("brier", 0.21),
        ("ece", 0.04),
    )


@patch(
    "gridiron_edge.models.game_prediction.prediction_policy.resolve_current_champion_with_metadata"
)
def test_loads_win_and_total_champions_independently(
    mock_resolve: MagicMock,
) -> None:
    def resolve_entry(
        model_name: str,
        *,
        repo: Path | None = None,
    ) -> dict[str, object]:
        del repo
        if model_name == "win_prob":
            return {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:20:00Z",
                "source_run_id": "run-win",
                "metrics": {"brier": 0.21},
            }
        return {
            "model_type": "xgboost",
            "promoted_at": "2026-07-01T14:21:00Z",
            "source_run_id": "run-total",
            "metrics": {"rmse": 12.8},
        }

    mock_resolve.side_effect = resolve_entry

    policy: PredictionPolicy = load_prediction_policy(
        _availability(
            week=8,
            win_random_forest_features_available=True,
            total_xgboost_features_available=True,
        )
    )

    assert policy.win.model_type == "random_forest"
    assert policy.total.model_type == "xgboost"
    assert policy.win.provenance is not None
    assert policy.win.provenance.source_run_id == "run-win"
    assert policy.total.provenance is not None
    assert policy.total.provenance.source_run_id == "run-total"
    assert mock_resolve.call_args_list == [
        call("win_prob", repo=None),
        call("total", repo=None),
    ]


@patch(
    "gridiron_edge.models.game_prediction.prediction_policy.resolve_current_champion_with_metadata"
)
def test_missing_total_champion_remains_explicit(
    mock_resolve: MagicMock,
) -> None:
    def resolve_entry(
        model_name: str,
        *,
        repo: Path | None = None,
    ) -> dict[str, object]:
        del repo
        if model_name == "total":
            raise ChampionNotFoundError("No total champion.")
        return {
            "model_type": "random_forest",
            "promoted_at": "2026-07-01T14:20:00Z",
            "source_run_id": "run-win",
            "metrics": {},
        }

    mock_resolve.side_effect = resolve_entry

    policy: PredictionPolicy = load_prediction_policy(
        _availability(
            week=8,
            win_random_forest_features_available=True,
            total_xgboost_features_available=True,
        )
    )

    assert policy.win.model_type == "random_forest"
    assert policy.total.status is PredictionModelStatus.UNAVAILABLE
    assert policy.total.model_type is None
    assert policy.total.rationale is PredictionPolicyRationale.CHAMPION_UNAVAILABLE


@patch(
    "gridiron_edge.models.game_prediction.prediction_policy.resolve_current_champion_with_metadata"
)
def test_win_override_skips_only_win_champion(
    mock_resolve: MagicMock,
) -> None:
    mock_resolve.return_value = {
        "model_type": "xgboost",
        "promoted_at": "2026-07-01T14:20:00Z",
        "source_run_id": "run-total",
        "metrics": {},
    }

    policy: PredictionPolicy = load_prediction_policy(
        _availability(
            week=8,
            win_logistic_features_available=True,
            total_xgboost_features_available=True,
        ),
        win_override="logistic",
    )

    assert policy.win.model_type == "logistic"
    assert policy.win.provenance is not None
    assert policy.win.provenance.source is PredictionModelSource.OVERRIDE
    assert policy.total.model_type == "xgboost"
    assert policy.total.provenance is not None
    assert policy.total.provenance.source is PredictionModelSource.CHAMPION
    mock_resolve.assert_called_once_with("total", repo=None)


@patch(
    "gridiron_edge.models.game_prediction.prediction_policy.resolve_current_champion_with_metadata"
)
def test_total_override_skips_only_total_champion(
    mock_resolve: MagicMock,
) -> None:
    mock_resolve.return_value = {
        "model_type": "random_forest",
        "promoted_at": "2026-07-01T14:20:00Z",
        "source_run_id": "run-win",
        "metrics": {},
    }

    policy: PredictionPolicy = load_prediction_policy(
        _availability(
            week=8,
            win_random_forest_features_available=True,
            total_xgboost_features_available=True,
        ),
        total_override="xgboost",
    )

    assert policy.win.model_type == "random_forest"
    assert policy.win.provenance is not None
    assert policy.win.provenance.source is PredictionModelSource.CHAMPION
    assert policy.total.model_type == "xgboost"
    assert policy.total.provenance is not None
    assert policy.total.provenance.source is PredictionModelSource.OVERRIDE
    mock_resolve.assert_called_once_with("win_prob", repo=None)


@patch(
    "gridiron_edge.models.game_prediction.prediction_policy.resolve_current_champion_with_metadata"
)
def test_two_overrides_require_no_champion_lookup(
    mock_resolve: MagicMock,
) -> None:
    policy: PredictionPolicy = load_prediction_policy(
        _availability(
            week=8,
            win_logistic_features_available=True,
            total_xgboost_features_available=True,
        ),
        win_override="logistic",
        total_override="xgboost",
    )

    assert policy.win.model_type == "logistic"
    assert policy.total.model_type == "xgboost"
    mock_resolve.assert_not_called()


@patch(
    "gridiron_edge.models.game_prediction.prediction_policy.resolve_current_champion_with_metadata"
)
def test_malformed_champion_metadata_is_not_hidden(
    mock_resolve: MagicMock,
) -> None:
    mock_resolve.return_value = {
        "promoted_at": "2026-07-01T14:20:00Z",
        "source_run_id": "run-1",
        "metrics": {},
    }

    with pytest.raises(
        ValueError,
        match="Champion entry must contain model_type",
    ):
        load_prediction_policy(
            _availability(
                week=8,
                total_xgboost_features_available=True,
            ),
            total_override="xgboost",
        )


def test_policy_module_has_no_compute_or_api_dependency() -> None:
    from gridiron_edge.models.game_prediction import prediction_policy

    source = Path(prediction_policy.__file__).read_text()

    assert "gridiron_edge.api" not in source
    assert "import pandas" not in source
    assert "ArtifactStore" not in source
    assert "run_features" not in source
    assert "predict_upcoming" not in source
    assert "write_manifest" not in source


def test_logistic_override_uses_its_own_feature_availability() -> None:
    policy = resolve_prediction_policy(
        _availability(
            elo_available=False,
            win_logistic_features_available=True,
            win_random_forest_features_available=False,
            win_xgboost_features_available=False,
        ),
        win_champion=None,
        total_champion=None,
        win_override="logistic",
    )

    assert policy.win.status is PredictionModelStatus.SELECTED
    assert policy.win.model_type == "logistic"


def test_random_forest_override_does_not_borrow_logistic_features() -> None:
    policy = resolve_prediction_policy(
        _availability(
            elo_available=True,
            win_logistic_features_available=True,
            win_random_forest_features_available=False,
        ),
        win_champion=None,
        total_champion=None,
        win_override="random_forest",
    )

    assert policy.win.status is PredictionModelStatus.UNAVAILABLE
    assert policy.win.model_type is None
    assert policy.win.rationale is PredictionPolicyRationale.OVERRIDE_INELIGIBLE


def test_win_champion_uses_exact_feature_requirement() -> None:
    champion = ModelProvenance(
        model_name="win_prob",
        model_type="xgboost",
        source=PredictionModelSource.CHAMPION,
    )

    policy = resolve_prediction_policy(
        _availability(
            elo_available=True,
            win_random_forest_features_available=True,
            win_xgboost_features_available=False,
        ),
        win_champion=champion,
        total_champion=None,
    )

    assert policy.win.status is PredictionModelStatus.SELECTED
    assert policy.win.model_type == "elo"
    assert policy.win.rationale is PredictionPolicyRationale.ELO_ONLY_AVAILABLE


def test_total_xgboost_does_not_borrow_random_forest_readiness() -> None:
    policy = resolve_prediction_policy(
        _availability(
            total_random_forest_features_available=True,
            total_xgboost_features_available=False,
        ),
        win_champion=None,
        total_champion=None,
        total_override="xgboost",
    )

    assert policy.total.status is PredictionModelStatus.UNAVAILABLE
    assert policy.total.model_type is None


def test_model_specific_availability_helpers() -> None:
    availability = _availability(
        elo_available=True,
        win_logistic_features_available=True,
        win_random_forest_features_available=False,
        win_xgboost_features_available=True,
        total_random_forest_features_available=True,
        total_xgboost_features_available=False,
    )

    assert availability.win_features_available("elo")
    assert availability.win_features_available("logistic")
    assert not availability.win_features_available("random_forest")
    assert availability.win_features_available("xgboost")
    assert not availability.win_features_available("unknown")

    assert availability.total_features_available("random_forest")
    assert not availability.total_features_available("xgboost")
    assert not availability.total_features_available("logistic")

# tests/unit/models/game_prediction/test_weekly_execution.py
"""Tests for policy-selected weekly game-model execution."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gridiron_edge.evaluation.forecast_contracts import ForecastRole
from gridiron_edge.evaluation.prediction_input_evidence import (
    BinaryArtifactReference,
    CalibrationResolutionSource,
    PredictionArtifactKind,
    PredictionArtifactState,
    PredictionPostProcessingEvidence,
    PredictionSourceState,
    SourceArtifactReference,
    SourceRevision,
    authenticate_prediction_input_evidence,
    create_prediction_feature_schema,
)
from gridiron_edge.models.game_prediction.prediction_execution import (
    StatisticalPredictionComputation,
    StatisticalPredictionExecution,
)
from gridiron_edge.models.game_prediction.prediction_policy import (
    ModelProvenance,
    PredictionAvailability,
    PredictionModelSource,
    resolve_prediction_policy,
)
from gridiron_edge.models.game_prediction.weekly_execution import (
    execute_development_weekly_prediction_policy,
    execute_weekly_prediction_policy,
)

SEASON = "2026-2027"
WEEK = 1
GENERATED_AT = datetime(2026, 9, 1, 12, tzinfo=UTC)
COMMIT = "a" * 40
DIGEST = "b" * 64
REGISTRY_PATH = "data/output/calibration/game_model_calibration.json"


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [SEASON, SEASON],
            "week": [WEEK, WEEK],
            "game_id": ["g1", "g2"],
            "game_day_of_week": ["Sun", "Sun"],
            "game_date": ["2026-09-13", "2026-09-13"],
            "game_time": ["13:00", "16:25"],
            "away_team": ["Away A", "Away B"],
            "home_team": ["Home A", "Home B"],
            "neutral_site": [0, 0],
        }
    )


def _availability() -> PredictionAvailability:
    return PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=True,
        win_logistic_features_available=True,
        win_random_forest_features_available=True,
        win_xgboost_features_available=True,
        total_random_forest_features_available=True,
        total_xgboost_features_available=True,
    )


def _unavailable_availability() -> PredictionAvailability:
    """Create availability with no eligible prediction family."""
    return PredictionAvailability(
        season=SEASON,
        week=WEEK,
        elo_available=False,
        win_logistic_features_available=False,
        win_random_forest_features_available=False,
        win_xgboost_features_available=False,
        total_random_forest_features_available=False,
        total_xgboost_features_available=False,
    )


def _policy():
    return resolve_prediction_policy(
        _availability(),
        win_champion=ModelProvenance(
            model_name="win_prob",
            model_type="logistic",
            source=PredictionModelSource.CHAMPION,
        ),
        total_champion=ModelProvenance(
            model_name="total",
            model_type="random_forest",
            source=PredictionModelSource.CHAMPION,
        ),
    )


def _unavailable_policy():
    """Resolve a policy whose Elo-dependent families are unavailable."""
    return resolve_prediction_policy(
        _unavailable_availability(),
        win_champion=ModelProvenance(
            model_name="win_prob",
            model_type="logistic",
            source=PredictionModelSource.CHAMPION,
        ),
        total_champion=ModelProvenance(
            model_name="total",
            model_type="random_forest",
            source=PredictionModelSource.CHAMPION,
        ),
    )


def _revision() -> SourceRevision:
    return SourceRevision(commit=COMMIT, tracked_worktree_clean=True)


def _sources() -> tuple[SourceArtifactReference, ...]:
    return (
        SourceArtifactReference(
            relative_path="data/cleaned/NFL_upcoming_schedule_rich.parquet",
            state=PredictionSourceState.PRESENT,
            content_digest="c" * 64,
            size_bytes=300,
        ),
        SourceArtifactReference(
            relative_path=REGISTRY_PATH,
            state=PredictionSourceState.ABSENT,
            content_digest=None,
            size_bytes=None,
        ),
    )


def _artifact(
    kind: PredictionArtifactKind,
    *,
    state: PredictionArtifactState,
    model_name: str,
    model_type: str,
) -> BinaryArtifactReference:
    filenames = {
        PredictionArtifactKind.EXTERNAL_CALIBRATOR: "calibrator.joblib",
        PredictionArtifactKind.MODEL: "model.joblib",
        PredictionArtifactKind.MODEL_METADATA: "metadata.json",
        PredictionArtifactKind.SCALER: "scaler.joblib",
    }
    return BinaryArtifactReference(
        kind=kind,
        source_relative_path=(f"data/models/{model_name}/{model_type}/{filenames[kind]}"),
        state=state,
        content_digest=DIGEST if state is PredictionArtifactState.PRESENT else None,
        size_bytes=100 if state is PredictionArtifactState.PRESENT else None,
    )


def _artifacts(
    *,
    model_name: str,
    model_type: str,
    scaler: PredictionArtifactState,
) -> tuple[BinaryArtifactReference, ...]:
    return (
        _artifact(
            PredictionArtifactKind.EXTERNAL_CALIBRATOR,
            state=PredictionArtifactState.ABSENT,
            model_name=model_name,
            model_type=model_type,
        ),
        _artifact(
            PredictionArtifactKind.MODEL,
            state=PredictionArtifactState.PRESENT,
            model_name=model_name,
            model_type=model_type,
        ),
        _artifact(
            PredictionArtifactKind.MODEL_METADATA,
            state=PredictionArtifactState.PRESENT,
            model_name=model_name,
            model_type=model_type,
        ),
        _artifact(
            PredictionArtifactKind.SCALER,
            state=scaler,
            model_name=model_name,
            model_type=model_type,
        ),
    )


def _win_predictions(*, game_ids: tuple[str, ...] = ("g1", "g2")) -> pd.DataFrame:
    count = len(game_ids)
    return pd.DataFrame(
        {
            "GAME_ID": list(game_ids),
            "AWAY_TEAM": ["Away A", "Away B"][:count],
            "HOME_TEAM": ["Home A", "Home B"][:count],
            "WEEK_NUM": [WEEK] * count,
            "AWAY_TEAM_ELO": [1500.0, 1490.0][:count],
            "HOME_TEAM_ELO": [1510.0, 1520.0][:count],
            "AWAY_WIN_PROB": [0.45, 0.40][:count],
            "HOME_WIN_PROB": [0.55, 0.60][:count],
            "model_spread": [-1.5, -3.0][:count],
            "margin_std": [13.5, 13.5][:count],
            "win_prob_lo": [0.40, 0.45][:count],
            "win_prob_hi": [0.70, 0.75][:count],
            "confidence_tier": ["Low", "Moderate"][:count],
        }
    )


def _total_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2"],
            "AWAY_TEAM": ["Away A", "Away B"],
            "HOME_TEAM": ["Home A", "Home B"],
            "WEEK_NUM": [WEEK, WEEK],
            "model_total": [44.5, 47.0],
        }
    )


def _computations(
    outputs: tuple[float, ...],
    *,
    game_ids: tuple[str, ...] = ("g1", "g2"),
    transformed: tuple[tuple[float, ...], ...] | None = None,
) -> tuple[StatisticalPredictionComputation, ...]:
    raw_values = ((1.0, 10.0), (2.0, 20.0))[: len(game_ids)]
    transformed_values = transformed or raw_values
    return tuple(
        StatisticalPredictionComputation(
            game_id=game_id,
            raw_feature_values=raw,
            transformed_feature_values=scaled,
            raw_estimator_output=output,
            post_estimator_output=output,
        )
        for game_id, raw, scaled, output in zip(
            game_ids,
            raw_values,
            transformed_values,
            outputs,
            strict=True,
        )
    )


def _win_execution(
    *,
    game_ids: tuple[str, ...] = ("g1", "g2"),
) -> StatisticalPredictionExecution:
    predictions = _win_predictions(game_ids=game_ids)
    schema = create_prediction_feature_schema(
        model_name="win_prob",
        model_type="logistic",
        task="classification",
        modeling_schema_version=5,
        epa_window=4,
        feature_set_name="test_features",
        ordered_columns=("feature_a", "feature_b"),
    )
    post_processing = PredictionPostProcessingEvidence(
        registry_reference=_sources()[1],
        registry_entry_updated_at=None,
        sigma=12.0,
        sigma_source=CalibrationResolutionSource.MODEL_FALLBACK,
        margin_std=13.5,
        margin_std_source=CalibrationResolutionSource.MODEL_FALLBACK,
        external_calibrator_state=PredictionArtifactState.ABSENT,
        embedded_estimator_calibration=False,
    )
    outputs = tuple(float(value) for value in predictions["HOME_WIN_PROB"])
    transformed = ((0.1, 1.0), (0.2, 2.0))[: len(game_ids)]
    return StatisticalPredictionExecution(
        predictions=predictions,
        computations=_computations(
            outputs,
            game_ids=game_ids,
            transformed=transformed,
        ),
        source_revision=_revision(),
        source_artifacts=_sources(),
        binary_artifacts=_artifacts(
            model_name="win_prob",
            model_type="logistic",
            scaler=PredictionArtifactState.PRESENT,
        ),
        feature_schema=schema,
        post_processing=post_processing,
    )


def _total_execution() -> StatisticalPredictionExecution:
    predictions = _total_predictions()
    schema = create_prediction_feature_schema(
        model_name="total",
        model_type="random_forest",
        task="regression",
        modeling_schema_version=5,
        epa_window=4,
        feature_set_name="test_features",
        ordered_columns=("feature_a", "feature_b"),
    )
    return StatisticalPredictionExecution(
        predictions=predictions,
        computations=_computations((44.5, 47.0)),
        source_revision=_revision(),
        source_artifacts=_sources(),
        binary_artifacts=_artifacts(
            model_name="total",
            model_type="random_forest",
            scaler=PredictionArtifactState.ABSENT,
        ),
        feature_schema=schema,
        post_processing=None,
    )


def test_executes_exact_selected_families_under_one_run(tmp_path: Path) -> None:
    win_model = MagicMock()
    win_model.predict_upcoming_with_evidence.return_value = _win_execution()
    win_model.predict_upcoming = MagicMock()
    total_model = MagicMock()
    total_model.predict_upcoming_with_evidence.return_value = _total_execution()
    total_model.predict_upcoming = MagicMock()

    def registry_get(key: str):
        return {
            "win_prob_logistic": lambda: win_model,
            "total_random_forest": lambda: total_model,
        }[key]

    with (
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.inspect_prediction_availability",
            return_value=_availability(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.load_prediction_policy",
            return_value=_policy(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.ModelRegistry.get",
            side_effect=registry_get,
        ),
    ):
        execution = execute_weekly_prediction_policy(
            _schedule(),
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
            run_id="run-1",
            generated_at=GENERATED_AT,
            source_revision=_revision(),
            source_artifacts=_sources(),
        )

    assert len(execution.events) == 4
    assert set(execution.events["model_name"]) == {"win_prob", "total"}
    assert set(execution.events["model_type"]) == {"logistic", "random_forest"}
    assert set(execution.events["run_id"]) == {"run-1"}
    assert execution.win_display is not None
    assert execution.win_display["GAME_ID"].tolist() == ["g1", "g2"]
    assert [
        (evidence.model_name, evidence.model_type) for evidence in execution.input_evidence
    ] == [
        ("win_prob", "logistic"),
        ("total", "random_forest"),
    ]
    win_model.predict_upcoming.assert_not_called()
    total_model.predict_upcoming.assert_not_called()
    win_model.predict_upcoming_with_evidence.assert_called_once()
    total_model.predict_upcoming_with_evidence.assert_called_once()
    assert set(execution.events["role"]) == {ForecastRole.LIVE.value}
    assert set(execution.events["run_id"]) == {"run-1"}


def test_rejects_partial_selected_family_before_return(tmp_path: Path) -> None:
    win_model = MagicMock()
    win_model.predict_upcoming_with_evidence.return_value = _win_execution(game_ids=("g1",))
    total_model = MagicMock()
    total_model.predict_upcoming_with_evidence.return_value = _total_execution()

    def registry_get(key: str):
        return {
            "win_prob_logistic": lambda: win_model,
            "total_random_forest": lambda: total_model,
        }[key]

    with (
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.inspect_prediction_availability",
            return_value=_availability(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.load_prediction_policy",
            return_value=_policy(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.ModelRegistry.get",
            side_effect=registry_get,
        ),
        pytest.raises(ValueError, match="Win prediction coverage"),
    ):
        execute_weekly_prediction_policy(
            _schedule(),
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
            run_id="run-1",
            generated_at=GENERATED_AT,
            source_revision=_revision(),
            source_artifacts=_sources(),
        )


def test_unavailable_lineage_blocks_models_before_execution(
    tmp_path: Path,
) -> None:
    unavailable = _unavailable_availability()
    policy = _unavailable_policy()

    with (
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.inspect_prediction_availability",
            return_value=unavailable,
        ) as inspect_availability,
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.load_prediction_policy",
            return_value=policy,
        ) as load_policy,
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.ModelRegistry.get",
        ) as registry_get,
        pytest.raises(
            ValueError,
            match="Prediction policy selected no available Win or Total model",
        ),
    ):
        execute_weekly_prediction_policy(
            _schedule(),
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
            run_id="blocked-run",
            generated_at=GENERATED_AT,
        )

    inspect_availability.assert_called_once()
    availability_args, availability_kwargs = inspect_availability.call_args
    pd.testing.assert_frame_equal(availability_args[0], _schedule())
    assert availability_kwargs == {
        "season": SEASON,
        "week": WEEK,
        "repo": tmp_path,
    }
    load_policy.assert_called_once_with(
        unavailable,
        repo=tmp_path,
        win_override=None,
        total_override=None,
    )
    registry_get.assert_not_called()


def test_executes_development_families_under_one_run(
    tmp_path: Path,
) -> None:
    win_model = MagicMock()
    win_model.predict_upcoming_with_evidence.return_value = _win_execution()
    win_model.predict_upcoming = MagicMock()

    total_model = MagicMock()
    total_model.predict_upcoming_with_evidence.return_value = _total_execution()
    total_model.predict_upcoming = MagicMock()

    def registry_get(key: str):
        return {
            "win_prob_logistic": lambda: win_model,
            "total_random_forest": lambda: total_model,
        }[key]

    with (
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.inspect_prediction_availability",
            return_value=_availability(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.load_prediction_policy",
            return_value=_policy(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.weekly_execution.ModelRegistry.get",
            side_effect=registry_get,
        ),
    ):
        execution = execute_development_weekly_prediction_policy(
            _schedule(),
            season=SEASON,
            week=WEEK,
            repo=tmp_path,
            run_id="development-run",
            generated_at=GENERATED_AT,
            source_revision=_revision(),
            source_artifacts=_sources(),
        )

    assert len(execution.events) == 4
    assert set(execution.events["model_name"]) == {
        "win_prob",
        "total",
    }
    assert set(execution.events["model_type"]) == {
        "logistic",
        "random_forest",
    }
    assert set(execution.events["run_id"]) == {"development-run"}
    assert set(execution.events["role"]) == {ForecastRole.DEVELOPMENT.value}
    assert {
        (
            evidence.model_name,
            evidence.model_type,
        )
        for evidence in execution.input_evidence
    } == {
        ("win_prob", "logistic"),
        ("total", "random_forest"),
    }

    win_model.predict_upcoming.assert_not_called()
    total_model.predict_upcoming.assert_not_called()
    win_model.predict_upcoming_with_evidence.assert_called_once()
    total_model.predict_upcoming_with_evidence.assert_called_once()

    for evidence in execution.input_evidence:
        family_events = execution.events.loc[
            (execution.events["model_name"] == evidence.model_name)
            & (execution.events["model_type"] == evidence.model_type),
            :,
        ]

        authenticate_prediction_input_evidence(
            evidence,
            forecast_events=family_events,
        )

# tests/unit/models/game_prediction/test_availability.py
"""Tests for read-only weekly prediction availability inspection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from gridiron_edge.models.artifact import BaseModelMetadata
from gridiron_edge.models.game_prediction._columns import FeatureSet
from gridiron_edge.models.game_prediction.availability import (
    _requires_verified_elo,
    inspect_prediction_availability,
)
import gridiron_edge.models.game_prediction.model  # noqa: F401
from gridiron_edge.models.registry import ModelRegistry


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2026-2027", "2026-2027", "2026-2027"],
            "week": [1, 1, 2],
            "game_id": ["g1", "g2", "g3"],
            "game_day_of_week": ["Sun", "Sun", "Sun"],
            "game_date": ["2026-09-13", "2026-09-13", "2026-09-20"],
            "game_time": ["13:00", "16:25", "13:00"],
            "away_team": ["Away A", "Away B", "Away C"],
            "home_team": ["Home A", "Home B", "Home C"],
            "neutral_site": [0, 0, 0],
        }
    )


def _elo(*, missing_home: bool = False) -> pd.DataFrame:
    rows = [
        ("Away A", "2026-2027", 1, 1500.0),
        ("Home A", "2026-2027", 1, 1510.0),
        ("Away B", "2026-2027", 1, 1490.0),
        ("Home B", "2026-2027", 1, 1520.0),
    ]
    if missing_home:
        rows.pop()
    return pd.DataFrame(rows, columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"])


def _metadata(model_name: str, model_type: str, task: str, columns: list[str]) -> BaseModelMetadata:
    return BaseModelMetadata(
        model_name=model_name,
        model_type=model_type,
        task=task,
        trained_at="2026-08-03T00:00:00",
        kind="game",
        feature_columns=columns,
    )


def _run(
    tmp_path: Path,
    *,
    elo: pd.DataFrame | None = None,
    complete_features: bool = True,
    artifact_keys: set[str] | None = None,
    lineage_available: bool = True,
    feature_columns: list[str] | None = None,
):
    artifact_keys = artifact_keys or set()
    resolved_feature_columns = feature_columns if feature_columns is not None else ["feature"]

    class FakeModel:
        def prediction_feature_set(self) -> FeatureSet:
            def feature_fn(frame: pd.DataFrame) -> pd.DataFrame:
                values = [1.0] * len(frame)
                result = pd.DataFrame(
                    {column: values.copy() for column in resolved_feature_columns},
                    index=frame.index,
                )
                if not complete_features:
                    result.iloc[-1, 0] = float("nan")
                return result

            return FeatureSet(
                "test",
                feature_fn,
                resolved_feature_columns,
            )

    class FakeStore:
        def __init__(self, repo: Path) -> None:
            self.repo = repo

        def is_trained(self, model_name: str, model_type: str) -> bool:
            return f"{model_name}_{model_type}" in artifact_keys

        def read_metadata(
            self,
            model_name: str,
            model_type: str,
        ) -> BaseModelMetadata:
            task = "classification" if model_name == "win_prob" else "regression"
            return _metadata(
                model_name,
                model_type,
                task,
                resolved_feature_columns,
            )

        def artifact_dir(self, model_name: str, model_type: str) -> Path:
            path = tmp_path / model_name / model_type
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.joblib").touch()
            return path

    source = _schedule()
    with (
        patch("gridiron_edge.models.game_prediction.availability.ArtifactStore", FakeStore),
        patch(
            "gridiron_edge.models.game_prediction.availability.ModelRegistry.get",
            return_value=FakeModel,
        ),
        patch(
            "gridiron_edge.models.game_prediction.availability.loaders.load_elo_state",
            return_value=_elo() if elo is None else elo,
        ),
        patch(
            "gridiron_edge.models.game_prediction.availability.run_features",
            side_effect=lambda *, df, feature_names, datasets: df.copy(),
        ),
        patch(
            "gridiron_edge.models.game_prediction.availability.verify_current_elo_lineage",
            return_value=lineage_available,
        ),
    ):
        result = inspect_prediction_availability(
            source,
            season="2026-2027",
            week=1,
            repo=tmp_path,
        )
    return result, source


def test_complete_week_is_available_without_inference(tmp_path: Path) -> None:
    keys = {
        "win_prob_logistic",
        "win_prob_random_forest",
        "win_prob_xgboost",
        "total_random_forest",
        "total_xgboost",
    }
    with (
        patch("gridiron_edge.models.artifact.ArtifactStore.load") as load_model,
        patch("gridiron_edge.models.artifact.ArtifactStore.load_scaler") as load_scaler,
    ):
        availability, _ = _run(tmp_path, artifact_keys=keys)

    assert all(
        value for key, value in availability.to_dict().items() if key not in {"season", "week"}
    )
    load_model.assert_not_called()
    load_scaler.assert_not_called()


def test_partial_feature_coverage_is_unavailable(tmp_path: Path) -> None:
    availability, _ = _run(
        tmp_path,
        artifact_keys={"win_prob_logistic"},
        complete_features=False,
    )
    assert not availability.win_logistic_features_available
    assert availability.elo_available


def test_models_remain_independent(tmp_path: Path) -> None:
    availability, _ = _run(
        tmp_path,
        artifact_keys={"win_prob_logistic", "total_xgboost"},
    )
    assert availability.win_logistic_features_available
    assert not availability.win_random_forest_features_available
    assert not availability.win_xgboost_features_available
    assert not availability.total_random_forest_features_available
    assert availability.total_xgboost_features_available


def test_missing_elo_for_one_game_is_unavailable(tmp_path: Path) -> None:
    availability, _ = _run(tmp_path, elo=_elo(missing_home=True))
    assert not availability.elo_available


def test_missing_lineage_blocks_every_elo_dependent_model(
    tmp_path: Path,
) -> None:
    keys = {
        "win_prob_logistic",
        "win_prob_random_forest",
        "win_prob_xgboost",
        "total_random_forest",
        "total_xgboost",
    }

    availability, _ = _run(
        tmp_path,
        artifact_keys=keys,
        lineage_available=False,
        feature_columns=[
            "AWAY_ELO",
            "HOME_ELO",
            "ELO_DIFF",
        ],
    )

    assert not availability.elo_available
    assert not availability.win_logistic_features_available
    assert not availability.win_random_forest_features_available
    assert not availability.win_xgboost_features_available
    assert not availability.total_random_forest_features_available
    assert not availability.total_xgboost_features_available


def test_verified_lineage_does_not_override_missing_weekly_elo(
    tmp_path: Path,
) -> None:
    availability, _ = _run(
        tmp_path,
        elo=_elo(missing_home=True),
        lineage_available=True,
    )

    assert not availability.elo_available


def test_other_week_does_not_change_denominator(tmp_path: Path) -> None:
    availability, _ = _run(tmp_path)
    assert availability.season == "2026-2027"
    assert availability.week == 1
    assert availability.elo_available


def test_input_schedule_is_not_mutated(tmp_path: Path) -> None:
    availability, source = _run(tmp_path)
    assert availability.elo_available
    pd.testing.assert_frame_equal(source, _schedule())


def test_duplicate_scoped_game_ids_are_rejected(tmp_path: Path) -> None:
    schedule = _schedule()
    schedule.loc[1, "game_id"] = "g1"
    with pytest.raises(ValueError, match="duplicate game_id"):
        inspect_prediction_availability(
            schedule,
            season="2026-2027",
            week=1,
            repo=tmp_path,
        )


def test_empty_requested_scope_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no games"):
        inspect_prediction_availability(
            _schedule(),
            season="2026-2027",
            week=9,
            repo=tmp_path,
        )


def test_malformed_lineage_is_not_hidden_as_unavailability(
    tmp_path: Path,
) -> None:
    with (
        patch(
            "gridiron_edge.models.game_prediction.availability.verify_current_elo_lineage",
            side_effect=ValueError("Unsupported Elo lineage schema_version: 999."),
        ),
        pytest.raises(
            ValueError,
            match="Unsupported Elo lineage schema_version: 999",
        ),
    ):
        inspect_prediction_availability(
            _schedule(),
            season="2026-2027",
            week=1,
            repo=tmp_path,
        )


@pytest.mark.parametrize(
    "feature_names",
    [
        ["AWAY_ELO"],
        ["HOME_ELO"],
        ["ELO_DIFF"],
        ["feature", "ELO_DIFF"],
    ],
)
def test_detects_direct_elo_feature_dependency(
    feature_names: list[str],
) -> None:
    assert _requires_verified_elo(feature_names)


def test_feature_contract_without_elo_is_independent() -> None:
    assert not _requires_verified_elo(
        [
            "WEEK_NUM",
            "IS_NEUTRAL_SITE",
        ]
    )


def test_missing_lineage_does_not_block_non_elo_feature_contract(
    tmp_path: Path,
) -> None:
    availability, _ = _run(
        tmp_path,
        artifact_keys={"total_xgboost"},
        lineage_available=False,
        feature_columns=[
            "WEEK_NUM",
            "IS_NEUTRAL_SITE",
        ],
    )

    assert not availability.elo_available
    assert not availability.win_logistic_features_available
    assert not availability.win_random_forest_features_available
    assert not availability.win_xgboost_features_available
    assert not availability.total_random_forest_features_available
    assert availability.total_xgboost_features_available


@pytest.mark.parametrize(
    ("model_name", "model_type"),
    [
        ("win_prob", "logistic"),
        ("win_prob", "random_forest"),
        ("win_prob", "xgboost"),
        ("total", "random_forest"),
        ("total", "xgboost"),
    ],
)
def test_current_registered_models_require_verified_elo(
    model_name: str,
    model_type: str,
) -> None:
    model = ModelRegistry.get(f"{model_name}_{model_type}")()
    feature_set = model.prediction_feature_set()

    required_elo_columns = {
        "AWAY_ELO",
        "HOME_ELO",
        "ELO_DIFF",
    }

    assert required_elo_columns.issubset(set(feature_set.feature_names))
    assert _requires_verified_elo(feature_set.feature_names)

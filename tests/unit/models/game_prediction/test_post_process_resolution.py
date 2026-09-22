# tests/unit/models/game_prediction/test_post_process_resolution.py
"""Tests for evidence-aware game prediction post-processing resolution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gridiron_edge.evaluation.prediction_input_evidence import (
    CalibrationResolutionSource,
)
from gridiron_edge.models.game_prediction.post_process import (
    _DEFAULT_MARGIN_STD,
    _MODEL_MARGIN_STDS,
    _MODEL_SIGMAS,
    _NFL_DEFAULT_SIGMA,
    PredictionPostProcessingResolution,
    enrich_predictions_with_resolution,
    resolve_prediction_post_processing,
)

MODEL_NAME = "win_prob"
MODEL_TYPE = "resolver_test"
COMPOSITE_KEY = f"{MODEL_NAME}_{MODEL_TYPE}"
UPDATED_AT = "2026-09-21T12:34:56+00:00"


@pytest.fixture(autouse=True)
def _restore_fallback_maps():
    """Restore mutable fallback registries after every resolver test."""
    saved_sigmas = dict(_MODEL_SIGMAS)
    saved_margin_stds = dict(_MODEL_MARGIN_STDS)
    try:
        yield
    finally:
        _MODEL_SIGMAS.clear()
        _MODEL_SIGMAS.update(saved_sigmas)
        _MODEL_MARGIN_STDS.clear()
        _MODEL_MARGIN_STDS.update(saved_margin_stds)


def _payload(
    *,
    sigma: object = 12.25,
    margin_std: object = 14.75,
    updated_at: object = UPDATED_AT,
) -> dict[str, object]:
    return {
        "sigma": sigma,
        "margin_std": margin_std,
        "updated_at": updated_at,
    }


def _resolve(
    tmp_path: Path,
    *,
    registry: dict[str, dict[str, object]],
    calibrator: object | None = None,
):
    with (
        patch(
            "gridiron_edge.models.game_prediction.post_process.load_model_calibrations",
            return_value=registry,
        ) as load_registry,
        patch(
            "gridiron_edge.models.game_prediction.post_process.load_calibrator",
            return_value=calibrator,
        ) as load_external_calibrator,
    ):
        resolution = resolve_prediction_post_processing(
            model_name=MODEL_NAME,
            model_type=MODEL_TYPE,
            repo=tmp_path,
        )
    return resolution, load_registry, load_external_calibrator


class TestPersistedResolution:
    def test_resolves_both_values_from_persisted_registry(
        self,
        tmp_path: Path,
    ) -> None:
        resolution, _, _ = _resolve(
            tmp_path,
            registry={COMPOSITE_KEY: _payload()},
        )

        assert resolution.sigma == pytest.approx(12.25)
        assert resolution.sigma_source is (CalibrationResolutionSource.PERSISTED_REGISTRY)
        assert resolution.margin_std == pytest.approx(14.75)
        assert resolution.margin_std_source is (CalibrationResolutionSource.PERSISTED_REGISTRY)
        assert resolution.registry_entry_updated_at == datetime(
            2026,
            9,
            21,
            12,
            34,
            56,
            tzinfo=UTC,
        )

    def test_sigma_persisted_and_margin_std_uses_model_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        _MODEL_MARGIN_STDS[(MODEL_NAME, MODEL_TYPE)] = 13.5

        resolution, _, _ = _resolve(
            tmp_path,
            registry={
                COMPOSITE_KEY: _payload(margin_std="not-numeric"),
            },
        )

        assert resolution.sigma == pytest.approx(12.25)
        assert resolution.sigma_source is (CalibrationResolutionSource.PERSISTED_REGISTRY)
        assert resolution.margin_std == pytest.approx(13.5)
        assert resolution.margin_std_source is (CalibrationResolutionSource.MODEL_FALLBACK)
        assert resolution.registry_entry_updated_at is not None

    def test_margin_std_persisted_and_sigma_uses_model_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        _MODEL_SIGMAS[(MODEL_NAME, MODEL_TYPE)] = 11.5

        resolution, _, _ = _resolve(
            tmp_path,
            registry={
                COMPOSITE_KEY: _payload(sigma="not-numeric"),
            },
        )

        assert resolution.sigma == pytest.approx(11.5)
        assert resolution.sigma_source is CalibrationResolutionSource.MODEL_FALLBACK
        assert resolution.margin_std == pytest.approx(14.75)
        assert resolution.margin_std_source is (CalibrationResolutionSource.PERSISTED_REGISTRY)
        assert resolution.registry_entry_updated_at is not None

    def test_z_timestamp_is_normalized_to_utc(self, tmp_path: Path) -> None:
        resolution, _, _ = _resolve(
            tmp_path,
            registry={
                COMPOSITE_KEY: _payload(updated_at="2026-09-21T12:34:56Z"),
            },
        )

        assert resolution.registry_entry_updated_at == datetime(
            2026,
            9,
            21,
            12,
            34,
            56,
            tzinfo=UTC,
        )

    def test_nonzero_offset_is_normalized_to_utc(self, tmp_path: Path) -> None:
        resolution, _, _ = _resolve(
            tmp_path,
            registry={
                COMPOSITE_KEY: _payload(updated_at="2026-09-21T06:34:56-06:00"),
            },
        )

        assert resolution.registry_entry_updated_at == datetime(
            2026,
            9,
            21,
            12,
            34,
            56,
            tzinfo=UTC,
        )
        assert resolution.registry_entry_updated_at.utcoffset() == timedelta(0)

    @pytest.mark.parametrize("updated_at", [None, "", "   "])
    def test_missing_updated_at_is_rejected_when_registry_value_is_used(
        self,
        tmp_path: Path,
        updated_at: object,
    ) -> None:
        with pytest.raises(ValueError, match="requires updated_at"):
            _resolve(
                tmp_path,
                registry={
                    COMPOSITE_KEY: _payload(updated_at=updated_at),
                },
            )

    def test_malformed_updated_at_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="updated_at is invalid"):
            _resolve(
                tmp_path,
                registry={
                    COMPOSITE_KEY: _payload(updated_at="not-a-timestamp"),
                },
            )

    def test_timezone_naive_updated_at_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be timezone-aware"):
            _resolve(
                tmp_path,
                registry={
                    COMPOSITE_KEY: _payload(updated_at="2026-09-21T12:34:56"),
                },
            )


class TestFallbackResolution:
    def test_resolves_both_values_from_model_fallbacks(
        self,
        tmp_path: Path,
    ) -> None:
        _MODEL_SIGMAS[(MODEL_NAME, MODEL_TYPE)] = 10.5
        _MODEL_MARGIN_STDS[(MODEL_NAME, MODEL_TYPE)] = 13.25

        resolution, _, _ = _resolve(tmp_path, registry={})

        assert resolution.sigma == pytest.approx(10.5)
        assert resolution.sigma_source is CalibrationResolutionSource.MODEL_FALLBACK
        assert resolution.margin_std == pytest.approx(13.25)
        assert resolution.margin_std_source is (CalibrationResolutionSource.MODEL_FALLBACK)
        assert resolution.registry_entry_updated_at is None

    def test_resolves_both_values_from_global_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        _MODEL_SIGMAS.pop((MODEL_NAME, MODEL_TYPE), None)
        _MODEL_MARGIN_STDS.pop((MODEL_NAME, MODEL_TYPE), None)

        resolution, _, _ = _resolve(tmp_path, registry={})

        assert resolution.sigma == pytest.approx(_NFL_DEFAULT_SIGMA)
        assert resolution.sigma_source is CalibrationResolutionSource.GLOBAL_DEFAULT
        assert resolution.margin_std == pytest.approx(_DEFAULT_MARGIN_STD)
        assert resolution.margin_std_source is (CalibrationResolutionSource.GLOBAL_DEFAULT)
        assert resolution.registry_entry_updated_at is None

    def test_registry_timestamp_is_ignored_when_entry_values_are_unused(
        self,
        tmp_path: Path,
    ) -> None:
        _MODEL_SIGMAS[(MODEL_NAME, MODEL_TYPE)] = 10.5
        _MODEL_MARGIN_STDS[(MODEL_NAME, MODEL_TYPE)] = 13.25

        resolution, _, _ = _resolve(
            tmp_path,
            registry={
                COMPOSITE_KEY: _payload(
                    sigma="invalid",
                    margin_std="invalid",
                    updated_at="also-invalid",
                ),
            },
        )

        assert resolution.sigma_source is CalibrationResolutionSource.MODEL_FALLBACK
        assert resolution.margin_std_source is (CalibrationResolutionSource.MODEL_FALLBACK)
        assert resolution.registry_entry_updated_at is None

    def test_boolean_registry_values_do_not_count_as_numeric(
        self,
        tmp_path: Path,
    ) -> None:
        _MODEL_SIGMAS[(MODEL_NAME, MODEL_TYPE)] = 10.5
        _MODEL_MARGIN_STDS[(MODEL_NAME, MODEL_TYPE)] = 13.25

        resolution, _, _ = _resolve(
            tmp_path,
            registry={
                COMPOSITE_KEY: _payload(
                    sigma=True,
                    margin_std=False,
                    updated_at="invalid-but-unused",
                ),
            },
        )

        assert resolution.sigma == pytest.approx(10.5)
        assert resolution.margin_std == pytest.approx(13.25)
        assert resolution.registry_entry_updated_at is None


class TestCalibratorAndLoadBehavior:
    def test_external_calibrator_is_preserved(self, tmp_path: Path) -> None:
        calibrator = MagicMock(name="external_calibrator")

        resolution, _, _ = _resolve(
            tmp_path,
            registry={},
            calibrator=calibrator,
        )

        assert resolution.calibrator is calibrator

    def test_external_calibrator_absence_is_explicit(self, tmp_path: Path) -> None:
        resolution, _, _ = _resolve(
            tmp_path,
            registry={},
            calibrator=None,
        )

        assert resolution.calibrator is None

    def test_loads_registry_and_calibrator_exactly_once(self, tmp_path: Path) -> None:
        _, load_registry, load_external_calibrator = _resolve(
            tmp_path,
            registry={},
        )

        load_registry.assert_called_once_with(tmp_path)
        load_external_calibrator.assert_called_once_with(
            MODEL_NAME,
            MODEL_TYPE,
            repo=tmp_path,
        )


class TestResolvedEnrichment:
    def test_uses_supplied_resolution_without_loading_inputs(
        self,
    ) -> None:
        resolution = PredictionPostProcessingResolution(
            sigma=12.0,
            sigma_source=(CalibrationResolutionSource.MODEL_FALLBACK),
            margin_std=13.5,
            margin_std_source=(CalibrationResolutionSource.MODEL_FALLBACK),
            registry_entry_updated_at=None,
            calibrator=None,
        )
        predictions = pd.DataFrame(
            {
                "GAME_ID": ["game-1"],
                "HOME_WIN_PROB": [0.60],
                "AWAY_WIN_PROB": [0.40],
            }
        )

        with (
            patch(
                "gridiron_edge.models.game_prediction.post_process.load_model_calibrations"
            ) as load_registry,
            patch(
                "gridiron_edge.models.game_prediction.post_process.load_calibrator"
            ) as load_calibrator,
        ):
            enriched = enrich_predictions_with_resolution(
                predictions,
                resolution=resolution,
            )

        load_registry.assert_not_called()
        load_calibrator.assert_not_called()
        assert enriched.loc[0, "HOME_WIN_PROB"] == pytest.approx(0.60)
        assert enriched.loc[0, "AWAY_WIN_PROB"] == pytest.approx(0.40)
        assert enriched.loc[0, "margin_std"] == pytest.approx(13.5)
        assert {
            "model_spread",
            "win_prob_lo",
            "win_prob_hi",
            "confidence_tier",
        }.issubset(enriched.columns)

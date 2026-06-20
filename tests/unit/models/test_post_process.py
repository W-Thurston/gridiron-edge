# tests/unit/models/test_post_process.py
"""Unit tests for post_process.py — spread, recalibration, bands, and enrichment."""

from __future__ import annotations

from collections.abc import Generator
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy import dtype, float64, ndarray
import pandas as pd
from pandas import DataFrame, Series
import pytest
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from gridiron_edge.models.game_prediction.post_process import (
    _CALIBRATOR_FILENAME,
    _DEFAULT_MARGIN_STD,
    _MODEL_MARGIN_STDS,
    _MODEL_SIGMAS,
    _NFL_DEFAULT_SIGMA,
    _PROB_CEIL,
    _PROB_FLOOR,
    _TIER_HIGH_PROB,
    _TIER_MODERATE_PROB,
    apply_recalibration,
    calibrate_spread_sigma,
    classify_confidence_tier,
    compute_margin_std,
    enrich_predictions,
    fit_recalibration,
    get_margin_std,
    get_sigma,
    load_calibrator,
    register_sigma,
    save_calibrator,
    spread_to_win_prob,
    win_prob_bands,
    win_prob_to_spread,
)

# ---------------------------------------------------------------------------
# TestWinProbToSpread
# ---------------------------------------------------------------------------


class TestWinProbToSpread:
    """Tests for win_prob_to_spread()."""

    def test_pickem(self) -> None:
        """50% win probability -> spread of 0.0 (pick'em)."""
        assert win_prob_to_spread(0.50) == pytest.approx(0.0, abs=1e-6)

    def test_home_favorite(self) -> None:
        """75% home win prob -> negative spread (home favored)."""
        spread: float = win_prob_to_spread(0.75)
        expected = -_NFL_DEFAULT_SIGMA * norm.ppf(0.75)
        assert spread == pytest.approx(expected, abs=0.01)
        assert spread < 0  # home favored = negative spread

    def test_away_favorite(self) -> None:
        """25% home win prob -> positive spread (away favored)."""
        spread: float = win_prob_to_spread(0.25)
        expected = -_NFL_DEFAULT_SIGMA * norm.ppf(0.25)
        assert spread == pytest.approx(expected, abs=0.01)
        assert spread > 0  # away favored = positive spread

    def test_symmetry(self) -> None:
        """spread(p) = -spread(1-p) for all valid probabilities."""
        for p in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            spread_p: float = win_prob_to_spread(p)
            spread_complement: float = win_prob_to_spread(1.0 - p)
            assert spread_p == pytest.approx(-spread_complement, abs=1e-6), (
                f"Symmetry failed for p={p}: "
                f"spread({p})={spread_p}, spread({1 - p})={spread_complement}"
            )

    def test_strong_favorite(self) -> None:
        """90% home win prob -> large negative spread."""
        spread: float = win_prob_to_spread(0.90)
        assert spread < -15  # approximately -17.8 with default sigma

    def test_near_zero_clamp(self) -> None:
        """Probability of 0.0 is clamped -- no -inf."""
        spread: float = win_prob_to_spread(0.0)
        assert math.isfinite(spread)
        # Should equal the spread at the floor
        assert spread == pytest.approx(
            win_prob_to_spread(_PROB_FLOOR),
            abs=1e-6,
        )

    def test_near_one_clamp(self) -> None:
        """Probability of 1.0 is clamped -- no +inf."""
        spread: float = win_prob_to_spread(1.0)
        assert math.isfinite(spread)
        assert spread == pytest.approx(
            win_prob_to_spread(_PROB_CEIL),
            abs=1e-6,
        )

    def test_custom_sigma(self) -> None:
        """Different sigma produces proportionally different spread."""
        sigma_a = 13.0
        sigma_b = 16.0
        p = 0.70
        spread_a: float = win_prob_to_spread(p, sigma=sigma_a)
        spread_b: float = win_prob_to_spread(p, sigma=sigma_b)
        # Both should be negative (home favored), but b is larger magnitude
        assert spread_a < 0
        assert spread_b < 0
        assert abs(spread_b) > abs(spread_a)
        # Ratio should equal sigma_b / sigma_a
        assert (spread_b / spread_a) == pytest.approx(
            sigma_b / sigma_a,
            abs=1e-6,
        )


# ---------------------------------------------------------------------------
# TestSpreadToWinProb
# ---------------------------------------------------------------------------


class TestSpreadToWinProb:
    """Tests for spread_to_win_prob()."""

    def test_pickem(self) -> None:
        """Spread of 0 -> 50% home win probability."""
        assert spread_to_win_prob(0.0) == pytest.approx(0.50, abs=1e-6)

    def test_home_favored(self) -> None:
        """Negative spread -> home win prob > 50%."""
        prob: float = spread_to_win_prob(-7.0)
        assert prob > 0.50

    def test_away_favored(self) -> None:
        """Positive spread -> home win prob < 50%."""
        prob: float = spread_to_win_prob(7.0)
        assert prob < 0.50

    def test_round_trip(self) -> None:
        """win_prob_to_spread -> spread_to_win_prob recovers original."""
        for p in [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
            spread: float = win_prob_to_spread(p)
            recovered: float = spread_to_win_prob(spread)
            assert recovered == pytest.approx(p, abs=1e-6), (
                f"Round-trip failed for p={p}: spread={spread}, recovered={recovered}"
            )

    def test_round_trip_from_spread(self) -> None:
        """spread_to_win_prob -> win_prob_to_spread recovers original."""
        for s in [-14.0, -7.0, -3.0, 0.0, 3.0, 7.0, 14.0]:
            prob: float = spread_to_win_prob(s)
            recovered: float = win_prob_to_spread(prob)
            assert recovered == pytest.approx(s, abs=0.01), (
                f"Round-trip failed for spread={s}: prob={prob}, recovered={recovered}"
            )


# ---------------------------------------------------------------------------
# TestGetSigma / register_sigma
# ---------------------------------------------------------------------------


class TestGetSigma:
    """Tests for get_sigma() and register_sigma()."""

    def setup_method(self) -> None:
        """Clear registered sigmas before each test."""
        self._saved = dict(_MODEL_SIGMAS)
        _MODEL_SIGMAS.clear()

    def teardown_method(self) -> None:
        """Restore registered sigmas after each test."""
        _MODEL_SIGMAS.clear()
        _MODEL_SIGMAS.update(self._saved)

    def test_default_fallback_none(self) -> None:
        """None model_version returns default sigma."""
        assert get_sigma(None) == _NFL_DEFAULT_SIGMA

    def test_default_fallback_unknown(self) -> None:
        """Unknown model_version returns default sigma."""
        assert get_sigma("nonexistent_model_v99") == _NFL_DEFAULT_SIGMA

    def test_registered_sigma(self) -> None:
        """Registered model version returns its calibrated sigma."""
        register_sigma("win_prob", "random_forest", 14.22)
        assert get_sigma("win_prob", "random_forest") == 14.22

    def test_register_overwrites(self) -> None:
        """Registering again overwrites the previous value."""
        register_sigma("win_prob", "xgboost", 13.50)
        register_sigma("win_prob", "xgboost", 14.10)
        assert get_sigma("win_prob", "xgboost") == 14.10

    def test_multiple_models(self) -> None:
        """Different models can have different sigmas."""
        register_sigma("win_prob", "random_forest", 14.22)
        register_sigma("win_prob", "xgboost", 13.50)
        register_sigma("win_prob", "logistic", 15.01)
        assert get_sigma("win_prob", "random_forest") == 14.22
        assert get_sigma("win_prob", "xgboost") == 13.50
        assert get_sigma("win_prob", "logistic") == 15.01


# ---------------------------------------------------------------------------
# TestCalibrateSigma
# ---------------------------------------------------------------------------


class TestCalibrateSigma:
    """Tests for calibrate_spread_sigma()."""

    def test_recovers_known_sigma(self) -> None:
        """Synthetic data generated with known sigma is recovered."""
        true_sigma = 15.0
        rng: Generator = np.random.default_rng(42)
        n = 500

        # Generate random home win probabilities (away from extremes)
        home_probs: Series = pd.Series(rng.uniform(0.20, 0.80, size=n))

        # True predicted margin = true_sigma * ppf(home_win_prob)
        predicted_margins = true_sigma * norm.ppf(home_probs.values)

        # Add noise (actual margins ~ predicted + noise)
        noise = rng.normal(0, 3.0, size=n)
        actual_margins: Series = pd.Series(predicted_margins + noise)

        recovered: float = calibrate_spread_sigma(home_probs, actual_margins)

        # Should recover close to the true sigma (noise adds some error)
        assert recovered == pytest.approx(true_sigma, abs=1.0)

    def test_recovers_exact_sigma_no_noise(self) -> None:
        """With zero noise, calibration recovers sigma exactly."""
        true_sigma = 12.5
        rng: Generator = np.random.default_rng(99)
        n = 200

        home_probs: Series = pd.Series(rng.uniform(0.25, 0.75, size=n))
        actual_margins: Series = pd.Series(
            true_sigma * norm.ppf(home_probs.values),
        )

        recovered: float = calibrate_spread_sigma(home_probs, actual_margins)
        assert recovered == pytest.approx(true_sigma, abs=0.01)

    def test_bounds(self) -> None:
        """Result stays within the valid NFL sigma range."""
        # Extreme margins that might push sigma out of range
        home_probs: Series = pd.Series([0.50, 0.50, 0.50, 0.50])
        actual_margins: Series = pd.Series([100.0, -100.0, 50.0, -50.0])

        sigma: float = calibrate_spread_sigma(home_probs, actual_margins)
        assert 8.0 <= sigma <= 22.0

    def test_empty_input_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            calibrate_spread_sigma(pd.Series(dtype=float), pd.Series(dtype=float))

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            calibrate_spread_sigma(
                pd.Series([0.5, 0.6]),
                pd.Series([3.0]),
            )


# ---------------------------------------------------------------------------
# TestEnrichPredictions
# ---------------------------------------------------------------------------


class TestEnrichPredictions:
    """Tests for enrich_predictions()."""

    def setup_method(self) -> None:
        """Clear registered sigmas before each test."""
        self._saved = dict(_MODEL_SIGMAS)
        _MODEL_SIGMAS.clear()

    def teardown_method(self) -> None:
        """Restore registered sigmas after each test."""
        _MODEL_SIGMAS.clear()
        _MODEL_SIGMAS.update(self._saved)

    def _make_predictions_df(self) -> pd.DataFrame:
        """Create a minimal predictions DataFrame for testing."""
        return pd.DataFrame(
            {
                "game_id": ["2024_01_KC_LAC", "2024_01_BUF_MIA", "2024_01_SF_SEA"],
                "home_win_prob": [0.65, 0.50, 0.35],
                "away_win_prob": [0.35, 0.50, 0.65],
            }
        )

    def test_adds_model_spread_column(self) -> None:
        """Enrichment adds a model_spread column."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df, recalibrate=False)
        assert "model_spread" in enriched.columns

    def test_does_not_mutate_input(self) -> None:
        """Input DataFrame is not modified."""
        df: DataFrame = self._make_predictions_df()
        original_columns: list[str] = list(df.columns)
        original_values: DataFrame = df.copy()
        enrich_predictions(df, recalibrate=False)
        assert list(df.columns) == original_columns
        pd.testing.assert_frame_equal(df, original_values)

    def test_spread_values_correct(self) -> None:
        """Spread values match win_prob_to_spread applied to each row."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df, recalibrate=False)
        for _, row in enriched.iterrows():
            expected: float = win_prob_to_spread(row["home_win_prob"])
            assert row["model_spread"] == pytest.approx(expected, abs=1e-6)

    def test_pickem_spread_zero(self) -> None:
        """A 50/50 game produces a spread of 0."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df, recalibrate=False)
        pickem_row: DataFrame = enriched.loc[enriched["home_win_prob"] == 0.50, :]
        assert pickem_row["model_spread"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_home_favorite_negative_spread(self) -> None:
        """Home favorite (65%) produces negative spread."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df, recalibrate=False)
        fav_row: DataFrame = enriched.loc[enriched["home_win_prob"] == 0.65, :]
        assert fav_row["model_spread"].iloc[0] < 0

    def test_away_favorite_positive_spread(self) -> None:
        """Away favorite (home 35%) produces positive spread."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df, recalibrate=False)
        dog_row: DataFrame = enriched.loc[enriched["home_win_prob"] == 0.35, :]
        assert dog_row["model_spread"].iloc[0] > 0

    def test_uses_model_sigma(self) -> None:
        """When a model sigma is registered, enrichment uses it."""
        register_sigma("win_prob", "random_forest", 15.0)
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(
            df,
            model_name="win_prob",
            model_type="random_forest",
            recalibrate=False,
        )
        for _, row in enriched.iterrows():
            expected: float = win_prob_to_spread(row["home_win_prob"], sigma=15.0)
            assert row["model_spread"] == pytest.approx(expected, abs=1e-6)

    def test_uppercase_column(self) -> None:
        """DataFrame with uppercase HOME_WIN_PROB is handled."""
        df = pd.DataFrame(
            {
                "GAME_ID": ["2024_01_KC_LAC"],
                "HOME_WIN_PROB": [0.70],
            }
        )
        enriched: DataFrame = enrich_predictions(df, recalibrate=False)
        assert "model_spread" in enriched.columns
        assert enriched["model_spread"].iloc[0] < 0  # home favored

    def test_missing_column_raises(self) -> None:
        """KeyError raised when no win probability column found."""
        df = pd.DataFrame({"game_id": ["X"], "score": [42]})
        with pytest.raises(KeyError, match="home_win_prob"):
            enrich_predictions(df, recalibrate=False)

    def test_preserves_existing_columns(self) -> None:
        """All original columns are preserved in enriched output."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df, recalibrate=False)
        for col in df.columns:
            assert col in enriched.columns


# ===========================================================================
# Isotonic Recalibration Tests
# ===========================================================================


def _make_calibrator() -> IsotonicRegression:
    """Fit a simple isotonic calibrator on synthetic underconfident data."""
    probs: ndarray = np.array([0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80])
    outcomes: ndarray = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    calibrator = IsotonicRegression(
        y_min=_PROB_FLOOR,
        y_max=_PROB_CEIL,
        out_of_bounds="clip",
    )
    calibrator.fit(probs, outcomes)
    return calibrator


# ---------------------------------------------------------------------------
# TestFitRecalibration
# ---------------------------------------------------------------------------


class TestFitRecalibration:
    """Tests for fit_recalibration()."""

    def _make_seasons_data(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Create synthetic underconfident model data across 5 seasons."""
        rng: Generator = np.random.default_rng(42)
        n_per_season = 100
        season_labels: list[str] = [
            "2019-2020",
            "2020-2021",
            "2021-2022",
            "2022-2023",
            "2023-2024",
        ]
        all_probs: list[float] = []
        all_outcomes: list[int] = []
        all_seasons: list[str] = []

        for season in season_labels:
            # Underconfident model: predictions are compressed toward 0.5
            raw_probs = rng.uniform(0.30, 0.70, size=n_per_season)
            # Actual outcomes have stronger signal
            outcomes = (rng.random(n_per_season) < (raw_probs * 1.2)).astype(int)
            all_probs.extend(raw_probs.tolist())
            all_outcomes.extend(outcomes.tolist())
            all_seasons.extend([season] * n_per_season)

        return (
            pd.Series(all_probs),
            pd.Series(all_outcomes),
            pd.Series(all_seasons),
        )

    def test_fits_monotonic_mapping(self) -> None:
        """Calibrator output is monotonically non-decreasing."""
        probs, outcomes, seasons = self._make_seasons_data()
        calibrator, _ = fit_recalibration(probs, outcomes, seasons, holdout_seasons=1)

        test_inputs: ndarray[tuple[Any, ...], dtype[float64]] = np.linspace(0.30, 0.70, 20)
        calibrated = calibrator.predict(test_inputs)
        diffs: ndarray = np.diff(calibrated)
        assert np.all(diffs >= -1e-10), "Calibrator output is not monotonically non-decreasing"

    def test_respects_temporal_split(self) -> None:
        """Diagnostics correctly identify train and holdout seasons."""
        probs, outcomes, seasons = self._make_seasons_data()
        _, diag = fit_recalibration(probs, outcomes, seasons, holdout_seasons=2)

        assert diag["train_seasons"] == ["2019-2020", "2020-2021", "2021-2022"]
        assert diag["holdout_seasons"] == ["2022-2023", "2023-2024"]
        assert diag["n_train"] == 300
        assert diag["n_holdout"] == 200

    def test_insufficient_seasons_raises(self) -> None:
        """Too few unique seasons for the requested split raises ValueError."""
        probs: Series[float] = pd.Series([0.5, 0.6])
        outcomes: Series[int] = pd.Series([0, 1])
        seasons: Series[str] = pd.Series(["2023-2024", "2024-2025"])

        with pytest.raises(ValueError, match="Need at least 3"):
            fit_recalibration(probs, outcomes, seasons, holdout_seasons=2)

    def test_diagnostics_keys(self) -> None:
        """Diagnostics dict contains all expected keys."""
        probs, outcomes, seasons = self._make_seasons_data()
        _, diag = fit_recalibration(probs, outcomes, seasons, holdout_seasons=1)

        expected_keys: set[str] = {
            "n_train",
            "n_holdout",
            "train_seasons",
            "holdout_seasons",
            "train_mean_pred",
            "train_mean_actual",
            "holdout_mean_pred",
            "holdout_mean_actual",
        }
        assert set(diag.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TestApplyRecalibration
# ---------------------------------------------------------------------------


class TestApplyRecalibration:
    """Tests for apply_recalibration()."""

    def test_output_range(self) -> None:
        """Output stays within (_PROB_FLOOR, _PROB_CEIL)."""
        calibrator = _make_calibrator()
        probs: ndarray = np.array([0.0, 0.001, 0.5, 0.999, 1.0])
        result: ndarray = apply_recalibration(probs, calibrator)
        assert np.all(result >= _PROB_FLOOR)
        assert np.all(result <= _PROB_CEIL)

    def test_preserves_ordering(self) -> None:
        """Monotonically increasing input -> non-decreasing output."""
        calibrator = _make_calibrator()
        probs: ndarray[tuple[Any, ...], dtype[float64]] = np.linspace(0.10, 0.90, 50)
        result: ndarray = apply_recalibration(probs, calibrator)
        diffs: ndarray = np.diff(result)
        assert np.all(diffs >= -1e-10)

    def test_accepts_series_and_array(self) -> None:
        """Works with both pd.Series and np.ndarray, same results."""
        calibrator = _make_calibrator()
        arr: ndarray = np.array([0.3, 0.5, 0.7])
        series: Series = pd.Series(arr)
        result_arr: ndarray = apply_recalibration(arr, calibrator)
        result_series: ndarray = apply_recalibration(series, calibrator)
        np.testing.assert_array_almost_equal(result_arr, result_series)


# ---------------------------------------------------------------------------
# TestSaveLoadCalibrator
# ---------------------------------------------------------------------------


class TestSaveLoadCalibrator:
    """Tests for save_calibrator() and load_calibrator()."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Save then load returns a calibrator with identical predictions."""
        calibrator = _make_calibrator()
        test_probs: ndarray = np.array([0.3, 0.5, 0.7])
        expected = calibrator.predict(test_probs)

        save_calibrator(calibrator, "win_prob", "random_forest", repo=tmp_path)
        loaded = load_calibrator("win_prob", "random_forest", repo=tmp_path)

        assert loaded is not None
        np.testing.assert_array_almost_equal(
            loaded.predict(test_probs),
            expected,
        )

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        """Loading from a path with no calibrator returns None."""
        result = load_calibrator("win_prob", "nonexistent_model", repo=tmp_path)
        assert result is None

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Save creates the _cal directory if it doesn't exist."""
        calibrator = _make_calibrator()
        cal_dir: Path = tmp_path / "data" / "models" / "win_prob" / "random_forest"
        assert not cal_dir.exists()

        save_calibrator(calibrator, "win_prob", "random_forest", repo=tmp_path)
        assert cal_dir.exists()
        assert (cal_dir / _CALIBRATOR_FILENAME).exists()


# ---------------------------------------------------------------------------
# TestEnrichWithRecalibration
# ---------------------------------------------------------------------------


class TestEnrichWithRecalibration:
    """Tests for enrich_predictions() with recalibration."""

    def _make_predictions_df(self) -> pd.DataFrame:
        """Create a minimal predictions DataFrame for testing."""
        return pd.DataFrame(
            {
                "game_id": ["2024_01_KC_LAC", "2024_01_BUF_MIA", "2024_01_SF_SEA"],
                "home_win_prob": [0.65, 0.50, 0.35],
                "away_win_prob": [0.35, 0.50, 0.65],
            }
        )

    def test_applies_calibrator_when_present(self, tmp_path: Path) -> None:
        """Recalibration adjusts probabilities when calibrator is saved."""
        calibrator = _make_calibrator()
        save_calibrator(calibrator, "win_prob", "random_forest", repo=tmp_path)

        df: DataFrame = self._make_predictions_df()
        original_probs: Series = df["home_win_prob"].copy()

        enriched: DataFrame = enrich_predictions(
            df,
            model_name="win_prob",
            model_type="random_forest",
            recalibrate=True,
            repo=tmp_path,
        )

        # At least one probability should differ after recalibration
        assert not enriched["home_win_prob"].equals(original_probs)

    def test_skips_when_no_calibrator(self, tmp_path: Path) -> None:
        """Without a saved calibrator, probabilities are unchanged."""
        df: DataFrame = self._make_predictions_df()
        original_probs: Series = df["home_win_prob"].copy()

        enriched: DataFrame = enrich_predictions(
            df,
            model_name="win_prob",
            model_type="random_forest",
            recalibrate=True,
            repo=tmp_path,
        )

        pd.testing.assert_series_equal(
            enriched["home_win_prob"],
            original_probs,
        )

    def test_skips_when_recalibrate_false(self, tmp_path: Path) -> None:
        """recalibrate=False skips calibration even if calibrator exists."""
        calibrator = _make_calibrator()
        save_calibrator(calibrator, "win_prob", "random_forest", repo=tmp_path)

        df: DataFrame = self._make_predictions_df()
        original_probs: Series = df["home_win_prob"].copy()

        enriched: DataFrame = enrich_predictions(
            df,
            model_name="win_prob",
            model_type="random_forest",
            recalibrate=False,
            repo=tmp_path,
        )

        pd.testing.assert_series_equal(
            enriched["home_win_prob"],
            original_probs,
        )

    def test_away_prob_updated(self, tmp_path: Path) -> None:
        """After recalibration, away_win_prob = 1 - home_win_prob."""
        calibrator = _make_calibrator()
        save_calibrator(calibrator, "win_prob", "random_forest", repo=tmp_path)

        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(
            df,
            model_name="win_prob",
            model_type="random_forest",
            recalibrate=True,
            repo=tmp_path,
        )

        expected_away: Series = 1.0 - enriched["home_win_prob"]
        pd.testing.assert_series_equal(
            enriched["away_win_prob"],
            expected_away,
            check_names=False,
        )


# ===========================================================================
# Uncertainty Bands & Confidence Tiers
# ===========================================================================


# ---------------------------------------------------------------------------
# TestComputeMarginStd
# ---------------------------------------------------------------------------


class TestComputeMarginStd:
    """Tests for compute_margin_std()."""

    def test_recovers_known_std(self) -> None:
        """Residual std matches the noise std used to generate data."""
        rng: Generator = np.random.default_rng(42)
        n = 500
        sigma = 14.0
        noise_std = 10.0

        probs: Series = pd.Series(rng.uniform(0.25, 0.75, size=n))
        predicted = sigma * norm.ppf(np.clip(probs.values, 0.001, 0.999))
        noise = rng.normal(0, noise_std, size=n)
        actual_margins: Series = pd.Series(predicted + noise)

        result: float = compute_margin_std(probs, actual_margins, sigma)
        assert result == pytest.approx(noise_std, abs=2.0)

    def test_zero_residuals(self) -> None:
        """Perfect predictions produce near-zero std."""
        rng: Generator = np.random.default_rng(99)
        sigma = 14.0
        probs: Series = pd.Series(rng.uniform(0.25, 0.75, size=200))
        actual_margins: Series = pd.Series(
            sigma * norm.ppf(np.clip(probs.values, 0.001, 0.999)),
        )

        result: float = compute_margin_std(probs, actual_margins, sigma)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_empty_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_margin_std(
                pd.Series(dtype=float),
                pd.Series(dtype=float),
                14.0,
            )

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_margin_std(
                pd.Series([0.5, 0.6]),
                pd.Series([3.0]),
                14.0,
            )


# ---------------------------------------------------------------------------
# TestGetMarginStd
# ---------------------------------------------------------------------------


class TestGetMarginStd:
    """Tests for get_margin_std()."""

    def setup_method(self) -> None:
        self._saved = dict(_MODEL_MARGIN_STDS)
        _MODEL_MARGIN_STDS.clear()

    def teardown_method(self) -> None:
        _MODEL_MARGIN_STDS.clear()
        _MODEL_MARGIN_STDS.update(self._saved)

    def test_default_fallback_none(self) -> None:
        assert get_margin_std(None) == _DEFAULT_MARGIN_STD

    def test_default_fallback_unknown(self) -> None:
        assert get_margin_std("nonexistent_v99") == _DEFAULT_MARGIN_STD

    def test_registered_value(self) -> None:
        _MODEL_MARGIN_STDS[("test", "model")] = 12.0
        assert get_margin_std("test", "model") == 12.0


# ---------------------------------------------------------------------------
# TestWinProbBands
# ---------------------------------------------------------------------------


class TestWinProbBands:
    """Tests for win_prob_bands()."""

    _SIGMA = 13.97
    _MS = 13.54

    def test_pickem_symmetric(self) -> None:
        """Bands for 50% are symmetric around 0.5."""
        lo, hi = win_prob_bands(0.50, margin_std=self._MS, sigma=self._SIGMA)
        assert lo == pytest.approx(1.0 - hi, abs=1e-6)

    def test_ordering(self) -> None:
        """lo < p < hi for a non-extreme probability."""
        lo, hi = win_prob_bands(0.65, margin_std=self._MS, sigma=self._SIGMA)
        assert lo < 0.65 < hi

    def test_favorite_narrower_than_pickem(self) -> None:
        """90% favorite has narrower band than 50% pick'em."""
        _, hi_50 = win_prob_bands(0.50, margin_std=self._MS, sigma=self._SIGMA)
        lo_50, _ = win_prob_bands(0.50, margin_std=self._MS, sigma=self._SIGMA)
        lo_90, hi_90 = win_prob_bands(0.90, margin_std=self._MS, sigma=self._SIGMA)
        assert (hi_90 - lo_90) < (hi_50 - lo_50)

    def test_symmetry(self) -> None:
        """Bands for p and (1-p) are mirrors."""
        lo_70, hi_70 = win_prob_bands(0.70, margin_std=self._MS, sigma=self._SIGMA)
        lo_30, hi_30 = win_prob_bands(0.30, margin_std=self._MS, sigma=self._SIGMA)
        assert lo_70 == pytest.approx(1.0 - hi_30, abs=1e-4)
        assert hi_70 == pytest.approx(1.0 - lo_30, abs=1e-4)

    def test_wider_z_gives_wider_bands(self) -> None:
        """Larger z produces wider bands."""
        lo_90, hi_90 = win_prob_bands(
            0.65,
            margin_std=self._MS,
            sigma=self._SIGMA,
            z=1.645,
        )
        lo_95, hi_95 = win_prob_bands(
            0.65,
            margin_std=self._MS,
            sigma=self._SIGMA,
            z=1.96,
        )
        assert (hi_95 - lo_95) > (hi_90 - lo_90)

    def test_wider_margin_std_gives_wider_bands(self) -> None:
        """Larger margin_std produces wider bands."""
        lo_sm, hi_sm = win_prob_bands(
            0.65,
            margin_std=10.0,
            sigma=self._SIGMA,
        )
        lo_lg, hi_lg = win_prob_bands(
            0.65,
            margin_std=15.0,
            sigma=self._SIGMA,
        )
        assert (hi_lg - lo_lg) > (hi_sm - lo_sm)


# ---------------------------------------------------------------------------
# TestClassifyConfidenceTier
# ---------------------------------------------------------------------------


class TestClassifyConfidenceTier:
    """Tests for classify_confidence_tier()."""

    def test_high_confidence(self) -> None:
        assert classify_confidence_tier(0.75) == "High"

    def test_moderate_confidence(self) -> None:
        assert classify_confidence_tier(0.65) == "Moderate"

    def test_low_confidence(self) -> None:
        assert classify_confidence_tier(0.52) == "Low"

    def test_boundary_high(self) -> None:
        """Exactly at _TIER_HIGH_PROB boundary (0.70) → High."""
        assert classify_confidence_tier(_TIER_HIGH_PROB) == "High"
        assert classify_confidence_tier(1.0 - _TIER_HIGH_PROB) == "High"

    def test_boundary_moderate(self) -> None:
        """Exactly at _TIER_MODERATE_PROB boundary (0.60) → Moderate."""
        assert classify_confidence_tier(_TIER_MODERATE_PROB) == "Moderate"
        assert classify_confidence_tier(1.0 - _TIER_MODERATE_PROB) == "Moderate"

    def test_symmetric_away_favorite(self) -> None:
        """Away favorite (prob < 0.5) gets same tier as equivalent home fav."""
        assert classify_confidence_tier(0.25) == "High"
        assert classify_confidence_tier(0.35) == "Moderate"
        assert classify_confidence_tier(0.48) == "Low"


# ---------------------------------------------------------------------------
# TestEnrichBands
# ---------------------------------------------------------------------------


class TestEnrichBands:
    """Tests for uncertainty band columns in enrich_predictions()."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": ["G1", "G2", "G3", "G4", "G5"],
                "home_win_prob": [0.90, 0.65, 0.50, 0.35, 0.10],
                "away_win_prob": [0.10, 0.35, 0.50, 0.65, 0.90],
            }
        )

    def test_adds_band_columns(self) -> None:
        enriched: DataFrame = enrich_predictions(self._make_df(), recalibrate=False)
        for col in ["margin_std", "win_prob_lo", "win_prob_hi", "confidence_tier"]:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_margin_std_constant(self) -> None:
        enriched: DataFrame = enrich_predictions(self._make_df(), recalibrate=False)
        assert enriched["margin_std"].nunique() == 1

    def test_band_ordering(self) -> None:
        enriched: DataFrame = enrich_predictions(self._make_df(), recalibrate=False)
        for _, row in enriched.iterrows():
            assert row["win_prob_lo"] < row["home_win_prob"] < row["win_prob_hi"], (
                f"Band ordering violated: {row['win_prob_lo']:.4f} < "
                f"{row['home_win_prob']:.4f} < {row['win_prob_hi']:.4f}"
            )

    def test_tier_values(self) -> None:
        enriched: DataFrame = enrich_predictions(self._make_df(), recalibrate=False)
        valid_tiers: set[str] = {"High", "Moderate", "Low"}
        for tier in enriched["confidence_tier"]:
            assert tier in valid_tiers

    def test_spread_columns_preserved(self) -> None:
        enriched: DataFrame = enrich_predictions(self._make_df(), recalibrate=False)
        assert "model_spread" in enriched.columns


# ---------------------------------------------------------------------------
# TestGetTotalStd
# ---------------------------------------------------------------------------


class TestGetTotalStd:
    """Tests for get_total_std() — artifact-driven total RMSE lookup."""

    def test_returns_rmse_when_artifact_present(self, tmp_path: Path) -> None:
        from gridiron_edge.models.artifact import ArtifactStore
        from gridiron_edge.models.game_prediction.base import GameModelMetadata
        from gridiron_edge.models.game_prediction.post_process import get_total_std

        meta = GameModelMetadata(
            model_name="total",
            model_type="random_forest",
            task="regression",
            trained_at="2026-06-20T00:00:00",
            metrics={"rmse": 10.5},
        )
        store = ArtifactStore(tmp_path)
        store.save(metadata=meta, model_obj={"x": 1})

        result: float = get_total_std("total", "random_forest", repo=tmp_path)
        assert result == pytest.approx(10.5)

    def test_returns_default_when_no_artifact(self, tmp_path: Path) -> None:
        from gridiron_edge.models.game_prediction.post_process import get_total_std

        result: float = get_total_std(
            "total",
            "elo",
            repo=tmp_path,
            default=13.0,
        )
        assert result == 13.0

    def test_returns_default_when_rmse_missing(self, tmp_path: Path) -> None:
        from gridiron_edge.models.artifact import ArtifactStore
        from gridiron_edge.models.game_prediction.base import GameModelMetadata
        from gridiron_edge.models.game_prediction.post_process import get_total_std

        meta = GameModelMetadata(
            model_name="total",
            model_type="random_forest",
            task="regression",
            trained_at="2026-06-20T00:00:00",
            metrics={"mae": 8.0},  # no rmse key
        )
        store = ArtifactStore(tmp_path)
        store.save(metadata=meta, model_obj={"x": 1})

        result: float = get_total_std(
            "total",
            "random_forest",
            repo=tmp_path,
            default=12.0,
        )
        assert result == 12.0

    def test_returns_default_when_rmse_is_nan(self, tmp_path: Path) -> None:
        from gridiron_edge.models.artifact import ArtifactStore
        from gridiron_edge.models.game_prediction.base import GameModelMetadata
        from gridiron_edge.models.game_prediction.post_process import get_total_std

        meta = GameModelMetadata(
            model_name="total",
            model_type="random_forest",
            task="regression",
            trained_at="2026-06-20T00:00:00",
            metrics={"rmse": float("nan")},
        )
        store = ArtifactStore(tmp_path)
        store.save(metadata=meta, model_obj={"x": 1})

        result: float = get_total_std(
            "total",
            "random_forest",
            repo=tmp_path,
            default=14.5,
        )
        assert result == 14.5

    def test_returns_default_when_model_args_none(self) -> None:
        from gridiron_edge.models.game_prediction.post_process import get_total_std

        assert get_total_std(None, "random_forest", default=11.0) == 11.0
        assert get_total_std("total", None, default=11.0) == 11.0
        assert get_total_std(None, None, default=11.0) == 11.0

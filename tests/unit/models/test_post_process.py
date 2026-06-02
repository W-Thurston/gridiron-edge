# tests/unit/models/test_post_process.py
"""Unit tests for post_process.py -- W2 Phase A + Phase A.5."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from gridiron_edge.models.game_prediction.post_process import (
    _CALIBRATOR_FILENAME,
    _MODEL_SIGMAS,
    _NFL_DEFAULT_SIGMA,
    _PROB_CEIL,
    _PROB_FLOOR,
    apply_recalibration,
    calibrate_spread_sigma,
    enrich_predictions,
    fit_recalibration,
    get_sigma,
    load_calibrator,
    register_sigma,
    save_calibrator,
    spread_to_win_prob,
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
        spread = win_prob_to_spread(0.75)
        expected = -_NFL_DEFAULT_SIGMA * norm.ppf(0.75)
        assert spread == pytest.approx(expected, abs=0.01)
        assert spread < 0  # home favored = negative spread

    def test_away_favorite(self) -> None:
        """25% home win prob -> positive spread (away favored)."""
        spread = win_prob_to_spread(0.25)
        expected = -_NFL_DEFAULT_SIGMA * norm.ppf(0.25)
        assert spread == pytest.approx(expected, abs=0.01)
        assert spread > 0  # away favored = positive spread

    def test_symmetry(self) -> None:
        """spread(p) = -spread(1-p) for all valid probabilities."""
        for p in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            spread_p = win_prob_to_spread(p)
            spread_complement = win_prob_to_spread(1.0 - p)
            assert spread_p == pytest.approx(-spread_complement, abs=1e-6), (
                f"Symmetry failed for p={p}: "
                f"spread({p})={spread_p}, spread({1 - p})={spread_complement}"
            )

    def test_strong_favorite(self) -> None:
        """90% home win prob -> large negative spread."""
        spread = win_prob_to_spread(0.90)
        assert spread < -15  # approximately -17.8 with default sigma

    def test_near_zero_clamp(self) -> None:
        """Probability of 0.0 is clamped -- no -inf."""
        spread = win_prob_to_spread(0.0)
        assert math.isfinite(spread)
        # Should equal the spread at the floor
        assert spread == pytest.approx(
            win_prob_to_spread(_PROB_FLOOR),
            abs=1e-6,
        )

    def test_near_one_clamp(self) -> None:
        """Probability of 1.0 is clamped -- no +inf."""
        spread = win_prob_to_spread(1.0)
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
        spread_a = win_prob_to_spread(p, sigma=sigma_a)
        spread_b = win_prob_to_spread(p, sigma=sigma_b)
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
        prob = spread_to_win_prob(-7.0)
        assert prob > 0.50

    def test_away_favored(self) -> None:
        """Positive spread -> home win prob < 50%."""
        prob = spread_to_win_prob(7.0)
        assert prob < 0.50

    def test_round_trip(self) -> None:
        """win_prob_to_spread -> spread_to_win_prob recovers original."""
        for p in [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
            spread = win_prob_to_spread(p)
            recovered = spread_to_win_prob(spread)
            assert recovered == pytest.approx(p, abs=1e-6), (
                f"Round-trip failed for p={p}: spread={spread}, recovered={recovered}"
            )

    def test_round_trip_from_spread(self) -> None:
        """spread_to_win_prob -> win_prob_to_spread recovers original."""
        for s in [-14.0, -7.0, -3.0, 0.0, 3.0, 7.0, 14.0]:
            prob = spread_to_win_prob(s)
            recovered = win_prob_to_spread(prob)
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
        register_sigma("random_forest_v3", 14.22)
        assert get_sigma("random_forest_v3") == 14.22

    def test_register_overwrites(self) -> None:
        """Registering again overwrites the previous value."""
        register_sigma("xgboost_v3", 13.50)
        register_sigma("xgboost_v3", 14.10)
        assert get_sigma("xgboost_v3") == 14.10

    def test_multiple_models(self) -> None:
        """Different models can have different sigmas."""
        register_sigma("rf_v3", 14.22)
        register_sigma("xgb_v3", 13.50)
        register_sigma("logistic_v4", 15.01)
        assert get_sigma("rf_v3") == 14.22
        assert get_sigma("xgb_v3") == 13.50
        assert get_sigma("logistic_v4") == 15.01


# ---------------------------------------------------------------------------
# TestCalibrateSigma
# ---------------------------------------------------------------------------


class TestCalibrateSigma:
    """Tests for calibrate_spread_sigma()."""

    def test_recovers_known_sigma(self) -> None:
        """Synthetic data generated with known sigma is recovered."""
        true_sigma = 15.0
        rng = np.random.default_rng(42)
        n = 500

        # Generate random home win probabilities (away from extremes)
        home_probs = pd.Series(rng.uniform(0.20, 0.80, size=n))

        # True predicted margin = true_sigma * ppf(home_win_prob)
        predicted_margins = true_sigma * norm.ppf(home_probs.values)

        # Add noise (actual margins ~ predicted + noise)
        noise = rng.normal(0, 3.0, size=n)
        actual_margins = pd.Series(predicted_margins + noise)

        recovered = calibrate_spread_sigma(home_probs, actual_margins)

        # Should recover close to the true sigma (noise adds some error)
        assert recovered == pytest.approx(true_sigma, abs=1.0)

    def test_recovers_exact_sigma_no_noise(self) -> None:
        """With zero noise, calibration recovers sigma exactly."""
        true_sigma = 12.5
        rng = np.random.default_rng(99)
        n = 200

        home_probs = pd.Series(rng.uniform(0.25, 0.75, size=n))
        actual_margins = pd.Series(
            true_sigma * norm.ppf(home_probs.values),
        )

        recovered = calibrate_spread_sigma(home_probs, actual_margins)
        assert recovered == pytest.approx(true_sigma, abs=0.01)

    def test_bounds(self) -> None:
        """Result stays within the valid NFL sigma range."""
        # Extreme margins that might push sigma out of range
        home_probs = pd.Series([0.50, 0.50, 0.50, 0.50])
        actual_margins = pd.Series([100.0, -100.0, 50.0, -50.0])

        sigma = calibrate_spread_sigma(home_probs, actual_margins)
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
        df = self._make_predictions_df()
        enriched = enrich_predictions(df, recalibrate=False)
        assert "model_spread" in enriched.columns

    def test_does_not_mutate_input(self) -> None:
        """Input DataFrame is not modified."""
        df = self._make_predictions_df()
        original_columns = list(df.columns)
        original_values = df.copy()
        enrich_predictions(df, recalibrate=False)
        assert list(df.columns) == original_columns
        pd.testing.assert_frame_equal(df, original_values)

    def test_spread_values_correct(self) -> None:
        """Spread values match win_prob_to_spread applied to each row."""
        df = self._make_predictions_df()
        enriched = enrich_predictions(df, recalibrate=False)
        for _, row in enriched.iterrows():
            expected = win_prob_to_spread(row["home_win_prob"])
            assert row["model_spread"] == pytest.approx(expected, abs=1e-6)

    def test_pickem_spread_zero(self) -> None:
        """A 50/50 game produces a spread of 0."""
        df = self._make_predictions_df()
        enriched = enrich_predictions(df, recalibrate=False)
        pickem_row = enriched.loc[enriched["home_win_prob"] == 0.50]
        assert pickem_row["model_spread"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_home_favorite_negative_spread(self) -> None:
        """Home favorite (65%) produces negative spread."""
        df = self._make_predictions_df()
        enriched = enrich_predictions(df, recalibrate=False)
        fav_row = enriched.loc[enriched["home_win_prob"] == 0.65]
        assert fav_row["model_spread"].iloc[0] < 0

    def test_away_favorite_positive_spread(self) -> None:
        """Away favorite (home 35%) produces positive spread."""
        df = self._make_predictions_df()
        enriched = enrich_predictions(df, recalibrate=False)
        dog_row = enriched.loc[enriched["home_win_prob"] == 0.35]
        assert dog_row["model_spread"].iloc[0] > 0

    def test_uses_model_sigma(self) -> None:
        """When a model sigma is registered, enrichment uses it."""
        register_sigma("rf_v3", 15.0)
        df = self._make_predictions_df()
        enriched = enrich_predictions(df, model_version="rf_v3", recalibrate=False)
        for _, row in enriched.iterrows():
            expected = win_prob_to_spread(row["home_win_prob"], sigma=15.0)
            assert row["model_spread"] == pytest.approx(expected, abs=1e-6)

    def test_uppercase_column(self) -> None:
        """DataFrame with uppercase HOME_WIN_PROB is handled."""
        df = pd.DataFrame(
            {
                "GAME_ID": ["2024_01_KC_LAC"],
                "HOME_WIN_PROB": [0.70],
            }
        )
        enriched = enrich_predictions(df, recalibrate=False)
        assert "model_spread" in enriched.columns
        assert enriched["model_spread"].iloc[0] < 0  # home favored

    def test_missing_column_raises(self) -> None:
        """KeyError raised when no win probability column found."""
        df = pd.DataFrame({"game_id": ["X"], "score": [42]})
        with pytest.raises(KeyError, match="home_win_prob"):
            enrich_predictions(df, recalibrate=False)

    def test_preserves_existing_columns(self) -> None:
        """All original columns are preserved in enriched output."""
        df = self._make_predictions_df()
        enriched = enrich_predictions(df, recalibrate=False)
        for col in df.columns:
            assert col in enriched.columns


# ===========================================================================
# Phase A.5 Tests — Isotonic Recalibration
# ===========================================================================


def _make_calibrator() -> IsotonicRegression:
    """Fit a simple isotonic calibrator on synthetic underconfident data."""
    probs = np.array([0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80])
    outcomes = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
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
        rng = np.random.default_rng(42)
        n_per_season = 100
        season_labels = [
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

        test_inputs = np.linspace(0.30, 0.70, 20)
        calibrated = calibrator.predict(test_inputs)
        diffs = np.diff(calibrated)
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
        probs = pd.Series([0.5, 0.6])
        outcomes = pd.Series([0, 1])
        seasons = pd.Series(["2023-2024", "2024-2025"])

        with pytest.raises(ValueError, match="Need at least 3"):
            fit_recalibration(probs, outcomes, seasons, holdout_seasons=2)

    def test_diagnostics_keys(self) -> None:
        """Diagnostics dict contains all expected keys."""
        probs, outcomes, seasons = self._make_seasons_data()
        _, diag = fit_recalibration(probs, outcomes, seasons, holdout_seasons=1)

        expected_keys = {
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
        probs = np.array([0.0, 0.001, 0.5, 0.999, 1.0])
        result = apply_recalibration(probs, calibrator)
        assert np.all(result >= _PROB_FLOOR)
        assert np.all(result <= _PROB_CEIL)

    def test_preserves_ordering(self) -> None:
        """Monotonically increasing input -> non-decreasing output."""
        calibrator = _make_calibrator()
        probs = np.linspace(0.10, 0.90, 50)
        result = apply_recalibration(probs, calibrator)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-10)

    def test_accepts_series_and_array(self) -> None:
        """Works with both pd.Series and np.ndarray, same results."""
        calibrator = _make_calibrator()
        arr = np.array([0.3, 0.5, 0.7])
        series = pd.Series(arr)
        result_arr = apply_recalibration(arr, calibrator)
        result_series = apply_recalibration(series, calibrator)
        np.testing.assert_array_almost_equal(result_arr, result_series)


# ---------------------------------------------------------------------------
# TestSaveLoadCalibrator
# ---------------------------------------------------------------------------


class TestSaveLoadCalibrator:
    """Tests for save_calibrator() and load_calibrator()."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Save then load returns a calibrator with identical predictions."""
        calibrator = _make_calibrator()
        test_probs = np.array([0.3, 0.5, 0.7])
        expected = calibrator.predict(test_probs)

        save_calibrator(calibrator, "rf_v3", repo=tmp_path)
        loaded = load_calibrator("rf_v3", repo=tmp_path)

        assert loaded is not None
        np.testing.assert_array_almost_equal(
            loaded.predict(test_probs),
            expected,
        )

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        """Loading from a path with no calibrator returns None."""
        result = load_calibrator("nonexistent_model", repo=tmp_path)
        assert result is None

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Save creates the _cal directory if it doesn't exist."""
        calibrator = _make_calibrator()
        cal_dir = tmp_path / "data" / "models" / "rf_v3_cal"
        assert not cal_dir.exists()

        save_calibrator(calibrator, "rf_v3", repo=tmp_path)
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
        save_calibrator(calibrator, "rf_v3", repo=tmp_path)

        df = self._make_predictions_df()
        original_probs = df["home_win_prob"].copy()

        enriched = enrich_predictions(
            df,
            model_version="rf_v3",
            recalibrate=True,
            repo=tmp_path,
        )

        # At least one probability should differ after recalibration
        assert not enriched["home_win_prob"].equals(original_probs)

    def test_skips_when_no_calibrator(self, tmp_path: Path) -> None:
        """Without a saved calibrator, probabilities are unchanged."""
        df = self._make_predictions_df()
        original_probs = df["home_win_prob"].copy()

        enriched = enrich_predictions(
            df,
            model_version="rf_v3",
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
        save_calibrator(calibrator, "rf_v3", repo=tmp_path)

        df = self._make_predictions_df()
        original_probs = df["home_win_prob"].copy()

        enriched = enrich_predictions(
            df,
            model_version="rf_v3",
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
        save_calibrator(calibrator, "rf_v3", repo=tmp_path)

        df = self._make_predictions_df()
        enriched = enrich_predictions(
            df,
            model_version="rf_v3",
            recalibrate=True,
            repo=tmp_path,
        )

        expected_away = 1.0 - enriched["home_win_prob"]
        pd.testing.assert_series_equal(
            enriched["away_win_prob"],
            expected_away,
            check_names=False,
        )

# tests/unit/models/test_post_process.py
"""Unit tests for post_process.py — W2 Phase A spread derivation."""

from __future__ import annotations

from collections.abc import Generator
import math

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pytest
from scipy.stats import norm

from gridiron_edge.models.game_prediction.post_process import (
    _MODEL_SIGMAS,
    _NFL_DEFAULT_SIGMA,
    _PROB_CEIL,
    _PROB_FLOOR,
    calibrate_spread_sigma,
    enrich_predictions,
    get_sigma,
    register_sigma,
    spread_to_win_prob,
    win_prob_to_spread,
)

# ---------------------------------------------------------------------------
# TestWinProbToSpread
# ---------------------------------------------------------------------------


class TestWinProbToSpread:
    """Tests for win_prob_to_spread()."""

    def test_pickem(self) -> None:
        """50% win probability → spread of 0.0 (pick'em)."""
        assert win_prob_to_spread(0.50) == pytest.approx(0.0, abs=1e-6)

    def test_home_favorite(self) -> None:
        """75% home win prob → negative spread (home favored)."""
        spread: float = win_prob_to_spread(0.75)
        expected = -_NFL_DEFAULT_SIGMA * norm.ppf(0.75)
        assert spread == pytest.approx(expected, abs=0.01)
        assert spread < 0  # home favored = negative spread

    def test_away_favorite(self) -> None:
        """25% home win prob → positive spread (away favored)."""
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
        """90% home win prob → large negative spread."""
        spread: float = win_prob_to_spread(0.90)
        assert spread < -15  # approximately -17.8 with default sigma

    def test_near_zero_clamp(self) -> None:
        """Probability of 0.0 is clamped — no -inf."""
        spread: float = win_prob_to_spread(0.0)
        assert math.isfinite(spread)
        # Should equal the spread at the floor
        assert spread == pytest.approx(
            win_prob_to_spread(_PROB_FLOOR),
            abs=1e-6,
        )

    def test_near_one_clamp(self) -> None:
        """Probability of 1.0 is clamped — no +inf."""
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
        """Spread of 0 → 50% home win probability."""
        assert spread_to_win_prob(0.0) == pytest.approx(0.50, abs=1e-6)

    def test_home_favored(self) -> None:
        """Negative spread → home win prob > 50%."""
        prob: float = spread_to_win_prob(-7.0)
        assert prob > 0.50

    def test_away_favored(self) -> None:
        """Positive spread → home win prob < 50%."""
        prob: float = spread_to_win_prob(7.0)
        assert prob < 0.50

    def test_round_trip(self) -> None:
        """win_prob_to_spread → spread_to_win_prob recovers original."""
        for p in [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
            spread: float = win_prob_to_spread(p)
            recovered: float = spread_to_win_prob(spread)
            assert recovered == pytest.approx(p, abs=1e-6), (
                f"Round-trip failed for p={p}: spread={spread}, recovered={recovered}"
            )

    def test_round_trip_from_spread(self) -> None:
        """spread_to_win_prob → win_prob_to_spread recovers original."""
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
        _MODEL_SIGMAS.clear()

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
        rng: Generator = np.random.default_rng(42)
        n = 500

        # Generate random home win probabilities (away from extremes)
        home_probs: Series = pd.Series(rng.uniform(0.20, 0.80, size=n))

        # True predicted margin = true_sigma * ppf(home_win_prob)
        predicted_margins = true_sigma * norm.ppf(home_probs.values)

        # Add noise (actual margins ≈ predicted + noise)
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
        _MODEL_SIGMAS.clear()

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
        enriched: DataFrame = enrich_predictions(df)
        assert "model_spread" in enriched.columns

    def test_does_not_mutate_input(self) -> None:
        """Input DataFrame is not modified."""
        df: DataFrame = self._make_predictions_df()
        original_columns: list[str] = list(df.columns)
        original_values: DataFrame = df.copy()
        enrich_predictions(df)
        assert list(df.columns) == original_columns
        pd.testing.assert_frame_equal(df, original_values)

    def test_spread_values_correct(self) -> None:
        """Spread values match win_prob_to_spread applied to each row."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df)
        for _, row in enriched.iterrows():
            expected: float = win_prob_to_spread(row["home_win_prob"])
            assert row["model_spread"] == pytest.approx(expected, abs=1e-6)

    def test_pickem_spread_zero(self) -> None:
        """A 50/50 game produces a spread of 0."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df)
        pickem_row = enriched.loc[enriched["home_win_prob"] == 0.50, :]
        assert pickem_row["model_spread"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_home_favorite_negative_spread(self) -> None:
        """Home favorite (65%) produces negative spread."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df)
        fav_row = enriched.loc[enriched["home_win_prob"] == 0.65, :]
        assert fav_row["model_spread"].iloc[0] < 0

    def test_away_favorite_positive_spread(self) -> None:
        """Away favorite (home 35%) produces positive spread."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df)
        dog_row = enriched.loc[enriched["home_win_prob"] == 0.35, :]
        assert dog_row["model_spread"].iloc[0] > 0

    def test_uses_model_sigma(self) -> None:
        """When a model sigma is registered, enrichment uses it."""
        register_sigma("rf_v3", 15.0)
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df, model_version="rf_v3")
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
        enriched: DataFrame = enrich_predictions(df)
        assert "model_spread" in enriched.columns
        assert enriched["model_spread"].iloc[0] < 0  # home favored

    def test_missing_column_raises(self) -> None:
        """KeyError raised when no win probability column found."""
        df = pd.DataFrame({"game_id": ["X"], "score": [42]})
        with pytest.raises(KeyError, match="home_win_prob"):
            enrich_predictions(df)

    def test_preserves_existing_columns(self) -> None:
        """All original columns are preserved in enriched output."""
        df: DataFrame = self._make_predictions_df()
        enriched: DataFrame = enrich_predictions(df)
        for col in df.columns:
            assert col in enriched.columns

# tests/unit/market/test_edge.py
"""Unit tests for edge.py - expected value, edge detection, and classification."""

from __future__ import annotations

import pytest
from scipy.stats import norm

from gridiron_edge.market.edge import (
    _MODERATE_THRESHOLD,
    _STRONG_THRESHOLD,
    MoneylineEdge,
    SpreadEdge,
    TotalEdge,
    classify_edge_strength,
    expected_value,
    moneyline_edge,
    spread_cover_prob,
    spread_edge,
    total_cover_prob,
    total_edge,
)
from gridiron_edge.market.odds_math import american_to_decimal

# ---------------------------------------------------------------------------
# TestExpectedValue
# ---------------------------------------------------------------------------


class TestExpectedValue:
    """Tests for expected_value()."""

    def test_positive_ev(self) -> None:
        """60% model prob at +100 (dec 2.0) -> EV = 0.6*2.0 - 1.0 = 0.20."""
        assert expected_value(0.6, 100) == pytest.approx(0.20, abs=1e-9)

    def test_negative_ev(self) -> None:
        """40% model prob at -110 (dec ~1.909) -> EV < 0."""
        ev: float = expected_value(0.4, -110)
        # 0.4 * (1 + 100/110) - 1.0 = 0.4 * 1.90909... - 1.0 = -0.23636...
        assert ev == pytest.approx(0.4 * american_to_decimal(-110) - 1.0, abs=1e-9)
        assert ev < 0

    def test_break_even(self) -> None:
        """Model prob exactly equals implied prob -> EV ~ 0."""
        # At -110, implied prob = 110/210 = 0.52381...
        # EV = 0.52381 * 1.90909 - 1.0 ≈ 0
        imp: float = 110.0 / 210.0
        assert expected_value(imp, -110) == pytest.approx(0.0, abs=1e-6)

    def test_heavy_favorite(self) -> None:
        """90% model prob at -300 (dec 1.3333) -> EV = 0.9*1.3333 - 1 = 0.2."""
        ev: float = expected_value(0.9, -300)
        assert ev == pytest.approx(0.9 * american_to_decimal(-300) - 1.0, abs=1e-9)

    def test_long_shot(self) -> None:
        """15% model prob at +500 (dec 6.0) -> EV = 0.15*6.0 - 1 = -0.1."""
        ev: float = expected_value(0.15, 500)
        assert ev == pytest.approx(0.15 * 6.0 - 1.0, abs=1e-9)

    def test_invalid_prob_zero_raises(self) -> None:
        """Probability of 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Probability"):
            expected_value(0.0, -110)

    def test_invalid_prob_one_raises(self) -> None:
        """Probability of 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Probability"):
            expected_value(1.0, -110)

    def test_zero_odds_raises(self) -> None:
        """American odds of zero should raise ValueError."""
        with pytest.raises(ValueError, match="zero"):
            expected_value(0.5, 0)


# ---------------------------------------------------------------------------
# TestMoneylineEdge
# ---------------------------------------------------------------------------


class TestMoneylineEdge:
    """Tests for moneyline_edge()."""

    def test_home_edge(self) -> None:
        """Model likes home (65%) more than market -> home MoneylineEdge."""
        result: MoneylineEdge | None = moneyline_edge(0.65, -150, 130)
        assert result is not None
        assert result.side == "home"
        assert result.model_prob == 0.65
        assert result.ev > 0
        assert result.kelly_frac > 0

    def test_away_edge(self) -> None:
        """Model likes away (65%) more than market -> away MoneylineEdge."""
        result: MoneylineEdge | None = moneyline_edge(0.35, -150, 130)
        assert result is not None
        assert result.side == "away"
        assert result.model_prob == pytest.approx(0.65, abs=1e-9)
        assert result.ev > 0

    def test_no_edge(self) -> None:
        """When model agrees with fair market, no edge exists."""
        # -110 / -110 -> fair prob ≈ 0.50 each (power devig).
        # Model at 0.50 should produce no edge at -110 juice.
        result: MoneylineEdge | None = moneyline_edge(0.50, -110, -110)
        assert result is None

    def test_pickem_with_model_lean(self) -> None:
        """Even-money odds with slight model lean -> edge on the lean side."""
        # +100 / +100 -> fair 50/50.  Model at 55% -> home edge.
        result: MoneylineEdge | None = moneyline_edge(0.55, 100, 100)
        assert result is not None
        assert result.side == "home"
        assert result.ev == pytest.approx(0.55 * 2.0 - 1.0, abs=1e-9)

    def test_edge_is_frozen(self) -> None:
        """MoneylineEdge is immutable (frozen dataclass)."""
        result: MoneylineEdge | None = moneyline_edge(0.65, -150, 130)
        assert result is not None
        with pytest.raises(AttributeError):
            result.ev = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestSpreadCoverProb
# ---------------------------------------------------------------------------


class TestSpreadCoverProb:
    """Tests for spread_cover_prob()."""

    def test_model_agrees_with_market(self) -> None:
        """When model spread == market spread, cover prob is 0.5."""
        assert spread_cover_prob(-7.0, -7.0, 13.0) == pytest.approx(0.5, abs=1e-9)

    def test_model_favors_home_more(self) -> None:
        """Model spread more negative -> home cover prob > 0.5."""
        # model=-7, market=-3 -> ((-3) - (-7)) / 13 = 4/13 -> Φ(0.3077)
        prob: float = spread_cover_prob(-7.0, -3.0, 13.0)
        expected: float = float(norm.cdf(4.0 / 13.0))
        assert prob == pytest.approx(expected, abs=1e-9)
        assert prob > 0.5

    def test_model_favors_home_less(self) -> None:
        """Model spread less negative -> home cover prob < 0.5."""
        # model=-3, market=-7 -> ((-7) - (-3)) / 13 = -4/13 -> Φ(-0.3077)
        prob: float = spread_cover_prob(-3.0, -7.0, 13.0)
        expected: float = float(norm.cdf(-4.0 / 13.0))
        assert prob == pytest.approx(expected, abs=1e-9)
        assert prob < 0.5

    def test_symmetry(self) -> None:
        """Swapping model/market roles gives complementary probabilities."""
        p1: float = spread_cover_prob(-7.0, -3.0, 13.0)
        p2: float = spread_cover_prob(-3.0, -7.0, 13.0)
        assert p1 + p2 == pytest.approx(1.0, abs=1e-9)

    def test_invalid_std_raises(self) -> None:
        """margin_std <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="margin_std"):
            spread_cover_prob(-7.0, -3.0, 0.0)
        with pytest.raises(ValueError, match="margin_std"):
            spread_cover_prob(-7.0, -3.0, -1.0)


# ---------------------------------------------------------------------------
# TestSpreadEdge
# ---------------------------------------------------------------------------


class TestSpreadEdge:
    """Tests for spread_edge()."""

    def test_home_cover_edge(self) -> None:
        """Model has home as much stronger -> home cover edge."""
        # model=-10, market=-3 -> big home-cover prob
        result: SpreadEdge | None = spread_edge(-10.0, -3.0, -110, -110, 13.0)
        assert result is not None
        assert result.side == "home"
        assert result.ev > 0
        assert result.point_edge == pytest.approx(7.0, abs=1e-9)
        assert result.cover_prob > 0.5

    def test_away_cover_edge(self) -> None:
        """Model has home as much weaker -> away cover edge."""
        # model=+3, market=-7 -> away covers easily
        result: SpreadEdge | None = spread_edge(3.0, -7.0, -110, -110, 13.0)
        assert result is not None
        assert result.side == "away"
        assert result.ev > 0
        assert result.cover_prob > 0.5

    def test_no_edge(self) -> None:
        """When model agrees with market, no spread edge."""
        result: SpreadEdge | None = spread_edge(-7.0, -7.0, -110, -110, 13.0)
        assert result is None

    def test_fields_populated(self) -> None:
        """All SpreadEdge fields should be populated correctly."""
        result: SpreadEdge | None = spread_edge(-10.0, -3.0, -110, -110, 13.0)
        assert result is not None
        assert result.model_spread == -10.0
        assert result.market_spread == -3.0
        assert result.odds == -110
        assert result.kelly_frac >= 0.0


# ---------------------------------------------------------------------------
# TestTotalCoverProb
# ---------------------------------------------------------------------------


class TestTotalCoverProb:
    """Tests for total_cover_prob()."""

    def test_model_higher_than_market(self) -> None:
        """Model total > market -> over prob > 0.5."""
        # model=50, market=45, std=13 -> Φ(5/13) = Φ(0.3846)
        prob: float = total_cover_prob(50.0, 45.0, 13.0)
        expected: float = float(norm.cdf(5.0 / 13.0))
        assert prob == pytest.approx(expected, abs=1e-9)
        assert prob > 0.5

    def test_model_lower_than_market(self) -> None:
        """Model total < market -> over prob < 0.5 (under favored)."""
        prob: float = total_cover_prob(40.0, 45.0, 13.0)
        expected: float = float(norm.cdf(-5.0 / 13.0))
        assert prob == pytest.approx(expected, abs=1e-9)
        assert prob < 0.5

    def test_model_agrees(self) -> None:
        """Model total == market total -> over prob = 0.5."""
        assert total_cover_prob(45.0, 45.0, 13.0) == pytest.approx(0.5, abs=1e-9)

    def test_invalid_std_raises(self) -> None:
        """total_std <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="total_std"):
            total_cover_prob(50.0, 45.0, 0.0)


# ---------------------------------------------------------------------------
# TestTotalEdge
# ---------------------------------------------------------------------------


class TestTotalEdge:
    """Tests for total_edge()."""

    def test_over_edge(self) -> None:
        """Model total well above market -> over edge."""
        result: TotalEdge | None = total_edge(55.0, 45.0, -110, -110, 13.0)
        assert result is not None
        assert result.side == "over"
        assert result.ev > 0
        assert result.point_edge == pytest.approx(10.0, abs=1e-9)

    def test_under_edge(self) -> None:
        """Model total well below market -> under edge."""
        result: TotalEdge | None = total_edge(35.0, 45.0, -110, -110, 13.0)
        assert result is not None
        assert result.side == "under"
        assert result.ev > 0
        assert result.point_edge == pytest.approx(10.0, abs=1e-9)

    def test_no_edge(self) -> None:
        """When model agrees with market, no total edge."""
        result: TotalEdge | None = total_edge(45.0, 45.0, -110, -110, 13.0)
        assert result is None

    def test_fields_populated(self) -> None:
        """All TotalEdge fields should be populated correctly."""
        result: TotalEdge | None = total_edge(55.0, 45.0, -110, -110, 13.0)
        assert result is not None
        assert result.model_total == 55.0
        assert result.market_total == 45.0
        assert result.odds == -110
        assert result.kelly_frac >= 0.0


# ---------------------------------------------------------------------------
# TestClassifyEdgeStrength
# ---------------------------------------------------------------------------


class TestClassifyEdgeStrength:
    """Tests for classify_edge_strength()."""

    def test_strong(self) -> None:
        """EV of 8% -> strong."""
        assert classify_edge_strength(0.08) == "strong"

    def test_moderate(self) -> None:
        """EV of 3% -> moderate."""
        assert classify_edge_strength(0.03) == "moderate"

    def test_lean(self) -> None:
        """EV of 1% -> lean."""
        assert classify_edge_strength(0.01) == "lean"

    def test_no_edge_negative(self) -> None:
        """EV of -1% -> no_edge."""
        assert classify_edge_strength(-0.01) == "no_edge"

    def test_boundary_strong(self) -> None:
        """EV of exactly 5% -> strong (inclusive)."""
        assert classify_edge_strength(_STRONG_THRESHOLD) == "strong"

    def test_boundary_moderate(self) -> None:
        """EV of exactly 2% -> moderate (inclusive)."""
        assert classify_edge_strength(_MODERATE_THRESHOLD) == "moderate"

    def test_zero_is_no_edge(self) -> None:
        """EV of exactly 0.0 -> no_edge (not lean)."""
        assert classify_edge_strength(0.0) == "no_edge"

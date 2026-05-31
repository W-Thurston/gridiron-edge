"""Unit tests for gridiron_edge.market.odds_math."""

from __future__ import annotations

import pytest

from gridiron_edge.market.odds_math import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    hold_pct,
    no_vig,
)

# ── american_to_decimal ───────────────────────────────────────────────────────


class TestAmericanToDecimal:
    """Tests for american_to_decimal()."""

    def test_positive_odds(self) -> None:
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_negative_odds(self) -> None:
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_even_money_positive(self) -> None:
        assert american_to_decimal(100) == pytest.approx(2.0)

    def test_even_money_negative(self) -> None:
        assert american_to_decimal(-100) == pytest.approx(2.0)

    def test_heavy_favorite(self) -> None:
        assert american_to_decimal(-500) == pytest.approx(1.2)

    def test_big_underdog(self) -> None:
        assert american_to_decimal(500) == pytest.approx(6.0)

    def test_extreme_favorite(self) -> None:
        assert american_to_decimal(-10000) == pytest.approx(1.01)

    def test_extreme_underdog(self) -> None:
        assert american_to_decimal(10000) == pytest.approx(101.0)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            american_to_decimal(0)


# ── american_to_implied_prob ──────────────────────────────────────────────────


class TestAmericanToImpliedProb:
    """Tests for american_to_implied_prob()."""

    def test_standard_vig_line(self) -> None:
        assert american_to_implied_prob(-110) == pytest.approx(110.0 / 210.0)

    def test_even_money_positive(self) -> None:
        assert american_to_implied_prob(100) == pytest.approx(0.5)

    def test_even_money_negative(self) -> None:
        assert american_to_implied_prob(-100) == pytest.approx(0.5)

    def test_heavy_favorite(self) -> None:
        assert american_to_implied_prob(-500) == pytest.approx(500.0 / 600.0)

    def test_big_underdog(self) -> None:
        assert american_to_implied_prob(500) == pytest.approx(100.0 / 600.0)

    def test_extreme_favorite(self) -> None:
        assert american_to_implied_prob(-10000) == pytest.approx(10000.0 / 10100.0)

    def test_extreme_underdog(self) -> None:
        assert american_to_implied_prob(10000) == pytest.approx(100.0 / 10100.0)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            american_to_implied_prob(0)


# ── decimal_to_american ───────────────────────────────────────────────────────


class TestDecimalToAmerican:
    """Tests for decimal_to_american()."""

    def test_even_money(self) -> None:
        assert decimal_to_american(2.0) == 100

    def test_positive_american(self) -> None:
        assert decimal_to_american(2.5) == 150

    def test_negative_american(self) -> None:
        assert decimal_to_american(1.5) == -200

    def test_heavy_favorite(self) -> None:
        assert decimal_to_american(1.2) == -500

    def test_big_underdog(self) -> None:
        assert decimal_to_american(6.0) == 500

    def test_extreme_favorite(self) -> None:
        assert decimal_to_american(1.01) == -10000

    def test_extreme_underdog(self) -> None:
        assert decimal_to_american(101.0) == 10000

    def test_at_one_raises(self) -> None:
        with pytest.raises(ValueError):
            decimal_to_american(1.0)

    def test_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            decimal_to_american(0.5)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            decimal_to_american(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            decimal_to_american(-1.5)


# ── Roundtrip: american → decimal → american ─────────────────────────────────


class TestRoundtrip:
    """american → decimal → american should be identity (within rounding)."""

    @pytest.mark.parametrize(
        "odds",
        [100, 150, -150, 200, -200, 300, -300, 500, -500, 10000, -10000],
    )
    def test_roundtrip(self, odds: int) -> None:
        dec: float = american_to_decimal(odds)
        back: int = decimal_to_american(dec)
        assert back == odds

    def test_negative_100_normalises_to_positive(self) -> None:
        """Even-money: -100 converts to dec 2.0, which maps back to +100."""
        dec: float = american_to_decimal(-100)
        back: int = decimal_to_american(dec)
        assert back == 100


# ── hold_pct ──────────────────────────────────────────────────────────────────


class TestHoldPct:
    """Tests for hold_pct()."""

    def test_standard_vig(self) -> None:
        h: float = hold_pct(-110, -110)
        assert h == pytest.approx(10.0 / 210.0)

    def test_no_vig_market(self) -> None:
        h: float = hold_pct(100, -100)
        assert h == pytest.approx(0.0)

    def test_skewed_market_positive_hold(self) -> None:
        h: float = hold_pct(-300, 240)
        assert h > 0.0


# ── no_vig ────────────────────────────────────────────────────────────────────


class TestNoVig:
    """Tests for no_vig()."""

    def test_additive_symmetric(self) -> None:
        p_a, p_b = no_vig(-110, -110, method="additive")
        assert p_a == pytest.approx(0.5)
        assert p_b == pytest.approx(0.5)

    def test_power_symmetric(self) -> None:
        p_a, p_b = no_vig(-110, -110, method="power")
        assert p_a == pytest.approx(0.5)
        assert p_b == pytest.approx(0.5)

    def test_sums_to_one_additive(self) -> None:
        p_a, p_b = no_vig(-300, 240, method="additive")
        assert p_a + p_b == pytest.approx(1.0)

    def test_sums_to_one_power(self) -> None:
        p_a, p_b = no_vig(-300, 240, method="power")
        assert p_a + p_b == pytest.approx(1.0)

    def test_power_is_default(self) -> None:
        default: tuple[float, float] = no_vig(-300, 240)
        explicit: tuple[float, float] = no_vig(-300, 240, method="power")
        assert default[0] == pytest.approx(explicit[0])
        assert default[1] == pytest.approx(explicit[1])

    def test_fair_market(self) -> None:
        """A +100 / -100 market is already fair."""
        p_a, p_b = no_vig(100, -100)
        assert p_a == pytest.approx(0.5)
        assert p_b == pytest.approx(0.5)

    def test_extreme_odds(self) -> None:
        p_a, p_b = no_vig(-10000, 10000)
        assert p_a + p_b == pytest.approx(1.0)
        assert p_a > 0.99

    def test_positive_both_sides(self) -> None:
        """Both sides positive (negative hold / arb scenario)."""
        p_a, p_b = no_vig(110, 110)
        assert p_a == pytest.approx(0.5)
        assert p_b == pytest.approx(0.5)
        assert p_a + p_b == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("odds_a", "odds_b"),
        [
            (-110, -110),
            (-150, 130),
            (-200, 170),
            (-300, 240),
            (-500, 380),
        ],
    )
    def test_power_probs_not_above_raw(self, odds_a: int, odds_b: int) -> None:
        """Fair probs from the power method should never exceed raw implied."""
        raw_a: float = american_to_implied_prob(odds_a)
        raw_b: float = american_to_implied_prob(odds_b)
        p_a, p_b = no_vig(odds_a, odds_b, method="power")
        assert p_a <= raw_a + 1e-9
        assert p_b <= raw_b + 1e-9

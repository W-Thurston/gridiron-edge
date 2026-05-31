"""Unit tests for gridiron_edge.market.kelly."""

from __future__ import annotations

import pytest

from gridiron_edge.market.kelly import kelly_fraction, kelly_stake

# ── kelly_fraction ────────────────────────────────────────────────────────────


class TestKellyFraction:
    """Tests for kelly_fraction()."""

    def test_no_edge_returns_zero(self) -> None:
        # +100 ⇒ implied 50 %;  model also 50 % ⇒ no edge.
        assert kelly_fraction(0.5, 100) == 0.0

    def test_negative_edge_returns_zero(self) -> None:
        assert kelly_fraction(0.4, 100) == 0.0

    def test_positive_edge(self) -> None:
        # b = 1.0;  f = (1 * 0.6 - 0.4) / 1 = 0.2
        assert kelly_fraction(0.6, 100) == pytest.approx(0.2)

    def test_large_edge(self) -> None:
        # b = 1.0;  f = (1 * 0.9 - 0.1) / 1 = 0.8
        assert kelly_fraction(0.9, 100) == pytest.approx(0.8)

    def test_no_edge_with_negative_odds(self) -> None:
        # -150 ⇒ implied 60 %;  model also 60 % ⇒ no edge.
        # b = 2/3;  f = (2/3 * 0.6 - 0.4) / (2/3) = 0.0
        assert kelly_fraction(0.6, -150) == pytest.approx(0.0, abs=1e-9)

    def test_positive_edge_with_negative_odds(self) -> None:
        # -150 ⇒ dec 1.6667;  b = 0.6667
        # f = (0.6667 * 0.7 - 0.3) / 0.6667 = 0.25
        assert kelly_fraction(0.7, -150) == pytest.approx(0.25, rel=1e-4)

    def test_prob_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Probability"):
            kelly_fraction(0.0, 100)

    def test_prob_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Probability"):
            kelly_fraction(1.0, 100)

    def test_prob_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="Probability"):
            kelly_fraction(-0.1, 100)

    def test_prob_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Probability"):
            kelly_fraction(1.1, 100)

    def test_odds_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            kelly_fraction(0.6, 0)


# ── kelly_stake ───────────────────────────────────────────────────────────────


class TestKellyStake:
    """Tests for kelly_stake()."""

    def test_quarter_kelly_default(self) -> None:
        # full = 0.2;  quarter = 0.25 * 0.2 = 0.05;  stake = 1000 * 0.05 = 50
        assert kelly_stake(0.6, 100, bankroll=1000.0) == pytest.approx(50.0)

    def test_half_kelly(self) -> None:
        # full = 0.2;  half = 0.5 * 0.2 = 0.1;  stake = 1000 * 0.1 = 100
        assert kelly_stake(0.6, 100, bankroll=1000.0, fraction=0.5) == pytest.approx(100.0)

    def test_full_kelly(self) -> None:
        # full = 0.2;  stake = 1000 * 1.0 * 0.2 = 200
        assert kelly_stake(0.6, 100, bankroll=1000.0, fraction=1.0) == pytest.approx(200.0)

    def test_no_edge_returns_zero(self) -> None:
        assert kelly_stake(0.5, 100, bankroll=1000.0) == 0.0

    def test_negative_edge_returns_zero(self) -> None:
        assert kelly_stake(0.4, 100, bankroll=1000.0) == 0.0

    def test_zero_bankroll_returns_zero(self) -> None:
        assert kelly_stake(0.6, 100, bankroll=0.0) == 0.0

    def test_negative_bankroll_raises(self) -> None:
        with pytest.raises(ValueError, match="Bankroll"):
            kelly_stake(0.6, 100, bankroll=-100.0)

    def test_fraction_zero_returns_zero(self) -> None:
        assert kelly_stake(0.6, 100, bankroll=1000.0, fraction=0.0) == 0.0

    def test_fraction_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="Fraction"):
            kelly_stake(0.6, 100, bankroll=1000.0, fraction=-0.1)

    def test_fraction_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Fraction"):
            kelly_stake(0.6, 100, bankroll=1000.0, fraction=1.5)

    def test_prob_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Probability"):
            kelly_stake(0.0, 100, bankroll=1000.0)

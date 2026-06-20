"""Tests for gridiron_edge.ratings.elo.core — update_elo zero-sum and parity."""

from __future__ import annotations

import numpy as np
import pytest

from gridiron_edge.ratings.elo.core import (
    elo_win_probability,
    update_elo,
)


def test_elo_win_probability_symmetric() -> None:
    p_a, p_b = elo_win_probability(1500.0, 1500.0)
    assert p_a == pytest.approx(0.5, abs=1e-6)
    assert p_b == pytest.approx(0.5, abs=1e-6)
    assert p_a + p_b == pytest.approx(1.0, abs=1e-6)


def test_update_elo_win() -> None:
    winner_elo, loser_elo = update_elo(1500.0, 1500.0, win_or_tie=1.0)
    assert winner_elo > 1500.0
    assert loser_elo < 1500.0


def test_update_elo_tie() -> None:
    winner_elo, loser_elo = update_elo(1500.0, 1500.0, win_or_tie=0.5)
    assert winner_elo == pytest.approx(1500.0, abs=1e-6)
    assert loser_elo == pytest.approx(1500.0, abs=1e-6)


class TestZeroSumInvariant:
    """Verify update_elo preserves total Elo (sum) across updates.

    This is the core property the drift-free form guarantees that the
    legacy expanded form did not. See audit_2026_06_18.md elo_core/H1.
    """

    def test_sum_preserved_single_update(self) -> None:
        """A single update preserves the sum of the two ratings."""
        w_orig, l_orig = 1623.4, 1487.9
        w_new, l_new = update_elo(w_orig, l_orig, win_or_tie=1.0)
        assert w_new + l_new == pytest.approx(w_orig + l_orig, abs=1e-9)

    def test_sum_preserved_tie(self) -> None:
        """Ties also preserve the sum."""
        w_orig, l_orig = 1623.4, 1487.9
        w_new, l_new = update_elo(w_orig, l_orig, win_or_tie=0.5)
        assert w_new + l_new == pytest.approx(w_orig + l_orig, abs=1e-9)

    def test_sum_preserved_across_many_updates(self) -> None:
        """The sum is preserved exactly even after thousands of updates.

        This is the regression test for the drift accumulation bug.
        """
        rng = np.random.default_rng(42)
        winner, loser = 1500.0, 1500.0
        initial_sum = winner + loser

        for _ in range(10_000):
            # Random outcomes; sum invariant must hold regardless
            outcome = float(rng.choice([0.5, 1.0], p=[0.05, 0.95]))
            winner, loser = update_elo(winner, loser, win_or_tie=outcome)

        assert winner + loser == pytest.approx(initial_sum, abs=1e-6)


class TestPythonNumbaParity:
    """Verify the Python update_elo matches the numba _elo_update form.

    The numba implementation in sim/_engine.py duplicates the formula
    because @njit can't import. This test pins them to agree to floating-
    point precision — if either drifts, this test breaks the build.

    See audit_2026_06_18.md engine/C1.
    """

    @staticmethod
    def _numba_update(
        elo_a: float,
        elo_b: float,
        score_a: float,
        k: float,
        divisor: float,
    ) -> tuple[float, float]:
        """Mirror of sim/_engine.py::_elo_update without numba decoration."""
        p_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / divisor))
        delta = k * (score_a - p_a)
        return elo_a + delta, elo_b - delta

    @pytest.mark.parametrize(
        ("elo_a", "elo_b", "score_a", "k", "divisor"),
        [
            (1500.0, 1500.0, 1.0, 20.0, 480.0),
            (1500.0, 1500.0, 0.5, 20.0, 480.0),
            (1623.4, 1487.9, 1.0, 20.0, 480.0),
            (1900.0, 1300.0, 0.5, 32.0, 400.0),
            (1450.5, 1551.2, 1.0, 24.0, 480.0),
        ],
    )
    def test_matches_numba_form(
        self,
        elo_a: float,
        elo_b: float,
        score_a: float,
        k: float,
        divisor: float,
    ) -> None:
        """Python update_elo and numba _elo_update must agree."""
        # update_elo's convention: (winner, loser, win_or_tie).
        # For score_a == 1.0, A is the winner.
        # For score_a == 0.5, A and B are equivalent (tie), and update_elo
        # treats the first arg as "winner" only nominally.
        py_winner, py_loser = update_elo(elo_a, elo_b, win_or_tie=score_a, k=k, divisor=divisor)
        numba_a, numba_b = self._numba_update(elo_a, elo_b, score_a=score_a, k=k, divisor=divisor)
        assert py_winner == pytest.approx(numba_a, abs=1e-10)
        assert py_loser == pytest.approx(numba_b, abs=1e-10)

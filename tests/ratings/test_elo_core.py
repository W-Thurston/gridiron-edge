import pytest

from gridiron_edge.ratings.elo.core import elo_win_probability, update_elo


def test_elo_win_probability_symmetric() -> None:
    p_a, p_b = elo_win_probability(1500.0, 1500.0)
    assert p_a == pytest.approx(0.5, abs=1e-6)
    assert p_b == pytest.approx(0.5, abs=1e-6)
    assert p_a + p_b == pytest.approx(1.0, abs=1e-6)


def test_update_elo_win() -> None:
    w, l = update_elo(1500.0, 1500.0, win_or_tie=1.0)
    assert w > 1500.0
    assert l < 1500.0


def test_update_elo_tie() -> None:
    w, l = update_elo(1500.0, 1500.0, win_or_tie=0.5)
    assert w == pytest.approx(1500.0, abs=1e-6)
    assert l == pytest.approx(1500.0, abs=1e-6)

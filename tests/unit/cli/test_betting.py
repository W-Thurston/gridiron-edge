"""Tests for cli/betting.py rendering helpers."""

from __future__ import annotations

import pytest

from gridiron_edge.cli.betting import _render_summary

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_stats(
    *,
    ev_gap: float = float("nan"),
    health: str = "unknown",
    n_model_bets: int = 0,
    mean_clv: float = float("nan"),
    n_clv_bets: int = 0,
) -> dict:
    """Build a minimal stats dict matching summary() output shape."""
    return {
        "wins": 1,
        "losses": 0,
        "pushes": 0,
        "total": 1,
        "win_pct": 1.0,
        "total_staked": 100.0,
        "total_pnl": 90.0,
        "roi_pct": 90.0,
        "mean_clv": mean_clv,
        "n_clv_bets": n_clv_bets,
        "ev_vs_actual_gap": ev_gap,
        "n_model_bets": n_model_bets,
        "calibration_health": health,
        "current_streak": 1,
        "current_streak_type": "W",
        "longest_win_streak": 1,
        "longest_loss_streak": 0,
    }


# ---------------------------------------------------------------------------
# Calibration display
# ---------------------------------------------------------------------------


class TestCalibrationDisplay:
    """Verify EV gap + Health rendering in bet summary."""

    def test_renders_ev_gap_when_present(self, capsys: pytest.CaptureFixture) -> None:
        stats = _make_stats(ev_gap=0.005, health="healthy", n_model_bets=5)
        _render_summary(stats)
        captured = capsys.readouterr()
        assert "EV gap:  +0.0050" in captured.out
        assert "(n=5)" in captured.out

    def test_renders_em_dash_when_ev_gap_nan(self, capsys: pytest.CaptureFixture) -> None:
        stats = _make_stats(ev_gap=float("nan"))
        _render_summary(stats)
        captured = capsys.readouterr()
        assert "EV gap:  —" in captured.out

    def test_renders_healthy_with_check(self, capsys: pytest.CaptureFixture) -> None:
        stats = _make_stats(ev_gap=0.001, health="healthy", n_model_bets=5)
        _render_summary(stats)
        captured = capsys.readouterr()
        assert "Health:  ✓ healthy" in captured.out

    def test_renders_degraded_with_warning(self, capsys: pytest.CaptureFixture) -> None:
        stats = _make_stats(ev_gap=-0.02, health="degraded", n_model_bets=5)
        _render_summary(stats)
        captured = capsys.readouterr()
        assert "Health:  ⚠ degraded" in captured.out

    def test_renders_unknown_with_dash(self, capsys: pytest.CaptureFixture) -> None:
        stats = _make_stats(health="unknown", n_model_bets=0)
        _render_summary(stats)
        captured = capsys.readouterr()
        assert "Health:  — unknown" in captured.out

    def test_negative_ev_gap_shows_sign(self, capsys: pytest.CaptureFixture) -> None:
        stats = _make_stats(ev_gap=-0.015, health="degraded", n_model_bets=10)
        _render_summary(stats)
        captured = capsys.readouterr()
        assert "EV gap:  -0.0150" in captured.out

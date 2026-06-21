"""Tests for viz/charts._format_records (charts/H1)."""

from __future__ import annotations

import numpy as np


class TestFormatRecords:
    """Verify vectorized record formatting matches the per-team loop semantics."""

    def test_no_ties_uses_wl_format(self) -> None:
        from gridiron_edge.viz.charts import _format_records

        wins = np.array([10, 8, 12])
        losses = np.array([7, 9, 5])
        ties = np.array([0, 0, 0])
        result = _format_records(wins, losses, ties)
        assert result == ["10-7", "8-9", "12-5"]

    def test_ties_use_wlt_format(self) -> None:
        from gridiron_edge.viz.charts import _format_records

        wins = np.array([8, 10])
        losses = np.array([8, 6])
        ties = np.array([1, 0])
        result = _format_records(wins, losses, ties)
        assert result == ["8-8-1", "10-6"]

    def test_mixed(self) -> None:
        from gridiron_edge.viz.charts import _format_records

        wins = np.array([10, 8, 0, 17])
        losses = np.array([7, 8, 17, 0])
        ties = np.array([0, 1, 0, 0])
        result = _format_records(wins, losses, ties)
        assert result == ["10-7", "8-8-1", "0-17", "17-0"]

    def test_preserves_team_order(self) -> None:
        from gridiron_edge.viz.charts import _format_records

        wins = np.array([1, 2, 3, 4, 5])
        losses = np.array([5, 4, 3, 2, 1])
        ties = np.array([0, 0, 0, 0, 0])
        result = _format_records(wins, losses, ties)
        assert result == ["1-5", "2-4", "3-3", "4-2", "5-1"]

    def test_int_coercion(self) -> None:
        """Float inputs should be coerced to int for clean string output."""
        from gridiron_edge.viz.charts import _format_records

        wins = np.array([10.0, 8.7])  # 8.7 truncates to 8
        losses = np.array([7.0, 9.0])
        ties = np.array([0.0, 0.0])
        result = _format_records(wins, losses, ties)
        assert result == ["10-7", "8-9"]

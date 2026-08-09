"""Unit tests for pure Closing Line Value calculations."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from gridiron_edge.market.clv import (
    closing_line_value,
    spread_clv,
    summarize_clv,
    total_clv,
)


class TestClosingLineValue:
    """Tests for probability-based CLV math."""

    def test_positive_clv(self) -> None:
        """A higher validated closing probability produces positive CLV."""
        result = closing_line_value(0.55, 0.60)
        assert result == pytest.approx(0.05 / 0.55)

    @pytest.mark.parametrize(
        ("bet_probability", "close_probability"),
        [(0.0, 0.55), (0.55, 1.0)],
    )
    def test_invalid_probability_raises(
        self,
        bet_probability: float,
        close_probability: float,
    ) -> None:
        """Probabilities must be strictly between zero and one."""
        with pytest.raises(ValueError, match="must be in"):
            closing_line_value(bet_probability, close_probability)


class TestSpreadClv:
    """Tests for spread point movement."""

    @pytest.mark.parametrize(
        ("bet", "close", "side", "expected"),
        [
            (-3.0, -7.0, "home", 4.0),
            (-7.0, -3.0, "home", -4.0),
            (-7.0, -3.0, "away", 4.0),
            (-3.0, -7.0, "away", -4.0),
        ],
    )
    def test_point_movement(
        self,
        bet: float,
        close: float,
        side: str,
        expected: float,
    ) -> None:
        """Point movement follows the selected side orientation."""
        assert spread_clv(bet, close, side) == pytest.approx(expected)

    def test_invalid_side_raises(self) -> None:
        """Unsupported sides are rejected."""
        with pytest.raises(ValueError, match="side"):
            spread_clv(-3.0, -7.0, "over")


class TestTotalClv:
    """Tests for total point movement."""

    @pytest.mark.parametrize(
        ("bet", "close", "side", "expected"),
        [
            (42.0, 45.0, "over", 3.0),
            (48.0, 45.0, "under", 3.0),
            (45.0, 42.0, "over", -3.0),
            (45.0, 48.0, "under", -3.0),
        ],
    )
    def test_point_movement(
        self,
        bet: float,
        close: float,
        side: str,
        expected: float,
    ) -> None:
        """Point movement follows the selected side orientation."""
        assert total_clv(bet, close, side) == pytest.approx(expected)

    def test_invalid_side_raises(self) -> None:
        """Unsupported sides are rejected."""
        with pytest.raises(ValueError, match="side"):
            total_clv(45.0, 48.0, "home")


class TestSummarizeClv:
    """Tests for summaries of already validated CLV values."""

    def test_summary(self) -> None:
        """Summary metrics use only explicitly supplied CLV values."""
        result = summarize_clv(pd.DataFrame({"clv": [0.10, 0.05, -0.02]}))
        assert result["mean_clv"] == pytest.approx(0.13 / 3)
        assert result["median_clv"] == pytest.approx(0.05)
        assert result["pct_positive_clv"] == pytest.approx(2 / 3)
        assert result["n_edges"] == 3.0

    @pytest.mark.parametrize(
        "frame",
        [pd.DataFrame(), pd.DataFrame({"clv": [None]})],
    )
    def test_unavailable_summary(self, frame: pd.DataFrame) -> None:
        """Missing validated values produce an unavailable summary."""
        result = summarize_clv(frame)
        assert math.isnan(result["mean_clv"])
        assert math.isnan(result["median_clv"])
        assert math.isnan(result["pct_positive_clv"])
        assert result["n_edges"] == 0.0

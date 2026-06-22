"""Pure odds-conversion and market-math functions.

No data dependencies.  Every function in this module is a leaf that operates on
scalar values and returns scalar values.
"""

from __future__ import annotations

from typing import Literal

NoVigMethod = Literal["power", "additive"]


# ── Odds conversion ────────────────────────────────────────────────────────────


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal (European) odds.

    Parameters
    ----------
    odds : int
        American-format odds (e.g. -110, +150).  Zero is invalid.

    Returns:
    -------
    float
        Equivalent decimal odds (always > 1.0).

    Raises:
    ------
    ValueError
        If *odds* is zero.
    """
    if odds == 0:
        raise ValueError("American odds of zero are undefined.")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to raw implied probability.

    The returned value includes the bookmaker's hold and therefore the sum
    across a two-way market will exceed 1.0.

    Parameters
    ----------
    odds : int
        American-format odds.  Zero is invalid.

    Returns:
    -------
    float
        Implied probability in [0, 1].

    Raises:
    ------
    ValueError
        If *odds* is zero.
    """
    if odds == 0:
        raise ValueError("American odds of zero are undefined.")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def decimal_to_american(dec: float) -> int:
    """Convert decimal (European) odds to American odds.

    Even-money (dec == 2.0) returns +100.

    Parameters
    ----------
    dec : float
        Decimal odds.  Must be strictly greater than 1.0.

    Returns:
    -------
    int
        Equivalent American odds (positive or negative).

    Raises:
    ------
    ValueError
        If *dec* is <= 1.0.
    """
    if dec <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {dec}")
    if dec >= 2.0:
        return round((dec - 1.0) * 100.0)
    return round(-100.0 / (dec - 1.0))


# ── Market-level helpers ───────────────────────────────────────────────────────


def hold_pct(odds_a: int, odds_b: int) -> float:
    """Return the bookmaker hold (overround) for a two-way market.

    Parameters
    ----------
    odds_a, odds_b : int
        American odds for each side of the market.

    Returns:
    -------
    float
        Hold as a fraction (e.g. 0.0476 for a typical -110/-110 market).
    """
    return american_to_implied_prob(odds_a) + american_to_implied_prob(odds_b) - 1.0


def no_vig(
    odds_a: int,
    odds_b: int,
    *,
    method: NoVigMethod = "power",
) -> tuple[float, float]:
    """Remove the bookmaker's vig and return fair probabilities.

    Parameters
    ----------
    odds_a, odds_b : int
        American odds for each side (e.g. -110, -110).
    method : {"power", "additive"}, default "power"
        Debiasing method.

        * ``"additive"`` - simple rescaling (divide each raw probability by
          their sum).
        * ``"power"`` - multiplicative / power method.  Finds exponent *k*
          such that ``raw_a ** k + raw_b ** k == 1``.  More accurate for
          skewed lines.

    Returns:
    -------
    tuple[float, float]
        Fair probabilities that sum to 1.0.
    """
    raw_a: float = american_to_implied_prob(odds_a)
    raw_b: float = american_to_implied_prob(odds_b)

    if method == "additive":
        total: float = raw_a + raw_b
        return raw_a / total, raw_b / total

    # Default: power method.
    return _power_devig(raw_a, raw_b)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _power_devig(
    raw_a: float,
    raw_b: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[float, float]:
    """Remove vig via the power (multiplicative) method.

    Finds exponent *k* such that ``raw_a ** k + raw_b ** k == 1`` using
    bisection, then returns ``(raw_a ** k, raw_b ** k)``.

    Parameters
    ----------
    raw_a, raw_b : float
        Raw implied probabilities (typically sum > 1 due to vig).
    tol : float
        Convergence tolerance for bisection.
    max_iter : int
        Maximum bisection iterations.

    Returns:
    -------
    tuple[float, float]
        Fair probabilities summing to 1.0.
    """
    total: float = raw_a + raw_b

    # Already fair - nothing to solve.
    if abs(total - 1.0) < tol:
        return raw_a, raw_b

    # Determine search bounds for k.
    # If total > 1 (normal vig), k > 1 shrinks values toward zero.
    # If total < 1 (negative hold / arb), k < 1 inflates values.
    if total > 1.0:
        lo, hi = 1.0, 1000.0
    else:
        lo, hi = 0.001, 1.0

    for _ in range(max_iter):
        mid: float = (lo + hi) / 2.0
        s = raw_a**mid + raw_b**mid
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo: float = mid
        else:
            hi: float = mid

    return raw_a**mid, raw_b**mid

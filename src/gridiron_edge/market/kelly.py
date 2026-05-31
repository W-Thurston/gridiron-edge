"""Kelly criterion staking functions.

No data dependencies.  Uses :func:`american_to_decimal` from
``gridiron_edge.market.odds_math`` for odds conversion.
"""

from __future__ import annotations

from gridiron_edge.market.odds_math import american_to_decimal

# ── Public API ────────────────────────────────────────────────────────────────


def kelly_fraction(model_prob: float, american_odds: int) -> float:
    """Return the full-Kelly fraction for a single wager.

    Parameters
    ----------
    model_prob : float
        Model's estimated win probability (exclusive 0-1).
    american_odds : int
        American-format odds offered by the book.

    Returns:
    -------
    float
        Optimal fraction of bankroll to wager (0.0 when edge is non-positive).

    Raises:
    ------
    ValueError
        If *model_prob* is not in (0, 1) or *american_odds* is zero.
    """
    _validate_prob(model_prob)
    dec: float = american_to_decimal(american_odds)  # raises if odds == 0
    b: float = dec - 1.0  # net decimal odds (payout per unit wagered)
    q: float = 1.0 - model_prob
    f: float = (b * model_prob - q) / b
    return max(f, 0.0)


def kelly_stake(
    model_prob: float,
    american_odds: int,
    bankroll: float,
    fraction: float = 0.25,
) -> float:
    """Return the dollar stake for a fractional-Kelly strategy.

    Parameters
    ----------
    model_prob : float
        Model's estimated win probability (exclusive 0-1).
    american_odds : int
        American-format odds offered by the book.
    bankroll : float
        Current bankroll in dollars.  Must be >= 0.
    fraction : float, default 0.25
        Fraction of full Kelly to apply (e.g. 0.25 for quarter-Kelly).
        Must be in [0, 1].

    Returns:
    -------
    float
        Recommended wager in dollars.  Zero when edge is non-positive or
        bankroll/fraction is zero.

    Raises:
    ------
    ValueError
        If *model_prob* not in (0, 1), *bankroll* < 0, or *fraction* not in
        [0, 1].
    """
    _validate_prob(model_prob)
    if bankroll < 0:
        raise ValueError(f"Bankroll must be >= 0, got {bankroll}")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"Fraction must be in [0, 1], got {fraction}")

    full: float = kelly_fraction(model_prob, american_odds)
    return bankroll * fraction * full


# ── Internal helpers ──────────────────────────────────────────────────────────


def _validate_prob(p: float) -> None:
    """Raise ``ValueError`` if *p* is not strictly between 0 and 1."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probability must be in (0, 1), got {p}")

# src/gridiron_edge/market/edge.py
"""Edge (expected value) calculations for game-level betting markets.

Pure scalar functions - no I/O, no pandas, no data dependencies.  Every
function operates on values already produced by other parts of the
pipeline and returns the edge / EV / Kelly information needed to make a
betting decision.

The module is a **pure-math leaf** following the same pattern as
``odds_math.py`` and ``kelly.py``.  It imports only from within the
``market`` package and from ``scipy.stats`` for the normal CDF.

Data model:
    MoneylineEdge    Result of a moneyline edge calculation
    SpreadEdge       Result of a spread edge calculation
    TotalEdge        Result of a total (over/under) edge calculation

Core functions:
    expected_value          EV of a single wager
    moneyline_edge          Best +EV moneyline side (or None)
    spread_cover_prob       P(home covers the market spread)
    spread_edge             Best +EV spread side (or None)
    total_cover_prob        P(over hits the market total)
    total_edge              Best +EV total side (or None)
    classify_edge_strength  EV -> "strong" / "moderate" / "lean" / "no_edge"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

# pyrefly: ignore [missing-import]
from scipy.stats import norm

from gridiron_edge.market.kelly import kelly_fraction
from gridiron_edge.market.odds_math import (
    american_to_decimal,
    no_vig,
)

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

EdgeStrength = Literal["strong", "moderate", "lean", "no_edge"]


@dataclass(frozen=True, slots=True)
class MoneylineEdge:
    """Result of a moneyline edge calculation.

    Attributes:
    ----------
    side : str
        ``"home"`` or ``"away"`` - the side with positive EV.
    model_prob : float
        Model's estimated win probability for *side*.
    market_prob : float
        No-vig implied probability from the market for *side*.
    ev : float
        Expected value as a decimal (e.g. 0.05 = +5%).
    kelly_frac : float
        Full-Kelly fraction of bankroll.
    odds : int
        American odds offered on *side*.
    """

    side: str
    model_prob: float
    market_prob: float
    ev: float
    kelly_frac: float
    odds: int


@dataclass(frozen=True, slots=True)
class SpreadEdge:
    """Result of a spread edge calculation.

    Attributes:
    ----------
    side : str
        ``"home"`` or ``"away"`` - the side to cover.
    model_spread : float
        Model's derived point spread (negative = home favored).
    market_spread : float
        Market spread for the home team (negative = home favored).
    point_edge : float
        Absolute difference between model and market spread.
    cover_prob : float
        Model's estimated probability that *side* covers.
    ev : float
        Expected value as a decimal.
    kelly_frac : float
        Full-Kelly fraction of bankroll.
    odds : int
        American odds offered on *side* covering.
    """

    side: str
    model_spread: float
    market_spread: float
    point_edge: float
    cover_prob: float
    ev: float
    kelly_frac: float
    odds: int


@dataclass(frozen=True, slots=True)
class TotalEdge:
    """Result of a total (over/under) edge calculation.

    Attributes:
    ----------
    side : str
        ``"over"`` or ``"under"``.
    model_total : float
        Model's projected combined score.
    market_total : float
        Market total (over/under line).
    point_edge : float
        Absolute difference between model and market total.
    cover_prob : float
        Model's estimated probability that *side* hits.
    ev : float
        Expected value as a decimal.
    kelly_frac : float
        Full-Kelly fraction of bankroll.
    odds : int
        American odds offered on *side*.
    """

    side: str
    model_total: float
    market_total: float
    point_edge: float
    cover_prob: float
    ev: float
    kelly_frac: float
    odds: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EV thresholds for edge classification. Starting values can be tuned
# empirically after historical validation.
_STRONG_THRESHOLD: Final[float] = 0.05
_MODERATE_THRESHOLD: Final[float] = 0.02


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expected_value(model_prob: float, american_odds: int) -> float:
    """Return the expected value of a single wager as a decimal.

    EV = model_prob * decimal_odds - 1.0

    A positive return means the bet is +EV; negative means -EV.

    Parameters
    ----------
    model_prob : float
        Model's estimated win probability for the wagered side.
        Must be strictly between 0 and 1.
    american_odds : int
        American-format odds offered by the book.  Zero is invalid.

    Returns:
    -------
    float
        Expected value as a decimal (e.g. 0.05 = +5% EV).

    Raises:
    ------
    ValueError
        If *model_prob* is not in (0, 1) or *american_odds* is zero.
    """
    _validate_prob(model_prob)
    dec: float = american_to_decimal(american_odds)
    return model_prob * dec - 1.0


def moneyline_edge(
    home_win_prob: float,
    ml_home: int,
    ml_away: int,
) -> MoneylineEdge | None:
    """Identify the +EV moneyline side, if any.

    Computes no-vig fair probabilities from the market odds, then
    checks whether the model's probability for either side exceeds
    the fair probability enough to generate positive expected value
    at the offered price.

    Parameters
    ----------
    home_win_prob : float
        Model's estimated home-team win probability (exclusive 0-1).
    ml_home : int
        American moneyline odds on the home team.
    ml_away : int
        American moneyline odds on the away team.

    Returns:
    -------
    MoneylineEdge | None
        Edge details for the +EV side, or ``None`` if neither side
        has positive EV.
    """
    _validate_prob(home_win_prob)
    away_win_prob: float = 1.0 - home_win_prob

    fair_home, fair_away = no_vig(ml_home, ml_away)

    ev_home: float = expected_value(home_win_prob, ml_home)
    ev_away: float = expected_value(away_win_prob, ml_away)

    # Pick the side with the larger positive EV.
    if ev_home > 0 and ev_home >= ev_away:
        return MoneylineEdge(
            side="home",
            model_prob=home_win_prob,
            market_prob=fair_home,
            ev=ev_home,
            kelly_frac=kelly_fraction(home_win_prob, ml_home),
            odds=ml_home,
        )
    if ev_away > 0:
        return MoneylineEdge(
            side="away",
            model_prob=away_win_prob,
            market_prob=fair_away,
            ev=ev_away,
            kelly_frac=kelly_fraction(away_win_prob, ml_away),
            odds=ml_away,
        )
    return None


def spread_cover_prob(
    model_spread: float,
    market_spread: float,
    margin_std: float,
) -> float:
    """Return the probability that the home team covers the market spread.

    Uses the probit model::

        P(home covers) = Φ((market_spread - model_spread) / margin_std)

    Both spreads follow the convention **negative = home favored**.
    When the model's spread is more negative than the market (model
    thinks home is stronger), ``market_spread - model_spread > 0``
    and ``P(home covers) > 0.5``.

    Parameters
    ----------
    model_spread : float
        Model-derived point spread (negative = home favored).
    market_spread : float
        Market point spread for the home team (negative = home favored).
    margin_std : float
        Standard deviation of margin-of-victory residuals.  Must be > 0.

    Returns:
    -------
    float
        Probability in (0, 1) that the home team covers.

    Raises:
    ------
    ValueError
        If *margin_std* is not positive.
    """
    if margin_std <= 0:
        raise ValueError(f"margin_std must be > 0, got {margin_std}")
    return float(norm.cdf((market_spread - model_spread) / margin_std))


def spread_edge(
    model_spread: float,
    market_spread_home: float,
    spread_odds_home: int,
    spread_odds_away: int,
    margin_std: float,
) -> SpreadEdge | None:
    """Identify the +EV spread side, if any.

    Parameters
    ----------
    model_spread : float
        Model-derived point spread (negative = home favored).
    market_spread_home : float
        Market spread for the home team (negative = home favored).
    spread_odds_home : int
        American odds on home covering the spread.
    spread_odds_away : int
        American odds on away covering the spread.
    margin_std : float
        Standard deviation of margin-of-victory residuals.

    Returns:
    -------
    SpreadEdge | None
        Edge details for the +EV side, or ``None`` if neither side
        has positive EV.
    """
    home_cover: float = spread_cover_prob(model_spread, market_spread_home, margin_std)
    away_cover: float = 1.0 - home_cover

    ev_home: float = expected_value(home_cover, spread_odds_home)
    ev_away: float = expected_value(away_cover, spread_odds_away)

    pt_edge: float = abs(model_spread - market_spread_home)

    if ev_home > 0 and ev_home >= ev_away:
        return SpreadEdge(
            side="home",
            model_spread=model_spread,
            market_spread=market_spread_home,
            point_edge=pt_edge,
            cover_prob=home_cover,
            ev=ev_home,
            kelly_frac=kelly_fraction(home_cover, spread_odds_home),
            odds=spread_odds_home,
        )
    if ev_away > 0:
        return SpreadEdge(
            side="away",
            model_spread=model_spread,
            market_spread=market_spread_home,
            point_edge=pt_edge,
            cover_prob=away_cover,
            ev=ev_away,
            kelly_frac=kelly_fraction(away_cover, spread_odds_away),
            odds=spread_odds_away,
        )
    return None


def total_cover_prob(
    model_total: float,
    market_total: float,
    total_std: float,
) -> float:
    """Return the probability that the game goes over the market total.

    Uses the probit model::

        P(over) = Φ((model_total - market_total) / total_std)

    When the model projects a higher total than the market, the over
    probability exceeds 0.5.

    Parameters
    ----------
    model_total : float
        Model's projected combined score.
    market_total : float
        Market total (over/under line).
    total_std : float
        Standard deviation of total-score residuals.  Must be > 0.

    Returns:
    -------
    float
        Probability in (0, 1) that the game goes over.

    Raises:
    ------
    ValueError
        If *total_std* is not positive.
    """
    if total_std <= 0:
        raise ValueError(f"total_std must be > 0, got {total_std}")
    return float(norm.cdf((model_total - market_total) / total_std))


def total_edge(
    model_total: float,
    market_total: float,
    over_odds: int,
    under_odds: int,
    total_std: float,
) -> TotalEdge | None:
    """Identify the +EV total side (over or under), if any.

    Parameters
    ----------
    model_total : float
        Model's projected combined score.
    market_total : float
        Market total (over/under line).
    over_odds : int
        American odds on the over.
    under_odds : int
        American odds on the under.
    total_std : float
        Standard deviation of total-score residuals.

    Returns:
    -------
    TotalEdge | None
        Edge details for the +EV side, or ``None`` if neither side
        has positive EV.
    """
    over_prob: float = total_cover_prob(model_total, market_total, total_std)
    under_prob: float = 1.0 - over_prob

    ev_over: float = expected_value(over_prob, over_odds)
    ev_under: float = expected_value(under_prob, under_odds)

    pt_edge: float = abs(model_total - market_total)

    if ev_over > 0 and ev_over >= ev_under:
        return TotalEdge(
            side="over",
            model_total=model_total,
            market_total=market_total,
            point_edge=pt_edge,
            cover_prob=over_prob,
            ev=ev_over,
            kelly_frac=kelly_fraction(over_prob, over_odds),
            odds=over_odds,
        )
    if ev_under > 0:
        return TotalEdge(
            side="under",
            model_total=model_total,
            market_total=market_total,
            point_edge=pt_edge,
            cover_prob=under_prob,
            ev=ev_under,
            kelly_frac=kelly_fraction(under_prob, under_odds),
            odds=under_odds,
        )
    return None


def classify_edge_strength(ev: float) -> EdgeStrength:
    """Classify an expected value into an edge-strength tier.

    Parameters
    ----------
    ev : float
        Expected value as a decimal (e.g. 0.05 = +5%).

    Returns:
    -------
    EdgeStrength
        One of ``"strong"``, ``"moderate"``, ``"lean"``, ``"no_edge"``.
    """
    if ev >= _STRONG_THRESHOLD:
        return "strong"
    if ev >= _MODERATE_THRESHOLD:
        return "moderate"
    if ev > 0.0:
        return "lean"
    return "no_edge"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_prob(p: float) -> None:
    """Raise ``ValueError`` if *p* is not strictly between 0 and 1."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probability must be in (0, 1), got {p}")

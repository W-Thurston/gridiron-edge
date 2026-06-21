# src/gridiron_edge/ratings/elo/core.py

# NOTE: numba counterparts of these functions exist in sim/season.py
# (_elo_win_prob, _elo_update). They are duplicated there because numba @njit
# functions cannot call regular Python functions at JIT compile time.
# If the Elo formula changes (e.g. a different divisor), update BOTH here
# AND the numba versions in sim/season.py.

#: Default Elo divisor. The classic Elo system uses 400; we use 480 to
#: reduce the sensitivity of win-probability to large rating gaps, which
#: better reflects parity in the NFL. The flat-K and zone-K tuning use
#: different divisors — pass ``divisor`` explicitly to override.
DEFAULT_ELO_DIVISOR: float = 480.0


def elo_win_probability(
    rating_team_a: float,
    rating_team_b: float,
    divisor: float = DEFAULT_ELO_DIVISOR,
) -> tuple[float, float]:
    """Compute win probabilities for two Elo ratings.

    Args:
        rating_team_a: Elo rating for team A (away team by convention).
        rating_team_b: Elo rating for team B (home team by convention).
        divisor: Win-probability divisor. Higher values compress probabilities
            toward 0.5. Defaults to ``DEFAULT_ELO_DIVISOR`` (480).

    Returns:
        ``(p_a, p_b)`` where ``p_a + p_b == 1.0`` (up to floating error).
    """
    p_a: float = 1.0 / (1.0 + 10 ** ((rating_team_b - rating_team_a) / divisor))
    p_b: float = 1.0 / (1.0 + 10 ** ((rating_team_a - rating_team_b) / divisor))
    return p_a, p_b


def update_elo(
    winning_team_elo: float,
    losing_team_elo: float,
    win_or_tie: float = 0.0,
    k: float = 20.0,
    divisor: float = DEFAULT_ELO_DIVISOR,
) -> tuple[float, float]:
    """Update Elo ratings for winner/loser given the outcome.

    Uses the zero-sum delta form: ``delta = k * (score - p_win)``,
    ``new_winner = winner + delta``, ``new_loser = loser - delta``.
    This is mathematically equivalent to the textbook expanded form
    in exact arithmetic but is drift-free under floating-point: the
    invariant ``new_winner + new_loser == winner + loser`` holds to
    machine precision regardless of how ``p_a + p_b`` rounds.

    The legacy expanded form (``new_winner = winner + k * (score_w - p_win)``,
    ``new_loser = loser + k * (score_l - p_lose)``) accumulates drift
    across thousands of updates because it implicitly assumes
    ``p_a + p_b == 1.0`` exactly, but ``elo_win_probability`` only
    guarantees this up to floating error.

    Matches the form used by ``sim/_engine.py::_elo_update`` (numba)
    and the inlined update in ``evaluation/tune.py::_simulate_and_score``.
    See ``audit_2026_06_18.md`` ``elo_core/H1`` and ``engine/C1``.

    Args:
        winning_team_elo: Elo for the winner (or team A if tie).
        losing_team_elo: Elo for the loser (or team B if tie).
        win_or_tie: ``1.0`` for a win, ``0.5`` for a tie.
        k: K-factor controlling update magnitude.
        divisor: Win-probability divisor passed through to
            ``elo_win_probability``. Defaults to ``DEFAULT_ELO_DIVISOR``.

    Returns:
        ``(new_winner_elo, new_loser_elo)`` with the zero-sum invariant
        preserved to machine precision.

    Raises:
        ValueError: If ``win_or_tie`` is not ``1.0`` or ``0.5``.
    """
    if win_or_tie not in (1.0, 0.5):
        msg: str = f"win_or_tie must be 1 (win) or 0.5 (tie). Your value: {win_or_tie}"
        raise ValueError(msg)

    p_winner, _ = elo_win_probability(
        winning_team_elo,
        losing_team_elo,
        divisor=divisor,
    )

    delta: float = k * (win_or_tie - p_winner)
    return winning_team_elo + delta, losing_team_elo - delta

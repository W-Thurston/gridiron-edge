# src/gridiron_edge/ratings/elo/core.py


def elo_win_probability(
    rating_team_a: float,
    rating_team_b: float,
) -> tuple[float, float]:
    """Compute win probabilities for two Elo ratings.

    Notes:
      - Uses 480 as the divisor (instead of classic 400) per the comment in the legacy code.

    Returns:
      (p_a, p_b) where p_a + p_b == 1.0 (up to floating error)

    """
    p_a: float = 1.0 / (1.0 + 10 ** ((rating_team_b - rating_team_a) / 480))
    p_b: float = 1.0 / (1.0 + 10 ** ((rating_team_a - rating_team_b) / 480))
    return p_a, p_b


def update_elo(
    winning_team_elo: float,
    losing_team_elo: float,
    win_or_tie: float = 0.0,
    k: float = 20.0,
) -> tuple[float, float]:
    """Update Elo ratings for winner/loser given the outcome.

    Args:
      winning_team_elo: Elo for the winner (or team A if tie)
      losing_team_elo: Elo for the loser (or team B if tie)
      win_or_tie: must be 1.0 (win) or 0.5 (tie)
      k: K-factor (default 20, legacy default)

    Returns:
      (new_winner_elo, new_loser_elo)

    """
    winners_chances, losers_chances = elo_win_probability(
        winning_team_elo,
        losing_team_elo,
    )

    if win_or_tie == 1:
        new_winner_elo: float = winning_team_elo + k * (1 - winners_chances)
        new_loser_elo: float = losing_team_elo + k * (0 - losers_chances)
    elif win_or_tie == 0.5:
        new_winner_elo = winning_team_elo + k * (0.5 - winners_chances)
        new_loser_elo = losing_team_elo + k * (0.5 - losers_chances)
    else:
        msg: str = f"win_or_tie must be 1 (win) or 0.5 (tie). Your value: {win_or_tie}"
        raise ValueError(msg)

    return new_winner_elo, new_loser_elo

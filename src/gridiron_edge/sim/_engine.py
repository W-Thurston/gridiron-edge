# src/gridiron_edge/sim/_engine.py

"""Numba-optimized simulation kernels and schedule analysis.

All @njit functions in this module are intentionally self-contained —
numba cannot call regular Python functions at JIT compile time, so the
Elo formula is duplicated here from ratings/elo/core.py. If the Elo
formula changes, update BOTH this module AND ratings/elo/core.py.

Public functions (non-numba):
    precompute_game_counts   — Precompute per-team game count arrays.

Numba kernels (not for direct external use):
    _elo_win_prob            — Win probability from two Elo ratings.
    _elo_update              — Update two Elo ratings after a game.
    apply_actuals_to_matrices — Accumulate completed game results.
    simulate_remaining_regular_season — Monte Carlo regular season sim.
"""

from __future__ import annotations

from numba import njit
import numpy as np

from gridiron_edge.sim._types import (
    N_TEAMS,
    N_WEEKS_REG,
    ScheduleArrays,
)

# ============================================================================
# ELO MODEL (NUMBA OPTIMIZED)
# ============================================================================


@njit(cache=True)
def _elo_win_prob(elo_a: float, elo_b: float, divisor: float) -> float:
    """Win probability for team A vs team B.

    Args:
        elo_a: Elo rating for team A (home team in simulation).
        elo_b: Elo rating for team B (away team in simulation).
        divisor: Win-probability divisor. Match the value used to build the
            Elo state table (default 480; tuned variants may differ).
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / divisor))


@njit(cache=True)
def _elo_update(
    elo_a: float,
    elo_b: float,
    score_a: float,
    k: float,
    divisor: float,
) -> tuple[float, float]:
    """Update Elo ratings after a game."""
    p_a = _elo_win_prob(elo_a, elo_b, divisor)
    delta = k * (score_a - p_a)
    return elo_a + delta, elo_b - delta


# ============================================================================
# RECORD ACCUMULATION (NUMBA OPTIMIZED)
# ============================================================================


@njit(cache=True)
def apply_actuals_to_matrices(
    schedule_home: np.ndarray,
    schedule_away: np.ndarray,
    week_offsets: np.ndarray,
    result: np.ndarray,
    final_actual_week: int,
    conf_id: np.ndarray,
    div_id: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Accumulate actual game results into record matrices.

    Returns:
        (pts_total, pts_conf, pts_div, gp_played,
         gp_vs, pts_vs, wins_vs, reg_win_counts)
    """
    pts_total = np.zeros(N_TEAMS, dtype=np.int16)
    pts_conf = np.zeros(N_TEAMS, dtype=np.int16)
    pts_div = np.zeros(N_TEAMS, dtype=np.int16)
    gp_played = np.zeros(N_TEAMS, dtype=np.int16)

    gp_vs = np.zeros((N_TEAMS, N_TEAMS), dtype=np.uint8)
    pts_vs = np.zeros((N_TEAMS, N_TEAMS), dtype=np.int8)
    wins_vs = np.zeros((N_TEAMS, N_TEAMS), dtype=np.uint8)

    reg_win_counts = np.zeros((N_TEAMS, N_WEEKS_REG + 1), dtype=np.int32)

    for w in range(1, final_actual_week + 1):
        start = week_offsets[w]
        end = week_offsets[w + 1]

        for gi in range(start, end):
            code = result[gi]
            if code == np.int8(-1):  # UNPLAYED
                continue

            h = int(schedule_home[gi])
            a = int(schedule_away[gi])

            gp_played[h] += 1
            gp_played[a] += 1

            gp_vs[h, a] = np.uint8(gp_vs[h, a] + 1)
            gp_vs[a, h] = np.uint8(gp_vs[a, h] + 1)

            same_conf = conf_id[h] == conf_id[a]
            same_div = div_id[h] == div_id[a]

            if code == np.int8(1):  # HOME_WIN
                pts_total[h] += 2
                if same_conf:
                    pts_conf[h] += 2
                if same_div:
                    pts_div[h] += 2
                pts_vs[h, a] = np.int8(pts_vs[h, a] + 2)
                wins_vs[h, a] = np.uint8(wins_vs[h, a] + 1)
                reg_win_counts[h, w] += 1

            elif code == np.int8(0):  # AWAY_WIN
                pts_total[a] += 2
                if same_conf:
                    pts_conf[a] += 2
                if same_div:
                    pts_div[a] += 2
                pts_vs[a, h] = np.int8(pts_vs[a, h] + 2)
                wins_vs[a, h] = np.uint8(wins_vs[a, h] + 1)
                reg_win_counts[a, w] += 1

            else:  # TIE
                pts_total[h] += 1
                pts_total[a] += 1
                if same_conf:
                    pts_conf[h] += 1
                    pts_conf[a] += 1
                if same_div:
                    pts_div[h] += 1
                    pts_div[a] += 1
                pts_vs[h, a] = np.int8(pts_vs[h, a] + 1)
                pts_vs[a, h] = np.int8(pts_vs[a, h] + 1)

    return (
        pts_total,
        pts_conf,
        pts_div,
        gp_played,
        gp_vs,
        pts_vs,
        wins_vs,
        reg_win_counts,
    )


# ============================================================================
# REGULAR SEASON SIMULATION (NUMBA OPTIMIZED)
# ============================================================================


@njit(cache=True)
def simulate_remaining_regular_season(
    n_sims: int,
    schedule_home: np.ndarray,
    schedule_away: np.ndarray,
    week_offsets: np.ndarray,
    final_actual_week: int,
    conf_id: np.ndarray,
    div_id: np.ndarray,
    elo_entering_next_week: np.ndarray,
    pts_total_actual: np.ndarray,
    pts_conf_actual: np.ndarray,
    pts_div_actual: np.ndarray,
    gp_vs_actual: np.ndarray,
    pts_vs_actual: np.ndarray,
    wins_vs_actual: np.ndarray,
    reg_win_counts_actual: np.ndarray,
    k_factor: float,
    p_tie: float,
    base_seed: int,
    divisor: float = 480.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Simulate remaining regular season games across n_sims Monte Carlo runs.

    Returns:
        (pts_total_by_sim, pts_conf_by_sim, pts_div_by_sim,
         gp_vs_by_sim, pts_vs_by_sim, wins_vs_by_sim,
         end_elo_by_sim, reg_win_counts)
    """
    pts_total_by_sim = np.zeros((n_sims, N_TEAMS), dtype=np.int16)
    pts_conf_by_sim = np.zeros((n_sims, N_TEAMS), dtype=np.int16)
    pts_div_by_sim = np.zeros((n_sims, N_TEAMS), dtype=np.int16)

    gp_vs_by_sim = np.zeros((n_sims, N_TEAMS, N_TEAMS), dtype=np.uint8)
    pts_vs_by_sim = np.zeros((n_sims, N_TEAMS, N_TEAMS), dtype=np.int8)
    wins_vs_by_sim = np.zeros((n_sims, N_TEAMS, N_TEAMS), dtype=np.uint8)

    end_elo_by_sim = np.zeros((n_sims, N_TEAMS), dtype=np.float32)
    reg_win_counts = reg_win_counts_actual.copy()

    for s in range(n_sims):
        np.random.seed(base_seed + s)

        elo = elo_entering_next_week.copy()
        pts_total = pts_total_actual.copy()
        pts_conf = pts_conf_actual.copy()
        pts_div = pts_div_actual.copy()
        gp_vs = gp_vs_actual.copy()
        pts_vs = pts_vs_actual.copy()
        wins_vs = wins_vs_actual.copy()

        for w in range(final_actual_week + 1, N_WEEKS_REG + 1):
            start = week_offsets[w]
            end = week_offsets[w + 1]

            for gi in range(start, end):
                h = int(schedule_home[gi])
                a = int(schedule_away[gi])

                gp_vs[h, a] = np.uint8(gp_vs[h, a] + 1)
                gp_vs[a, h] = np.uint8(gp_vs[a, h] + 1)

                same_conf = conf_id[h] == conf_id[a]
                same_div = div_id[h] == div_id[a]

                eh = float(elo[h])
                ea = float(elo[a])
                p_home = _elo_win_prob(eh, ea, divisor)

                u = np.random.random()
                if u < p_tie:
                    pts_total[h] += 1
                    pts_total[a] += 1
                    if same_conf:
                        pts_conf[h] += 1
                        pts_conf[a] += 1
                    if same_div:
                        pts_div[h] += 1
                        pts_div[a] += 1
                    pts_vs[h, a] = np.int8(pts_vs[h, a] + 1)
                    pts_vs[a, h] = np.int8(pts_vs[a, h] + 1)
                    new_h, new_a = _elo_update(eh, ea, 0.5, k_factor, divisor)
                else:
                    u2 = (u - p_tie) / (1.0 - p_tie)
                    if u2 < p_home:
                        pts_total[h] += 2
                        if same_conf:
                            pts_conf[h] += 2
                        if same_div:
                            pts_div[h] += 2
                        pts_vs[h, a] = np.int8(pts_vs[h, a] + 2)
                        wins_vs[h, a] = np.uint8(wins_vs[h, a] + 1)
                        reg_win_counts[h, w] += 1
                        new_h, new_a = _elo_update(eh, ea, 1.0, k_factor, divisor)
                    else:
                        pts_total[a] += 2
                        if same_conf:
                            pts_conf[a] += 2
                        if same_div:
                            pts_div[a] += 2
                        pts_vs[a, h] = np.int8(pts_vs[a, h] + 2)
                        wins_vs[a, h] = np.uint8(wins_vs[a, h] + 1)
                        reg_win_counts[a, w] += 1
                        new_h, new_a = _elo_update(eh, ea, 0.0, k_factor, divisor)

                elo[h] = np.float32(new_h)
                elo[a] = np.float32(new_a)

        pts_total_by_sim[s] = pts_total
        pts_conf_by_sim[s] = pts_conf
        pts_div_by_sim[s] = pts_div
        gp_vs_by_sim[s] = gp_vs
        pts_vs_by_sim[s] = pts_vs
        wins_vs_by_sim[s] = wins_vs
        end_elo_by_sim[s] = elo

    return (
        pts_total_by_sim,
        pts_conf_by_sim,
        pts_div_by_sim,
        gp_vs_by_sim,
        pts_vs_by_sim,
        wins_vs_by_sim,
        end_elo_by_sim,
        reg_win_counts,
    )


# ============================================================================
# SCHEDULE ANALYSIS
# ============================================================================


def precompute_game_counts(
    schedule: ScheduleArrays,
    conf_id: np.ndarray,
    div_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute total, conference, and division game counts for each team.

    Returns:
        (gp_total, gp_conf, gp_div, opp_mask)
    """
    gp_total = np.zeros(N_TEAMS, dtype=np.int16)
    gp_conf = np.zeros(N_TEAMS, dtype=np.int16)
    gp_div = np.zeros(N_TEAMS, dtype=np.int16)
    opp_mask = np.zeros(N_TEAMS, dtype=np.uint32)

    for i in range(schedule.home.shape[0]):
        h = int(schedule.home[i])
        a = int(schedule.away[i])

        gp_total[h] += 1
        gp_total[a] += 1

        opp_mask[h] |= np.uint32(1) << np.uint32(a)
        opp_mask[a] |= np.uint32(1) << np.uint32(h)

        if conf_id[h] == conf_id[a]:
            gp_conf[h] += 1
            gp_conf[a] += 1

        if div_id[h] == div_id[a]:
            gp_div[h] += 1
            gp_div[a] += 1

    return gp_total, gp_conf, gp_div, opp_mask

# src/gridiron_edge/sim/playoffs.py
"""Playoff simulation: tiebreakers, conference seeding, and bracket simulation.

All numba-compiled functions use @njit(cache=True) for performance.
Called by season.py's run_full_simulation().
"""

from __future__ import annotations

from typing import Final

from numba import njit
import numpy as np

# ---------------------------------------------------------------------------
# Constants (duplicated from season.py to keep this module self-contained
# for numba - numba cannot import from sibling modules at JIT time)
# season.py asserts these stay in sync at import time.
# ---------------------------------------------------------------------------

N_TEAMS: Final[int] = 32
N_PLAYOFF_ROUNDS: Final[int] = 4

ROUND_WC: Final[int] = 0
ROUND_DIV: Final[int] = 1
ROUND_CONF: Final[int] = 2
ROUND_SB: Final[int] = 3


# ============================================================================
# TIEBREAKER UTILITIES (NUMBA OPTIMIZED)
# ============================================================================


@njit(cache=True)
def _cmp_ratio(num_a: int, den_a: int, num_b: int, den_b: int) -> int:
    """Compare two ratios without division (avoids floating point issues)."""
    left: int = num_a * den_b
    right: int = num_b * den_a
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


@njit(cache=True)
def _overall_cmp(a: int, b: int, pts_total: np.ndarray, gp_total: np.ndarray) -> int:
    """Compare overall win percentage between two teams."""
    return _cmp_ratio(
        int(pts_total[a]), int(2 * gp_total[a]), int(pts_total[b]), int(2 * gp_total[b])
    )


@njit(cache=True)
def _head_to_head_two_clubs(pts_vs: np.ndarray, gp_vs: np.ndarray, a: int, b: int) -> int:
    """Compare head-to-head record between two teams."""
    g = int(gp_vs[a, b])
    if g == 0:
        return 0
    return _cmp_ratio(int(pts_vs[a, b]), 2 * g, int(pts_vs[b, a]), 2 * g)


@njit(cache=True)
def _common_games_two_clubs(
    pts_vs: np.ndarray,
    gp_vs: np.ndarray,
    opp_mask: np.ndarray,
    a: int,
    b: int,
    min_games: int,
) -> int:
    """Compare record in common games between two teams."""
    common = opp_mask[a] & opp_mask[b]
    common &= ~(np.uint32(1) << np.uint32(a))
    common &= ~(np.uint32(1) << np.uint32(b))

    num_a = den_a = num_b = den_b = 0

    for opp in range(N_TEAMS):
        if (common >> np.uint32(opp)) & np.uint32(1):
            g_a = int(gp_vs[a, opp])
            g_b = int(gp_vs[b, opp])
            if g_a > 0:
                num_a += int(pts_vs[a, opp])
                den_a += 2 * g_a
            if g_b > 0:
                num_b += int(pts_vs[b, opp])
                den_b += 2 * g_b

    if den_a < 2 * min_games or den_b < 2 * min_games:
        return 0
    return _cmp_ratio(num_a, den_a, num_b, den_b)


@njit(cache=True)
def _strength_of_victory_two_clubs(
    wins_vs: np.ndarray,
    pts_total: np.ndarray,
    gp_total: np.ndarray,
    a: int,
    b: int,
) -> int:
    """Compare strength of victory between two teams."""
    sov_a = sov_b = 0.0

    for opp in range(N_TEAMS):
        w_a = int(wins_vs[a, opp])
        if w_a > 0:
            den = float(2 * gp_total[opp])
            sov_a += float(w_a) * (float(pts_total[opp]) / den)

        w_b = int(wins_vs[b, opp])
        if w_b > 0:
            den = float(2 * gp_total[opp])
            sov_b += float(w_b) * (float(pts_total[opp]) / den)

    if sov_a > sov_b:
        return 1
    if sov_a < sov_b:
        return -1
    return 0


@njit(cache=True)
def _strength_of_schedule_two_clubs(
    gp_vs: np.ndarray,
    pts_total: np.ndarray,
    gp_total: np.ndarray,
    a: int,
    b: int,
) -> int:
    """Compare strength of schedule between two teams."""
    sos_a = sos_b = 0.0

    for opp in range(N_TEAMS):
        g_a = int(gp_vs[a, opp])
        if g_a > 0:
            den = float(2 * gp_total[opp])
            sos_a += float(g_a) * (float(pts_total[opp]) / den)

        g_b = int(gp_vs[b, opp])
        if g_b > 0:
            den = float(2 * gp_total[opp])
            sos_b += float(g_b) * (float(pts_total[opp]) / den)

    if sos_a > sos_b:
        return 1
    if sos_a < sos_b:
        return -1
    return 0


@njit(cache=True)
def compare_division_two_clubs(
    a: int,
    b: int,
    pts_total: np.ndarray,
    pts_conf: np.ndarray,
    pts_div: np.ndarray,
    gp_total: np.ndarray,
    gp_conf: np.ndarray,
    gp_div: np.ndarray,
    pts_vs: np.ndarray,
    gp_vs: np.ndarray,
    wins_vs: np.ndarray,
    opp_mask: np.ndarray,
) -> int:
    """Apply NFL division tiebreaker rules for two teams."""
    c: int = _overall_cmp(a, b, pts_total, gp_total)
    if c != 0:
        return a if c == 1 else b

    c = _head_to_head_two_clubs(pts_vs, gp_vs, a, b)
    if c != 0:
        return a if c == 1 else b

    c = _cmp_ratio(int(pts_div[a]), int(2 * gp_div[a]), int(pts_div[b]), int(2 * gp_div[b]))
    if c != 0:
        return a if c == 1 else b

    c = _common_games_two_clubs(pts_vs, gp_vs, opp_mask, a, b, min_games=0)
    if c != 0:
        return a if c == 1 else b

    c = _cmp_ratio(int(pts_conf[a]), int(2 * gp_conf[a]), int(pts_conf[b]), int(2 * gp_conf[b]))
    if c != 0:
        return a if c == 1 else b

    c = _strength_of_victory_two_clubs(wins_vs, pts_total, gp_total, a, b)
    if c != 0:
        return a if c == 1 else b

    c = _strength_of_schedule_two_clubs(gp_vs, pts_total, gp_total, a, b)
    if c != 0:
        return a if c == 1 else b

    return a if np.random.random() < 0.5 else b


@njit(cache=True)
def compare_wildcard_two_clubs(
    a: int,
    b: int,
    pts_total: np.ndarray,
    pts_conf: np.ndarray,
    gp_total: np.ndarray,
    gp_conf: np.ndarray,
    pts_vs: np.ndarray,
    gp_vs: np.ndarray,
    wins_vs: np.ndarray,
    opp_mask: np.ndarray,
) -> int:
    """Apply NFL wildcard tiebreaker rules for two teams."""
    c: int = _overall_cmp(a, b, pts_total, gp_total)
    if c != 0:
        return a if c == 1 else b

    c = _head_to_head_two_clubs(pts_vs, gp_vs, a, b)
    if c != 0:
        return a if c == 1 else b

    c = _cmp_ratio(int(pts_conf[a]), int(2 * gp_conf[a]), int(pts_conf[b]), int(2 * gp_conf[b]))
    if c != 0:
        return a if c == 1 else b

    c = _common_games_two_clubs(pts_vs, gp_vs, opp_mask, a, b, min_games=4)
    if c != 0:
        return a if c == 1 else b

    c = _strength_of_victory_two_clubs(wins_vs, pts_total, gp_total, a, b)
    if c != 0:
        return a if c == 1 else b

    c = _strength_of_schedule_two_clubs(gp_vs, pts_total, gp_total, a, b)
    if c != 0:
        return a if c == 1 else b

    return a if np.random.random() < 0.5 else b


@njit(cache=True)
def compare_seeding_two_clubs(
    a: int,
    b: int,
    div_id: np.ndarray,
    pts_total: np.ndarray,
    pts_conf: np.ndarray,
    pts_div: np.ndarray,
    gp_total: np.ndarray,
    gp_conf: np.ndarray,
    gp_div: np.ndarray,
    pts_vs: np.ndarray,
    gp_vs: np.ndarray,
    wins_vs: np.ndarray,
    opp_mask: np.ndarray,
) -> int:
    """Apply appropriate tiebreaker based on division membership."""
    if div_id[a] == div_id[b]:
        return compare_division_two_clubs(
            a,
            b,
            pts_total,
            pts_conf,
            pts_div,
            gp_total,
            gp_conf,
            gp_div,
            pts_vs,
            gp_vs,
            wins_vs,
            opp_mask,
        )
    return compare_wildcard_two_clubs(
        a,
        b,
        pts_total,
        pts_conf,
        gp_total,
        gp_conf,
        pts_vs,
        gp_vs,
        wins_vs,
        opp_mask,
    )


# ============================================================================
# MULTI-TEAM TIEBREAKERS (NUMBA OPTIMIZED)
# ============================================================================


@njit(cache=True)
def _best_ratio_mask(nums: np.ndarray, dens: np.ndarray) -> np.ndarray:
    """Find teams with best ratio, return binary mask."""
    n = nums.shape[0]
    best_mask = np.zeros(n, dtype=np.uint8)

    best_i: int = -1
    for i in range(n):
        if dens[i] > 0:
            best_i = i
            break

    if best_i == -1:
        best_mask[:] = 1
        return best_mask

    best_num = nums[best_i]
    best_den = dens[best_i]
    best_mask[best_i] = 1

    for i in range(n):
        if i == best_i or dens[i] <= 0:
            continue
        left = nums[i] * best_den
        right = best_num * dens[i]
        if left > right:
            best_mask[:] = 0
            best_mask[i] = 1
            best_num = nums[i]
            best_den = dens[i]
        elif left == right:
            best_mask[i] = 1

    return best_mask


@njit(cache=True)
def _filter_by_mask_int16(items: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Filter array by binary mask."""
    k = 0
    for i in range(items.shape[0]):
        if mask[i] == 1:
            k += 1

    out = np.empty(k, dtype=np.int16)
    j = 0
    for i in range(items.shape[0]):
        if mask[i] == 1:
            out[j] = items[i]
            j += 1

    return out


@njit(cache=True)
def resolve_division_three_plus(
    clubs: np.ndarray,
    pts_total: np.ndarray,
    pts_conf: np.ndarray,
    pts_div: np.ndarray,
    gp_total: np.ndarray,
    gp_conf: np.ndarray,
    gp_div: np.ndarray,
    pts_vs: np.ndarray,
    gp_vs: np.ndarray,
    wins_vs: np.ndarray,
    opp_mask: np.ndarray,
) -> int:
    """Resolve 3+ team division tiebreaker using mini-league head-to-head."""
    current = clubs.copy()
    step = 1

    while True:
        n = current.shape[0]
        if n == 1:
            return int(current[0])
        if n == 2:
            return compare_division_two_clubs(
                int(current[0]),
                int(current[1]),
                pts_total,
                pts_conf,
                pts_div,
                gp_total,
                gp_conf,
                gp_div,
                pts_vs,
                gp_vs,
                wins_vs,
                opp_mask,
            )

        if step == 1:
            nums = np.zeros(n, dtype=np.int32)
            dens = np.zeros(n, dtype=np.int32)
            for i in range(n):
                t = int(current[i])
                num = den = 0
                for j in range(n):
                    if i == j:
                        continue
                    o = int(current[j])
                    g = int(gp_vs[t, o])
                    if g > 0:
                        num += int(pts_vs[t, o])
                        den += 2 * g
                nums[i] = num
                dens[i] = den

            mask = _best_ratio_mask(nums, dens)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 1
                continue

        elif step == 2:
            nums = np.zeros(n, dtype=np.int32)
            dens = np.zeros(n, dtype=np.int32)
            for i in range(n):
                t = int(current[i])
                nums[i] = int(pts_div[t])
                dens[i] = int(2 * gp_div[t])

            mask = _best_ratio_mask(nums, dens)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 1
                continue

        elif step == 3:
            common = np.uint32(0xFFFFFFFF)
            for i in range(n):
                common &= opp_mask[int(current[i])]
            for i in range(n):
                common &= ~(np.uint32(1) << np.uint32(int(current[i])))

            nums = np.zeros(n, dtype=np.int32)
            dens = np.zeros(n, dtype=np.int32)
            for i in range(n):
                t = int(current[i])
                num = den = 0
                for opp in range(N_TEAMS):
                    if (common >> np.uint32(opp)) & np.uint32(1):
                        g = int(gp_vs[t, opp])
                        if g > 0:
                            num += int(pts_vs[t, opp])
                            den += 2 * g
                nums[i] = num
                dens[i] = den

            mask = _best_ratio_mask(nums, dens)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 1
                continue

        elif step == 4:
            nums = np.zeros(n, dtype=np.int32)
            dens = np.zeros(n, dtype=np.int32)
            for i in range(n):
                t = int(current[i])
                nums[i] = int(pts_conf[t])
                dens[i] = int(2 * gp_conf[t])

            mask = _best_ratio_mask(nums, dens)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 1
                continue

        elif step == 5:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                t = int(current[i])
                sov = 0.0
                for opp in range(N_TEAMS):
                    w = int(wins_vs[t, opp])
                    if w > 0:
                        sov += float(w) * (float(pts_total[opp]) / float(2 * gp_total[opp]))
                vals[i] = sov

            best = np.max(vals)
            mask = (vals == best).astype(np.uint8)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 1
                continue

        elif step == 6:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                t = int(current[i])
                sos = 0.0
                for opp in range(N_TEAMS):
                    g = int(gp_vs[t, opp])
                    if g > 0:
                        sos += float(g) * (float(pts_total[opp]) / float(2 * gp_total[opp]))
                vals[i] = sos

            best = np.max(vals)
            mask = (vals == best).astype(np.uint8)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 1
                continue

        step += 1
        if step > 6:
            idx = int(np.random.random() * current.shape[0])
            return int(current[idx])


@njit(cache=True)
def _wildcard_sweep_status(clubs: np.ndarray, gp_vs: np.ndarray, wins_vs: np.ndarray) -> int:
    """Check for head-to-head sweep in wildcard tiebreaker.

    Returns:
        team_id if a team swept all others
        -2 if a team lost all games
        -1 otherwise
    """
    n = clubs.shape[0]

    for i in range(n):
        t = int(clubs[i])
        swept = True
        for j in range(n):
            if i == j:
                continue
            o = int(clubs[j])
            g = int(gp_vs[t, o])
            if g <= 0 or int(wins_vs[t, o]) != g:
                swept = False
                break
        if swept:
            return t

    for i in range(n):
        t = int(clubs[i])
        winless = True
        for j in range(n):
            if i == j:
                continue
            o = int(clubs[j])
            g = int(gp_vs[t, o])
            if g <= 0 or int(wins_vs[t, o]) != 0:
                winless = False
                break
        if winless:
            return -2

    return -1


@njit(cache=True)
def resolve_wildcard_three_plus(
    clubs: np.ndarray,
    div_id: np.ndarray,
    pts_total: np.ndarray,
    pts_conf: np.ndarray,
    pts_div: np.ndarray,
    gp_total: np.ndarray,
    gp_conf: np.ndarray,
    gp_div: np.ndarray,
    pts_vs: np.ndarray,
    gp_vs: np.ndarray,
    wins_vs: np.ndarray,
    opp_mask: np.ndarray,
) -> int:
    """Resolve 3+ team wildcard tiebreaker."""
    divs = np.empty(clubs.shape[0], dtype=np.int8)
    nd = 0
    for i in range(clubs.shape[0]):
        d = div_id[int(clubs[i])]
        seen = False
        for j in range(nd):
            if divs[j] == d:
                seen = True
                break
        if not seen:
            divs[nd] = d
            nd += 1

    reduced = np.empty(nd, dtype=np.int16)
    for j in range(nd):
        d = divs[j]
        tmp = np.empty(clubs.shape[0], dtype=np.int16)
        k = 0
        for i in range(clubs.shape[0]):
            t = int(clubs[i])
            if div_id[t] == d:
                tmp[k] = np.int16(t)
                k += 1

        if k == 1:
            reduced[j] = tmp[0]
        elif k == 2:
            reduced[j] = np.int16(
                compare_division_two_clubs(
                    int(tmp[0]),
                    int(tmp[1]),
                    pts_total,
                    pts_conf,
                    pts_div,
                    gp_total,
                    gp_conf,
                    gp_div,
                    pts_vs,
                    gp_vs,
                    wins_vs,
                    opp_mask,
                )
            )
        else:
            reduced[j] = np.int16(
                resolve_division_three_plus(
                    tmp[:k],
                    pts_total,
                    pts_conf,
                    pts_div,
                    gp_total,
                    gp_conf,
                    gp_div,
                    pts_vs,
                    gp_vs,
                    wins_vs,
                    opp_mask,
                )
            )

    current = reduced
    if current.shape[0] == 1:
        return int(current[0])
    if current.shape[0] == 2:
        return compare_wildcard_two_clubs(
            int(current[0]),
            int(current[1]),
            pts_total,
            pts_conf,
            gp_total,
            gp_conf,
            pts_vs,
            gp_vs,
            wins_vs,
            opp_mask,
        )

    step = 2
    while True:
        n: int = current.shape[0]
        if n == 1:
            return int(current[0])
        if n == 2:
            return compare_wildcard_two_clubs(
                int(current[0]),
                int(current[1]),
                pts_total,
                pts_conf,
                gp_total,
                gp_conf,
                pts_vs,
                gp_vs,
                wins_vs,
                opp_mask,
            )

        if step == 2:
            sweep: int = _wildcard_sweep_status(current, gp_vs, wins_vs)
            if sweep >= 0:
                return sweep
            if sweep == -2:
                loser: int = -1
                for i in range(n):
                    t = int(current[i])
                    winless = True
                    for j in range(n):
                        if i == j:
                            continue
                        o = int(current[j])
                        g = int(gp_vs[t, o])
                        if g <= 0 or int(wins_vs[t, o]) != 0:
                            winless = False
                            break
                    if winless:
                        loser = t
                        break

                tmp2 = np.empty(n - 1, dtype=np.int16)
                k = 0
                for i in range(n):
                    t = int(current[i])
                    if t != loser:
                        tmp2[k] = np.int16(t)
                        k += 1
                current = tmp2
                step = 2
                continue

        elif step == 3:
            nums = np.zeros(n, dtype=np.int32)
            dens = np.zeros(n, dtype=np.int32)
            for i in range(n):
                t = int(current[i])
                nums[i] = int(pts_conf[t])
                dens[i] = int(2 * gp_conf[t])

            mask = _best_ratio_mask(nums, dens)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 2
                continue

        elif step == 4:
            common = np.uint32(0xFFFFFFFF)
            for i in range(n):
                common &= opp_mask[int(current[i])]
            for i in range(n):
                common &= ~(np.uint32(1) << np.uint32(int(current[i])))

            nums = np.zeros(n, dtype=np.int32)
            dens = np.zeros(n, dtype=np.int32)
            applicable = True
            for i in range(n):
                t = int(current[i])
                num = den = 0
                for opp in range(N_TEAMS):
                    if (common >> np.uint32(opp)) & np.uint32(1):
                        g = int(gp_vs[t, opp])
                        if g > 0:
                            num += int(pts_vs[t, opp])
                            den += 2 * g
                nums[i] = num
                dens[i] = den
                if den < 8:
                    applicable = False

            if applicable:
                mask = _best_ratio_mask(nums, dens)
                new_current = _filter_by_mask_int16(current, mask)
                if new_current.shape[0] < current.shape[0]:
                    current = new_current
                    step = 2
                    continue

        elif step == 5:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                t = int(current[i])
                sov = 0.0
                for opp in range(N_TEAMS):
                    w = int(wins_vs[t, opp])
                    if w > 0:
                        sov += float(w) * (float(pts_total[opp]) / float(2 * gp_total[opp]))
                vals[i] = sov

            best: float = np.max(vals)
            mask = (vals == best).astype(np.uint8)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 2
                continue

        elif step == 6:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                t = int(current[i])
                sos = 0.0
                for opp in range(N_TEAMS):
                    g = int(gp_vs[t, opp])
                    if g > 0:
                        sos += float(g) * (float(pts_total[opp]) / float(2 * gp_total[opp]))
                vals[i] = sos

            best = np.max(vals)
            mask = (vals == best).astype(np.uint8)
            new_current = _filter_by_mask_int16(current, mask)
            if new_current.shape[0] < current.shape[0]:
                current = new_current
                step = 2
                continue

        step += 1
        if step > 6:
            idx = int(np.random.random() * current.shape[0])
            return int(current[idx])


# ============================================================================
# CONFERENCE SEEDING (NUMBA OPTIMIZED)
# ============================================================================


@njit(cache=True)
def pick_division_winner(
    teams4: np.ndarray,
    pts_total: np.ndarray,
    pts_conf: np.ndarray,
    pts_div: np.ndarray,
    gp_total: np.ndarray,
    gp_conf: np.ndarray,
    gp_div: np.ndarray,
    pts_vs: np.ndarray,
    gp_vs: np.ndarray,
    wins_vs: np.ndarray,
    opp_mask: np.ndarray,
) -> int:
    """Determine division winner from 4 teams."""
    best = int(teams4[0])
    for i in range(1, 4):
        t = int(teams4[i])
        if _overall_cmp(t, best, pts_total, gp_total) == 1:
            best: int = t

    tmp = np.empty(4, dtype=np.int16)
    k = 0
    for i in range(4):
        t = int(teams4[i])
        if _overall_cmp(t, best, pts_total, gp_total) == 0:
            tmp[k] = np.int16(t)
            k += 1

    if k == 1:
        return int(tmp[0])
    if k == 2:
        return compare_division_two_clubs(
            int(tmp[0]),
            int(tmp[1]),
            pts_total,
            pts_conf,
            pts_div,
            gp_total,
            gp_conf,
            gp_div,
            pts_vs,
            gp_vs,
            wins_vs,
            opp_mask,
        )

    return resolve_division_three_plus(
        tmp[:k],
        pts_total,
        pts_conf,
        pts_div,
        gp_total,
        gp_conf,
        gp_div,
        pts_vs,
        gp_vs,
        wins_vs,
        opp_mask,
    )


@njit(cache=True)
def seed_conference(
    conf_team_ids: np.ndarray,
    div_id: np.ndarray,
    pts_total: np.ndarray,
    pts_conf: np.ndarray,
    pts_div: np.ndarray,
    gp_total: np.ndarray,
    gp_conf: np.ndarray,
    gp_div: np.ndarray,
    pts_vs: np.ndarray,
    gp_vs: np.ndarray,
    wins_vs: np.ndarray,
    opp_mask: np.ndarray,
    conf_div_base: int,
) -> np.ndarray:
    """Determine playoff seeding for a conference (seeds 1-7).

    Args:
        conf_div_base: 0 for AFC divisions, 4 for NFC divisions.

    Returns:
        Array of 7 team IDs representing seeds 1-7.
    """
    div_winners = np.empty(4, dtype=np.int16)
    for d in range(4):
        target_div = conf_div_base + d
        teams4 = np.empty(4, dtype=np.int16)
        k = 0
        for i in range(conf_team_ids.shape[0]):
            t = int(conf_team_ids[i])
            if div_id[t] == target_div:
                teams4[k] = np.int16(t)
                k += 1

        div_winners[d] = np.int16(
            pick_division_winner(
                teams4,
                pts_total,
                pts_conf,
                pts_div,
                gp_total,
                gp_conf,
                gp_div,
                pts_vs,
                gp_vs,
                wins_vs,
                opp_mask,
            )
        )

    seeds14 = div_winners.copy()
    for i in range(4):
        for j in range(i + 1, 4):
            a = int(seeds14[i])
            b = int(seeds14[j])
            winner: int = compare_seeding_two_clubs(
                a,
                b,
                div_id,
                pts_total,
                pts_conf,
                pts_div,
                gp_total,
                gp_conf,
                gp_div,
                pts_vs,
                gp_vs,
                wins_vs,
                opp_mask,
            )
            if winner == b:
                seeds14[i], seeds14[j] = seeds14[j], seeds14[i]

    remain = np.empty(conf_team_ids.shape[0] - 4, dtype=np.int16)
    r = 0
    for i in range(conf_team_ids.shape[0]):
        t = int(conf_team_ids[i])
        is_winner = False
        for j in range(4):
            if t == int(div_winners[j]):
                is_winner = True
                break
        if not is_winner:
            remain[r] = np.int16(t)
            r += 1

    wc = np.empty(3, dtype=np.int16)
    pool = remain
    pool_n = pool.shape[0]

    for pick in range(3):
        best = int(pool[0])
        for i in range(1, pool_n):
            t = int(pool[i])
            if _overall_cmp(t, best, pts_total, gp_total) == 1:
                best = t

        tmp = np.empty(pool_n, dtype=np.int16)
        k = 0
        for i in range(pool_n):
            t = int(pool[i])
            if _overall_cmp(t, best, pts_total, gp_total) == 0:
                tmp[k] = np.int16(t)
                k += 1

        if k == 1:
            chosen = int(tmp[0])
        elif k == 2:
            chosen = compare_seeding_two_clubs(
                int(tmp[0]),
                int(tmp[1]),
                div_id,
                pts_total,
                pts_conf,
                pts_div,
                gp_total,
                gp_conf,
                gp_div,
                pts_vs,
                gp_vs,
                wins_vs,
                opp_mask,
            )
        else:
            chosen: int = resolve_wildcard_three_plus(
                tmp[:k],
                div_id,
                pts_total,
                pts_conf,
                pts_div,
                gp_total,
                gp_conf,
                gp_div,
                pts_vs,
                gp_vs,
                wins_vs,
                opp_mask,
            )

        wc[pick] = np.int16(chosen)

        new_pool = np.empty(pool_n - 1, dtype=np.int16)
        j = 0
        for i in range(pool_n):
            if int(pool[i]) != chosen:
                new_pool[j] = pool[i]
                j += 1
        pool = new_pool
        pool_n = pool.shape[0]

    seeds = np.empty(7, dtype=np.int16)
    for i in range(4):
        seeds[i] = seeds14[i]
    for i in range(3):
        seeds[4 + i] = wc[i]

    return seeds


# ============================================================================
# PLAYOFF SIMULATION (NUMBA OPTIMIZED)
# ============================================================================


@njit(cache=True)
def _simulate_one_game(
    team_a: int,
    team_b: int,
    elo: np.ndarray,
    divisor: float,
) -> int:
    """Simulate single playoff game using Elo ratings."""
    p_a = 1.0 / (1.0 + 10.0 ** ((float(elo[team_b]) - float(elo[team_a])) / divisor))
    return team_a if np.random.random() < p_a else team_b


@njit(cache=True)
def simulate_playoffs(
    pts_total_by_sim: np.ndarray,
    pts_conf_by_sim: np.ndarray,
    pts_div_by_sim: np.ndarray,
    gp_total: np.ndarray,
    gp_conf: np.ndarray,
    gp_div: np.ndarray,
    gp_vs_by_sim: np.ndarray,
    pts_vs_by_sim: np.ndarray,
    wins_vs_by_sim: np.ndarray,
    opp_mask: np.ndarray,
    end_elo_by_sim: np.ndarray,
    conf_id: np.ndarray,
    div_id: np.ndarray,
    base_seed: int,
    fixed_playoff_winners: np.ndarray,
    divisor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate playoffs for all regular season simulations.

    Args:
        fixed_playoff_winners: (N_PLAYOFF_ROUNDS, N_TEAMS, N_TEAMS) int16 array.
            fixed[rnd, lo, hi] = winner_team_idx for known outcomes (lo < hi), else -1.

    Returns:
        Tuple of (po_win_counts, make_playoffs_counts, bye_counts).
    """
    n_sims = end_elo_by_sim.shape[0]
    po = np.zeros((N_TEAMS, N_PLAYOFF_ROUNDS), dtype=np.int32)
    make_playoffs = np.zeros(N_TEAMS, dtype=np.int32)
    bye_counts = np.zeros(N_TEAMS, dtype=np.int32)

    fixed = fixed_playoff_winners

    afc = np.empty(16, dtype=np.int16)
    nfc = np.empty(16, dtype=np.int16)
    ai = ni = 0
    for t in range(N_TEAMS):
        if conf_id[t] == 0:
            afc[ai] = np.int16(t)
            ai += 1
        else:
            nfc[ni] = np.int16(t)
            ni += 1

    for s in range(n_sims):
        np.random.seed(base_seed + 10_000_000 + s)

        elo = end_elo_by_sim[s]
        pts_total = pts_total_by_sim[s]
        pts_conf = pts_conf_by_sim[s]
        pts_div = pts_div_by_sim[s]
        gp_vs = gp_vs_by_sim[s]
        pts_vs = pts_vs_by_sim[s]
        wins_vs = wins_vs_by_sim[s]

        seeds_afc = seed_conference(
            afc,
            div_id,
            pts_total,
            pts_conf,
            pts_div,
            gp_total,
            gp_conf,
            gp_div,
            pts_vs,
            gp_vs,
            wins_vs,
            opp_mask,
            0,
        )
        seeds_nfc = seed_conference(
            nfc,
            div_id,
            pts_total,
            pts_conf,
            pts_div,
            gp_total,
            gp_conf,
            gp_div,
            pts_vs,
            gp_vs,
            wins_vs,
            opp_mask,
            4,
        )

        bye_counts[int(seeds_afc[0])] += 1
        bye_counts[int(seeds_nfc[0])] += 1

        for i in range(7):
            make_playoffs[int(seeds_afc[i])] += 1
            make_playoffs[int(seeds_nfc[i])] += 1

        # B023 fix: pass elo explicitly rather than closing over the loop variable.
        # Inner helpers receive elo_arr as a parameter so each simulation uses
        # the correct per-iteration ratings regardless of rebinding.
        def _fixed_winner(rnd: int, a: int, b: int) -> int:
            lo: int = min(a, b)
            hi: int = max(a, b)
            return int(fixed[rnd, lo, hi])

        def _wc_round(
            seeds: np.ndarray,
            elo_arr: np.ndarray,
        ) -> tuple[int, int, int]:
            t1, t2 = int(seeds[1]), int(seeds[6])
            w1: int = _fixed_winner(ROUND_WC, t1, t2)
            if w1 < 0:
                w1 = _simulate_one_game(t1, t2, elo_arr, divisor)

            t1, t2 = int(seeds[2]), int(seeds[5])
            w2: int = _fixed_winner(ROUND_WC, t1, t2)
            if w2 < 0:
                w2 = _simulate_one_game(t1, t2, elo_arr, divisor)

            t1, t2 = int(seeds[3]), int(seeds[4])
            w3: int = _fixed_winner(ROUND_WC, t1, t2)
            if w3 < 0:
                w3 = _simulate_one_game(t1, t2, elo_arr, divisor)

            po[w1, ROUND_WC] += 1
            po[w2, ROUND_WC] += 1
            po[w3, ROUND_WC] += 1
            return w1, w2, w3

        afc_wc: tuple[int, int, int] = _wc_round(seeds_afc, elo)
        nfc_wc: tuple[int, int, int] = _wc_round(seeds_nfc, elo)

        def _div_round(
            seeds: np.ndarray,
            wc_winners: tuple[int, int, int],
            elo_arr: np.ndarray,
        ) -> tuple[int, int]:
            s1 = int(seeds[0])
            w = np.array([wc_winners[0], wc_winners[1], wc_winners[2]], dtype=np.int16)

            idx_low: int = -1
            for i in range(1, 7):
                t = int(seeds[i])
                if t == int(w[0]) or t == int(w[1]) or t == int(w[2]):
                    idx_low = i
            low_team = int(seeds[idx_low])

            r = np.empty(2, dtype=np.int16)
            ri = 0
            for j in range(3):
                if int(w[j]) != low_team:
                    r[ri] = w[j]
                    ri += 1

            t1, t2 = s1, low_team
            d1: int = _fixed_winner(ROUND_DIV, t1, t2)
            if d1 < 0:
                d1 = _simulate_one_game(t1, t2, elo_arr, divisor)

            t1, t2 = int(r[0]), int(r[1])
            d2: int = _fixed_winner(ROUND_DIV, t1, t2)
            if d2 < 0:
                d2 = _simulate_one_game(t1, t2, elo_arr, divisor)

            po[d1, ROUND_DIV] += 1
            po[d2, ROUND_DIV] += 1
            return d1, d2

        afc_div: tuple[int, int] = _div_round(seeds_afc, afc_wc, elo)
        nfc_div: tuple[int, int] = _div_round(seeds_nfc, nfc_wc, elo)

        t1, t2 = afc_div[0], afc_div[1]
        afc_champ: int = _fixed_winner(ROUND_CONF, t1, t2)
        if afc_champ < 0:
            afc_champ = _simulate_one_game(t1, t2, elo, divisor)

        t1, t2 = nfc_div[0], nfc_div[1]
        nfc_champ: int = _fixed_winner(ROUND_CONF, t1, t2)
        if nfc_champ < 0:
            nfc_champ = _simulate_one_game(t1, t2, elo, divisor)

        po[afc_champ, ROUND_CONF] += 1
        po[nfc_champ, ROUND_CONF] += 1

        t1, t2 = afc_champ, nfc_champ
        sb_winner: int = _fixed_winner(ROUND_SB, t1, t2)
        if sb_winner < 0:
            sb_winner = _simulate_one_game(t1, t2, elo, divisor)
        po[sb_winner, ROUND_SB] += 1

    return po, make_playoffs, bye_counts

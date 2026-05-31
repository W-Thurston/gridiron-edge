# src/gridiron_edge/sim/season.py

"""NFL Season Monte Carlo Simulation — data loading, output formatting, and orchestration.

Simulation flow:
    1. Load historical results, schedules, and Elo ratings
    2. Apply actual game results through the current week
    3. Simulate remaining regular season games using Elo-based probabilities
    4. Determine playoff seeding using official NFL tiebreakers
    5. Simulate playoff brackets
    6. Aggregate results across thousands of simulations

Usage:
    poetry run gridiron sim run --season-year 2025-2026 --week 12

Public API is re-exported via sim/__init__.py. Import constants and data
containers from sim._types; import kernels from sim._engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gridiron_edge.sim import playoffs as _playoffs_mod
from gridiron_edge.sim._engine import (
    apply_actuals_to_matrices,
    precompute_game_counts,
    simulate_remaining_regular_season,
)
from gridiron_edge.sim._types import (
    AWAY_WIN,
    CONF_CODES,
    DIV_CODES,
    HOME_WIN,
    N_PLAYOFF_ROUNDS,
    N_TEAMS,
    N_WEEKS_REG,
    ROUND_CONF,
    ROUND_DIV,
    ROUND_SB,
    ROUND_WC,
    TIE,
    UNPLAYED,
    ScheduleArrays,
    SimPaths,
    SimulationConfig,
    SimulationResults,
    TeamIndex,
    _log_phase,
)
from gridiron_edge.sim.playoffs import simulate_playoffs

if TYPE_CHECKING:
    from logging import Logger

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sync assertions — verify playoffs.py constants match _types.py at import time.
# playoffs.py duplicates these because numba @njit cannot import from siblings.
# ---------------------------------------------------------------------------
assert _playoffs_mod.N_TEAMS == N_TEAMS, (
    f"sim/playoffs.py N_TEAMS ({_playoffs_mod.N_TEAMS}) out of sync with _types.py ({N_TEAMS})"
)
assert _playoffs_mod.N_PLAYOFF_ROUNDS == N_PLAYOFF_ROUNDS, (
    f"sim/playoffs.py N_PLAYOFF_ROUNDS ({_playoffs_mod.N_PLAYOFF_ROUNDS}) out of sync"
)
assert _playoffs_mod.ROUND_WC == ROUND_WC
assert _playoffs_mod.ROUND_DIV == ROUND_DIV
assert _playoffs_mod.ROUND_CONF == ROUND_CONF
assert _playoffs_mod.ROUND_SB == ROUND_SB


# ============================================================================
# DATA LOADING AND PARSING
# ============================================================================


def _parse_game_id(game_id: str) -> tuple[str, str]:
    """Parse game ID (YYYY_WW_AWAY_HOME) into (away_short, home_short)."""
    parts = game_id.split("_")
    if len(parts) != 4:
        msg = f"Invalid GAME_ID format: {game_id}"
        raise ValueError(msg)
    return parts[2], parts[3]


def load_long_to_short_mapping(mapping_csv: Path) -> dict[str, str]:
    """Load team name mapping from long names to short codes."""
    df = pd.read_csv(mapping_csv)
    required = {"NFL_LONG_NAME", "NFL_SHORT_NAME"}
    if not required.issubset(df.columns):
        raise ValueError(f"Mapping file missing columns: {required - set(df.columns)}")

    mapping = dict(zip(df["NFL_LONG_NAME"], df["NFL_SHORT_NAME"], strict=False))
    if len(mapping) < N_TEAMS:
        raise ValueError(f"Insufficient teams in mapping: {len(mapping)} (expected >= {N_TEAMS})")
    return mapping


def build_team_index_from_results(
    df_wk_by_wk: pd.DataFrame,
    long_to_short: dict[str, str],
    season_year: str,
    df_schedule: pd.DataFrame | None = None,
) -> TeamIndex:
    """Build TeamIndex from actual game results for the season.

    When no completed games exist for ``season_year`` (e.g. at the start of
    a season), falls back to ``df_schedule`` to derive the team list.

    Args:
        df_wk_by_wk: Historical games DataFrame.
        long_to_short: Long name → short code mapping.
        season_year: Season label (e.g. ``"2026-2027"``).
        df_schedule: Upcoming schedule DataFrame. Used as fallback when no
            completed games exist for ``season_year``.

    Returns:
        A ``TeamIndex`` covering all 32 NFL teams.
    """
    if not {"YEAR", "GAME_ID"}.issubset(df_wk_by_wk.columns):
        raise ValueError("wk_by_wk must include YEAR and GAME_ID columns")

    season_ids = df_wk_by_wk.loc[df_wk_by_wk["YEAR"] == season_year, "GAME_ID"].astype(str)

    shorts: set[str] = set()
    for gid in season_ids.tolist():
        a_s, h_s = _parse_game_id(gid)
        shorts.add(a_s)
        shorts.add(h_s)

    if len(shorts) < N_TEAMS and df_schedule is not None:
        for col in ("AWAY_TEAM", "HOME_TEAM"):
            if col in df_schedule.columns:
                for long_name in df_schedule[col].dropna().unique():
                    short = long_to_short.get(str(long_name))
                    if short:
                        shorts.add(short)

    short_names = sorted(shorts)
    if len(short_names) != N_TEAMS:
        raise ValueError(f"Expected {N_TEAMS} teams, got {len(short_names)}: {short_names}")

    short_to_id = {s: i for i, s in enumerate(short_names)}
    return TeamIndex(
        short_names=short_names,
        short_to_id=short_to_id,
        long_to_short=long_to_short,
    )


def add_game_id_to_schedule(
    df_schedule: pd.DataFrame,
    long_to_short: dict[str, str],
) -> pd.DataFrame:
    """Add standardized GAME_ID column to schedule DataFrame."""
    required = {"YEAR", "WEEK_NUM", "AWAY_TEAM", "HOME_TEAM"}
    if not required.issubset(df_schedule.columns):
        raise ValueError(f"Schedule missing columns: {required - set(df_schedule.columns)}")

    sched = df_schedule.copy()
    away_short = sched["AWAY_TEAM"].map(long_to_short)
    home_short = sched["HOME_TEAM"].map(long_to_short)

    missing = sorted(
        set(
            sched.loc[away_short.isna(), "AWAY_TEAM"].tolist()
            + sched.loc[home_short.isna(), "HOME_TEAM"].tolist()
        )
    )
    if missing:
        raise ValueError(f"Missing long->short mappings: {', '.join(missing)}")

    year4 = sched["YEAR"].astype(str).str.slice(0, 4)
    week2 = sched["WEEK_NUM"].astype(int).map(lambda w: f"{w:02d}").astype(str)
    sep = pd.Series(["_"] * len(sched), index=sched.index, dtype=str)
    sched["GAME_ID"] = year4 + sep + week2 + sep + away_short + sep + home_short
    return sched


def build_schedule_arrays(
    df_schedule_with_gid: pd.DataFrame,
    df_wk_by_wk: pd.DataFrame,
    team_index: TeamIndex,
    season_year: str,
) -> tuple[ScheduleArrays, int]:
    """Build numpy arrays representing the season schedule and results.

    Returns:
        (ScheduleArrays, final_actual_week)
    """
    sched = df_schedule_with_gid.loc[df_schedule_with_gid["YEAR"] == season_year].copy()
    sched = sched.loc[(sched["WEEK_NUM"] >= 1) & (sched["WEEK_NUM"] <= N_WEEKS_REG)]

    sort_cols = [c for c in ("WEEK_NUM", "GAME_DATE", "GAMETIME") if c in sched.columns]
    if sort_cols:
        sched = sched.sort_values(sort_cols, kind="mergesort")
    sched = sched.reset_index(drop=True)

    away_ids: list[int] = []
    home_ids: list[int] = []
    for gid in sched["GAME_ID"].astype(str).tolist():
        a_s, h_s = _parse_game_id(gid)
        away_ids.append(team_index.short_to_id[a_s])
        home_ids.append(team_index.short_to_id[h_s])

    away = np.asarray(away_ids, dtype=np.int16)
    home = np.asarray(home_ids, dtype=np.int16)
    week = sched["WEEK_NUM"].to_numpy(np.int16)

    wk_vals = df_wk_by_wk.loc[df_wk_by_wk["YEAR"] == season_year, "WEEK_NUM"].to_numpy()
    final_actual_week = int(wk_vals.max()) if wk_vals.size else 0

    needed = {"YEAR", "GAME_ID", "WINNER", "WIN_OR_TIE"}
    if not needed.issubset(df_wk_by_wk.columns):
        raise ValueError(f"wk_by_wk missing columns: {needed - set(df_wk_by_wk.columns)}")

    season_res = df_wk_by_wk.loc[
        df_wk_by_wk["YEAR"] == season_year, ["GAME_ID", "WINNER", "WIN_OR_TIE"]
    ].copy()

    gid_to_code: dict[str, np.int8] = {}
    for row in season_res.itertuples(index=False):
        gid = str(row.GAME_ID)
        win_or_tie = float(row.WIN_OR_TIE)  # type: ignore[arg-type]

        if win_or_tie == 0.5:
            gid_to_code[gid] = TIE
            continue

        winner_long = str(row.WINNER)
        winner_short = team_index.long_to_short.get(winner_long)
        if winner_short is None:
            raise ValueError(f"Missing long->short mapping for WINNER: {winner_long}")

        a_s, h_s = _parse_game_id(gid)
        if winner_short == h_s:
            gid_to_code[gid] = HOME_WIN
        elif winner_short == a_s:
            gid_to_code[gid] = AWAY_WIN
        else:
            gid_to_code[gid] = UNPLAYED

    result = np.full(len(sched), UNPLAYED, dtype=np.int8)
    for i, gid in enumerate(sched["GAME_ID"].astype(str).tolist()):
        code = gid_to_code.get(gid)
        if code is not None:
            result[i] = code

    week_offsets = np.zeros(N_WEEKS_REG + 2, dtype=np.int32)
    counts = np.bincount(week.astype(np.int32), minlength=N_WEEKS_REG + 1)
    running = 0
    for w in range(1, N_WEEKS_REG + 1):
        week_offsets[w] = running
        running += int(counts[w])
    week_offsets[N_WEEKS_REG + 1] = running

    return (
        ScheduleArrays(week=week, home=home, away=away, result=result, week_offsets=week_offsets),
        final_actual_week,
    )


def load_week_elo_vector(
    df_elo: pd.DataFrame,
    team_index: TeamIndex,
    season_year: str,
    target_week: int,
) -> tuple[np.ndarray, int]:
    """Load Elo ratings for all teams entering a specific week.

    Returns:
        (elo_vector shape (32,), week_used)
    """
    required: set[str] = {"NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"}
    if not required.issubset(df_elo.columns):
        raise ValueError(f"ELO file missing columns: {required - set(df_elo.columns)}")

    season = df_elo.loc[df_elo["NFL_YEAR"] == season_year, ["NFL_TEAM", "NFL_WEEK", "ELO"]].copy()
    if season.empty:
        raise ValueError(f"No Elo rows found for season {season_year}")

    season["NFL_WEEK"] = season["NFL_WEEK"].astype(int)
    season = season.loc[season["ELO"].notna()]

    weeks_le = season.loc[season["NFL_WEEK"] <= target_week, "NFL_WEEK"].unique()
    if weeks_le.size == 0:
        raise ValueError(f"No Elo data for {season_year} at/before week {target_week}")

    chosen_week: int | None = None
    for w in sorted(weeks_le.tolist(), reverse=True):
        if season.loc[season["NFL_WEEK"] == w, "NFL_TEAM"].nunique() == N_TEAMS:
            chosen_week = int(w)
            break

    if chosen_week is None:
        raise ValueError(
            f"No complete Elo week found at/before week {target_week} for {season_year}"
        )

    wk = season.loc[season["NFL_WEEK"] == chosen_week, ["NFL_TEAM", "ELO"]].copy()
    wk["SHORT"] = wk["NFL_TEAM"].map(team_index.long_to_short)

    if wk["SHORT"].isna().any():
        missing = wk.loc[wk["SHORT"].isna(), "NFL_TEAM"].unique().tolist()
        raise ValueError(f"Missing long->short mapping for Elo teams: {missing}")

    elo_vec = np.full(N_TEAMS, np.nan, dtype=np.float32)
    for row in wk.itertuples(index=False):
        short = str(row.SHORT)
        tid = team_index.short_to_id.get(short)
        if tid is not None:
            elo_vec[tid] = np.float32(float(row.ELO))  # type: ignore[arg-type]

    if np.isnan(elo_vec).any():
        missing_ids = np.where(np.isnan(elo_vec))[0].tolist()
        missing_shorts = [team_index.short_names[i] for i in missing_ids]
        raise ValueError(f"Elo vector incomplete for week {chosen_week}. Missing: {missing_shorts}")

    return elo_vec, chosen_week


def build_conf_div_arrays_from_csv(
    team_index: TeamIndex,
    conf_div_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Build conference and division assignment arrays from CSV.

    Returns:
        (conf_id, div_id) arrays of shape (32,).
    """
    df = pd.read_csv(conf_div_path)
    required = {"NFL_TEAM", "CONFERENCE", "DIVISION"}
    if not required.issubset(df.columns):
        raise ValueError(f"conf/div file missing columns: {required - set(df.columns)}")

    conf_id = np.full(N_TEAMS, -1, dtype=np.int8)
    div_id = np.full(N_TEAMS, -1, dtype=np.int8)

    for row in df.itertuples(index=False):
        long_name = str(row.NFL_TEAM)
        conf = str(row.CONFERENCE)
        div = str(row.DIVISION)

        short = team_index.long_to_short.get(long_name)
        if short is None:
            raise ValueError(f"Missing long->short mapping for conf/div team: {long_name}")

        tid = team_index.short_to_id.get(short)
        if tid is None:
            raise ValueError(f"Team {short} not found in team_index")

        c = CONF_CODES.get(conf)
        d = DIV_CODES.get(div)
        if c is None:
            raise ValueError(f"Unknown CONFERENCE: {conf}")
        if d is None:
            raise ValueError(f"Unknown DIVISION: {div}")

        conf_id[tid] = np.int8(c)
        div_id[tid] = np.int8(d)

    if (conf_id < 0).any() or (div_id < 0).any():
        missing = [
            team_index.short_names[i] for i in range(N_TEAMS) if conf_id[i] < 0 or div_id[i] < 0
        ]
        raise ValueError(f"Conference/division incomplete. Missing: {missing}")

    return conf_id, div_id


def extract_fixed_playoff_winners(
    *,
    df_wk_by_wk: pd.DataFrame,
    df_schedule: pd.DataFrame,
    team_index: TeamIndex,
    season_year: str,
) -> np.ndarray:
    """Extract completed playoff game results into a fixed-winner lookup array.

    Returns:
        (N_PLAYOFF_ROUNDS, N_TEAMS, N_TEAMS) int16 array.
        fixed[rnd, lo, hi] = winner_team_idx for known outcomes (lo < hi), else -1.
    """

    def team_name_to_id(name: str) -> int | None:
        if name in team_index.short_to_id:
            return int(team_index.short_to_id[name])
        short = team_index.long_to_short.get(name)
        if short is not None and short in team_index.short_to_id:
            return int(team_index.short_to_id[short])
        return None

    fixed = np.full((N_PLAYOFF_ROUNDS, N_TEAMS, N_TEAMS), -1, dtype=np.int16)

    round_by_week = {19: ROUND_WC, 20: ROUND_DIV, 21: ROUND_CONF, 22: ROUND_SB}

    game_to_pair: dict[str, tuple[int, int]] = {}
    for _, row in df_schedule.iterrows():
        gid = str(row["GAME_ID"])
        a = team_name_to_id(str(row["AWAY_TEAM"]))
        b = team_name_to_id(str(row["HOME_TEAM"]))
        if a is not None and b is not None:
            game_to_pair[gid] = (a, b)

    df = df_wk_by_wk.loc[df_wk_by_wk["YEAR"] == season_year].copy()
    df = df.loc[df["WEEK_NUM"].isin(round_by_week.keys())]

    if df.empty:
        return fixed

    for _, row in df.iterrows():
        wk = int(row["WEEK_NUM"])
        rnd = int(round_by_week[wk])
        gid = str(row["GAME_ID"])
        if gid not in game_to_pair:
            continue

        a, b = game_to_pair[gid]
        w = team_name_to_id(str(row["WINNER"]))
        if w is None:
            continue

        lo = min(a, b)
        hi = max(a, b)
        fixed[rnd, lo, hi] = np.int16(w)

    return fixed


# ============================================================================
# OUTPUT DATAFRAMES
# ============================================================================


def build_projections_df(
    team_index: TeamIndex,
    pts_total_by_sim: np.ndarray,
    po_win_counts: np.ndarray,
    make_playoffs_counts: np.ndarray,
    bye_counts: np.ndarray,
    n_sims: int,
) -> pd.DataFrame:
    """Build summary DataFrame with playoff probabilities, sorted by SB win %."""
    avg_wins = (pts_total_by_sim.astype(np.float64) / 2.0).mean(axis=0)

    p_make_po = make_playoffs_counts.astype(np.float64) / float(n_sims)
    p_reach_div = (po_win_counts[:, ROUND_WC] + bye_counts).astype(np.float64) / float(n_sims)
    p_reach_conf = po_win_counts[:, ROUND_DIV].astype(np.float64) / float(n_sims)
    p_reach_sb = po_win_counts[:, ROUND_CONF].astype(np.float64) / float(n_sims)
    p_win_sb = po_win_counts[:, ROUND_SB].astype(np.float64) / float(n_sims)

    return pd.DataFrame(
        {
            "TEAM": team_index.short_names,
            "AVG_WINS": avg_wins,
            "P_MAKE_PLAYOFFS": p_make_po,
            "P_REACH_DIV": p_reach_div,
            "P_REACH_CONF": p_reach_conf,
            "P_REACH_SB": p_reach_sb,
            "P_WIN_SB": p_win_sb,
        }
    ).sort_values(["P_WIN_SB", "AVG_WINS"], ascending=[False, False])


def build_season_grid_df(
    team_index: TeamIndex,
    reg_win_counts: np.ndarray,
    po_win_counts: np.ndarray,
    n_sims: int,
) -> pd.DataFrame:
    """Build detailed grid with weekly win probabilities and playoff round rates."""
    reg_counts = reg_win_counts[:, 1 : N_WEEKS_REG + 1]
    counts_22 = np.concatenate([reg_counts, po_win_counts], axis=1)

    col_counts = [f"W{w:02d}_WIN_CT" for w in range(1, N_WEEKS_REG + 1)] + [
        "WC_WIN_CT",
        "DIV_WIN_CT",
        "CONF_WIN_CT",
        "SB_WIN_CT",
    ]
    col_rates = [c.replace("_CT", "_P") for c in col_counts]

    df_counts = pd.DataFrame(counts_22, columns=col_counts)
    df_rates = pd.DataFrame(counts_22 / float(n_sims), columns=col_rates)

    return pd.concat([pd.Series(team_index.short_names, name="TEAM"), df_counts, df_rates], axis=1)


def _summarize_fixed_playoff_winners(fixed: np.ndarray) -> dict[str, int]:
    """Count how many fixed (non -1) matchups exist per round."""
    tri_mask = np.triu(np.ones((N_TEAMS, N_TEAMS), dtype=bool), k=1)
    round_names = {ROUND_WC: "WC", ROUND_DIV: "DIV", ROUND_CONF: "CONF", ROUND_SB: "SB"}
    return {
        round_names[rnd]: int(np.sum((fixed[rnd] != -1) & tri_mask))
        for rnd in range(N_PLAYOFF_ROUNDS)
    }


# ============================================================================
# ORCHESTRATION
# ============================================================================


def run_full_simulation(
    *,
    paths: SimPaths | None = None,
    config: SimulationConfig | None = None,
    render: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full season + playoff simulation.

    Args:
        paths: SimPaths (defaults to SimPaths.from_settings()).
        config: SimulationConfig (defaults to SimulationConfig()).
        render: If ``True``, renders the playoff probability table PNG via
            ``gridiron_edge.viz.charts.render_playoff_table`` after simulation.

    Returns:
        (df_projections, df_season_grid)
        df_projections: Per-team playoff probability summary.
        df_season_grid: Weekly win counts + playoff round rates.
    """
    import time

    paths = paths or SimPaths.from_settings()
    config = config or SimulationConfig()

    t0_total = time.perf_counter()

    with _log_phase("Validate paths"):
        paths.validate()
        logger.info("Data dir: %s", paths.data_cleaned)

    with _log_phase("Load CSV inputs"):
        df_schedule = pd.read_csv(paths.schedule_file)
        df_wk_by_wk = pd.read_csv(paths.wk_by_wk_file)
        df_elo = pd.read_csv(paths.elo_file)

        if df_schedule.empty:
            raise FileNotFoundError(
                "Upcoming schedule is empty — run 'gridiron ingest upcoming' first "
                "to fetch the schedule before simulating."
            )

        season_year = str(df_schedule["YEAR"].iloc[0])
        logger.info("Season year: %s", season_year)
        logger.info("Schedule rows: %d", len(df_schedule))
        logger.info("Week-by-week rows: %d", len(df_wk_by_wk))
        logger.info("Elo rows: %d", len(df_elo))

    with _log_phase("Build team index + schedule arrays"):
        long_to_short = load_long_to_short_mapping(paths.mapping_file)
        team_index = build_team_index_from_results(
            df_wk_by_wk, long_to_short, season_year, df_schedule=df_schedule
        )
        logger.info("Teams detected: %d", len(team_index.short_names))

        df_schedule = add_game_id_to_schedule(df_schedule, long_to_short)
        schedule, final_actual_week = build_schedule_arrays(
            df_schedule,
            df_wk_by_wk,
            team_index,
            season_year,
        )
        logger.info("Final actual week: %d", final_actual_week)

    with _log_phase("Load conference/division mapping + precompute counts"):
        conf_id, div_id = build_conf_div_arrays_from_csv(team_index, paths.conf_div_file)
        gp_total, gp_conf, gp_div, opp_mask = precompute_game_counts(
            schedule,
            conf_id,
            div_id,
        )

    with _log_phase("Apply actual results to matrices"):
        (
            pts_total_actual,
            pts_conf_actual,
            pts_div_actual,
            gp_played_actual,
            gp_vs_actual,
            pts_vs_actual,
            wins_vs_actual,
            reg_win_counts_actual,
        ) = apply_actuals_to_matrices(
            schedule.home,
            schedule.away,
            schedule.week_offsets,
            schedule.result,
            final_actual_week,
            conf_id,
            div_id,
        )

    with _log_phase("Load Elo entering next week"):
        elo_entering_next_week, week_used = load_week_elo_vector(
            df_elo,
            team_index,
            season_year,
            final_actual_week + 1,
        )
        logger.info("Using Elo week %d for entering week %d", week_used - 1, final_actual_week + 1)

    with _log_phase(f"Simulate remaining regular season (n_sims={config.n_sims:,})"):
        (
            pts_total_by_sim,
            pts_conf_by_sim,
            pts_div_by_sim,
            gp_vs_by_sim,
            pts_vs_by_sim,
            wins_vs_by_sim,
            end_elo_by_sim,
            reg_win_counts,
        ) = simulate_remaining_regular_season(
            config.n_sims,
            schedule.home,
            schedule.away,
            schedule.week_offsets,
            final_actual_week,
            conf_id,
            div_id,
            elo_entering_next_week,
            pts_total_actual,
            pts_conf_actual,
            pts_div_actual,
            gp_vs_actual,
            pts_vs_actual,
            wins_vs_actual,
            reg_win_counts_actual,
            float(config.k_factor),
            float(config.p_tie),
            int(config.base_seed),
            float(config.divisor),
        )

    with _log_phase("Extract fixed playoff outcomes"):
        fixed_playoff_winners = extract_fixed_playoff_winners(
            df_wk_by_wk=df_wk_by_wk,
            df_schedule=df_schedule,
            team_index=team_index,
            season_year=season_year,
        )
        summary = _summarize_fixed_playoff_winners(fixed_playoff_winners)
        logger.info(
            "Fixed playoff matchups: WC=%d DIV=%d CONF=%d SB=%d",
            summary.get("WC", 0),
            summary.get("DIV", 0),
            summary.get("CONF", 0),
            summary.get("SB", 0),
        )

    with _log_phase("Simulate playoffs"):
        po_win_counts, make_playoffs_counts, bye_counts = simulate_playoffs(
            pts_total_by_sim,
            pts_conf_by_sim,
            pts_div_by_sim,
            gp_total,
            gp_conf,
            gp_div,
            gp_vs_by_sim,
            pts_vs_by_sim,
            wins_vs_by_sim,
            opp_mask,
            end_elo_by_sim,
            conf_id,
            div_id,
            int(config.base_seed),
            fixed_playoff_winners,
        )

    with _log_phase("Build output dataframes"):
        df_projections = build_projections_df(
            team_index,
            pts_total_by_sim,
            po_win_counts,
            make_playoffs_counts,
            bye_counts,
            config.n_sims,
        )
        df_season_grid = build_season_grid_df(
            team_index,
            reg_win_counts,
            po_win_counts,
            config.n_sims,
        )
        logger.info("df_projections: %s", df_projections.shape)
        logger.info("df_season_grid: %s", df_season_grid.shape)

    with _log_phase("Save outputs"):
        paths.output_temp_dir.mkdir(parents=True, exist_ok=True)
        proj_path = paths.output_temp_dir / "projections_summary.csv"
        grid_path = paths.output_temp_dir / "season_grid.csv"
        df_projections.to_csv(proj_path, index=False)
        df_season_grid.to_csv(grid_path, index=False)
        logger.info("Wrote: %s", proj_path)
        logger.info("Wrote: %s", grid_path)

    if render:
        with _log_phase("Render playoff probability table"):
            from gridiron_edge.viz.charts import build_viz_table_df, render_playoff_table

            sim_results = SimulationResults(
                pts_total_by_sim=pts_total_by_sim,
                po_win_counts=po_win_counts,
                make_playoffs_counts=make_playoffs_counts,
                bye_counts=bye_counts,
                reg_win_counts=reg_win_counts,
                pts_total_actual=pts_total_actual,
                gp_played_actual=gp_played_actual,
                gp_total=gp_total,
                div_id=div_id,
            )
            df_viz = build_viz_table_df(
                sim_results,
                team_index=team_index,
                df_elo=df_elo,
                final_actual_week=final_actual_week,
                season_year=season_year,
                logo_dir=paths.logo_dir,
            )
            out_dir = paths.output_images_dir / f"Elo_Rankings/{season_year[:4]}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = (
                out_dir / f"{season_year[:4]}_playoff_table_wk{final_actual_week:02d}.png"
            )
            render_playoff_table(df=df_viz, output_path=out_img_path)
            logger.info("Playoff table written: %s", out_img_path)

    logger.info("Total runtime: %.2fs", time.perf_counter() - t0_total)

    return df_projections, df_season_grid

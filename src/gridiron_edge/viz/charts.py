# src/gridiron_edge/viz/charts.py
"""Visualization: playoff probability table and supporting data builder.

render_playoff_table() renders the plottable image.
build_viz_table_df() builds the formatted DataFrame it consumes.

Both are intentionally presentation-focused and depend on matplotlib/plottable.
Called after run_full_simulation() in gridiron_edge.sim.season.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import image

from gridiron_edge.sim.season import (
    DIV_CODE_TO_LABEL,
    N_TEAMS,
    N_WEEKS_REG,
    ROUND_CONF,
    ROUND_DIV,
    ROUND_SB,
    ROUND_WC,
    TeamIndex,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def build_viz_table_df(
    team_index: TeamIndex,
    pts_total_by_sim: np.ndarray,
    pts_total_actual: np.ndarray,
    gp_played_actual: np.ndarray,
    gp_total: np.ndarray,
    make_playoffs_counts: np.ndarray,
    bye_counts: np.ndarray,
    po_win_counts: np.ndarray,
    div_id: np.ndarray,
    df_elo: pd.DataFrame,
    final_actual_week: int,
    season_year: str,
    logo_dir: Path,
) -> pd.DataFrame:
    """Build formatted DataFrame for visualization with logos and probabilities.

    Shapes raw simulation outputs into a single table suitable for rendering
    with plottable. Indexed by logo path (for image cells), sorted by Elo.

    Args:
        team_index: Team mappings.
        pts_total_by_sim: Total points across simulations (n_sims, 32).
        pts_total_actual: Actual total points through the last completed week (32,).
        gp_played_actual: Actual games played through the last completed week (32,).
        gp_total: Total scheduled games for each team (32,).
        make_playoffs_counts: Playoff appearances (32,).
        bye_counts: First-round byes (32,).
        po_win_counts: Playoff round wins (32, 4).
        div_id: Division id per team (32,).
        df_elo: Elo history dataframe.
        final_actual_week: Last completed week.
        season_year: Season label as used in datasets (e.g. "2025-2026").
        logo_dir: Directory containing team logo PNGs named by long team name.

    Returns:
        DataFrame indexed by logo path, sorted descending by Elo.
    """
    n_sims = int(pts_total_by_sim.shape[0])

    # --- Playoff probabilities ---
    p_make_po = make_playoffs_counts.astype(np.float64) / n_sims
    p_reach_div = (po_win_counts[:, ROUND_WC] + bye_counts).astype(np.float64) / n_sims
    p_reach_conf = po_win_counts[:, ROUND_DIV].astype(np.float64) / n_sims
    p_make_sb = po_win_counts[:, ROUND_CONF].astype(np.float64) / n_sims
    p_win_sb = po_win_counts[:, ROUND_SB].astype(np.float64) / n_sims

    # --- Division labels ---
    division_labels = [DIV_CODE_TO_LABEL[int(div_id[i])] for i in range(N_TEAMS)]

    # --- Current Elo ---
    df_elo_this = df_elo.loc[
        (df_elo["NFL_YEAR"] == season_year)
        & (df_elo["NFL_WEEK"].astype(int) == final_actual_week + 1),
        ["NFL_TEAM", "ELO"],
    ].copy()
    df_elo_this["SHORT"] = df_elo_this["NFL_TEAM"].map(team_index.long_to_short)
    df_elo_this = df_elo_this.dropna(subset=["SHORT"])
    elo_by_short = dict(zip(df_elo_this["SHORT"], df_elo_this["ELO"].astype(float), strict=False))
    elo_vals = np.array([elo_by_short.get(s, np.nan) for s in team_index.short_names], dtype=float)

    # --- Prior Elo for 1-week delta ---
    if final_actual_week == 0:
        prior_year_parts = season_year.split("-")
        if len(prior_year_parts) == 2 and prior_year_parts[0].isdigit():
            prior_year = f"{int(prior_year_parts[0]) - 1}-{prior_year_parts[0]}"
        else:
            prior_year = season_year

        last_week = df_elo.loc[df_elo["NFL_YEAR"] == prior_year, "NFL_WEEK"].astype(int).max()
        df_elo_last = df_elo.loc[
            (df_elo["NFL_YEAR"] == prior_year) & (df_elo["NFL_WEEK"].astype(int) == int(last_week)),
            ["NFL_TEAM", "ELO"],
        ].copy()
    else:
        df_elo_last = df_elo.loc[
            (df_elo["NFL_YEAR"] == season_year)
            & (df_elo["NFL_WEEK"].astype(int) == final_actual_week),
            ["NFL_TEAM", "ELO"],
        ].copy()

    df_elo_last["SHORT"] = df_elo_last["NFL_TEAM"].map(team_index.long_to_short)
    df_elo_last = df_elo_last.dropna(subset=["SHORT"])
    elo_last_by_short = dict(
        zip(df_elo_last["SHORT"], df_elo_last["ELO"].astype(float), strict=False)
    )
    elo_last_vals = np.array(
        [elo_last_by_short.get(s, np.nan) for s in team_index.short_names], dtype=float
    )
    rank_change = np.nan_to_num(elo_vals - elo_last_vals, nan=0.0)

    # --- Current / projected record strings ---
    w_base = np.zeros(N_TEAMS, dtype=int)
    l_base = np.zeros(N_TEAMS, dtype=int)
    t_base = np.zeros(N_TEAMS, dtype=int)

    for i in range(N_TEAMS):
        ties = int(pts_total_actual[i]) % 2
        wins = (int(pts_total_actual[i]) - ties) // 2
        losses = int(gp_played_actual[i]) - wins - ties
        w_base[i], l_base[i], t_base[i] = wins, losses, ties

    current_record = [
        f"{w_base[i]}-{l_base[i]}" if t_base[i] == 0 else f"{w_base[i]}-{l_base[i]}-{t_base[i]}"
        for i in range(N_TEAMS)
    ]

    if final_actual_week >= N_WEEKS_REG:
        projected = current_record
    else:
        avg_wins_total = (pts_total_by_sim.astype(np.float64) / 2.0).mean(axis=0)
        wins_rounded = np.rint(avg_wins_total).astype(int)

        games_total = gp_total.astype(int)
        g_act = gp_played_actual.astype(int)
        ties_proj = t_base.copy()

        remaining = games_total - g_act
        wins_rounded = np.clip(wins_rounded, w_base, w_base + remaining)
        losses_proj = games_total - wins_rounded - ties_proj

        projected = [
            f"{int(wins_rounded[i])}-{int(losses_proj[i])}"
            if int(ties_proj[i]) == 0
            else f"{int(wins_rounded[i])}-{int(losses_proj[i])}-{int(ties_proj[i])}"
            for i in range(N_TEAMS)
        ]

    # --- Logos ---
    flag_paths = list(logo_dir.glob("*.png"))
    long_to_flagpath = {p.stem: p for p in flag_paths}
    short_to_long = {short: long for long, short in team_index.long_to_short.items()}

    logo_paths: list[Path | None] = []
    missing: list[str] = []
    for short in team_index.short_names:
        long_name = short_to_long.get(short)
        if long_name is None:
            logo_paths.append(None)
            missing.append(f"{short} (no short->long mapping)")
            continue

        p = long_to_flagpath.get(long_name)
        if p is None:
            logo_paths.append(None)
            missing.append(f"{short} -> '{long_name}' (no matching PNG stem)")
        else:
            logo_paths.append(p)

    if missing:
        logger.warning(
            "Missing logos: %s",
            "; ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""),
        )

    out = pd.DataFrame(
        {
            "LOGO": logo_paths,
            "ELO": elo_vals,
            "RANK_CHANGE": rank_change,
            "DIVISION": division_labels,
            "CURRENT_RECORD": current_record,
            "PROJECTED": projected,
            "MAKE PLAYOFFS": p_make_po,
            "DIVISIONAL ROUND": p_reach_div,
            "CONFERENCE CHAMPIONSHIP": p_reach_conf,
            "MAKE SUPER BOWL": p_make_sb,
            "WIN SUPER BOWL": p_win_sb,
        }
    )

    return out.set_index("LOGO").sort_values(["ELO"], ascending=False)


def render_playoff_table(
    df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    """Render the plottable playoff probability table to an image file.

    Args:
        df: DataFrame from build_viz_table_df() — indexed by logo path,
            sorted by Elo descending.
        output_path: Where to write the PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    knockout_cols: list[str] = [
        "WIN SUPER BOWL",
        "MAKE SUPER BOWL",
        "CONFERENCE CHAMPIONSHIP",
        "DIVISIONAL ROUND",
        "MAKE PLAYOFFS",
    ]

    cmap = LinearSegmentedColormap.from_list(
        name="bugw",
        colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"],
        N=256,
    )

    col_defs = [
        ColumnDefinition(
            name="ELO",
            title="Elo",
            formatter="{:.0f}",
            textprops={"ha": "left", "fontsize": 13},
            width=0.25,
        ),
        ColumnDefinition(
            name="NFL_TEAM",
            title="Team",
            textprops={"ha": "left", "weight": "bold"},
            width=1,
        ),
        ColumnDefinition(
            name="LOGO",
            title="",
            textprops={"ha": "center"},
            width=0.25,
            plot_fn=image,
        ),
        ColumnDefinition(
            name="RANK_CHANGE",
            title="1-Week\nElo Adj.",
            formatter="{:+.0f}",
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "round4", "pad": 0.35},
                "fontsize": 11,
            },
            cmap=normed_cmap(df["RANK_CHANGE"], cmap=mpl.cm.RdYlGn, num_stds=2.5),  # type: ignore[attr-defined]
            width=0.6,
        ),
        ColumnDefinition(
            name="DIVISION",
            title="Division",
            textprops={"ha": "left", "fontsize": 13},
            width=0.68,
        ),
        ColumnDefinition(
            name="CURRENT_RECORD",
            title="Curr.",
            width=0.4,
            textprops={"ha": "right"},
            group="Record",
            border="left",
        ),
        ColumnDefinition(
            name="PROJECTED",
            title="Proj.",
            width=0.4,
            textprops={"ha": "right"},
            group="Record",
        ),
        ColumnDefinition(
            name=knockout_cols[0],
            title=knockout_cols[0].replace(" ", "\n", 1).title(),
            formatter=decimal_to_percent,
            cmap=cmap,
            group="Playoff Chances",
        ),
        ColumnDefinition(
            name=knockout_cols[1],
            title=knockout_cols[1].replace(" ", "\n", 1).title(),
            formatter=decimal_to_percent,
            cmap=cmap,
            group="Playoff Chances",
        ),
        ColumnDefinition(
            name=knockout_cols[2],
            title="Conf.\nChamp.",
            formatter=decimal_to_percent,
            cmap=cmap,
            group="Playoff Chances",
        ),
        ColumnDefinition(
            name=knockout_cols[3],
            title=knockout_cols[3].replace(" ", "\n", 1).title(),
            formatter=decimal_to_percent,
            cmap=cmap,
            group="Playoff Chances",
        ),
        ColumnDefinition(
            name=knockout_cols[4],
            title=knockout_cols[4].replace(" ", "\n", 1).title(),
            formatter=decimal_to_percent,
            cmap=cmap,
            group="Playoff Chances",
            border="left",
        ),
    ]

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["savefig.bbox"] = "tight"

    fig, ax = plt.subplots(figsize=(20, 22))

    Table(
        df,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 14},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
    )

    fig.savefig(output_path, facecolor=ax.get_facecolor(), dpi=200)
    plt.close(fig)
    logger.info("Wrote: %s", output_path)

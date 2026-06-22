# src/gridiron_edge/viz/predictions.py

"""Weekly matchup predictions visualisation.

Generates the weekly predictions image: a matchup table with team logos,
win probabilities, team colour gradient bars highlighting the predicted
winner, and an optional DraftKings underdog highlight.

Migrated from ``notebooks/exploratory/Weekly_Prediction_Visualisation.ipynb``.

Improvements over the notebook:
- Logo paths keyed by full long team name (no splitting/mapping needed).
- Time-separator rows built explicitly rather than the index > 15 trick.
- DK odds are optional — if no snapshot exists the underdog box is skipped.
- Output paths derived from ``get_settings()`` rather than hardcoded.
- Both PNG and static HTML outputs are written in one call.
"""

from __future__ import annotations

from html import escape
import logging
from logging import Logger
from pathlib import Path
from typing import Literal

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from plottable import ColDef, ColumnDefinition, Table
from plottable.plots import image

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.ingest.odds.store import load_current_odds
from gridiron_edge.ratings.elo.core import elo_win_probability

logger: Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team colour palette — one primary colour per franchise (last name keyed).
# ---------------------------------------------------------------------------

TEAM_COLORS: dict[str, str] = {
    "Arizona Cardinals": "#97233F",
    "Atlanta Falcons": "#A71930",
    "Baltimore Ravens": "#241773",
    "Buffalo Bills": "#00338D",
    "Carolina Panthers": "#0085CA",
    "Chicago Bears": "#C83803",
    "Cincinnati Bengals": "#FB4F14",
    "Cleveland Browns": "#311D00",
    "Dallas Cowboys": "#003594",
    "Denver Broncos": "#FB4F14",
    "Detroit Lions": "#0076B6",
    "Green Bay Packers": "#203731",
    "Houston Texans": "#03202F",
    "Indianapolis Colts": "#002C5F",
    "Jacksonville Jaguars": "#006778",
    "Kansas City Chiefs": "#E31837",
    "Las Vegas Raiders": "#A5ACAF",
    "Los Angeles Chargers": "#FFC20E",
    "Los Angeles Rams": "#003594",
    "Miami Dolphins": "#008E97",
    "Minnesota Vikings": "#4F2683",
    "New England Patriots": "#002244",
    "New Orleans Saints": "#D3BC8D",
    "New York Giants": "#0B2265",
    "New York Jets": "#125740",
    "Philadelphia Eagles": "#004C54",
    "Pittsburgh Steelers": "#FFB612",
    "San Francisco 49ers": "#AA0000",
    "Seattle Seahawks": "#002244",
    "Tampa Bay Buccaneers": "#D50A0A",
    "Tennessee Titans": "#4B92DB",
    "Washington Commanders": "#5A1414",
}

_BACKGROUND_COLOR: str = "#28282B"
_EMPTY_LOGO: str = "Empty.png"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gradient_image(
    ax: plt.Axes,
    direction: float = 0.3,
    cmap_range: tuple[float, float] = (0, 1),
    **kwargs: float | str | None,  # extent, cmap, transform, etc.
) -> None:
    """Draw a directional gradient image on an axes.

    Args:
        ax: Target matplotlib Axes.
        direction: Gradient direction — 0 = vertical, 1 = horizontal.
        cmap_range: Fraction (cmin, cmax) of the colormap to use.
        **kwargs: Forwarded to ``Axes.imshow()``.
    """
    phi: float = direction * np.pi / 2
    v: ndarray = np.array([np.cos(phi), np.sin(phi)])
    x_mat = np.array([[v @ [1, 0], v @ [1, 1]], [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    x_mat = a + (b - a) / x_mat.max() * x_mat
    ax.imshow(
        x_mat,
        interpolation="bicubic",
        clim=(0, 1),
        aspect="auto",
        # pyrefly: ignore [bad-argument-type]
        **kwargs,
    )


def _gradient_bar(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: float,
    height: float,
    cmap: LinearSegmentedColormap,
) -> None:
    """Draw gradient-filled bars at positions (x, y).

    Args:
        ax: Target matplotlib Axes.
        x: Array of left x-coordinates.
        y: Array of bottom y-coordinates.
        width: Width of each bar.
        height: Height of each bar.
        cmap: Colormap to use for the gradient fill.
    """
    for left, bottom in zip(x, y, strict=False):
        _gradient_image(
            ax,
            direction=1,
            # pyrefly: ignore [bad-argument-type]
            extent=(left, left + width, bottom, bottom + height),
            # pyrefly: ignore [bad-argument-type]
            cmap=cmap,
        )


def _build_logo_map(logo_dir: Path) -> dict[str, Path]:
    """Build a mapping from full team name to logo PNG path.

    Args:
        logo_dir: Directory containing ``{Team Name}.png`` files.

    Returns:
        Dict mapping team long name to absolute PNG path.
    """
    return {f.stem: f for f in logo_dir.glob("*.png")}


def _empty_logo_path(logo_dir: Path) -> Path:
    """Return the path to the empty placeholder logo.

    Args:
        logo_dir: Team logos directory.

    Returns:
        Path to ``Empty.png``.
    """
    return logo_dir / _EMPTY_LOGO


def _build_predictions_df(
    df_schedule: pd.DataFrame,
    logo_map: dict[str, Path],
    empty_logo: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the display DataFrame with time-separator rows for the table.

    Inserts one separator row per unique game time above the first game
    of that time slot. Separator rows show only the GAME_TIME value;
    all other fields are NaN.

    Args:
        df_schedule: Schedule DataFrame with win probability columns added.
            Expected columns: ``AWAY_TEAM``, ``HOME_TEAM``, ``GAME_DATE``,
            ``GAMETIME``, ``GAME_DAY_OF_WEEK``, ``AWAY_TEAM_WIN_PROB``,
            ``HOME_TEAM_WIN_PROB``.
        logo_map: Full-name → Path mapping from ``_build_logo_map()``.
        empty_logo: Path to the empty placeholder logo.

    Returns:
        Tuple of ``(display_df, table_cols)`` where ``display_df`` is
        ready to pass to ``plottable.Table`` and ``table_cols`` lists the
        column names in display order.
    """
    df: DataFrame = df_schedule.copy()

    # --- Logo columns (keyed by full long name) ---
    _empty_str = str(empty_logo)
    df["AWAY_TEAM_LOGO"] = df["AWAY_TEAM"].map(logo_map).fillna(_empty_str)
    df["HOME_TEAM_LOGO"] = df["HOME_TEAM"].map(logo_map).fillna(_empty_str)

    # --- Display name = last word of team name ---
    df["AWAY_TEAM_SHORT"] = df["AWAY_TEAM"].str.split(" ").str[-1]
    df["HOME_TEAM_SHORT"] = df["HOME_TEAM"].str.split(" ").str[-1]

    # --- Game time label ---
    def _to_12hr(row: pd.Series) -> str:
        try:
            parts: list[str] = str(row["GAMETIME"]).split(":")
            hour, minute = int(parts[0]), int(parts[1])
            suffix: Literal["AM", "PM"] = "AM" if hour < 12 else "PM"
            hour12: int = hour % 12 or 12
            return f"{row['GAME_DAY_OF_WEEK'][:3]}\n{hour12}:{minute:02d} {suffix}"
        except (ValueError, IndexError, KeyError):
            return f"{row['GAME_DAY_OF_WEEK'][:3]}\n{row['GAMETIME']}"

    df["GAME_TIME"] = df.apply(_to_12hr, axis=1)

    # --- Sort by date + time ---
    df["_sort_key"] = pd.to_datetime(
        df["GAME_DATE"].astype(str).str.cat(df["GAMETIME"].astype(str), sep=" "),
        errors="coerce",
    )
    df = df.sort_values("_sort_key").reset_index(drop=True)

    # --- Build separator rows explicitly ---
    # One separator per unique game time, inserted before the first game of
    # that slot. This replaces the fragile index > 15 trick from the notebook.
    separator_rows: list[pd.DataFrame] = []
    seen_times: set[str] = set()

    for _, row in df.iterrows():
        gt = row["GAME_TIME"]
        if gt not in seen_times:
            seen_times.add(gt)
            sep = pd.DataFrame(
                {
                    "AWAY_TEAM_WIN_PROB": [np.nan],
                    "AWAY_TEAM_SHORT": [np.nan],
                    "AWAY_TEAM_LOGO": [str(empty_logo)],
                    "GAME_TIME": [gt],
                    "HOME_TEAM_LOGO": [str(empty_logo)],
                    "HOME_TEAM_SHORT": [np.nan],
                    "HOME_TEAM_WIN_PROB": [np.nan],
                    "EMPTY_COL_END": [""],
                    "_is_separator": [True],
                    "_sort_key": [row["_sort_key"]],
                    "AWAY_TEAM": [np.nan],
                    "HOME_TEAM": [np.nan],
                    "AWAY_TEAM_ELO": [np.nan],
                    "HOME_TEAM_ELO": [np.nan],
                }
            )
            separator_rows.append(sep)

    df["_is_separator"] = False

    all_rows: DataFrame = pd.concat([df, *separator_rows], ignore_index=True)
    all_rows = all_rows.sort_values(
        ["_sort_key", "_is_separator"],
        ascending=[True, False],
    ).reset_index(drop=True)

    all_rows["EMPTY_COL_START"] = ""
    all_rows["EMPTY_COL_END"] = ""

    table_cols: list[str] = [
        "AWAY_TEAM_WIN_PROB",
        "AWAY_TEAM_SHORT",
        "AWAY_TEAM_LOGO",
        "GAME_TIME",
        "HOME_TEAM_LOGO",
        "HOME_TEAM_SHORT",
        "HOME_TEAM_WIN_PROB",
        "EMPTY_COL_END",
    ]

    return all_rows, table_cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_predictions_df(
    *,
    year: str,
    week: int,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Build the raw predictions DataFrame for a given week.

    Merges Elo ratings onto the upcoming schedule and computes win
    probabilities. This DataFrame is the canonical prediction output —
    it is written to CSV and also used as input to the visualisation.

    Args:
        year: NFL season label (e.g. ``"2026-2027"``).
        week: NFL week number.
        repo: Repository root path.

    Returns:
        DataFrame with columns: ``WEEK_NUM``, ``GAME_DATE``,
        ``GAME_DAY_OF_WEEK``, ``GAMETIME``, ``AWAY_TEAM``, ``HOME_TEAM``,
        ``AWAY_TEAM_ELO``, ``HOME_TEAM_ELO``, ``AWAY_WIN_PROB``,
        ``HOME_WIN_PROB``.
    """
    settings = get_settings()
    resolved_repo: Path = repo or settings.repo_root

    elo_path: Path = dataset_path(resolved_repo, "elo_state")
    schedule_path: Path = dataset_path(resolved_repo, "schedule_upcoming")

    df_elo: DataFrame = pd.read_csv(elo_path)
    df_schedule: DataFrame = pd.read_csv(schedule_path)

    df_schedule = df_schedule.loc[
        (df_schedule["YEAR"] == year) & (df_schedule["WEEK_NUM"] == week), :
    ].copy()

    if df_schedule.empty:
        logger.warning("No upcoming games found for %s week %d in %s", year, week, schedule_path)
        return pd.DataFrame()

    # Merge Elo for away team
    df_schedule = (
        pd.merge(
            df_schedule,
            df_elo,
            how="left",
            left_on=["AWAY_TEAM", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        )
        .drop(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"])
        .rename(columns={"ELO": "AWAY_TEAM_ELO"})
    )

    # Merge Elo for home team
    df_schedule = (
        pd.merge(
            df_schedule,
            df_elo,
            how="left",
            left_on=["HOME_TEAM", "YEAR", "WEEK_NUM"],
            right_on=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"],
        )
        .drop(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK"])
        .rename(columns={"ELO": "HOME_TEAM_ELO"})
    )

    df_schedule = df_schedule.dropna(subset=["AWAY_TEAM_ELO", "HOME_TEAM_ELO"])

    if df_schedule.empty:
        logger.warning(
            "Elo data missing for %s week %d — schedule has games but Elo state "
            "does not yet cover this week. Run `gridiron ratings elo fit` first.",
            year,
            week,
        )
        return pd.DataFrame()

    # Compute win probabilities
    probs = df_schedule.apply(
        lambda x: elo_win_probability(x["AWAY_TEAM_ELO"], x["HOME_TEAM_ELO"]),
        axis=1,
    )
    df_schedule[["AWAY_WIN_PROB", "HOME_WIN_PROB"]] = pd.DataFrame(
        probs.tolist(),
        index=df_schedule.index,
    )

    # Formatted percentage strings for display
    df_schedule["AWAY_TEAM_WIN_PROB"] = df_schedule["AWAY_WIN_PROB"].map(
        lambda x: f"{x * 100:.1f} %"
    )
    df_schedule["HOME_TEAM_WIN_PROB"] = df_schedule["HOME_WIN_PROB"].map(
        lambda x: f"{x * 100:.1f} %"
    )

    return df_schedule.drop(columns=["YEAR"])


def render_predictions_image(
    df_schedule: pd.DataFrame,
    *,
    year: str,
    week: int,
    repo: Path | None = None,
) -> Path:
    """Render the weekly predictions matchup image.

    Produces a PNG file at
    ``data/output/predictions/{year[:4]}/week_{week:02d}_predictions.png``.

    DK odds are loaded from the current snapshot if available. If no odds
    snapshot exists, the underdog highlight is silently skipped.

    Args:
        df_schedule: DataFrame from ``build_predictions_df()``.
        year: NFL season label (e.g. ``"2026-2027"``).
        week: NFL week number.
        repo: Repository root path.

    Returns:
        Absolute path to the written PNG file.
    """
    settings = get_settings()
    resolved_repo: Path = repo or settings.repo_root

    logo_dir: Path = resolved_repo / "data" / "images" / "Team Logos"
    logo_map: dict[str, Path] = _build_logo_map(logo_dir)
    empty_logo: Path = _empty_logo_path(logo_dir)

    # --- Load DK odds (optional) ---
    df_ml: DataFrame | None = load_current_odds(market="moneyline", repo=resolved_repo)
    moneylines: dict[str, float] = {}
    if df_ml is not None and not df_ml.empty:
        for _, row in df_ml.iterrows():
            team: str = str(row["away_team"]) if row["side"] == "away" else str(row["home_team"])
            short: str = team.split(" ")[-1]
            moneylines[short] = float(row["odds"])

    # --- Build display DataFrame ---
    display_df, table_cols = _build_predictions_df(df_schedule, logo_map, empty_logo)

    # --- Plot ---
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["text.color"] = "white"

    n_rows: int = len(display_df)
    fig, ax = plt.subplots(figsize=(14, max(8, n_rows * 0.65)))

    # Background gradient
    _gradient_image(
        ax,
        direction=1,
        # pyrefly: ignore [bad-argument-type]
        extent=(0, 1, 0, 1),
        # pyrefly: ignore [bad-argument-type]
        transform=ax.transAxes,
        # pyrefly: ignore [bad-argument-type]
        cmap=LinearSegmentedColormap.from_list(
            "bg",
            ["black", "#181818", "black"],
        ),
    )

    plt.axis("off")
    plt.subplots_adjust(top=1, right=1, bottom=0, left=0)

    col_defs: list[ColumnDefinition] = [
        ColDef("EMPTY_COL_START", width=0.05, title="", textprops={"ha": "center"}),
        ColDef(
            "AWAY_TEAM_WIN_PROB",
            width=0.2,
            title="",
            textprops={"ha": "right"},
            formatter=lambda x: "" if pd.isna(x) else x,
        ),
        ColDef("AWAY_TEAM_LOGO", width=0.1, title="", plot_fn=image),
        ColDef(
            "AWAY_TEAM_SHORT",
            width=0.2,
            title="",
            textprops={"ha": "right"},
            formatter=lambda x: "" if pd.isna(x) else x,
        ),
        ColDef(
            "GAME_TIME",
            width=0.1,
            title="",
            textprops={"ha": "center"},
            formatter=lambda x: "" if pd.isna(x) else str(x),
        ),
        ColDef(
            "HOME_TEAM_SHORT",
            width=0.2,
            title="",
            textprops={"ha": "left"},
            formatter=lambda x: "" if pd.isna(x) else x,
        ),
        ColDef("HOME_TEAM_LOGO", width=0.1, title="", plot_fn=image),
        ColDef(
            "HOME_TEAM_WIN_PROB",
            width=0.2,
            title="",
            textprops={"ha": "left"},
            formatter=lambda x: "" if pd.isna(x) else x,
        ),
        ColDef("EMPTY_COL_END", width=0.05, title="", textprops={"ha": "center"}),
    ]

    table = Table(
        # pyrefly: ignore [bad-argument-type]
        df=display_df,
        column_definitions=col_defs,
        row_dividers=False,
        col_label_divider=False,
        footer_divider=False,
        columns=table_cols,
        index_col="EMPTY_COL_START",
        col_label_cell_kw={"alpha": 0.0},
        textprops={"fontsize": 16, "ha": "center"},
    )

    # --- Column headers ---
    top = table.rows[0]
    top_ax = top.cells[3].ax

    # DK Underdog legend box
    lx = top.cells[1].x
    by = top.cells[3].y - 1
    h = top.cells[3].height
    w = top.cells[0].width + top.cells[1].width - 0.05
    underdog_rect = Rectangle(
        (lx, by),
        w - 0.01,
        h - 0.03,
        linewidth=1,
        edgecolor="y",
        facecolor="none",
    )
    top_ax.add_patch(underdog_rect)
    top_ax.annotate(
        "DK Underdog",
        ((lx + w / 2), by + h / 2),
        color="w",
        weight="bold",
        fontsize=15,
        ha="center",
        va="center",
    )

    # Away Team header — x coords derived from column width proportions,
    # y coord derived from cell position so it tracks row height correctly.
    header_y = top.cells[2].y - 1
    header_h = top.cells[2].height
    underline_y = header_y + header_h * 0.8

    ax.annotate(
        "Away Team",
        (0.45, header_y + header_h / 2),
        color="w",
        weight="bold",
        fontsize=20,
        ha="center",
        va="center",
    )
    ax.hlines(underline_y, 0.35, 0.55, colors="w")

    # Home Team header
    ax.annotate(
        "Home Team",
        (0.75, header_y + header_h / 2),
        color="w",
        weight="bold",
        fontsize=20,
        ha="center",
        va="center",
    )
    ax.hlines(underline_y, 0.65, 0.85, colors="w")

    # --- Row rendering ---
    seen_game_time = ""
    for idx in range(len(table.rows)):
        row = table.rows[idx]
        row.set_alpha(0.0)
        row.cells[2].text.set_weight("bold")
        row.cells[6].text.set_weight("bold")

        # Deduplicate game time label
        gt = row.cells[4].content
        if gt == seen_game_time:
            row.cells[4].text.set_alpha(0.0)
        else:
            seen_game_time = gt

        if pd.isna(row.cells[1].content):
            continue

        away_prob: float = float(str(row.cells[1].content).rstrip(" %")) / 100
        home_prob: float = float(str(row.cells[7].content).rstrip(" %")) / 100
        away_short = str(row.cells[2].content)
        home_short = str(row.cells[6].content)

        away_long: str = next(
            (t for t in TEAM_COLORS if t.split(" ")[-1] == away_short),
            away_short,
        )
        home_long: str = next(
            (t for t in TEAM_COLORS if t.split(" ")[-1] == home_short),
            home_short,
        )

        away_color: str = TEAM_COLORS.get(away_long, "#333333")
        home_color: str = TEAM_COLORS.get(home_long, "#333333")

        if away_prob > home_prob:
            # Away team wins — gradient from left
            cur_ax = row.cells[3].ax
            lx = row.cells[1].x
            by = row.cells[3].y
            h = row.cells[3].height
            w: int = sum(row.cells[i].width for i in range(1, 5))

            row.cells[7].text.set_alpha(0.0)
            _gradient_bar(
                cur_ax,
                np.array([lx]),
                np.array([by]),
                width=w,
                height=h,
                cmap=LinearSegmentedColormap.from_list(
                    "away_grad",
                    ["black", away_color, away_color, away_color, "#181818"],
                ),
            )

            if moneylines:
                away_ml: float | None = moneylines.get(away_short)
                home_ml: float | None = moneylines.get(home_short)
                if away_ml is not None and home_ml is not None and away_ml > home_ml:
                    rect = Rectangle(
                        (lx, by),
                        w,
                        h,
                        linewidth=1,
                        edgecolor="y",
                        facecolor="none",
                    )
                    cur_ax.add_patch(rect)
        else:
            # Home team wins — gradient from right
            cur_ax = row.cells[5].ax
            lx = row.cells[4].x
            by = row.cells[4].y
            h = row.cells[5].height
            w = sum(row.cells[i].width for i in range(4, 8))

            row.cells[1].text.set_alpha(0.0)
            _gradient_bar(
                cur_ax,
                np.array([lx]),
                np.array([by]),
                width=w,
                height=h,
                cmap=LinearSegmentedColormap.from_list(
                    "home_grad",
                    ["black", home_color, home_color, home_color, "#181818"],
                ).reversed(),
            )

            if moneylines:
                away_ml = moneylines.get(away_short)
                home_ml = moneylines.get(home_short)
                if away_ml is not None and home_ml is not None and away_ml < home_ml:
                    rect = Rectangle(
                        (lx, by),
                        w,
                        h,
                        linewidth=1,
                        edgecolor="y",
                        facecolor="none",
                    )
                    cur_ax.add_patch(rect)

    # --- Save PNG ---
    out_dir: Path = resolved_repo / "data" / "output" / "predictions" / year[:4]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = out_dir / f"week_{week:02d}_predictions.png"
    fig.savefig(
        out_path,
        facecolor=fig.get_facecolor(),
        dpi=200,
        transparent=True,
        pad_inches=0,
    )
    plt.close(fig)
    logger.info("Predictions image written: %s", out_path)
    return out_path


def render_predictions_html(
    df_schedule: pd.DataFrame,
    *,
    year: str,
    week: int,
    repo: Path | None = None,
) -> Path:
    """Render a static HTML predictions table for the given week.

    Produces a self-contained HTML file at
    ``data/output/predictions/{year[:4]}/week_{week:02d}_predictions.html``.

    Inline CSS uses team colours for the predicted-winner row background.
    No external dependencies — the file is shareable as-is.

    Args:
        df_schedule: DataFrame from ``build_predictions_df()``.
        year: NFL season label (e.g. ``"2026-2027"``).
        week: NFL week number.
        repo: Repository root path.

    Returns:
        Absolute path to the written HTML file.
    """
    settings = get_settings()
    resolved_repo: Path = repo or settings.repo_root

    df_ml: DataFrame | None = load_current_odds(market="moneyline", repo=resolved_repo)
    moneylines: dict[str, float] = {}
    if df_ml is not None and not df_ml.empty:
        for _, row in df_ml.iterrows():
            team: str = str(row["away_team"]) if row["side"] == "away" else str(row["home_team"])
            moneylines[team] = float(row["odds"])

    rows_html: list[str] = []
    prev_time = ""

    df_sorted: DataFrame = df_schedule.sort_values(["GAME_DATE", "GAMETIME"]).reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        try:
            parts: list[str] = str(row["GAMETIME"]).split(":")
            hour, minute = int(parts[0]), int(parts[1])
            suffix: Literal["AM", "PM"] = "AM" if hour < 12 else "PM"
            hour12: int = hour % 12 or 12
            time_12hr: str = f"{hour12}:{minute:02d} {suffix}"
        except (ValueError, IndexError, KeyError):
            time_12hr = row["GAMETIME"]
        game_time: str = f"{row['GAME_DAY_OF_WEEK']} {time_12hr}"
        if game_time != prev_time:
            prev_time: str = game_time
            rows_html.append(f'<tr class="time-sep"><td colspan="5">{escape(game_time)}</td></tr>')

        away = str(row["AWAY_TEAM"])
        home = str(row["HOME_TEAM"])
        away_prob = float(row["AWAY_WIN_PROB"])
        home_prob = float(row["HOME_WIN_PROB"])
        away_pct: str = f"{away_prob * 100:.1f}%"
        home_pct: str = f"{home_prob * 100:.1f}%"

        winner: str = away if away_prob >= home_prob else home
        winner_color: str = TEAM_COLORS.get(winner, "#333")
        loser_cell_style = "color: #888;"

        away_ml: float | None = moneylines.get(away)
        home_ml: float | None = moneylines.get(home)
        away_is_dk_dog: bool = (
            away_ml is not None
            and home_ml is not None
            and away_ml > home_ml
            and away_prob >= home_prob
        )
        home_is_dk_dog: bool = (
            away_ml is not None
            and home_ml is not None
            and home_ml > away_ml
            and home_prob > away_prob
        )

        dog_style = "outline: 2px solid gold; outline-offset: -2px;"
        row_style: str = f"background: linear-gradient(to right, #111, {winner_color}88, #111);"

        away_style: Literal["", "color: #888;"] = "" if away_prob >= home_prob else loser_cell_style
        home_style: Literal["", "color: #888;"] = "" if home_prob > away_prob else loser_cell_style

        away_dog: Literal["", "outline: 2px solid gold; outline-offset: -2px;"] = (
            dog_style if away_is_dk_dog else ""
        )
        home_dog: Literal["", "outline: 2px solid gold; outline-offset: -2px;"] = (
            dog_style if home_is_dk_dog else ""
        )

        away_label = escape(away.split(" ")[-1])
        home_label = escape(home.split(" ")[-1])
        game_time_label = escape(game_time)
        away_pct_label = escape(away_pct)
        home_pct_label = escape(home_pct)

        rows_html.append(f"""
        <tr style="{row_style}">
            <td style="text-align:right; font-weight:bold; {away_style}
            {away_dog}">{away_pct_label}</td>
            <td style="text-align:right; font-weight:bold; {away_style}">{away_label}</td>
            <td style="text-align:center; color:#aaa; font-size:0.85em;">{game_time_label}</td>
            <td style="text-align:left; font-weight:bold; {home_style}">{home_label}</td>
            <td style="text-align:left; font-weight:bold; {home_style}
            {home_dog}">{home_pct_label}</td>
        </tr>""")

    html: str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Week {week} Predictions — {escape(year)}</title>
<style>
  body {{ background: #1a1a1a; color: #eee; font-family: 'Segoe UI', Arial, sans-serif;
          display: flex; justify-content: center; padding: 2rem; }}
  h1 {{ text-align: center; margin-bottom: 1.5rem; font-size: 1.4rem; color: #ccc; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 640px; }}
  th {{ padding: 0.5rem 1rem; text-align: center; color: #aaa;
        font-size: 0.9rem; border-bottom: 1px solid #444; }}
  td {{ padding: 0.6rem 1rem; }}
  tr.time-sep td {{ text-align: center; color: #aaa; font-size: 0.8rem;
                    padding: 0.3rem; border-top: 1px solid #333; }}
  tr:not(.time-sep):hover {{ filter: brightness(1.15); }}
  .legend {{ text-align: center; margin-bottom: 1rem; font-size: 0.8rem; color: #888; }}
  .dk-legend {{ display: inline-block; border: 1.5px solid gold;
                padding: 2px 8px; margin-left: 6px; font-size: 0.8rem; color: gold; }}
</style>
</head>
<body>
<div>
  <h1>NFL Week {week} Predictions &mdash; {escape(year)}</h1>
  <p class="legend">
    Elo win probability &nbsp;|&nbsp;
    <span class="dk-legend">DK Underdog</span>
  </p>
  <table>
    <thead>
      <tr>
        <th>Prob.</th>
        <th>Away Team</th>
        <th>Time</th>
        <th>Home Team</th>
        <th>Prob.</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</div>
</body>
</html>"""

    out_dir: Path = resolved_repo / "data" / "output" / "predictions" / year[:4]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = out_dir / f"week_{week:02d}_predictions.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Predictions HTML written: %s", out_path)
    return out_path

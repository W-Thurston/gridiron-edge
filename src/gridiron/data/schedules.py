from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import requests

from gridiron.utils.ids import canon_team, human_game_id
from gridiron.utils.time import ET, UTC
from gridiron.utils.validation import schedules_schema


def _head_ok(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        return r.ok
    except requests.RequestException:
        return False


def _coalesce_kickoff_et(df: pd.DataFrame) -> pd.Series:
    """
    Normalize to tz-aware ET kickoff.
    nflverse schedules typically have 'gameday' (YYYY-MM-DD) and 'gametime' (HH:MM:SS ET),
    or a single timestamp column. We handle both patterns.
    """
    if {"gameday", "gametime"}.issubset(df.columns):
        ts = pd.to_datetime(df["gameday"] + " " + df["gametime"], errors="coerce")
        return ts.dt.tz_localize(ET, nonexistent="NaT", ambiguous="NaT")
    # fallback column names if they ever appear
    for cand in ("start_time", "game_time", "kickoff"):
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], errors="coerce", utc=False)
            return (
                ts.dt.tz_localize(ET, nonexistent="NaT", ambiguous="NaT")
                if ts.dt.tz is None
                else ts.dt.tz_convert(ET)
            )
    raise ValueError("Cannot determine kickoff time columns in schedules data.")


def load_schedules_standardized(
    seasons: Iterable[int],
    csv_url: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    if verbose:
        print(f"[schedules] Reading CSV → {csv_url}")
    df = pd.read_csv(csv_url, low_memory=False)

    # rename/select canon columns from nfldata games.csv
    rename = {
        "season": "season",
        "week": "week",
        "game_type": "game_type",
        "home_team": "home_team",
        "away_team": "away_team",
        "game_id": "game_id_nflfastr",
    }
    present = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=present).copy()

    df["home_team"] = df["home_team"].map(lambda x: canon_team(str(x)))
    df["away_team"] = df["away_team"].map(lambda x: canon_team(str(x)))

    df["kickoff_et"] = _coalesce_kickoff_et(df)
    df["kickoff_utc"] = df["kickoff_et"].dt.tz_convert(UTC)

    # Ensure tz-aware dtype
    if str(df["kickoff_utc"].dtype) == "datetime64[ns]":
        df["kickoff_utc"] = df["kickoff_utc"].dt.tz_localize("UTC")

    df["game_id"] = [
        human_game_id(int(season), int(week), away, home)
        for season, week, away, home in zip(
            df["season"], df["week"], df["away_team"], df["home_team"], strict=False
        )
    ]

    for col in ("home_score", "away_score"):
        if col not in df.columns:
            df[col] = np.nan

    std = (
        df[
            [
                "season",
                "week",
                "game_type",
                "game_id",
                "game_id_nflfastr",
                "home_team",
                "away_team",
                "kickoff_et",
                "kickoff_utc",
                "home_score",
                "away_score",
            ]
        ]
        .sort_values(["season", "week", "kickoff_utc"])
        .reset_index(drop=True)
    )

    std = schedules_schema().validate(std)
    std = std[std["season"].isin({int(y) for y in seasons})].reset_index(drop=True)

    # Drop rows with missing kickoff_utc (can happen when gametime is NaN in the source)
    missing = std["kickoff_utc"].isna().sum()
    if missing:
        print(
            f"[schedules] Warning: dropping {missing} rows with missing kickoff_utc in requested seasons"
        )
    std = std.dropna(subset=["kickoff_utc"]).reset_index(drop=True)

    if verbose:
        print(f"[schedules] Loaded {len(std)} rows for seasons={sorted(set(seasons))}")
    return std

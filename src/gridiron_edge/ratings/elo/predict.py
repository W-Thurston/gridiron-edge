# src/gridiron_edge/ratings/elo/predict.py

"""Elo-based predictions for scheduled NFL games."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.paths import repo_root
from gridiron_edge.datasets.loaders import (
    load_elo_state,
    load_schedule_upcoming,
)
from gridiron_edge.ratings.elo.core import elo_win_probability


class EloPredictionStatus(StrEnum):
    """Availability state for one scheduled Elo prediction."""

    READY = "ready"
    MISSING_AWAY_ELO = "missing_away_elo"
    MISSING_HOME_ELO = "missing_home_elo"
    MISSING_BOTH_ELO = "missing_both_elo"


_SCHEDULE_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "YEAR",
    "WEEK_NUM",
    "GAME_ID",
    "AWAY_TEAM",
    "HOME_TEAM",
)

_ELO_COLUMNS: Final[tuple[str, ...]] = (
    "NFL_TEAM",
    "NFL_YEAR",
    "NFL_WEEK",
    "ELO",
)


def _require_columns(
    frame: DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require a stable input schema."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _validate_elo_identity(
    elo_state: DataFrame,
) -> None:
    """Reject duplicate team, season, and week Elo identities."""
    duplicated = elo_state.duplicated(
        subset=[
            "NFL_TEAM",
            "NFL_YEAR",
            "NFL_WEEK",
        ],
        keep=False,
    )
    if not duplicated.any():
        return

    duplicate_rows = (
        elo_state.loc[
            duplicated,
            [
                "NFL_TEAM",
                "NFL_YEAR",
                "NFL_WEEK",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "NFL_YEAR",
                "NFL_WEEK",
                "NFL_TEAM",
            ],
            kind="stable",
        )
    )

    identities = [
        (f"{row['NFL_TEAM']}/{row['NFL_YEAR']}/{row['NFL_WEEK']}")
        for _, row in duplicate_rows.iterrows()
    ]

    raise ValueError("Elo state contains duplicate identities: " + ", ".join(identities))


def predict_schedule_with_elo(
    schedule: DataFrame,
    elo_state: DataFrame,
    *,
    year: str,
    week: int,
) -> DataFrame:
    """Attach Elo ratings and numeric probabilities to scheduled games.

    Schedule truth is authoritative. Every schedule row in the requested
    season and week remains in the result, including games with missing
    Elo ratings.

    Missing ratings produce null probabilities and an explicit prediction
    status. Ratings are never replaced with a default value.

    Args:
        schedule: Focused upcoming schedule rows.
        elo_state: Elo state keyed by team, season, and week.
        year: NFL season label, such as ``"2026-2027"``.
        week: NFL week number.

    Returns:
        Scoped schedule rows with numeric Elo ratings, numeric complementary
        probabilities, and ``PREDICTION_STATUS``.
    """
    _require_columns(
        schedule,
        _SCHEDULE_IDENTITY_COLUMNS,
        label="Schedule",
    )
    _require_columns(
        elo_state,
        _ELO_COLUMNS,
        label="Elo state",
    )
    _validate_elo_identity(elo_state)

    scoped_schedule = schedule.loc[
        (schedule["YEAR"].astype(str) == year) & (schedule["WEEK_NUM"] == week),
        :,
    ].copy()

    scoped_schedule["_SCHEDULE_ORDER"] = range(len(scoped_schedule))

    away_state = elo_state.rename(
        columns={
            "NFL_TEAM": "AWAY_TEAM",
            "NFL_YEAR": "YEAR",
            "NFL_WEEK": "WEEK_NUM",
            "ELO": "AWAY_TEAM_ELO",
        }
    ).loc[
        :,
        [
            "AWAY_TEAM",
            "YEAR",
            "WEEK_NUM",
            "AWAY_TEAM_ELO",
        ],
    ]

    predicted = scoped_schedule.merge(
        away_state,
        how="left",
        on=[
            "AWAY_TEAM",
            "YEAR",
            "WEEK_NUM",
        ],
        sort=False,
        validate="many_to_one",
    )

    home_state = elo_state.rename(
        columns={
            "NFL_TEAM": "HOME_TEAM",
            "NFL_YEAR": "YEAR",
            "NFL_WEEK": "WEEK_NUM",
            "ELO": "HOME_TEAM_ELO",
        }
    ).loc[
        :,
        [
            "HOME_TEAM",
            "YEAR",
            "WEEK_NUM",
            "HOME_TEAM_ELO",
        ],
    ]

    predicted = predicted.merge(
        home_state,
        how="left",
        on=[
            "HOME_TEAM",
            "YEAR",
            "WEEK_NUM",
        ],
        sort=False,
        validate="many_to_one",
    )

    away_missing = predicted["AWAY_TEAM_ELO"].isna()
    home_missing = predicted["HOME_TEAM_ELO"].isna()

    both_missing = away_missing & home_missing
    away_only_missing = away_missing & ~home_missing
    home_only_missing = ~away_missing & home_missing
    ready = ~away_missing & ~home_missing

    statuses = pd.Series(
        EloPredictionStatus.READY.value,
        index=predicted.index,
        dtype="string",
    )
    statuses = statuses.mask(
        both_missing,
        EloPredictionStatus.MISSING_BOTH_ELO.value,
    )
    statuses = statuses.mask(
        away_only_missing,
        EloPredictionStatus.MISSING_AWAY_ELO.value,
    )
    statuses = statuses.mask(
        home_only_missing,
        EloPredictionStatus.MISSING_HOME_ELO.value,
    )

    probabilities = [
        elo_win_probability(
            float(away_elo),
            float(home_elo),
        )
        if is_ready
        else (
            pd.NA,
            pd.NA,
        )
        for away_elo, home_elo, is_ready in zip(
            predicted["AWAY_TEAM_ELO"],
            predicted["HOME_TEAM_ELO"],
            ready,
            strict=True,
        )
    ]

    predicted["AWAY_WIN_PROB"] = pd.Series(
        [away_probability for away_probability, _ in probabilities],
        index=predicted.index,
        dtype="Float64",
    )
    predicted["HOME_WIN_PROB"] = pd.Series(
        [home_probability for _, home_probability in probabilities],
        index=predicted.index,
        dtype="Float64",
    )
    predicted["PREDICTION_STATUS"] = statuses

    return (
        predicted.sort_values(
            "_SCHEDULE_ORDER",
            kind="stable",
        )
        .drop(columns=["_SCHEDULE_ORDER"])
        .reset_index(drop=True)
    )


def predict_elo_for_week(
    *,
    year: str,
    week: int,
    repo: Path | None = None,
) -> DataFrame:
    """Load schedule and Elo state and predict the requested week."""
    resolved_repo = repo or repo_root()

    schedule = load_schedule_upcoming(resolved_repo)
    elo_state = load_elo_state(resolved_repo)

    return predict_schedule_with_elo(
        schedule,
        elo_state,
        year=year,
        week=week,
    )


def format_elo_prediction_percentages(
    predictions: DataFrame,
) -> DataFrame:
    """Add human-readable percentage columns without recalculating."""
    formatted = predictions.copy()

    formatted["AWAY_TEAM_WIN_PROB"] = formatted["AWAY_WIN_PROB"].map(
        lambda value: pd.NA if pd.isna(value) else f"{float(value) * 100:.1f} %"
    )
    formatted["HOME_TEAM_WIN_PROB"] = formatted["HOME_WIN_PROB"].map(
        lambda value: pd.NA if pd.isna(value) else f"{float(value) * 100:.1f} %"
    )

    return formatted


def predict_elo_only(
    *,
    year: str,
    week: int,
    repo: Path | None = None,
) -> Path:
    """Compute Elo predictions and write a versioned CSV."""
    resolved_repo = repo or repo_root()

    predictions = predict_elo_for_week(
        year=year,
        week=week,
        repo=resolved_repo,
    )
    output = format_elo_prediction_percentages(predictions)

    out_dir = resolved_repo / "data" / "output" / "predictions" / year[:4]
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = out_dir / f"week_{week:02d}_predictions.csv"
    output.to_csv(
        out_path,
        index=False,
    )

    return out_path

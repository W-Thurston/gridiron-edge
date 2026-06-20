# src/gridiron_edge/evaluation/archive.py

"""Prediction archive — append-only log of all model predictions.

Every call to ``gridiron output predictions`` appends to this log so that
predictions can be compared against actual outcomes, evaluated for
calibration, and tracked against closing lines for CLV analysis.

Storage layout:
    data/output/predictions/predictions_log.parquet

Schema:
    predicted_at    datetime64[ns]  UTC timestamp of the prediction run.
                                    Live predictions use the actual run time.
                                    Backfilled predictions use the time of the
                                    backfill run; ``is_backfilled`` is the
                                    canonical flag for distinguishing them.
    is_backfilled   bool            True for historical backfill predictions,
                                    False for live pre-game predictions.
    model_name      str             Model purpose, e.g. "win_prob", "total".
    model_type      str             Model algorithm, e.g. "random_forest",
                                    "xgboost", "logistic", "elo".
    season          str             "2026-2027"
    week            int             NFL week number
    game_id         str             "2026_01_KC_LAC" (canonical GAME_ID)
    game_date       str             "2026-09-05"
    away_team       str             long name ("Kansas City Chiefs")
    home_team       str             long name ("Los Angeles Chargers")
    away_elo        float           Elo rating used for away team
    home_elo        float           Elo rating used for home team
    away_win_prob   float           predicted away win probability [0, 1]
    home_win_prob   float           predicted home win probability [0, 1]
    model_spread    float           NFL point spread (negative = home favored)
    model_total     float           predicted total points
    ...
"""

from __future__ import annotations

import datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings

logger: Logger = logging.getLogger(__name__)

# Ordered columns enforced on every write.
_ARCHIVE_COLUMNS: list[str] = [
    "predicted_at",
    "is_backfilled",
    "model_name",
    "model_type",
    "season",
    "week",
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "away_elo",
    "home_elo",
    "away_win_prob",
    "home_win_prob",
    "model_spread",
    "model_total",
    "projected_home_score",
    "projected_away_score",
    "margin_std",
    "win_prob_lo",
    "win_prob_hi",
    "confidence_tier",
]

# Deduplication key — one prediction per game per (model_name, model_type).
# A later run with the same pair for the same game overwrites the earlier one,
# so predictions are always the most recent for that model/game pair.
_DEDUP_KEY: Final[list[str]] = [
    "game_id",
    "model_name",
    "model_type",
]


def _archive_path(repo: Path | None = None) -> Path:
    """Return the predictions log path, creating the directory if needed.

    Args:
        repo: Repository root. Defaults to ``get_settings().repo_root``.

    Returns:
        Absolute path to ``data/output/predictions/predictions_log.parquet``.
    """
    root: Path = repo or get_settings().repo_root
    directory: Path = root / "data" / "output" / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "predictions_log.parquet"


def build_archive_rows(
    df_predictions: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    season: str,
    week: int,
    predicted_at: datetime.datetime | None = None,
    is_backfilled: bool = False,
) -> pd.DataFrame:
    """Convert a predictions DataFrame into archive-schema rows.

    Args:
        df_predictions: Output of ``build_predictions_df()`` — contains
            ``GAME_ID``, ``GAME_DATE``, ``AWAY_TEAM``, ``HOME_TEAM``,
            ``AWAY_TEAM_ELO``, ``HOME_TEAM_ELO``, ``AWAY_WIN_PROB``,
            ``HOME_WIN_PROB``.
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``, ``"elo"``).
        season: NFL season label (e.g. ``"2026-2027"``).
        week: NFL week number.
        predicted_at: UTC timestamp of the prediction run. Defaults to now.
        is_backfilled: ``True`` for historical backfill predictions,
            ``False`` for live pre-game predictions.

    Returns:
        DataFrame conforming to ``_ARCHIVE_COLUMNS``.
    """
    ts = predicted_at or datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None)

    rows = pd.DataFrame(
        {
            "predicted_at": ts,
            "is_backfilled": is_backfilled,
            "model_name": model_name,
            "model_type": model_type,
            "season": season,
            "week": week,
            "game_id": df_predictions["GAME_ID"],
            "game_date": df_predictions.get("GAME_DATE", pd.Series([""] * len(df_predictions))),
            "away_team": df_predictions["AWAY_TEAM"],
            "home_team": df_predictions["HOME_TEAM"],
            "away_elo": df_predictions["AWAY_TEAM_ELO"],
            "home_elo": df_predictions["HOME_TEAM_ELO"],
            "away_win_prob": df_predictions["AWAY_WIN_PROB"],
            "home_win_prob": df_predictions["HOME_WIN_PROB"],
        }
    )

    # Enrichment columns — filled by enrich_predictions() at prediction
    # time. Default to NaN / empty for backward compatibility with callers
    # that don't enrich before archiving.
    for col in _ARCHIVE_COLUMNS:
        if col not in rows.columns:
            rows[col] = float("nan") if col != "confidence_tier" else ""

    return rows.loc[:, _ARCHIVE_COLUMNS].reset_index(drop=True)


def write_archive_rows(
    new_rows: pd.DataFrame,
    *,
    repo: Path | None = None,
) -> Path:
    """Write pre-built archive rows to the prediction log.

    Low-level function used by both ``append_to_prediction_log`` (single
    week) and bulk backfill operations. Deduplicates on
    ``(game_id, model_name, model_type)`` — the most recently written row
    wins.

    Args:
        new_rows: DataFrame already conforming to ``_ARCHIVE_COLUMNS``.
        repo: Repository root. Defaults to ``get_settings().repo_root``.

    Returns:
        Absolute path to the archive file.
    """
    path: Path = _archive_path(repo)

    if path.exists():
        existing: DataFrame = pd.read_parquet(path)
        mask = existing.set_index(_DEDUP_KEY).index.isin(new_rows.set_index(_DEDUP_KEY).index)
        existing = existing.loc[~mask].copy()
        combined: DataFrame = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()

    combined = combined.sort_values(
        ["season", "week", "game_id", "model_name", "model_type"]
    ).reset_index(drop=True)

    combined.to_parquet(path, index=False)
    logger.info(
        "Prediction archive: %d total rows (%d new) → %s",
        len(combined),
        len(new_rows),
        path,
    )
    return path


def append_to_prediction_log(
    df_predictions: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    season: str,
    week: int,
    predicted_at: datetime.datetime | None = None,
    is_backfilled: bool = False,
    repo: Path | None = None,
) -> Path:
    """Append predictions to the archive log.

    Converts a raw predictions DataFrame (output of
    ``build_predictions_df()``) into archive rows and delegates to
    ``write_archive_rows()``. Deduplicates on
    ``(game_id, model_name, model_type)`` so re-running predictions for
    the same week overwrites rather than duplicates.

    Args:
        df_predictions: Output of ``build_predictions_df()``.
        model_name: Model purpose (e.g. ``"win_prob"``).
        model_type: Model algorithm (e.g. ``"random_forest"``, ``"elo"``).
        season: NFL season label (e.g. ``"2026-2027"``).
        week: NFL week number.
        predicted_at: UTC timestamp. Defaults to now.
        is_backfilled: ``True`` for historical backfill predictions,
            ``False`` for live pre-game predictions.
        repo: Repository root. Defaults to ``get_settings().repo_root``.

    Returns:
        Absolute path to the archive file.
    """
    if df_predictions.empty:
        logger.warning("Prediction archive: empty DataFrame — nothing written.")
        return _archive_path(repo)

    new_rows: DataFrame = build_archive_rows(
        df_predictions,
        model_name=model_name,
        model_type=model_type,
        season=season,
        week=week,
        predicted_at=predicted_at,
        is_backfilled=is_backfilled,
    )
    return write_archive_rows(new_rows, repo=repo)


def load_prediction_log(
    *,
    season: str | None = None,
    week: int | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    repo: Path | None = None,
) -> pd.DataFrame:
    """Load the prediction archive with optional filters.

    Args:
        season: Filter to a specific season (e.g. ``"2026-2027"``).
        week: Filter to a specific week number.
        model_name: Filter to a specific model purpose (e.g. ``"win_prob"``).
        model_type: Filter to a specific model algorithm
            (e.g. ``"random_forest"``).
        repo: Repository root. Defaults to ``get_settings().repo_root``.

    Returns:
        Archive DataFrame. Empty DataFrame (with correct columns) if no
        archive exists yet.
    """
    path: Path = _archive_path(repo)
    if not path.exists():
        return pd.DataFrame(columns=_ARCHIVE_COLUMNS)

    df = pd.read_parquet(path)

    # Backward compat: add enrichment columns if archive predates them.
    for col in _ARCHIVE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan") if col != "confidence_tier" else ""

    if season is not None:
        df = df.loc[df["season"] == season]
    if week is not None:
        df = df.loc[df["week"] == week]
    if model_name is not None:
        df = df.loc[df["model_name"] == model_name]
    if model_type is not None:
        df = df.loc[df["model_type"] == model_type]

    # pyrefly: ignore [bad-return]
    return df.reset_index(drop=True)

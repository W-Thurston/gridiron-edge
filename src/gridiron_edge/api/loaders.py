# src/gridiron_edge/api/loaders.py

"""Thin loader wrappers for API consumption.

Bridges domain data loaders (datasets/, betting/, evaluation/) into
FastAPI-friendly functions that take Settings and return DataFrames or
domain objects.

Per D19, every wrapper passes `settings.repo_root` explicitly to the
underlying loader — the domain-loader fallback to `get_settings()` is
never used from the API path.
"""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import Settings


def load_bets_df(settings: Settings, *, status: str | None = None) -> pd.DataFrame:
    """Return bets from the ledger, optionally filtered by status."""
    from gridiron_edge.betting.ledger import load_bets

    return load_bets(status=status, repo=settings.repo_root)


def load_bankroll_txns_df(
    settings: Settings,
    *,
    txn_type: str | None = None,
) -> pd.DataFrame:
    """Return the raw bankroll transaction log."""
    from gridiron_edge.betting.bankroll import load_transactions

    return load_transactions(txn_type=txn_type, repo=settings.repo_root)


def load_bankroll_history_df(settings: Settings) -> pd.DataFrame:
    """Return the running-balance curve as a DataFrame.

    Columns: timestamp, txn_type, amount, signed_amount, running_balance.
    """
    from gridiron_edge.betting.bankroll import balance_history

    return balance_history(repo=settings.repo_root)


def load_current_bankroll(settings: Settings) -> float:
    """Return the current bankroll balance."""
    from gridiron_edge.betting.bankroll import current_balance

    return current_balance(repo=settings.repo_root)


def resolve_current_week(settings: Settings) -> tuple[int, int, str]:
    """Resolve the current NFL (season, week) from the schedule.

    Returns (season, week, source_label). If the schedule is unavailable
    or empty, falls back to (current_nfl_season(), 1, "fallback").
    """
    from gridiron_edge.core.settings import current_nfl_season
    from gridiron_edge.datasets.loaders import load_schedule_upcoming

    season = current_nfl_season()

    try:
        schedule = load_schedule_upcoming(settings.repo_root)
    except FileNotFoundError:
        return (season, 1, "fallback")

    if schedule.empty:
        return (season, 1, "fallback")

    # The schedule is upcoming games; the first row after sorting is the
    # nearest upcoming week, which we treat as "current" for scheduling.
    # NOTE: column names assumed to be `season` and `week`. If the actual
    # columns differ, this is the one line to update.
    sort_cols = [c for c in ("season", "week") if c in schedule.columns]
    if not sort_cols:
        return (season, 1, "fallback")

    latest = schedule.sort_values(sort_cols).iloc[0]
    return (int(latest["season"]), int(latest["week"]), "schedule")


def load_evaluation_df(
    settings: Settings,
    *,
    model_name: str | None = None,
    model_type: str | None = None,
    season: str | None = None,
) -> DataFrame:
    """Return the evaluation DataFrame (predictions joined to outcomes)."""
    from gridiron_edge.evaluation.metrics import build_evaluation_df

    return build_evaluation_df(
        model_name=model_name,
        model_type=model_type,
        season=season,
        repo=settings.repo_root,
    )


def load_games_df(settings: Settings) -> pd.DataFrame:
    """Return the cleaned historical games table."""
    from gridiron_edge.datasets.loaders import load_games

    return load_games(settings.repo_root)


def load_elo_state_df(settings: Settings) -> pd.DataFrame:
    """Return the full Elo history (team, season, week, ELO)."""
    from gridiron_edge.datasets.loaders import load_elo_state

    return load_elo_state(settings.repo_root)


def load_team_name_map(settings: Settings) -> dict[str, str]:
    """Return the long → short team name mapping as a dict.

    Example: {"Baltimore Ravens": "BAL", "Kansas City Chiefs": "KC", ...}
    """
    from gridiron_edge.datasets.loaders import load_teams_long_short

    df = load_teams_long_short(settings.repo_root)
    return dict(zip(df["NFL_LONG_NAME"], df["NFL_SHORT_NAME"], strict=True))


def resolve_current_season_week(settings: Settings) -> tuple[str, int]:
    """Resolve the current (season, week) from the games table.

    Uses the latest completed game. Returns ("", 0) if games is empty.
    """
    from gridiron_edge.datasets.loaders import load_games

    games = load_games(settings.repo_root)
    if games.empty:
        return ("", 0)

    games_sorted = games.sort_values(["YEAR", "WEEK_NUM"])
    latest = games_sorted.iloc[-1]
    return (str(latest["YEAR"]), int(latest["WEEK_NUM"]))


def load_projections_summary_df(
    settings: Settings,
) -> tuple[pd.DataFrame, str | None]:
    """Load the projections summary CSV.

    Returns:
        Tuple of (dataframe, csv_mtime_iso). The mtime is the CSV file's
        last-modified time as an ISO string, useful for staleness display.
        Returns (empty_df, None) if the CSV doesn't exist.
    """
    from datetime import UTC, datetime

    path = settings.repo_root / "data" / "output" / "temp" / "projections_summary.csv"

    if not path.exists():
        return pd.DataFrame(), None

    df = pd.read_csv(path)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    return df, mtime


def load_games_for_week(
    settings: Settings,
    *,
    season: str,
    week: int,
) -> pd.DataFrame:
    """Load champion-model predictions for all games in (season, week).

    Filters the prediction archive to the current win_prob champion's
    output. Joins to the games table for schedule truth (game_date,
    game_time, venue) and converts team names to short codes.

    Args:
        settings: API settings, source of repo_root.
        season: Season label, e.g. "2026-2027".
        week: Week number.

    Returns:
        DataFrame with one row per game. Columns:
            game_id, game_date, week, season,
            away_team, home_team (short codes),
            home_win_prob, away_win_prob,
            model_spread, model_total,
            projected_home_score, projected_away_score,
            confidence_tier,
            win_prob_lo, win_prob_hi (uncertainty band).
        Empty DataFrame if no games match.

    Raises:
        ChampionNotFoundError: If the champion manifest is missing or
            has no win_prob entry.
    """
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.evaluation.champion_resolver import resolve_current_champion

    _, model_type = resolve_current_champion("win_prob", repo=settings.repo_root)

    archive: DataFrame = load_prediction_log(
        season=season,
        week=week,
        model_name="win_prob",
        model_type=model_type,
        repo=settings.repo_root,
    )

    if archive.empty:
        return archive

    games: DataFrame = load_games_df(settings)
    long_to_short: dict[str, str] = load_team_name_map(settings)

    return _finalize_games_frame(archive, games, long_to_short)


def load_game(
    settings: Settings,
    *,
    game_id: str,
) -> dict | None:
    """Load champion-model prediction for one game.

    Same champion-filtering and enrichment as ``load_games_for_week``,
    but returns a dict for a single game rather than a DataFrame.

    Args:
        settings: API settings, source of repo_root.
        game_id: Composite game_id, e.g. "2026_01_KC_LAC".

    Returns:
        Dict of the fields listed in ``load_games_for_week``'s docstring,
        or ``None`` if the game_id is not in the archive.

    Raises:
        ChampionNotFoundError: If the champion manifest is missing or
            has no win_prob entry.
    """
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.evaluation.champion_resolver import resolve_current_champion

    _, model_type = resolve_current_champion("win_prob", repo=settings.repo_root)

    archive: DataFrame = load_prediction_log(
        model_name="win_prob",
        model_type=model_type,
        repo=settings.repo_root,
    )
    if archive.empty:
        return None
    archive = archive.loc[archive["game_id"] == game_id, :].copy()

    if archive.empty:
        return None

    games: DataFrame = load_games_df(settings)
    long_to_short: dict[str, str] = load_team_name_map(settings)

    enriched: DataFrame = _finalize_games_frame(archive, games, long_to_short)
    if enriched.empty:
        return None

    return enriched.iloc[0].to_dict()


def load_edges_for_week(
    settings: Settings,
    *,
    season: str,
    week: int,
    min_ev: float = 0.0,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
) -> pd.DataFrame:
    """Load ranked edges for (season, week) using the champion model.

    Resolves the current win_prob champion, loads its predictions
    filtered by (season, week), joins to the current DK odds snapshot,
    computes edges via ``market.recommendations.build_edge_report``, and
    ranks by EV. Team names are converted to short codes.

    Args:
        settings: API settings, source of repo_root.
        season: Season label, e.g. "2026-2027".
        week: Week number.
        min_ev: Minimum EV threshold. Rows with ev <= min_ev are excluded.
        bankroll: Bankroll for Kelly stake sizing.
        kelly_multiplier: Fraction of full Kelly (e.g. 0.25 for quarter).

    Returns:
        DataFrame with columns from ``_REPORT_COLUMNS`` in
        ``market.recommendations``, ranked by EV descending. Team name
        columns hold short codes.

    Raises:
        ChampionNotFoundError: If the champion manifest is missing or
            has no win_prob entry.
        OddsUnavailableError: If the current odds snapshot is missing
            or empty.
    """
    from gridiron_edge.api.exceptions import OddsUnavailableError
    from gridiron_edge.evaluation.archive import load_prediction_log
    from gridiron_edge.evaluation.champion_resolver import resolve_current_champion
    from gridiron_edge.ingest.odds.store import load_current_odds
    from gridiron_edge.market.recommendations import build_edge_report, rank_edges
    from gridiron_edge.models.game_prediction.post_process import (
        get_margin_std,
        get_total_std,
    )

    _, model_type = resolve_current_champion("win_prob", repo=settings.repo_root)

    predictions: DataFrame = load_prediction_log(
        season=season,
        week=week,
        model_name="win_prob",
        model_type=model_type,
        repo=settings.repo_root,
    )
    if predictions.empty:
        return pd.DataFrame()

    odds: DataFrame | None = load_current_odds(repo=settings.repo_root)
    if odds is None or odds.empty:
        raise OddsUnavailableError(
            f"No current odds snapshot at "
            f"{settings.repo_root}/data/odds/dk_odds_current.parquet. "
            f"Run `gridiron ingest fetch-odds` to refresh."
        )

    margin_std: float = get_margin_std("win_prob", model_type)
    total_std: float = get_total_std("total", model_type, default=13.0)

    edge_report: DataFrame = build_edge_report(
        predictions,
        odds,
        margin_std=margin_std,
        total_std=total_std,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
    )
    if edge_report.empty:
        return pd.DataFrame()

    ranked: DataFrame = rank_edges(edge_report, min_ev=min_ev)
    if ranked.empty:
        return pd.DataFrame()

    long_to_short: dict[str, str] = load_team_name_map(settings)
    ranked["away_team"] = ranked["away_team"].map(long_to_short).fillna(ranked["away_team"])
    ranked["home_team"] = ranked["home_team"].map(long_to_short).fillna(ranked["home_team"])

    return ranked


def _finalize_games_frame(
    archive: pd.DataFrame,
    games: pd.DataFrame,
    long_to_short: dict[str, str],
) -> pd.DataFrame:
    """Convert archive rows to the API-facing games shape.

    Shared by ``load_games_for_week`` and ``load_game``. Joins the
    archive to the games table for schedule truth, converts long team
    names to short codes, and selects the API-relevant columns.
    """
    # Join to games for schedule truth. Left join preserves archive
    # rows that might reference upcoming games not yet in the games
    # table (games table is populated post-week).
    merged: DataFrame = archive.merge(
        games[["GAME_ID", "YEAR", "WEEK_NUM", "GAME_DATE"]],
        left_on="game_id",
        right_on="GAME_ID",
        how="left",
        suffixes=("", "_games"),
    )

    # Prefer games table for game_date when available (schedule truth);
    # fall back to archive's game_date.
    merged["game_date"] = merged["GAME_DATE"].fillna(merged["game_date"])

    # Team name conversion to short codes.
    merged["away_team"] = merged["away_team"].map(long_to_short).fillna(merged["away_team"])
    merged["home_team"] = merged["home_team"].map(long_to_short).fillna(merged["home_team"])

    columns: list[str] = [
        "game_id",
        "game_date",
        "week",
        "season",
        "away_team",
        "home_team",
        "home_win_prob",
        "away_win_prob",
        "model_spread",
        "model_total",
        "projected_home_score",
        "projected_away_score",
        "confidence_tier",
        "win_prob_lo",
        "win_prob_hi",
    ]
    return merged.loc[:, columns].copy()

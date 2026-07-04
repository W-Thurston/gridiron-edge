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

from datetime import UTC, datetime
from pathlib import Path

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


def compute_elo_deltas(
    elo_state: DataFrame,
    long_to_short: dict[str, str],
) -> DataFrame:
    """Compute per-team Elo delta from prior NFL week within same season.

    For the latest (season, week) in the Elo state table, computes:
        elo_delta = current_week_elo - prior_week_elo

    Prior week is `NFL_WEEK - 1` within the same `NFL_YEAR`. Teams with
    no prior-week Elo (Week 1 of a season, or fresh checkouts) get null.

    The Elo state table stores long team names (e.g. "Arizona Cardinals").
    This function converts to short codes (e.g. "ARI") using the provided
    map for join compatibility with the projections summary CSV.

    Args:
        elo_state: DataFrame with columns NFL_TEAM (long), NFL_YEAR, NFL_WEEK, ELO.
        long_to_short: Mapping from long team names to short codes.

    Returns:
        DataFrame with columns team_abbr, elo_delta. One row per team.
        Empty if elo_state is empty.
    """
    if elo_state.empty:
        return pd.DataFrame(columns=["team_abbr", "elo_delta"])

    # Resolve latest (season, week) — max NFL_WEEK for latest NFL_YEAR.
    latest_year = str(elo_state["NFL_YEAR"].max())
    year_rows = elo_state.loc[elo_state["NFL_YEAR"] == latest_year, :]
    latest_week = int(year_rows["NFL_WEEK"].max())

    # Week 1 → no prior week within same season → return null deltas.
    if latest_week == 1:
        current = year_rows.loc[
            year_rows["NFL_WEEK"] == latest_week,
            ["NFL_TEAM"],
        ].copy()
        current["team_abbr"] = current["NFL_TEAM"].map(long_to_short).fillna(current["NFL_TEAM"])
        current["elo_delta"] = None
        return current.loc[:, ["team_abbr", "elo_delta"]].copy()

    # Current-week Elo.
    current = year_rows.loc[
        year_rows["NFL_WEEK"] == latest_week,
        ["NFL_TEAM", "ELO"],
    ].rename(columns={"ELO": "current_elo"})

    # Prior-week Elo.
    prior = year_rows.loc[
        year_rows["NFL_WEEK"] == latest_week - 1,
        ["NFL_TEAM", "ELO"],
    ].rename(columns={"ELO": "prior_elo"})

    # Join on long team name.
    merged = current.merge(prior, on="NFL_TEAM", how="left")
    merged["elo_delta"] = merged["current_elo"] - merged["prior_elo"]

    # Convert to short codes.
    merged["team_abbr"] = merged["NFL_TEAM"].map(long_to_short).fillna(merged["NFL_TEAM"])

    return merged.loc[:, ["team_abbr", "elo_delta"]].copy()


def load_projections_summary_df(
    settings: Settings,
) -> tuple[pd.DataFrame, str | None]:
    """Load the projections summary CSV, joined with per-team Elo deltas.

    Reads projections_summary.csv and joins per-team Elo delta from the
    Elo state table (prior NFL week within same season). Populates the
    ``week_over_week_delta`` column on the returned DataFrame.

    Returns:
        Tuple of (dataframe, csv_mtime_iso). The mtime is the projections
        CSV file's last-modified time as an ISO string, useful for
        staleness display. Returns (empty_df, None) if the CSV doesn't
        exist.

    Notes:
        Teams without a prior-week Elo entry (Week 1 of a season, or
        fresh checkouts without historical data) get null for
        ``week_over_week_delta``. Frontend renders these as em-dashes.
    """
    path: Path = settings.repo_root / "data" / "output" / "temp" / "projections_summary.csv"

    if not path.exists():
        return pd.DataFrame(), None

    df = pd.read_csv(path)
    mtime: str = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()

    # Compute per-team Elo deltas and join.
    elo_state: DataFrame = load_elo_state_df(settings)
    long_to_short: dict[str, str] = load_team_name_map(settings)
    deltas: DataFrame = compute_elo_deltas(elo_state, long_to_short)

    if deltas.empty:
        df["week_over_week_delta"] = None
    else:
        df = df.merge(
            deltas,
            left_on="TEAM",
            right_on="team_abbr",
            how="left",
        )
        df["week_over_week_delta"] = df["elo_delta"]
        df = df.drop(columns=["team_abbr", "elo_delta"], errors="ignore")

    return df, mtime


def load_team_percentiles_df(
    settings: Settings,
) -> pd.DataFrame:
    """Load the latest team percentile artifact.

    Reads the most recent file from ``data/output/rankings/percentiles/``.
    Returns empty DataFrame if no artifact exists yet (e.g., before the
    first sim run has completed).

    Returns:
        DataFrame with columns team_abbr, season, week, rating_pct,
        avg_wins_pct, make_playoffs_pct, win_sb_pct.
    """
    from gridiron_edge.evaluation.percentiles import load_latest_team_percentiles

    return load_latest_team_percentiles(settings.repo_root)


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


def _parse_season_int(season: str) -> int:
    """Parse the leading int year from a season label.

    Examples:
        "2026-2027" -> 2026
        "2026"      -> 2026
    """
    try:
        return int(season.split("-")[0])
    except (ValueError, IndexError, AttributeError) as exc:
        msg: str = f"Cannot parse season {season!r}; expected 'YYYY' or 'YYYY-YYYY+1'."
        raise ValueError(msg) from exc


def load_props_for_week(
    settings: Settings,
    *,
    season: str,
    week: int,
    stat_type: str | None = None,
    position: str | None = None,
) -> pd.DataFrame:
    """Load champion-model prop predictions for (season, week).

    Iterates registered prop stat families (or just ``stat_type`` when
    provided), resolves each family's current champion via
    :func:`resolve_current_champion`, and filters the prop archive to
    that champion's rows. Families without a resolved champion are
    silently skipped.

    Args:
        settings: API settings, source of repo_root.
        season: Season label, e.g. "2026-2027".
        week: Week number.
        stat_type: Optional single-family filter. When set, only this
            family is processed.
        position: Optional position filter applied to the returned
            rows (e.g. "QB", "RB").

    Returns:
        DataFrame with columns from ``_ARCHIVE_COLUMNS`` in
        ``evaluation.prop_archive``, filtered to the champion for each
        family. Empty DataFrame if no families produced rows after
        filtering.

    Raises:
        ChampionNotFoundError: If zero families resolved a champion.
            Not raised when families resolve but their archive is empty.
    """
    from gridiron_edge.evaluation.champion_resolver import (
        ChampionNotFoundError,
        resolve_current_champion,
    )
    from gridiron_edge.evaluation.prop_archive import load_prop_archive
    from gridiron_edge.models.catalog import PROP_STAT_FAMILIES

    season_int: int = _parse_season_int(season)
    families: list[str] = [stat_type] if stat_type else PROP_STAT_FAMILIES

    resolved_frames: list[pd.DataFrame] = []
    resolved_families: list[str] = []

    for family in families:
        try:
            _, model_type = resolve_current_champion(
                family,
                repo=settings.repo_root,
            )
        except ChampionNotFoundError:
            continue

        family_rows: DataFrame = load_prop_archive(
            repo=settings.repo_root,
            stat_type=family,
            season=season_int,
        )
        if family_rows.empty:
            resolved_families.append(family)
            continue

        filtered = family_rows.loc[
            (family_rows["model_type"] == model_type) & (family_rows["week"] == week),
            :,
        ]
        if position is not None:
            filtered = filtered.loc[filtered["position"] == position, :]

        resolved_families.append(family)
        if not filtered.empty:
            resolved_frames.append(filtered)

    if not resolved_families:
        raise ChampionNotFoundError(
            "No prop champions registered. Run `gridiron full-retrain` or "
            "`gridiron props champion --write-manifest`."
        )

    if not resolved_frames:
        return pd.DataFrame()

    return pd.concat(resolved_frames, ignore_index=True)


def load_prop(
    settings: Settings,
    *,
    game_id: str,
    player_id: str,
    stat_type: str,
) -> dict | None:
    """Load champion-model prop prediction for one (game_id, player_id, stat_type).

    Args:
        settings: API settings, source of repo_root.
        game_id: Composite game_id.
        player_id: Player identifier.
        stat_type: Prop stat family (e.g. "qb_pass_yards").

    Returns:
        Dict with archive-schema fields, or None if the composite
        doesn't match any archived prediction under the current champion.

    Raises:
        ChampionNotFoundError: If the champion manifest has no entry
            for ``stat_type``.
    """
    from gridiron_edge.evaluation.champion_resolver import resolve_current_champion
    from gridiron_edge.evaluation.prop_archive import load_prop_archive

    _, model_type = resolve_current_champion(
        stat_type,
        repo=settings.repo_root,
    )

    archive: DataFrame = load_prop_archive(
        repo=settings.repo_root,
        stat_type=stat_type,
    )
    if archive.empty:
        return None

    match = archive.loc[
        (archive["game_id"] == game_id)
        & (archive["player_id"] == player_id)
        & (archive["model_type"] == model_type),
        :,
    ]
    if match.empty:
        return None

    return match.iloc[0].to_dict()

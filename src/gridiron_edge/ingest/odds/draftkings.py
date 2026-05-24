# src/gridiron_edge/ingest/odds/draftkings.py

"""DraftKings odds ingestion facade.

Pulls current NFL odds from DraftKings, converts to long format,
appends to the historical odds ledger, and writes a current snapshot
for downstream viz use.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

from pandas import DataFrame

from gridiron_edge.ingest.odds.store import (
    append_to_odds_ledger,
    wide_to_long,
    write_current_odds_snapshot,
)
from gridiron_edge.ingest.pfr.collector_impl import PFR_Data_Collector

logger: Logger = logging.getLogger(__name__)


def fetch_dk_odds(
    *,
    season: str | None = None,
    week: int | None = None,
    repo: Path | None = None,
) -> tuple[Path, Path]:
    """Pull DraftKings odds and persist to ledger + snapshot.

    Fetches current NFL moneyline, spread, and total odds from the
    DraftKings sportsbook API, converts to long format, appends to the
    historical odds ledger, and writes a current snapshot for the
    predictions visualisation.

    Args:
        season: NFL season label (e.g. ``"2026-2027"``).
        week: NFL week number being fetched.
        repo: Repository root path. Defaults to ``get_settings().repo_root``.

    Returns:
        Tuple of ``(ledger_path, snapshot_path)``.
    """
    from gridiron_edge.ingest.nflverse.games import _current_nfl_season

    _curr: int = _current_nfl_season()
    resolved_season: str = season or f"{_curr}-{_curr + 1}"
    resolved_week: int = week or 1  # default to 1; pass explicit week when known
    logger.info("Fetching DraftKings odds for %s week %d", resolved_season, resolved_week)

    collector = PFR_Data_Collector()
    df_wide: DataFrame = collector.pull_dk_sportsbook_odds_refactored()

    if df_wide.empty:
        logger.warning("No DraftKings odds returned for %s week %d", resolved_season, resolved_week)
        return (
            repo / "data" / "odds" / "dk_odds_log.parquet" if repo else Path(),
            repo / "data" / "odds" / "dk_odds_current.parquet" if repo else Path(),
        )

    df_long: DataFrame = wide_to_long(
        df_wide,
        sportsbook="draftkings",
        season=resolved_season,
        week=resolved_week,
    )

    ledger_path: Path = append_to_odds_ledger(df_long, repo=repo)
    snapshot_path: Path = write_current_odds_snapshot(df_long, repo=repo)

    logger.info(
        "DK odds: %d rows written to ledger and snapshot (season=%s week=%d)",
        len(df_long),
        resolved_season,
        resolved_week,
    )
    return ledger_path, snapshot_path

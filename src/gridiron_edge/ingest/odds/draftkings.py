# src/gridiron_edge/ingest/odds/draftkings.py

"""DraftKings NFL odds ingestion.

Pulls current NFL odds from the DraftKings sportsbook API, parses moneyline,
spread, and total markets into a team-oriented long-format DataFrame, and
persists to the historical odds ledger and current snapshot.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pandas import DataFrame
import requests

from gridiron_edge.core.settings import current_nfl_season
from gridiron_edge.ingest.odds.store import (
    append_to_odds_ledger,
    wide_to_long,
    write_current_odds_snapshot,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NFL week window
# ---------------------------------------------------------------------------


def _current_nfl_week_bounds(
    anchor: str = "tuesday",
    tz: str = "America/New_York",
    now: datetime | None = None,
) -> tuple[datetime, datetime, datetime]:
    """Return the start, end, and current local datetimes for the current NFL week.

    The week window runs from the most recent ``anchor`` weekday at 00:00
    through the following ``anchor`` weekday at 00:00. Default anchor is
    Tuesday, matching the NFL's weekly reset cycle.

    Args:
        anchor: Weekday name marking the week boundary. One of
            ``"monday"``, ``"tuesday"``, ..., ``"sunday"``.
        tz: IANA timezone string for localising the window.
        now: Override for the current datetime. Defaults to
            ``datetime.now(tzinfo)`` if ``None``.

    Returns:
        A tuple of ``(week_start_local, week_end_local, now_local)``.
    """
    anchor_idx: int = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }[anchor.lower()]
    tzinfo = ZoneInfo(tz)
    now_local: datetime = (now or datetime.now(tzinfo)).astimezone(tzinfo)
    delta_days: int = (now_local.weekday() - anchor_idx) % 7
    week_start: datetime = (now_local - timedelta(days=delta_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    week_end: datetime = week_start + timedelta(days=7)
    return week_start, week_end, now_local


# ---------------------------------------------------------------------------
# Payload parsing helpers
# ---------------------------------------------------------------------------


def _classify_market(name: str) -> str:
    n: str = (name or "").lower()
    if "money" in n:
        return "moneyline"
    if "spread" in n:
        return "spread"
    if "total" in n:
        return "total"
    return ""


def _norm_display_odds_american(sel: dict) -> int | str | None:
    """Extract and normalise an American odds value from a DraftKings selection dict.

    Tries the keys ``oddsAmerican``, ``price``, ``americanOdds``, and
    ``american`` within the ``displayOdds`` sub-dict, coercing numeric
    strings to ``int`` where possible.

    Args:
        sel: A DraftKings market selection dict potentially containing
            a ``displayOdds`` mapping.

    Returns:
        The normalised odds value as ``int`` or ``str``, or ``None`` if
        no matching key is found.
    """
    for k in ("oddsAmerican", "price", "americanOdds", "american"):
        v = sel.get("displayOdds", {}).get(k)
        if v is not None:
            return (
                int(v)
                if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip("-").isdigit())
                else v
            )
    return None


def _norm_point(sel: dict) -> float | str | None:
    """Extract and normalise a point/spread/total value from a DraftKings selection.

    Tries keys ``line``, ``point``, ``points``, and ``total``, coercing
    to ``float`` where possible.

    Args:
        sel: A DraftKings market selection dict.

    Returns:
        The normalised numeric value as ``float``, the raw value as
        ``str`` if coercion fails, or ``None`` if no key is found.
    """
    for k in ("line", "point", "points", "total"):
        if sel.get(k) is not None:
            try:
                return float(sel[k])
            except Exception:
                return sel[k]
    return None


def _label_lower(sel: dict) -> str:
    """Return the lowercased ``outcomeType`` label from a selection dict.

    Args:
        sel: A DraftKings market selection dict.

    Returns:
        Lowercased outcome type string, or ``""`` if the key is absent.
    """
    return (sel.get("outcomeType") or "").lower()


def _side_from_label(
    lbl: str,
    home_name: str | None,
    away_name: str | None,
) -> str | None:
    """Infer whether a DraftKings outcome label refers to the home or away team.

    Matches against the literal strings ``"home"`` / ``"away"`` first,
    then falls back to substring matching against the team names.

    Args:
        lbl: Lowercased outcome type label from the DraftKings API.
        home_name: Home team name, or ``None`` if unavailable.
        away_name: Away team name, or ``None`` if unavailable.

    Returns:
        ``"home"``, ``"away"``, or ``None`` if the side cannot be determined.
    """
    if "home" in lbl:
        return "home"
    if "away" in lbl:
        return "away"
    if home_name and home_name.lower() in lbl:
        return "home"
    if away_name and away_name.lower() in lbl:
        return "away"
    return None


def _extract_game_lines(payload: dict) -> pd.DataFrame:  # noqa: PLR0912
    """Parse a DraftKings API payload into a per-event lines DataFrame.

    Extracts event metadata (teams, start time) and associated market
    selections (moneyline, spread, total), returning one row per event.

    Args:
        payload: Raw JSON payload from the DraftKings sportsbook API.

    Returns:
        DataFrame with columns: ``event_id``, ``start_time``,
        ``home_team``, ``away_team``, ``ml_home``, ``ml_away``,
        ``spread_value_home``, ``spread_odds_home``, ``spread_value_away``,
        ``spread_odds_away``, ``total_OU_value``, ``over_total_odds``,
        ``under_total_odds``.
    """
    events: dict = {}
    for e in payload.get("events", []):
        home: dict = next(
            (p for p in e.get("participants", []) if p.get("venueRole") == "Home"),
            {},
        )
        away: dict = next(
            (p for p in e.get("participants", []) if p.get("venueRole") == "Away"),
            {},
        )
        events[e["id"]] = {
            "event_id": e["id"],
            "start": e.get("startEventDate") or e.get("startDate"),
            "home": home.get("name"),
            "away": away.get("name"),
        }

    markets_by_event: defaultdict = defaultdict(list)
    for m in payload.get("markets", []):
        markets_by_event[m.get("eventId")].append(m)

    selections_by_market: defaultdict = defaultdict(list)
    for s in payload.get("selections", []):
        link = s.get("marketId") or s.get("parentMarketId")
        if link:
            selections_by_market[link].append(s)
        elif s.get("id"):
            m = re.search(r"(\d{6,})", str(s["id"]))
            if m:
                selections_by_market[m.group(1)].append(s)

    rows: list = []
    for ev_id, ev in events.items():
        row: dict = {
            "event_id": ev_id,
            "start_time": ev["start"],
            "home_team": ev["home"],
            "away_team": ev["away"],
            "ml_home": None,
            "ml_away": None,
            "spread_value_home": None,
            "spread_odds_home": None,
            "spread_value_away": None,
            "spread_odds_away": None,
            "total_OU_value": None,
            "over_total_odds": None,
            "under_total_odds": None,
        }
        for m in markets_by_event.get(ev_id, []):
            kind: str = _classify_market(m.get("name"))
            if kind not in {"moneyline", "spread", "total"}:
                continue
            sels: list = list(selections_by_market.get(m["id"], []))
            if not sels and isinstance(m.get("id"), str) and "_" in m["id"]:
                sels = list(selections_by_market.get(m["id"].split("_")[-1], []))

            for s in sels:
                lbl: str = _label_lower(s)
                display_odds_american: int | str | None = _norm_display_odds_american(s)
                points: float | str | None = _norm_point(s)
                if kind == "moneyline":
                    if lbl == "home":
                        row["ml_home"] = (
                            int(display_odds_american)
                            if display_odds_american is not None
                            else None
                        )
                    elif lbl == "away":
                        row["ml_away"] = (
                            int(display_odds_american)
                            if display_odds_american is not None
                            else None
                        )
                elif kind == "spread":
                    if lbl == "home":
                        row["spread_value_home"] = points
                        row["spread_odds_home"] = display_odds_american
                    elif lbl == "away":
                        row["spread_value_away"] = points
                        row["spread_odds_away"] = display_odds_american
                elif kind == "total":
                    row["total_OU_value"] = points
                    if lbl == "over":
                        row["over_total_odds"] = display_odds_american
                    if lbl == "under":
                        row["under_total_odds"] = display_odds_american

        rows.append(row)
    return pd.DataFrame(rows)


def _event_rows_to_team_rows(df_events: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-event odds rows into a per-team long-form DataFrame.

    Each event produces two rows — one for the home team and one for
    the away team — with all relevant odds fields attached.

    Args:
        df_events: Per-event lines DataFrame from ``_extract_game_lines``.

    Returns:
        Long-form DataFrame with one row per team per event.
    """
    rows: list[dict[str, int | Any]] = []
    for _, r in df_events.iterrows():
        rows.append(
            {
                "team": r["away_team"],
                "opponent": r["home_team"],
                "location": 0,
                "event_id": r["event_id"],
                "start_time": r["start_time"],
                "moneyline": r["ml_away"],
                "spread_value": r["spread_value_away"],
                "spread_odds": r["spread_odds_away"],
                "total_OU_value": r["total_OU_value"],
                "over_total_odds": r["over_total_odds"],
                "under_total_odds": r["under_total_odds"],
            }
        )
        rows.append(
            {
                "team": r["home_team"],
                "opponent": r["away_team"],
                "location": 1,
                "event_id": r["event_id"],
                "start_time": r["start_time"],
                "moneyline": r["ml_home"],
                "spread_value": r["spread_value_home"],
                "spread_odds": r["spread_odds_home"],
                "total_OU_value": r["total_OU_value"],
                "over_total_odds": r["over_total_odds"],
                "under_total_odds": r["under_total_odds"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_dk_odds_wide(
    region: str = "US-SB",
    league_id: str = "88808",
    subcategory_id: str = "4518",
    session: requests.Session | None = None,
    payload_override: dict | None = None,
) -> pd.DataFrame:
    """Pull DraftKings NFL odds and return a team-oriented wide DataFrame.

    Fetches current Moneyline, Spread, and Total markets for the current
    NFL week. Games that have already started are excluded.

    Args:
        region: DraftKings region code (e.g. ``"US-SB"``).
        league_id: DraftKings NFL league identifier.
        subcategory_id: DraftKings subcategory identifier for the desired
            market group.
        session: Optional ``requests.Session`` to reuse.
        payload_override: If provided, skips the network call and parses
            this dict as the API payload. Useful for unit testing.

    Returns:
        Long-form DataFrame with one row per team per upcoming game,
        containing odds for moneyline, spread, and total markets.
    """
    if payload_override is None:
        base: str = f"https://sportsbook-nash.draftkings.com/sites/{region}/api/sportscontent/controldata/league/leagueSubcategory/v1/markets"
        events_q: str = f"$filter=leagueId eq '{league_id}' AND clientMetadata/Subcategories/any(s: s/Id eq '{subcategory_id}')"  # noqa: E501
        markets_q: str = f"$filter=clientMetadata/subCategoryId eq '{subcategory_id}' AND tags/all(t: t ne 'SportcastBetBuilder')"  # noqa: E501
        params: dict[str, str] = {
            "eventsQuery": events_q,
            "marketsQuery": markets_q,
            "include": "Events",
            "entity": "events",
            "format": "json",
        }
        sess = session or requests.Session()
        headers: dict[str, str] = {
            "accept": "application/json",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",  # noqa: E501
            "accept-language": "en-US,en;q=0.9",
        }
        resp = sess.get(base, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    else:
        payload = payload_override

    df_events: DataFrame = _extract_game_lines(payload)
    df_teams: DataFrame = _event_rows_to_team_rows(df_events)
    df_teams = df_teams.sort_values(
        ["start_time", "event_id", "location"],
        ignore_index=True,
    )

    _week_start, week_end, now_local = _current_nfl_week_bounds(
        anchor="tuesday",
        tz="America/New_York",
    )

    # pyrefly: ignore [missing-attribute]
    dt_local = pd.to_datetime(df_teams["start_time"], utc=True).dt.tz_convert(
        "America/New_York",
    )
    mask = (dt_local >= now_local) & (dt_local < week_end)
    df_teams = df_teams.loc[mask].copy()
    df_teams["start_time"] = dt_local.loc[mask].dt.tz_localize(None)
    df_teams = df_teams.sort_values(["start_time", "event_id", "location"], ignore_index=True)

    for c in df_teams.columns:
        if isinstance(df_teams[c].dtype, pd.DatetimeTZDtype):
            df_teams[c] = pd.to_datetime(df_teams[c]).dt.tz_convert(None)  # type: ignore[attr-defined]

    odds_cols: list[str] = [
        c
        for c in df_teams.columns
        if c
        in {
            "event_id",
            "moneyline",
            "ml_away",
            "spread_value",
            "spread_odds",
            "total_OU_value",
            "over_total_odds",
            "under_total_odds",
        }
    ]
    for c in odds_cols:
        if df_teams[c].dtype in (float, int):
            continue
        df_teams[c] = df_teams[c].str.replace("\u2212", "-")
        # pyrefly: ignore [missing-attribute]
        df_teams[c] = pd.to_numeric(df_teams[c], errors="coerce").astype("float")

    return df_teams


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
    _curr: int = current_nfl_season()
    resolved_season: str = season or f"{_curr}-{_curr + 1}"
    resolved_week: int = week or 1
    logger.info("Fetching DraftKings odds for %s week %d", resolved_season, resolved_week)

    df_wide: DataFrame = fetch_dk_odds_wide()

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

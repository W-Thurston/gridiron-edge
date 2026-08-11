"""Parse The Odds API v4 NFL payloads into canonical quote rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame
import requests

from gridiron_edge.ingest.odds.store import (
    QUOTE_COLUMNS,
    append_to_odds_ledger,
    validate_quote_rows,
    write_current_odds_snapshot,
)

THE_ODDS_API_PROVIDER: Final[str] = "the_odds_api"
THE_ODDS_API_SPORT_KEY: Final[str] = "americanfootball_nfl"

_MARKET_NAMES: Final[dict[str, str]] = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "total",
}

_REQUIRED_SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "week",
    "game_id",
    "game_date",
    "away_team",
    "home_team",
)


class OddsPayloadError(ValueError):
    """Raised when a provider payload violates the supported v4 contract."""


def _text(value: object, *, label: str) -> str:
    """Return one required nonempty provider text value."""
    if not isinstance(value, str) or not value.strip():
        raise OddsPayloadError(f"{label} must be a nonempty string.")
    return value.strip()


def _utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    """Return one valid timezone-aware UTC provider timestamp."""
    if not isinstance(value, str | int | float | datetime):
        raise OddsPayloadError(f"{label} must be a valid datetime.")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise OddsPayloadError(f"{label} must be a valid datetime.") from exc
    if pd.isna(timestamp):
        raise OddsPayloadError(f"{label} must be a valid datetime.")
    if timestamp.tzinfo is None:
        raise OddsPayloadError(f"{label} must be timezone-aware UTC.")
    if timestamp.utcoffset() != timedelta(0):
        raise OddsPayloadError(f"{label} must use UTC.")
    return timestamp


def _number(value: object, *, label: str) -> float:
    """Return one finite numeric provider value."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OddsPayloadError(f"{label} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise OddsPayloadError(f"{label} must be finite.")
    return numeric


def _objects(value: object, *, label: str) -> list[Mapping[str, object]]:
    """Return one provider array containing JSON objects only."""
    if not isinstance(value, list):
        raise OddsPayloadError(f"{label} must be an array.")
    if any(not isinstance(item, Mapping) for item in value):
        raise OddsPayloadError(f"{label} must contain objects.")
    return list(value)


def _normalize_team(value: object, *, label: str) -> str:
    """Normalize provider and schedule team names for exact matching."""
    return " ".join(_text(value, label=label).casefold().split())


def _schedule_lookup(
    schedule: DataFrame,
    *,
    season: str,
    week: int,
) -> dict[tuple[str, str], Mapping[object, object]]:
    """Index one canonical weekly schedule by normalized away/home names."""
    missing = sorted(set(_REQUIRED_SCHEDULE_COLUMNS) - set(schedule.columns))
    if missing:
        raise ValueError("Rich schedule is missing required columns: " + ", ".join(missing))

    scoped = schedule.loc[
        (schedule["season"].astype(str) == season) & (schedule["week"] == week),
        :,
    ]
    lookup: dict[tuple[str, str], Mapping[object, object]] = {}
    for row in scoped.to_dict(orient="records"):
        key = (
            _normalize_team(row["away_team"], label="schedule away_team"),
            _normalize_team(row["home_team"], label="schedule home_team"),
        )
        if key in lookup:
            raise ValueError("Rich schedule contains duplicate team matchups in scope.")
        lookup[key] = row
    return lookup


def _side(
    *,
    market: str,
    outcome_name: str,
    away_team: str,
    home_team: str,
) -> str:
    """Map one provider outcome name to a canonical quote side."""
    normalized = " ".join(outcome_name.casefold().split())
    if market in {"moneyline", "spread"}:
        if normalized == away_team:
            return "away"
        if normalized == home_team:
            return "home"
        raise OddsPayloadError("Team outcome does not match the provider event teams.")
    if normalized == "over":
        return "over"
    if normalized == "under":
        return "under"
    raise OddsPayloadError("Total outcome must be Over or Under.")


def parse_the_odds_api_payload(
    payload: object,
    schedule: DataFrame,
    *,
    season: str,
    week: int,
    fetched_at: datetime,
) -> DataFrame:
    """Parse supported pregame NFL odds into the canonical quote contract.

    Events outside the requested canonical schedule scope and events whose
    commence time is not later than ``fetched_at`` are excluded. All supported
    bookmaker rows are preserved independently.
    """
    if not season.strip():
        raise ValueError("season must not be empty.")
    if week < 1:
        raise ValueError("week must be at least 1.")
    observed_at = _utc_timestamp(fetched_at, label="fetched_at")
    events = _objects(payload, label="payload")
    schedule_by_teams = _schedule_lookup(schedule, season=season, week=week)

    rows: list[dict[str, object]] = []
    for event in events:
        if event.get("sport_key") != THE_ODDS_API_SPORT_KEY:
            continue
        event_id = _text(event.get("id"), label="event id")
        commence_time = _utc_timestamp(event.get("commence_time"), label="commence_time")
        if commence_time <= observed_at:
            continue

        away_key = _normalize_team(event.get("away_team"), label="event away_team")
        home_key = _normalize_team(event.get("home_team"), label="event home_team")
        schedule_row = schedule_by_teams.get((away_key, home_key))
        if schedule_row is None:
            continue

        for bookmaker in _objects(event.get("bookmakers"), label="bookmakers"):
            sportsbook = _text(bookmaker.get("key"), label="bookmaker key")
            bookmaker_updated_at = _utc_timestamp(
                bookmaker.get("last_update"),
                label="bookmaker last_update",
            )
            for provider_market in _objects(bookmaker.get("markets"), label="markets"):
                provider_market_key = provider_market.get("key")
                if provider_market_key not in _MARKET_NAMES:
                    continue
                market = _MARKET_NAMES[str(provider_market_key)]
                updated_at_value = provider_market.get("last_update")
                updated_at = (
                    bookmaker_updated_at
                    if updated_at_value is None
                    else _utc_timestamp(updated_at_value, label="market last_update")
                )
                for outcome in _objects(provider_market.get("outcomes"), label="outcomes"):
                    outcome_name = _text(outcome.get("name"), label="outcome name")
                    odds = _number(outcome.get("price"), label="outcome price")
                    if odds == 0:
                        raise OddsPayloadError("outcome price must not be zero.")
                    line = None
                    if market in {"spread", "total"}:
                        line = _number(outcome.get("point"), label="outcome point")
                    rows.append(
                        {
                            "fetched_at": observed_at,
                            "provider": THE_ODDS_API_PROVIDER,
                            "provider_event_id": event_id,
                            "sportsbook": sportsbook,
                            "sportsbook_updated_at": updated_at,
                            "commence_time": commence_time,
                            "is_live": False,
                            "season": season,
                            "week": week,
                            "game_id": str(schedule_row["game_id"]),
                            "game_date": schedule_row["game_date"],
                            "away_team": str(schedule_row["away_team"]),
                            "home_team": str(schedule_row["home_team"]),
                            "market": market,
                            "side": _side(
                                market=market,
                                outcome_name=outcome_name,
                                away_team=away_key,
                                home_team=home_key,
                            ),
                            "odds": odds,
                            "line": line,
                        }
                    )

    frame = DataFrame(rows, columns=list(QUOTE_COLUMNS))
    return validate_quote_rows(frame)


THE_ODDS_API_BASE_URL: Final[str] = "https://api.the-odds-api.com/v4"
THE_ODDS_API_REGIONS: Final[str] = "us"
THE_ODDS_API_MARKETS: Final[str] = "h2h,spreads,totals"


class OddsRequestError(RuntimeError):
    """Raised when The Odds API request or response is unusable."""


@dataclass(frozen=True, slots=True)
class OddsApiUsage:
    """Quota metadata returned in The Odds API response headers."""

    requests_remaining: int | None = None
    requests_used: int | None = None
    request_cost: int | None = None


@dataclass(frozen=True, slots=True)
class OddsApiResponse:
    """Validated provider payload paired with optional quota metadata."""

    payload: list[Mapping[str, object]]
    usage: OddsApiUsage


def _optional_header_int(headers: Mapping[str, str], name: str) -> int | None:
    """Parse one optional nonnegative integer response header."""
    value = headers.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise OddsRequestError(f"Response header {name!r} must be an integer.") from exc
    if parsed < 0:
        raise OddsRequestError(f"Response header {name!r} must not be negative.")
    return parsed


def fetch_the_odds_api_payload(
    *,
    api_key: str,
    session: requests.Session | None = None,
    timeout: float = 15.0,
) -> OddsApiResponse:
    """Fetch current NFL featured-market odds from The Odds API v4."""
    if not api_key.strip():
        raise ValueError("api_key must not be empty.")
    if timeout <= 0:
        raise ValueError("timeout must be positive.")

    client = session or requests.Session()
    url = f"{THE_ODDS_API_BASE_URL}/sports/{THE_ODDS_API_SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": THE_ODDS_API_REGIONS,
        "markets": THE_ODDS_API_MARKETS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        response = client.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload: object = response.json()
    except requests.RequestException as exc:
        raise OddsRequestError("The Odds API request failed.") from exc
    except ValueError as exc:
        raise OddsRequestError("The Odds API response was not valid JSON.") from exc

    events = _objects(payload, label="payload")
    usage = OddsApiUsage(
        requests_remaining=_optional_header_int(
            response.headers,
            "x-requests-remaining",
        ),
        requests_used=_optional_header_int(
            response.headers,
            "x-requests-used",
        ),
        request_cost=_optional_header_int(
            response.headers,
            "x-requests-last",
        ),
    )
    return OddsApiResponse(payload=events, usage=usage)


class OddsIngestError(RuntimeError):
    """Raised when a provider pull cannot safely update quote artifacts."""


class OddsIngestPartialPersistenceError(OddsIngestError):
    """Raised when history persists but current snapshot replacement fails."""


@dataclass(frozen=True, slots=True)
class OddsIngestResult:
    """Summary of one successful current-market ingestion."""

    quote_count: int
    game_count: int
    sportsbook_count: int
    ledger_path: Path
    snapshot_path: Path
    usage: OddsApiUsage


def ingest_the_odds_api_current(
    *,
    api_key: str,
    schedule: DataFrame,
    season: str,
    week: int,
    repo: Path | None = None,
    session: requests.Session | None = None,
    timeout: float = 15.0,
    fetched_at: datetime | None = None,
) -> OddsIngestResult:
    """Fetch, parse, and atomically persist current pregame NFL quotes.

    Request, JSON, payload, and matching failures happen before either quote
    artifact is written. Partial canonical schedule coverage is allowed; a pull
    with no usable matched quote rows is rejected.
    """
    observed_at = fetched_at or datetime.now(UTC)
    response = fetch_the_odds_api_payload(
        api_key=api_key,
        session=session,
        timeout=timeout,
    )
    if not response.payload:
        raise OddsIngestError("The Odds API returned no events; quote artifacts were not updated.")

    quotes = parse_the_odds_api_payload(
        response.payload,
        schedule,
        season=season,
        week=week,
        fetched_at=observed_at,
    )
    if quotes.empty:
        raise OddsIngestError(
            "The Odds API returned no usable matched pregame quotes; "
            "quote artifacts were not updated."
        )

    ledger_path = append_to_odds_ledger(quotes, repo=repo)
    try:
        snapshot_path = write_current_odds_snapshot(quotes, repo=repo)
    except Exception as exc:
        raise OddsIngestPartialPersistenceError(
            "Quote observations were persisted to the historical ledger, "
            "but the current snapshot was not replaced."
        ) from exc
    return OddsIngestResult(
        quote_count=len(quotes),
        game_count=int(quotes["game_id"].nunique()),
        sportsbook_count=int(quotes["sportsbook"].nunique()),
        ledger_path=ledger_path,
        snapshot_path=snapshot_path,
        usage=response.usage,
    )

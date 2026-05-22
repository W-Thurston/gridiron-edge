from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pandas import DataFrame
import pytz
import requests
from requests.adapters import HTTPAdapter
import timezonefinder
from urllib3.util.retry import Retry

from gridiron_edge.core.settings import ensure_data_dirs
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.metrics.travel.geo import to_decimal_degrees


class PFR_Data_Collector:  # noqa: N801
    """Data collector for Pro Football Reference (PFR) and external APIs.

    Handles raw data ingestion via Scrapy spiders, schedule cleaning,
    weather enrichment via OpenWeatherMap, and odds ingestion from
    DraftKings Sportsbook.

    Attributes:
        repo_root: Absolute path to the repository root.
        raw_historical_data_file: Path to raw scraped game results CSV.
        cleaned_historical_data_file: Path to cleaned game results CSV.
        raw_upcoming_schedule_data_file: Path to raw upcoming schedule CSV.
        cleaned_upcoming_schedule_data_file: Path to cleaned schedule CSV.
        ELO_visualization_file: Path to the Elo rankings Excel workbook.
        long_to_short_team_name_file: Path to team name mapping CSV.
        stadium_file: Path to stadium reference CSV.
        weather_file: Path to weather-enriched game data CSV.
    """

    def __init__(self, repo: Path | None = None) -> None:
        settings = ensure_data_dirs()
        root: Path = repo or settings.repo_root
        self.repo_root = root
        self.long_to_short_team_name_file = str(dataset_path(root, "teams_long_short"))
        self.stadium_file = str(dataset_path(root, "stadiums"))
        self.weather_file = str(dataset_path(root, "weather_enriched"))

    @staticmethod
    def _convert_12hour_to_24hour(time_12hour: str) -> str:
        """Convert a 12-hour time string (e.g. ``"1:00PM"``) to 24-hour format.

        Args:
            time_12hour: Time string in ``"%I:%M%p"`` format.

        Returns:
            Time string in ``"%H:%M:%S"`` format.
        """
        in_time: datetime = datetime.strptime(time_12hour, "%I:%M%p")
        out_time: str = datetime.strftime(in_time, "%H:%M:%S")

        return out_time

    @staticmethod
    def _is_time_format(input: str, format: str) -> bool:
        """Check whether a string matches a given ``strptime`` format.

        Args:
            input: The string to test.
            format: A ``datetime.strptime``-compatible format string.

        Returns:
            ``True`` if ``input`` parses successfully under ``format``,
            ``False`` otherwise.
        """
        try:
            datetime.strptime(input, format)
            return True
        except ValueError:
            return False

    def _pull_open_weather_map_data(
        self,
        row: pd.Series,
        tf: timezonefinder.TimezoneFinder,
        session: requests.Session,
        owm_api_key: str,
    ) -> pd.Series:
        """Pull weather data for a single game row from OpenWeatherMap.

        Converts the game's location and time to a UTC Unix timestamp,
        calls the OWM One Call API timemachine endpoint, and appends
        weather fields (temperature, humidity, wind, etc.) to the row.

        Args:
            row: A pandas Series representing one game, containing at minimum
                ``LATITUDE``, ``LONGITUDE``, ``GAME_DATE``, and ``GAMETIME``.
            tf: A ``TimezoneFinder`` instance for timezone resolution.
            session: A ``requests.Session`` configured with retry logic.
            owm_api_key: OpenWeatherMap API key.

        Returns:
            The input ``row`` with weather fields appended, or the original
            row on API failure.
        """
        try:
            ## Make sure Lat & Lon are in decimal format
            lat: float = to_decimal_degrees(row.LATITUDE)
            lon: float = to_decimal_degrees(row.LONGITUDE)

            ## Build a timestamp that is accepted by the API
            tz_name: str | None = tf.certain_timezone_at(lat=lat, lng=lon)
            local = pytz.timezone(tz_name if tz_name is not None else "UTC")

            date: str = f"{row.GAME_DATE.year}-{row.GAME_DATE.month}-{row.GAME_DATE.day}"

            if self._is_time_format(
                f"{date} {self._convert_12hour_to_24hour(row.GAMETIME)}",
                "%Y-%m-%d %H:%M:%S",
            ):
                naive: datetime = datetime.strptime(
                    f"{date} {self._convert_12hour_to_24hour(row.GAMETIME)}",
                    "%Y-%m-%d %H:%M:%S",
                )
            elif self._is_time_format(
                f"{date} {self._convert_12hour_to_24hour(row.GAMETIME)}",
                "%m-%d-%Y %H:%M:%S",
            ):
                naive = datetime.strptime(
                    f"{date} {self._convert_12hour_to_24hour(row.GAMETIME)}",
                    "%m-%d-%Y %H:%M:%S",
                )
            else:
                naive = datetime.strptime(
                    f"{date} {self._convert_12hour_to_24hour(row.GAMETIME)}",
                    "%m/%d/%Y %H:%M:%S",
                )

            local_dt: datetime = local.localize(naive, is_dst=None)
            utc_dt: datetime = local_dt.astimezone(pytz.utc)
            time = int(utc_dt.timestamp())

            ## Build the API's URL
            url: str = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={time}&appid={owm_api_key}"

            ## Pull API's response as JSON
            owm_response = session.get(url).json()

            ## Gather wanted data fields from JSON response
            row["TEMP"] = owm_response["data"][0].get("temp", "NULL_VALUE")
            row["FEELS_LIKE"] = owm_response["data"][0].get("feels_like", "NULL_VALUE")
            row["PRESSURE"] = owm_response["data"][0].get("pressure", "NULL_VALUE")
            row["HUMIDITY"] = owm_response["data"][0].get("humidity", "NULL_VALUE")
            row["DEW_POINT"] = owm_response["data"][0].get("dew_point", "NULL_VALUE")
            row["CLOUDS"] = owm_response["data"][0].get("clouds", "NULL_VALUE")
            row["VISIBILITY"] = owm_response["data"][0].get("visibility", "NULL_VALUE")
            row["WIND_SPEED"] = owm_response["data"][0].get("wind_speed", "NULL_VALUE")
            row["WIND_DEG"] = owm_response["data"][0].get("wind_deg", "NULL_VALUE")
            row["WEATHER_MAIN"] = owm_response["data"][0]["weather"][0].get("main", "NULL_VALUE")
            row["WEATHER_DESC"] = owm_response["data"][0]["weather"][0].get(
                "description", "NULL_VALUE"
            )

            return row
        except Exception:
            print(f"ID: {row.GAME_ID}")
            return row

    def pull_weather_data(self, year: str, owm_api_key: str) -> None:
        """Fetch and append weather data for the most recent completed week.

        Reads the cleaned games and stadium datasets, resolves stadium
        coordinates, and calls the OpenWeatherMap API for each game in
        the most recently completed season week. Appends results to the
        weather-enriched dataset.

        Args:
            year: Season year string (e.g. ``"2024-2025"``).
            owm_api_key: OpenWeatherMap API key.
        """
        ## Read in wk_by_wk data
        df: DataFrame = pd.read_csv(dataset_path(self.repo_root, "games"))
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df.sort_values(["GAME_DATE", "GAMETIME", "GAME_ID"], ascending=True, inplace=True)

        ## Read in stadium data
        df_stadium: DataFrame = pd.read_csv(self.stadium_file)

        ## Merge Stadium Latitude and Longitude on to wk_by_wk data
        temp_df = df.loc[:, ["GAME_ID", "GAME_DATE", "GAMETIME", "YEAR", "STADIUM"]].copy()
        temp_df = temp_df.merge(
            df_stadium.loc[:, ["YEAR", "STADIUM", "LATITUDE", "LONGITUDE"]],
            how="left",
            on=["YEAR", "STADIUM"],
        ).drop_duplicates()
        temp_df.sort_values(
            ["GAME_DATE", "GAMETIME", "GAME_ID"],
            ascending=True,
            inplace=True,
            ignore_index=True,
        )

        ## Reduce temp_df down to just the new week's worth of data
        temp_df = temp_df.iloc[
            df.loc[(df["YEAR"] == year) & (df["WEEK_NUM"] == df.iloc[-1, :]["WEEK_NUM"]), :].index,
            :,
        ]

        ## Create a TimezoneFinder object
        tzf = timezonefinder.TimezoneFinder()

        ## Set up the Requests.Session for pulling data from openweathermap
        sess = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        # pyrefly: ignore [bad-argument-type]
        adapter = HTTPAdapter(max_retries=retry)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)

        ## Pull weather data
        temp_df = temp_df.progress_apply(  # type: ignore[attr-defined]
            lambda x: self._pull_open_weather_map_data(
                row=x, tf=tzf, session=sess, owm_api_key=owm_api_key
            ),
            axis=1,
        )

        ## Append new Weather data to weather data file
        temp_df.drop(
            ["GAME_DATE", "GAMETIME", "YEAR", "STADIUM", "LATITUDE", "LONGITUDE"],
            axis=1,
        ).to_csv(self.weather_file, mode="a", index=False, header=False)

    @staticmethod
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
        # most recent anchor at 00:00
        delta_days: int = (now_local.weekday() - anchor_idx) % 7
        week_start: datetime = (now_local - timedelta(days=delta_days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        week_end: datetime = week_start + timedelta(days=7)
        return week_start, week_end, now_local

    @staticmethod
    def _classify_market(name: str) -> str:
        n: str = (name or "").lower()
        if "money" in n:
            return "moneyline"
        if "spread" in n:
            return "spread"
        if "total" in n:
            return "total"
        return ""

    @staticmethod
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
                    if isinstance(v, (int, float))
                    or (isinstance(v, str) and v.strip("-").isdigit())
                    else v
                )
        return None

    @staticmethod
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

    @staticmethod
    def _label_lower(sel: dict) -> str:
        """Return the lowercased ``outcomeType`` label from a selection dict.

        Args:
            sel: A DraftKings market selection dict.

        Returns:
            Lowercased outcome type string, or ``""`` if the key is absent.
        """
        return (sel.get("outcomeType") or "").lower()

    @staticmethod
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

    def _extract_game_lines(self, payload: dict) -> pd.DataFrame:
        """Parse a DraftKings API payload into a per-event lines DataFrame.

        Extracts event metadata (teams, start time) and associated market
        selections (moneyline, spread, total), returning one row per team
        per market type per event.

        Args:
            payload: Raw JSON payload from the DraftKings sportsbook API.

        Returns:
            DataFrame with columns: ``event_id``, ``start_time``,
            ``home_team``, ``away_team``, ``ml_home``, ``ml_away``,
            ``spread_value_home``, ``spread_odds_home``, ``spread_value_away``,
            ``spread_odds_away``, ``total_OU_value``, ``over_total_odds``,
            ``under_total_odds``.
        """
        # events → {event_id: names/times}
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

        # markets grouped by event
        markets_by_event: defaultdict = defaultdict(list)
        for m in payload.get("markets", []):
            markets_by_event[m.get("eventId")].append(m)

        # selections grouped by market id (with a numeric-suffix fallback)
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
                kind: str = self._classify_market(m.get("name"))
                if kind not in {"moneyline", "spread", "total"}:
                    continue
                sels: list = list(selections_by_market.get(m["id"], []))
                if not sels and isinstance(m.get("id"), str) and "_" in m["id"]:
                    # numeric suffix fallback (e.g., '1_79743033' → '79743033')
                    sels = list(selections_by_market.get(m["id"].split("_")[-1], []))

                for s in sels:
                    lbl: str = self._label_lower(s)
                    display_odds_american: int | str | None = self._norm_display_odds_american(s)
                    points: float | str | None = self._norm_point(s)
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

    def _event_rows_to_team_rows(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """Pivot per-event odds rows into a per-team long-form DataFrame.

        Each event produces two rows — one for the home team and one for
        the away team — with all relevant odds fields attached.

        Args:
            df_events: Per-event lines DataFrame from ``_extract_game_lines``.

        Returns:
            Long-form DataFrame with one row per team per event, containing
            ``team``, ``opponent``, ``location`` (1=home, 0=away),
            ``event_id``, ``start_time``, ``moneyline``, ``spread_value``,
            ``spread_odds``, ``total_OU_value``, ``over_total_odds``, and
            ``under_total_odds``.
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

    def pull_dk_sportsbook_odds_refactored(
        self,
        region: str = "US-SB",
        league_id: str = "88808",
        subcategory_id: str = "4518",
        session: requests.Session | None = None,
        payload_override: dict | None = None,
    ) -> pd.DataFrame:
        """Pull DraftKings NFL odds (Moneyline, Spread, Total) from the sportsbook API.

        Fetches current Moneyline, Spread, and Total markets for the current
        NFL week and returns them as a team-oriented DataFrame. Games that
        have already started are excluded.

        Args:
            region: DraftKings region code (e.g. ``"US-SB"``).
            league_id: DraftKings NFL league identifier.
            subcategory_id: DraftKings subcategory identifier for the desired
                market group (default targets the main NFL betting markets).
            session: Optional ``requests.Session`` to reuse. A new session
                is created if ``None``.
            payload_override: If provided, skips the network call and parses
                this dict as the API payload. Useful for unit testing.

        Returns:
            Long-form DataFrame with one row per team per upcoming game,
            containing odds for moneyline, spread, and total markets.
        """
        if payload_override is None:
            base: str = f"https://sportsbook-nash.draftkings.com/sites/{region}/api/sportscontent/controldata/league/leagueSubcategory/v1/markets"
            events_q: str = f"$filter=leagueId eq '{league_id}' AND clientMetadata/Subcategories/any(s: s/Id eq '{subcategory_id}')"
            markets_q: str = f"$filter=clientMetadata/subCategoryId eq '{subcategory_id}' AND tags/all(t: t ne 'SportcastBetBuilder')"
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
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
                "accept-language": "en-US,en;q=0.9",
            }
            resp = sess.get(base, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
        else:
            payload: dict = payload_override

        df_events: DataFrame = self._extract_game_lines(payload)
        df_teams: DataFrame = self._event_rows_to_team_rows(df_events)
        df_teams = df_teams.sort_values(
            ["start_time", "event_id", "location"],
            ignore_index=True,
        )
        # keep only upcoming games in the current NFL week ---
        _week_start, week_end, now_local = self._current_nfl_week_bounds(
            anchor="tuesday",
            tz="America/New_York",
        )

        # Parse and localize start_time (DraftKings times are ISO, usually UTC/Z)
        # pyrefly: ignore [missing-attribute]
        dt_local = pd.to_datetime(df_teams["start_time"], utc=True).dt.tz_convert(
            "America/New_York",
        )

        # “Coming up this week”: from *now* until the end of the week window
        mask = (dt_local >= now_local) & (dt_local < week_end)
        df_teams = df_teams.loc[mask].copy()

        # If you prefer “all games in the current NFL week (even earlier today)”, use:
        # mask = (dt_local >= week_start) & (dt_local < week_end)

        # Optional: keep the localized datetime for downstream sorting/printing
        df_teams["start_time"] = dt_local.loc[mask].dt.tz_localize(None)

        # Final tidy sort
        df_teams = df_teams.sort_values(["start_time", "event_id", "location"], ignore_index=True)

        # Strip tz from ANY datetime col that might still be tz-aware
        for c in df_teams.columns:
            if isinstance(df_teams[c].dtype, pd.DatetimeTZDtype):
                df_teams[c] = pd.to_datetime(df_teams[c]).dt.tz_convert(None)  # type: ignore[attr-defined]

        # identify all odds columns in your current schema
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

    def pull_dk_sportsbook_odds(self) -> DataFrame:
        """Pull current DraftKings NFL odds using the default endpoint parameters.

        Convenience wrapper around ``pull_dk_sportsbook_odds_refactored``
        using default region, league, and subcategory identifiers.

        Returns:
            Long-form DataFrame with one row per team per upcoming game.
        """
        return self.pull_dk_sportsbook_odds_refactored()

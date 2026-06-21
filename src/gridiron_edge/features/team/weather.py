# src/gridiron_edge/features/team/weather.py

"""Weather and venue features.

Produces game-level environmental context that affects play style and
outcomes — particularly suppression of passing EPA in wind and cold.
All features are game-level (not team-specific), so the same value
appears in both TEAM_A and TEAM_B rows for a given game.

Produces:
    IS_DOME            int    1 if game played in a domed or retractable-roof
                              stadium, 0 otherwise.  Source: ROOF column.
                              Always populated — no OWM data required.
    WIND_SPEED_MPH     float  Wind speed at kickoff in mph.
                              0.0 for dome games (controlled environment).
                              NaN if OWM data unavailable for outdoor games.
                              Source: weather_enriched WIND_SPEED (m/s -> mph).
    TEMP_F             float  Ambient temperature at kickoff in Fahrenheit.
                              72.0 for dome games (standard controlled temp).
                              NaN if OWM data unavailable for outdoor games.
                              Source: weather_enriched TEMP (Kelvin -> degF).
    PRECIP_FLAG        int    1 if precipitation detected, 0 if clear.
                              0 for dome games.
                              NaN if OWM data unavailable for outdoor games.
                              Source: weather_enriched WEATHER_MAIN column.
    FEELS_LIKE_F       float  Feels-like temperature at kickoff in Fahrenheit.
                              72.0 for dome games.
                              NaN if OWM data unavailable for outdoor games.
                              Source: weather_enriched FEELS_LIKE (K -> degF).
    HUMIDITY_PCT       float  Relative humidity at kickoff (0-100 scale).
                              50.0 for dome games.
                              NaN if OWM data unavailable for outdoor games.
                              Source: weather_enriched HUMIDITY.
    VISIBILITY_M       float  Visibility in meters at kickoff.
                              10000.0 for dome games.
                              NaN filled with 10000.0 (clear-sky default).
                              Source: weather_enriched VISIBILITY.
    SNOW_FLAG          int    1 if snow detected, 0 otherwise.
                              0 for dome games.
                              NaN if OWM data unavailable for outdoor games.
                              Source: weather_enriched WEATHER_MAIN.
    LOW_VIS_FLAG       int    1 if fog/mist/haze/smoke detected, 0 otherwise.
                              0 for dome games.
                              NaN if OWM data unavailable for outdoor games.
                              Source: weather_enriched WEATHER_MAIN.
    WIND_CHILL_DELTA   float  TEMP_F minus FEELS_LIKE_F.
                              0.0 for dome games.
                              Positive = wind chill / cold feels worse than temp.
                              NaN if OWM data unavailable for outdoor games.

Design notes:
    - IS_DOME values: "dome"/"retractable" -> 1, "outdoors"/"open" -> 0.
      Retractable treated as dome (conservative -- roof typically closed
      during the weather conditions that matter most for game outcomes).

    - Dome games receive definitional rather than observed values:
        WIND_SPEED_MPH = 0.0    (no wind in a controlled environment)
        TEMP_F         = 72.0   (standard HVAC set-point for NFL domes)
        PRECIP_FLAG    = 0      (no precipitation indoors)
      These override any OWM values that may exist for the location.

    - Outdoor games without OWM coverage remain NaN.  _prepare_data
      excludes NaN feature rows from training, so those games are
      automatically withheld.  As OWM historical coverage expands via
      ``gridiron ingest weather --all-years``, more rows become usable.

    - Unit conversions from OWM raw:
        TEMP:       Kelvin to Fahrenheit  =  (K - 273.15) x 9/5 + 32
        WIND_SPEED: m/s to mph            =  m/s x 2.237
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from gridiron_edge.core.enums import COVERED_STADIUMS
from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# ROOF values that indicate a covered playing surface (stadium has
# a roof at all, regardless of whether it's open on game day). Used
# to override OWM weather data with controlled-environment defaults.
# See :data:`gridiron_edge.core.enums.COVERED_STADIUMS` for the
# canonical semantic grouping.
_DOME_ROOF_VALUES: Final[frozenset[str]] = frozenset(r.value for r in COVERED_STADIUMS)

# OWM WEATHER_MAIN values that indicate precipitation
_PRECIP_WEATHER_MAINS: Final[frozenset[str]] = frozenset(
    {"Rain", "Snow", "Drizzle", "Thunderstorm"}
)

# Unit conversion constants
_KELVIN_TO_CELSIUS: Final[float] = 273.15
_MPS_TO_MPH: Final[float] = 2.23694

# Standard controlled temperature for dome stadiums (degF)
_DOME_TEMP_F: Final[float] = 72.0

# Standard controlled humidity for dome stadiums (%)
_DOME_HUMIDITY_PCT: Final[float] = 50.0

# Standard controlled visibility for dome stadiums (meters)
_DOME_VISIBILITY_M: Final[float] = 10000.0

# Standard controlled feels-like for dome stadiums (degF) — same as temp
_DOME_FEELS_LIKE_F: Final[float] = 72.0

# OWM WEATHER_MAIN values that indicate snow
_SNOW_WEATHER_MAINS: Final[frozenset[str]] = frozenset({"Snow"})

# OWM WEATHER_MAIN values that indicate low-visibility conditions
_LOW_VIS_WEATHER_MAINS: Final[frozenset[str]] = frozenset({"Fog", "Mist", "Haze", "Smoke"})

# Default visibility when OWM returns NaN (clear-sky assumption)
_DEFAULT_VISIBILITY_M: Final[float] = 10000.0

# Columns produced by _process_weather.
_WEATHER_OUTPUT_COLS: Final[list[str]] = [
    "GAME_ID",
    "WIND_SPEED_MPH",
    "TEMP_F",
    "PRECIP_FLAG",
    "FEELS_LIKE_F",
    "HUMIDITY_PCT",
    "VISIBILITY_M",
    "SNOW_FLAG",
    "LOW_VIS_FLAG",
    "WIND_CHILL_DELTA",
]


@FeatureRegistry.register("weather")
class WeatherFeature:
    """Weather and venue features: dome flag, wind, temperature, precipitation.

    IS_DOME is always populated from the stadium reference ROOF column.
    Wind, temperature, and precipitation are populated from OWM data where
    available and set to controlled-environment constants for dome games.
    Outdoor games without OWM coverage receive NaN and are excluded from
    training by _prepare_data.
    """

    spec = FeatureSpec(
        name="weather",
        produces=[
            "IS_DOME",
            "WIND_SPEED_MPH",
            "TEMP_F",
            "PRECIP_FLAG",
            "FEELS_LIKE_F",
            "HUMIDITY_PCT",
            "VISIBILITY_M",
            "SNOW_FLAG",
            "LOW_VIS_FLAG",
            "WIND_CHILL_DELTA",
        ],
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute weather and venue features and join onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame with GAME_ID, YEAR, WEEK_NUM,
                TEAM_A, TEAM_B columns.
            datasets: Provides ``games()`` (for ROOF column) and
                ``weather_enriched()`` (optional OWM data).

        Returns:
            Input DataFrame with weather and venue columns appended:
            IS_DOME, WIND_SPEED_MPH, TEMP_F, PRECIP_FLAG, FEELS_LIKE_F,
            HUMIDITY_PCT, VISIBILITY_M, SNOW_FLAG, LOW_VIS_FLAG, and
            WIND_CHILL_DELTA.  Dome games are fully populated with
            controlled-environment defaults.  Outdoor games without OWM
            data have NaN in the weather columns.
        """
        games = datasets.games()

        # -- IS_DOME from ROOF column (always available) -------------------
        # Vectorized via .isin() instead of per-row apply (weather/M2).
        roof = games[["GAME_ID", "ROOF"]].copy()
        _dome_lower: frozenset[str] = frozenset(v.lower() for v in _DOME_ROOF_VALUES)
        roof["IS_DOME"] = roof["ROOF"].str.lower().str.strip().isin(_dome_lower).astype(int)
        dome_lookup = roof[["GAME_ID", "IS_DOME"]].drop_duplicates("GAME_ID")

        # -- Weather columns from weather_enriched (optional) --------------
        weather_df = self._load_weather(datasets)
        if weather_df is not None and not weather_df.empty:
            weather_lookup = self._process_weather(weather_df)
        else:
            weather_lookup = pd.DataFrame(columns=_WEATHER_OUTPUT_COLS)

        # -- Merge onto modeling DataFrame ---------------------------------
        df = df.merge(dome_lookup, how="left", on="GAME_ID")

        if not weather_lookup.empty:
            df = df.merge(weather_lookup, how="left", on="GAME_ID")
        else:
            df["WIND_SPEED_MPH"] = float("nan")
            df["TEMP_F"] = float("nan")
            df["PRECIP_FLAG"] = float("nan")
            df["FEELS_LIKE_F"] = float("nan")
            df["HUMIDITY_PCT"] = float("nan")
            df["VISIBILITY_M"] = float("nan")
            df["SNOW_FLAG"] = float("nan")
            df["LOW_VIS_FLAG"] = float("nan")
            df["WIND_CHILL_DELTA"] = float("nan")

        # -- Dome games: override with definitional controlled values ------
        # Applied after the OWM merge so they always take precedence.
        dome_mask = df["IS_DOME"] == 1
        df.loc[dome_mask, "WIND_SPEED_MPH"] = 0.0
        df.loc[dome_mask, "TEMP_F"] = _DOME_TEMP_F
        df.loc[dome_mask, "PRECIP_FLAG"] = 0
        df.loc[dome_mask, "FEELS_LIKE_F"] = _DOME_FEELS_LIKE_F
        df.loc[dome_mask, "HUMIDITY_PCT"] = _DOME_HUMIDITY_PCT
        df.loc[dome_mask, "VISIBILITY_M"] = _DOME_VISIBILITY_M
        df.loc[dome_mask, "SNOW_FLAG"] = 0
        df.loc[dome_mask, "LOW_VIS_FLAG"] = 0
        df.loc[dome_mask, "WIND_CHILL_DELTA"] = 0.0

        return df

    def _load_weather(self, datasets: DatasetAccessor) -> pd.DataFrame | None:
        """Load the weather_enriched dataset if it exists."""
        try:
            return datasets.weather_enriched()
        except (AttributeError, FileNotFoundError):
            return None

    def _process_weather(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Convert OWM raw columns to model-ready features.

        Args:
            weather_df: Raw weather_enriched DataFrame with OWM columns.

        Returns:
            DataFrame with columns defined in ``_WEATHER_OUTPUT_COLS``.
            One row per game.
        """
        w = weather_df.copy()

        def _coerce(col: str) -> pd.Series:
            """Return *col* as a numeric Series, or all-NaN if missing."""
            if col in w.columns:
                # pyrefly: ignore [bad-return]
                return pd.to_numeric(w[col], errors="coerce")
            return pd.Series(float("nan"), index=w.index)

        # -- Kelvin -> Fahrenheit conversions ------------------------------
        w["TEMP_F"] = (_coerce("TEMP") - _KELVIN_TO_CELSIUS) * 9 / 5 + 32
        w["FEELS_LIKE_F"] = (_coerce("FEELS_LIKE") - _KELVIN_TO_CELSIUS) * 9 / 5 + 32

        # -- Unit conversions & pass-throughs ------------------------------
        w["WIND_SPEED_MPH"] = _coerce("WIND_SPEED") * _MPS_TO_MPH
        w["HUMIDITY_PCT"] = _coerce("HUMIDITY").astype(float)
        w["VISIBILITY_M"] = _coerce("VISIBILITY").fillna(_DEFAULT_VISIBILITY_M)

        # -- WEATHER_MAIN-derived flags ------------------------------------
        if "WEATHER_MAIN" in w.columns:
            main_str = w["WEATHER_MAIN"].astype(str)
            main_valid = w["WEATHER_MAIN"].notna() & (main_str != "")
            for col, categories in (
                ("PRECIP_FLAG", _PRECIP_WEATHER_MAINS),
                ("SNOW_FLAG", _SNOW_WEATHER_MAINS),
                ("LOW_VIS_FLAG", _LOW_VIS_WEATHER_MAINS),
            ):
                w[col] = np.where(
                    main_valid,
                    main_str.isin(categories).astype(int),
                    float("nan"),
                )
        else:
            w["PRECIP_FLAG"] = float("nan")
            w["SNOW_FLAG"] = float("nan")
            w["LOW_VIS_FLAG"] = float("nan")

        # -- Derived features ----------------------------------------------
        w["WIND_CHILL_DELTA"] = w["TEMP_F"] - w["FEELS_LIKE_F"]

        if "GAME_ID" not in w.columns:
            return pd.DataFrame(columns=_WEATHER_OUTPUT_COLS)

        # pyrefly: ignore [no-matching-overload]
        return w[_WEATHER_OUTPUT_COLS].drop_duplicates("GAME_ID").reset_index(drop=True)

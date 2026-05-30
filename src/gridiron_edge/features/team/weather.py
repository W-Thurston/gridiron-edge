# src/gridiron_edge/features/team/weather.py

"""Weather and venue features.

Produces game-level environmental context that affects play style and
outcomes — particularly suppression of passing EPA in wind and cold.
All features are game-level (not team-specific), so the same value
appears in both TEAM_A and TEAM_B rows for a given game.

Produces:

    IS_DOME         int     1 if game played in a domed or retractable-roof
                            stadium, 0 otherwise.  Source: ROOF column.
                            Always populated — no OWM data required.

    WIND_SPEED_MPH  float   Wind speed at kickoff in mph.
                            0.0 for dome games (controlled environment).
                            NaN if OWM data unavailable for outdoor games.
                            Source: weather_enriched WIND_SPEED (m/s -> mph).

    TEMP_F          float   Ambient temperature at kickoff in Fahrenheit.
                            72.0 for dome games (standard controlled temp).
                            NaN if OWM data unavailable for outdoor games.
                            Source: weather_enriched TEMP (Kelvin -> degF).

    PRECIP_FLAG     int     1 if precipitation detected, 0 if clear.
                            0 for dome games.
                            NaN if OWM data unavailable for outdoor games.
                            Source: weather_enriched WEATHER_MAIN column.

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

import pandas as pd

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# ROOF values that indicate a covered playing surface
_DOME_ROOF_VALUES: Final[frozenset[str]] = frozenset({"dome", "retractable"})

# OWM WEATHER_MAIN values that indicate precipitation
_PRECIP_WEATHER_MAINS: Final[frozenset[str]] = frozenset(
    {"Rain", "Snow", "Drizzle", "Thunderstorm"}
)

# Unit conversion constants
_KELVIN_TO_CELSIUS: Final[float] = 273.15
_MPS_TO_MPH: Final[float] = 2.23694

# Standard controlled temperature for dome stadiums (degF)
_DOME_TEMP_F: Final[float] = 72.0


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
        produces=["IS_DOME", "WIND_SPEED_MPH", "TEMP_F", "PRECIP_FLAG"],
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute weather and venue features and join onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame with GAME_ID, YEAR, WEEK_NUM,
                TEAM_A, TEAM_B columns.
            datasets: Provides ``games()`` (for ROOF column) and
                ``weather_enriched()`` (optional OWM data).

        Returns:
            Input DataFrame with IS_DOME, WIND_SPEED_MPH, TEMP_F, and
            PRECIP_FLAG columns appended.  Dome games are fully populated.
            Outdoor games without OWM data have NaN in the weather columns.
        """
        games = datasets.games()

        # -- IS_DOME from ROOF column (always available) -------------------
        roof = games[["GAME_ID", "ROOF"]].copy()
        roof["IS_DOME"] = (
            roof["ROOF"]
            .str.lower()
            .str.strip()
            .apply(lambda r: 1 if r in {v.lower() for v in _DOME_ROOF_VALUES} else 0)
        )
        dome_lookup = roof[["GAME_ID", "IS_DOME"]].drop_duplicates("GAME_ID")

        # -- Weather columns from weather_enriched (optional) --------------
        weather_df = self._load_weather(datasets)
        if weather_df is not None and not weather_df.empty:
            weather_lookup = self._process_weather(weather_df)
        else:
            weather_lookup = pd.DataFrame(
                columns=["GAME_ID", "WIND_SPEED_MPH", "TEMP_F", "PRECIP_FLAG"]
            )

        # -- Merge onto modeling DataFrame ---------------------------------
        df = df.merge(dome_lookup, how="left", on="GAME_ID")

        if not weather_lookup.empty:
            df = df.merge(weather_lookup, how="left", on="GAME_ID")
        else:
            df["WIND_SPEED_MPH"] = float("nan")
            df["TEMP_F"] = float("nan")
            df["PRECIP_FLAG"] = float("nan")

        # -- Dome games: override with definitional controlled values ------
        # Applied after the OWM merge so they always take precedence.
        dome_mask = df["IS_DOME"] == 1
        df.loc[dome_mask, "WIND_SPEED_MPH"] = 0.0
        df.loc[dome_mask, "TEMP_F"] = _DOME_TEMP_F
        df.loc[dome_mask, "PRECIP_FLAG"] = 0

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
            DataFrame with columns: GAME_ID, WIND_SPEED_MPH, TEMP_F,
            PRECIP_FLAG.  One row per game.
        """
        w = weather_df.copy()

        if "TEMP" in w.columns:
            temp_numeric = pd.to_numeric(w["TEMP"], errors="coerce")
            w["TEMP_F"] = (temp_numeric - _KELVIN_TO_CELSIUS) * 9 / 5 + 32
        else:
            w["TEMP_F"] = float("nan")

        if "WIND_SPEED" in w.columns:
            wind_numeric = pd.to_numeric(w["WIND_SPEED"], errors="coerce")
            w["WIND_SPEED_MPH"] = wind_numeric * _MPS_TO_MPH
        else:
            w["WIND_SPEED_MPH"] = float("nan")

        if "WEATHER_MAIN" in w.columns:
            w["PRECIP_FLAG"] = w["WEATHER_MAIN"].apply(
                lambda x: 1
                if str(x) in _PRECIP_WEATHER_MAINS
                else 0
                if pd.notna(x) and str(x) != ""
                else float("nan")
            )
        else:
            w["PRECIP_FLAG"] = float("nan")

        if "GAME_ID" not in w.columns:
            return pd.DataFrame(columns=["GAME_ID", "WIND_SPEED_MPH", "TEMP_F", "PRECIP_FLAG"])

        return (
            w[["GAME_ID", "WIND_SPEED_MPH", "TEMP_F", "PRECIP_FLAG"]]
            .drop_duplicates("GAME_ID")
            .reset_index(drop=True)
        )

# src/gridiron_edge/features/team/weather.py

"""Canonical game-level weather and venue features.

Attaches nullable environmental context to one canonical game row.
Historical and upcoming schedule metadata share the same output schema.

Covered venues receive controlled-environment defaults. Outdoor games
without enriched weather retain explicit null weather values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.enums import COVERED_STADIUMS
from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team._game_metadata import (
    build_game_metadata_lookup,
    load_optional_upcoming_metadata,
)

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# ROOF values that indicate a covered playing surface (stadium has
# a roof at all, regardless of whether it's open on game day). Used
# to override OWM weather data with controlled-environment defaults.
# See :data:`gridiron_edge.core.enums.COVERED_STADIUMS` for the
# canonical semantic grouping.
_DOME_ROOF_VALUES: Final[frozenset[str]] = frozenset(r.value for r in COVERED_STADIUMS)
_ROOF_VALUE_ALIASES: Final[dict[str, str]] = {
    "retractable roof": "retractable",
}

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

# Standard controlled feels-like for dome stadiums (degF) - same as temp
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

_WEATHER_FEATURE_COLUMNS: Final[list[str]] = [
    "IS_DOME",
    *[column for column in _WEATHER_OUTPUT_COLS if column != "GAME_ID"],
]


def _load_weather(
    datasets: DatasetAccessor,
) -> DataFrame | None:
    """Load enriched weather data when available."""
    try:
        return datasets.weather_enriched()
    except (AttributeError, FileNotFoundError):
        return None


def _process_weather(
    weather_df: DataFrame,
) -> DataFrame:
    """Convert enriched weather data to canonical model features."""
    weather = weather_df.copy()

    def _coerce(
        column: str,
    ) -> Series:
        """Return a numeric source column or an all-null Series."""
        if column in weather.columns:
            # pyrefly: ignore [bad-return]
            return pd.to_numeric(
                weather[column],
                errors="coerce",
            )

        return Series(
            float("nan"),
            index=weather.index,
            dtype=float,
        )

    weather["TEMP_F"] = (_coerce("TEMP") - _KELVIN_TO_CELSIUS) * 9 / 5 + 32
    weather["FEELS_LIKE_F"] = (_coerce("FEELS_LIKE") - _KELVIN_TO_CELSIUS) * 9 / 5 + 32

    weather["WIND_SPEED_MPH"] = _coerce("WIND_SPEED") * _MPS_TO_MPH
    weather["HUMIDITY_PCT"] = _coerce("HUMIDITY").astype(float)
    weather["VISIBILITY_M"] = _coerce("VISIBILITY").fillna(_DEFAULT_VISIBILITY_M)

    if "WEATHER_MAIN" in weather.columns:
        main_text = weather["WEATHER_MAIN"].astype(str)
        main_available = weather["WEATHER_MAIN"].notna() & main_text.ne("")

        for column, categories in (
            (
                "PRECIP_FLAG",
                _PRECIP_WEATHER_MAINS,
            ),
            (
                "SNOW_FLAG",
                _SNOW_WEATHER_MAINS,
            ),
            (
                "LOW_VIS_FLAG",
                _LOW_VIS_WEATHER_MAINS,
            ),
        ):
            weather[column] = np.where(
                main_available,
                main_text.isin(categories).astype(int),
                float("nan"),
            )
    else:
        weather["PRECIP_FLAG"] = float("nan")
        weather["SNOW_FLAG"] = float("nan")
        weather["LOW_VIS_FLAG"] = float("nan")

    weather["WIND_CHILL_DELTA"] = weather["TEMP_F"] - weather["FEELS_LIKE_F"]

    if "GAME_ID" not in weather.columns:
        return DataFrame(columns=_WEATHER_OUTPUT_COLS)

    return (
        weather.loc[
            :,
            _WEATHER_OUTPUT_COLS,
        ]
        .drop_duplicates("GAME_ID")
        .reset_index(drop=True)
    )


def _fill_missing_roof_from_stadiums(
    metadata: DataFrame,
    stadiums: DataFrame,
) -> DataFrame:
    """Fill missing game roof values from canonical stadium metadata."""
    if metadata.empty or stadiums.empty:
        return metadata

    required = {"STADIUM", "ROOF"}
    missing = sorted(required - set(stadiums.columns))
    if missing:
        raise ValueError("Stadium reference is missing required columns: " + ", ".join(missing))

    reference = stadiums.loc[:, ["STADIUM", "ROOF"]].copy()
    reference["STADIUM"] = reference["STADIUM"].fillna("").astype(str).str.strip()
    reference["ROOF"] = reference["ROOF"].astype("string").str.strip()
    reference = reference.loc[reference["STADIUM"].ne(""), :].drop_duplicates()

    roof_counts = reference.dropna(subset=["ROOF"]).groupby("STADIUM")["ROOF"].nunique()
    if roof_counts.gt(1).any():
        raise ValueError("Stadium reference contains conflicting roof identities.")

    roof_lookup = (
        reference.dropna(subset=["ROOF"])
        .drop_duplicates(subset=["STADIUM"], keep="first")
        .rename(columns={"ROOF": "_REFERENCE_ROOF"})
    )
    result = metadata.merge(
        roof_lookup,
        how="left",
        on="STADIUM",
        sort=False,
        validate="many_to_one",
    )
    result["ROOF"] = result["ROOF"].where(
        result["ROOF"].notna(),
        result["_REFERENCE_ROOF"],
    )
    return result.drop(columns=["_REFERENCE_ROOF"])


def _add_optional_weather_metadata_columns(
    frame: DataFrame,
    *,
    historical: bool,
) -> DataFrame:
    """Add nullable weather metadata columns absent from a source."""
    result = frame.copy()

    optional_columns = ("GAME_DATE", "STADIUM") if historical else ("game_date", "stadium")

    for column in optional_columns:
        if column not in result.columns:
            result[column] = pd.NA

    return result


_CLIMATOLOGY_CONTINUOUS: Final[tuple[str, ...]] = (
    "WIND_SPEED_MPH",
    "TEMP_F",
    "FEELS_LIKE_F",
    "HUMIDITY_PCT",
    "VISIBILITY_M",
)
_CLIMATOLOGY_FLAGS: Final[tuple[str, ...]] = (
    "PRECIP_FLAG",
    "SNOW_FLAG",
    "LOW_VIS_FLAG",
)


def _binary_mode(values: Series) -> float:
    """Return the observed binary mode, or NaN when unavailable."""
    modes = values.dropna().mode()
    return float(modes.iloc[0]) if not modes.empty else float("nan")


def _normalize_roof_values(values: Series) -> Series:
    """Normalize roof vocabulary before covered-venue classification."""
    return values.astype("string").str.lower().str.strip().replace(_ROOF_VALUE_ALIASES)


def _build_weather_climatology(
    games: DataFrame,
    weather_df: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """Build outdoor stadium-month and league-month climatology."""
    climate_columns = [
        *_CLIMATOLOGY_CONTINUOUS,
        *_CLIMATOLOGY_FLAGS,
    ]
    stadium_columns = ["STADIUM", "MONTH", *climate_columns]
    league_columns = ["MONTH", *climate_columns]

    required = {"GAME_ID", "GAME_DATE", "STADIUM", "ROOF"}
    if games.empty or weather_df.empty or required - set(games.columns):
        return (
            DataFrame(columns=stadium_columns),
            DataFrame(columns=league_columns),
        )

    history = games.loc[
        :,
        ["GAME_ID", "GAME_DATE", "STADIUM", "ROOF"],
    ].copy()
    history["GAME_DATE"] = pd.to_datetime(
        history["GAME_DATE"],
        errors="coerce",
    )
    # pyrefly: ignore [missing-attribute]
    history["MONTH"] = history["GAME_DATE"].dt.month
    roof_text = _normalize_roof_values(history["ROOF"])
    history = history.loc[
        ~roof_text.isin(_DOME_ROOF_VALUES),
        :,
    ]
    history = history.merge(
        _process_weather(weather_df),
        how="inner",
        on="GAME_ID",
        validate="one_to_one",
    )
    history = history.loc[
        history["MONTH"].notna() & history["STADIUM"].notna(),
        :,
    ].copy()
    if history.empty:
        return (
            DataFrame(columns=stadium_columns),
            DataFrame(columns=league_columns),
        )

    aggregations = {
        **dict.fromkeys(_CLIMATOLOGY_CONTINUOUS, "median"),
        **dict.fromkeys(_CLIMATOLOGY_FLAGS, _binary_mode),
    }
    stadium = (
        history.groupby(
            ["STADIUM", "MONTH"],
            dropna=False,
        )
        .agg(aggregations)
        .reset_index()
    )
    league = history.groupby("MONTH", dropna=False).agg(aggregations).reset_index()
    return stadium, league


def _fill_missing_outdoor_weather(
    result: DataFrame,
    *,
    stadium_climatology: DataFrame,
    league_climatology: DataFrame,
) -> DataFrame:
    """Fill null outdoor fields, preferring stadium-month history."""
    output = result.copy()
    # pyrefly: ignore [missing-attribute]
    output["MONTH"] = pd.to_datetime(
        output["_WEATHER_GAME_DATE"],
        errors="coerce",
    ).dt.month
    climate_columns = (
        *_CLIMATOLOGY_CONTINUOUS,
        *_CLIMATOLOGY_FLAGS,
    )
    stadium_renames = {column: f"_STADIUM_CLIMATE_{column}" for column in climate_columns}
    league_renames = {column: f"_LEAGUE_CLIMATE_{column}" for column in climate_columns}
    stadium_lookup = stadium_climatology.rename(
        columns={
            "STADIUM": "_WEATHER_STADIUM",
            **stadium_renames,
        }
    )
    output = output.merge(
        stadium_lookup,
        how="left",
        on=["_WEATHER_STADIUM", "MONTH"],
        sort=False,
        validate="many_to_one",
    )
    output = output.merge(
        league_climatology.rename(columns=league_renames),
        how="left",
        on="MONTH",
        sort=False,
        validate="many_to_one",
    )

    outdoor = output["IS_DOME"].eq(0)
    for column in climate_columns:
        fallback = output[f"_STADIUM_CLIMATE_{column}"].combine_first(
            output[f"_LEAGUE_CLIMATE_{column}"]
        )
        missing = outdoor & output[column].isna()
        output.loc[missing, column] = fallback.loc[missing]

    return output.drop(
        columns=[
            "MONTH",
            *[
                f"_{scope}_CLIMATE_{column}"
                for scope in ("STADIUM", "LEAGUE")
                for column in climate_columns
            ],
        ]
    )


@FeatureRegistry.register("home_away_weather")
class HomeAwayWeatherFeature:
    """Attach schedule-complete game-level weather and dome features."""

    spec = FeatureSpec(
        name="home_away_weather",
        produces=list(_WEATHER_FEATURE_COLUMNS),
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach nullable roof state and weather by canonical game ID."""
        if "GAME_ID" not in df.columns:
            raise ValueError("Home/away game frame is missing required columns: GAME_ID")

        output_columns: list[str] = list(self.spec.produces)
        source: DataFrame = df.copy().drop(
            columns=output_columns,
            errors="ignore",
        )
        source["_INPUT_ORDER"] = range(len(source))

        historical_games = datasets.games()
        upcoming_games = load_optional_upcoming_metadata(datasets)

        historical_metadata = _add_optional_weather_metadata_columns(
            historical_games,
            historical=True,
        )
        upcoming_metadata = _add_optional_weather_metadata_columns(
            upcoming_games,
            historical=False,
        )

        metadata: DataFrame = build_game_metadata_lookup(
            historical=historical_metadata,
            upcoming=upcoming_metadata,
            historical_mapping={
                "GAME_ID": "GAME_ID",
                "GAME_DATE": "GAME_DATE",
                "STADIUM": "STADIUM",
                "ROOF": "ROOF",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "game_date": "GAME_DATE",
                "stadium": "STADIUM",
                "roof": "ROOF",
            },
        )
        missing_roof = metadata["ROOF"].isna()

        if missing_roof.any():
            try:
                stadiums = datasets.stadiums()
            except (AttributeError, FileNotFoundError):
                stadiums = DataFrame()

            metadata = _fill_missing_roof_from_stadiums(
                metadata,
                stadiums,
            )
        roof_text: Series[str] = _normalize_roof_values(metadata["ROOF"])
        dome_values: Series[int] = roof_text.isin(
            frozenset(value.lower() for value in _DOME_ROOF_VALUES)
        ).astype("Int64")
        dome_values = dome_values.mask(metadata["ROOF"].isna())
        metadata["IS_DOME"] = dome_values

        weather_df = _load_weather(datasets)
        if weather_df is not None and not weather_df.empty:
            weather_lookup = _process_weather(weather_df)
        else:
            weather_lookup = DataFrame(columns=_WEATHER_OUTPUT_COLS)

        weather_metadata = metadata.loc[
            :,
            [
                "GAME_ID",
                "GAME_DATE",
                "STADIUM",
                "IS_DOME",
            ],
        ].rename(
            columns={
                "GAME_DATE": "_WEATHER_GAME_DATE",
                "STADIUM": "_WEATHER_STADIUM",
            }
        )

        result: DataFrame = source.merge(
            weather_metadata,
            how="left",
            on="GAME_ID",
            sort=False,
            validate="many_to_one",
        )
        if weather_lookup.empty:
            for column in _WEATHER_OUTPUT_COLS:
                if column != "GAME_ID":
                    result[column] = float("nan")
        else:
            result = result.merge(
                weather_lookup,
                how="left",
                on="GAME_ID",
                sort=False,
                validate="many_to_one",
            )

        if weather_df is not None and not weather_df.empty:
            stadium_climate, league_climate = _build_weather_climatology(
                historical_games,
                weather_df,
            )
            if not stadium_climate.empty or not league_climate.empty:
                result = _fill_missing_outdoor_weather(
                    result,
                    stadium_climatology=stadium_climate,
                    league_climatology=league_climate,
                )

        result["WIND_CHILL_DELTA"] = result["TEMP_F"] - result["FEELS_LIKE_F"]
        dome_mask: Series[bool] = result["IS_DOME"].eq(1)
        result.loc[dome_mask, "WIND_SPEED_MPH"] = 0.0
        result.loc[dome_mask, "TEMP_F"] = _DOME_TEMP_F
        result.loc[dome_mask, "PRECIP_FLAG"] = 0
        result.loc[dome_mask, "FEELS_LIKE_F"] = _DOME_FEELS_LIKE_F
        result.loc[dome_mask, "HUMIDITY_PCT"] = _DOME_HUMIDITY_PCT
        result.loc[dome_mask, "VISIBILITY_M"] = _DOME_VISIBILITY_M
        result.loc[dome_mask, "SNOW_FLAG"] = 0
        result.loc[dome_mask, "LOW_VIS_FLAG"] = 0
        result.loc[dome_mask, "WIND_CHILL_DELTA"] = 0.0

        return (
            result.sort_values("_INPUT_ORDER", kind="stable")
            .drop(
                columns=[
                    "_INPUT_ORDER",
                    "_WEATHER_GAME_DATE",
                    "_WEATHER_STADIUM",
                ]
            )
            .reset_index(drop=True)
        )

# src/gridiron_edge/features/team/surface.py

"""Canonical game-level playing-surface feature.

Resolves the playing surface for one canonical game row from historical and
upcoming schedule metadata, filling gaps from the stadium reference at
(stadium, season) resolution, and reduces it to an artificial-turf flag.

Some stadiums change surface between seasons, so a stadium-only fill would
be wrong; the reference is keyed by season as well.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.features.team._game_metadata import (
    build_game_metadata_lookup,
    load_optional_upcoming_metadata,
)

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Natural surfaces. ``dessograss`` is a hybrid natural-grass system.
_NATURAL_SURFACES: Final[frozenset[str]] = frozenset({"grass", "dessograss"})

# Artificial surfaces recorded by nflverse.
_ARTIFICIAL_SURFACES: Final[frozenset[str]] = frozenset(
    {"fieldturf", "astroturf", "sportturf", "matrixturf", "astroplay", "a_turf"}
)

_SURFACE_COLUMN: Final[str] = "IS_ARTIFICIAL_TURF"


def _normalize_surface(values: pd.Series) -> pd.Series:
    """Strip and lowercase surface labels, leaving nulls and blanks null."""
    text = values.astype("string").str.strip().str.lower()
    return text.mask(text.eq(""))


def _fill_missing_surface_from_stadiums(
    metadata: DataFrame,
    stadiums: DataFrame,
) -> DataFrame:
    """Fill null surfaces from the stadium reference at (stadium, season)."""
    required = {"STADIUM", "YEAR", "SURFACE"}
    if metadata.empty or stadiums.empty or required - set(stadiums.columns):
        return metadata

    reference = stadiums.loc[:, ["STADIUM", "YEAR", "SURFACE"]].copy()
    reference["STADIUM"] = reference["STADIUM"].astype("string").str.strip()
    reference["SURFACE"] = _normalize_surface(reference["SURFACE"])
    reference = reference.dropna(subset=["STADIUM", "SURFACE"]).drop_duplicates()

    conflicts = reference.groupby(["STADIUM", "YEAR"])["SURFACE"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Stadium reference contains conflicting surface identities.")

    lookup = reference.rename(columns={"SURFACE": "_REFERENCE_SURFACE"})
    result = metadata.merge(
        lookup,
        how="left",
        on=["STADIUM", "YEAR"],
        sort=False,
        validate="many_to_one",
    )
    result["SURFACE"] = result["SURFACE"].where(
        result["SURFACE"].notna(),
        result["_REFERENCE_SURFACE"],
    )
    return result.drop(columns=["_REFERENCE_SURFACE"])


def _artificial_turf_flag(surface: pd.Series) -> pd.Series:
    """Map normalized surface labels to a nullable artificial-turf flag.

    A non-null label outside the known vocabulary raises: a new label is a
    data-contract change, and emitting null would silently disable the live
    feature family for the whole week.
    """
    unknown = sorted(
        set(surface.dropna().unique()) - _NATURAL_SURFACES - _ARTIFICIAL_SURFACES,
    )
    if unknown:
        raise ValueError("Unrecognized playing surface values: " + ", ".join(unknown))
    flag = surface.isin(_ARTIFICIAL_SURFACES).astype("Int64")
    return flag.mask(surface.isna())


@FeatureRegistry.register("home_away_surface")
class HomeAwaySurfaceFeature:
    """Attach the artificial-turf flag by canonical game identity."""

    spec = FeatureSpec(
        name="home_away_surface",
        produces=[_SURFACE_COLUMN],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach surface state, filling gaps from the stadium reference."""
        if "GAME_ID" not in df.columns:
            raise ValueError("Home/away game frame is missing required columns: GAME_ID")

        source = df.copy().drop(columns=[_SURFACE_COLUMN], errors="ignore")
        source["_INPUT_ORDER"] = range(len(source))

        lookup = build_game_metadata_lookup(
            historical=datasets.games(),
            upcoming=load_optional_upcoming_metadata(datasets),
            historical_mapping={
                "GAME_ID": "GAME_ID",
                "YEAR": "YEAR",
                "STADIUM": "STADIUM",
                "SURFACE": "SURFACE",
            },
            upcoming_mapping={
                "game_id": "GAME_ID",
                "season": "YEAR",
                "stadium": "STADIUM",
                "surface": "SURFACE",
            },
        )
        lookup["SURFACE"] = _normalize_surface(lookup["SURFACE"])

        if lookup["SURFACE"].isna().any():
            try:
                stadiums = datasets.stadiums()
            except (AttributeError, FileNotFoundError):
                stadiums = DataFrame()
            lookup = _fill_missing_surface_from_stadiums(lookup, stadiums)

        lookup[_SURFACE_COLUMN] = _artificial_turf_flag(lookup["SURFACE"])

        result = source.merge(
            lookup[["GAME_ID", _SURFACE_COLUMN]],
            how="left",
            on="GAME_ID",
            sort=False,
            validate="many_to_one",
        )
        return (
            result.sort_values("_INPUT_ORDER", kind="stable")
            .drop(columns=["_INPUT_ORDER"])
            .reset_index(drop=True)
        )

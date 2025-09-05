from __future__ import annotations

import pandas as pd

from gridiron.features.registry import register_feature


@register_feature("home_field", tags=["context"])
def home_field(games_df: pd.DataFrame, asof_utc: pd.Series) -> pd.Series:
    return pd.Series(1, index=games_df.index)


@register_feature("is_divisional", tags=["context"])
def is_divisional(games_df: pd.DataFrame, asof_utc: pd.Series) -> pd.Series:
    # Placeholder - will be computed when divisions map is introduced
    return pd.Series(0, index=games_df.index)


@register_feature("rest_days_home", tags=["context"])
def rest_days_home(games_df: pd.DataFrame, asof_utc: pd.Series) -> pd.Series:
    # For Phase 0 placeholder, set NaN; implemented in Phase 1 with historical joins
    return pd.Series(pd.NA, index=games_df.index, dtype="float")


@register_feature("rest_days_away", tags=["context"])
def rest_days_away(games_df: pd.DataFrame, asof_utc: pd.Series) -> pd.Series:
    return pd.Series(pd.NA, index=games_df.index, dtype="float")


@register_feature("travel_tz_diff_away", tags=["context"])
def travel_tz_diff_away(games_df: pd.DataFrame, asof_utc: pd.Series) -> pd.Series:
    return pd.Series(0, index=games_df.index, dtype="int")

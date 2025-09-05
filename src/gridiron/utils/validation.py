from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column


def schedules_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "season": Column(int, Check.ge(1999)),
            "week": Column(int, Check.between(1, 22)),
            "game_type": Column(str),
            "game_id": Column(str, unique=True),
            "game_id_nflfastr": Column(object, nullable=True),
            "home_team": Column(str),
            "away_team": Column(str),
            "kickoff_et": Column(object, nullable=True, coerce=False),
            "kickoff_utc": Column(object, nullable=True, coerce=False),
            "home_score": Column(object, nullable=True),
            "away_score": Column(object, nullable=True),
        },
        strict=False,
        coerce=True,
    )


def predictions_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "game_id": Column(str),
            "home_team": Column(str),
            "away_team": Column(str),
            "season": Column(int),
            "week": Column(int),
            "p_home_win": Column(float, Check.in_range(0.0, 1.0)),
            "p_away_win": Column(float, Check.in_range(0.0, 1.0)),
            "snapshot_utc": Column(object, nullable=False, coerce=False),
            "home_score": Column(object, nullable=True),
            "away_score": Column(object, nullable=True),
        },
        strict=False,
        coerce=True,
    )


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    return schema.validate(df, lazy=True)

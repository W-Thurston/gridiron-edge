from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from gridiron.models.moneyline_dummy import MoneylineDummyModel
from gridiron.utils.time import SnapshotPolicy, assign_snapshot_utc_weekwise
from gridiron.utils.validation import predictions_schema, validate


def _week_key(df: pd.DataFrame) -> Iterable[tuple[int, int]]:
    for k, _ in df.groupby(["season", "week"]):
        yield k


def _ensure_tz_series(s: pd.Series, tz: str) -> pd.Series:
    """Return a tz-aware datetime Series in the requested tz (no copies if already ok)."""
    # If it's not any datetime dtype, parse first
    if not (is_datetime64_any_dtype(s) or isinstance(s.dtype, pd.DatetimeTZDtype)):
        s = pd.to_datetime(s, errors="coerce")

    # Now s is datetime-like. If tz-aware → convert; else → localize.
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(tz)
    else:
        return s.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")


def build_features(
    df: pd.DataFrame, feature_names: list[str], snapshot_policy: SnapshotPolicy
) -> pd.DataFrame:
    out = df.copy()

    # Harden timestamp dtypes
    out["kickoff_et"] = _ensure_tz_series(out["kickoff_et"], "America/New_York")
    out["kickoff_utc"] = _ensure_tz_series(out["kickoff_utc"], "UTC")

    # Compute one snapshot timestamp per (season, week) and join back
    def _snap_row(g: pd.DataFrame) -> pd.DataFrame:
        # g.name is (season, week) when grouping by ["season","week"]
        season_key, week_key = g.name
        # helper returns same timestamp for all rows in the group; take first
        snap = assign_snapshot_utc_weekwise(g, snapshot_policy).iloc[0]
        return pd.DataFrame({"season": [season_key], "week": [week_key], "snapshot_utc": [snap]})

    # On pandas ≥2.2, include_groups=False avoids the deprecation warning.
    try:
        snap_per_week = (
            out.groupby(["season", "week"], sort=True)
            .apply(_snap_row, include_groups=False)
            .reset_index(drop=True)
        )
    except TypeError:
        # Fallback for older pandas that don't support include_groups
        snap_per_week = (
            out.groupby(["season", "week"], sort=True).apply(_snap_row).reset_index(drop=True)
        )

    out = out.merge(snap_per_week, on=["season", "week"], how="left")

    # Phase 0 uses context-only features; no extra features added when feature_names=[]
    for _name in feature_names:
        pass

    return out


def _fit_dummy(train_df: pd.DataFrame, mode: str) -> MoneylineDummyModel:
    # target: 1=home win where scores are known
    y = (train_df["home_score"] > train_df["away_score"]).astype("Int64").dropna().astype(int)
    x = train_df.loc[y.index, []]  # no real features yet
    model = MoneylineDummyModel(mode=mode).fit(x, y)
    return model


def _predict_for_week(week_df: pd.DataFrame, model: MoneylineDummyModel) -> pd.DataFrame:
    p_home = model.predict_proba(week_df[[]])
    res = week_df[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "snapshot_utc",
        ]
    ].copy()
    res["p_home_win"] = p_home.values
    res["p_away_win"] = 1.0 - res["p_home_win"]
    return validate(res, predictions_schema())


def walk_forward(
    train_hist: pd.DataFrame,
    predict_df: pd.DataFrame,
    snapshot_policy: SnapshotPolicy,
    mode: str = "coinflip",
    n_jobs: int = -1,
) -> pd.DataFrame:
    preds = []
    # Iterate predict weeks in chronological order
    for (_season, _week), wk in predict_df.groupby(["season", "week"], sort=True):
        # training window: all history strictly before the earliest kickoff of this week
        cutoff = wk["kickoff_utc"].min()
        train = train_hist[train_hist["kickoff_utc"] < cutoff]
        model = _fit_dummy(train, mode=mode)
        # features + predictions for this week
        wk_feats = build_features(wk, feature_names=[], snapshot_policy=snapshot_policy)
        preds.append(_predict_for_week(wk_feats, model))
    return pd.concat(preds, ignore_index=True)

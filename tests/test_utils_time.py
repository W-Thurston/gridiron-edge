import pandas as pd

from gridiron.utils.time import ET, SnapshotPolicy, assign_snapshot_utc_weekwise


def test_snapshot_early_wed_before_thu_kickoff():
    df = pd.DataFrame(
        {
            "season": [2022, 2022],
            "week": [1, 1],
            "kickoff_et": [
                pd.Timestamp("2022-09-08 20:20", tz=ET),  # Thu night
                pd.Timestamp("2022-09-11 13:00", tz=ET),  # Sun
            ],
        }
    )
    s = assign_snapshot_utc_weekwise(df, SnapshotPolicy("EARLY_WED_10ET"))
    # snapshot should be Wednesday 10:00 ET (the day before Thursday opener)
    assert s.dt.tz is not None
    assert s.iloc[0] == s.iloc[1]
    assert s.iloc[0].tz_convert(ET).hour == 10
    assert s.iloc[0].tz_convert(ET).weekday() == 2  # Wed

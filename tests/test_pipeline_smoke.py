from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir

from gridiron.api.backtest import BacktestConfig, run_backtest
from gridiron.utils.time import SnapshotPolicy


def test_backtest_smoke(tmp_path: Path):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        paths = compose(config_name="paths")

    cfg = BacktestConfig(
        train_seasons=[2010, 2011],
        predict_seasons=[2012],
        snapshot_policy=SnapshotPolicy("EARLY_WED_10ET"),
        mode="coinflip",
        artifacts_dir=tmp_path,
        schedules_csv_url=paths.nflverse_schedules_csv,
    )
    out = run_backtest(cfg)

    pred_path = Path(out["artifacts_dir"]) / "predictions.parquet"
    assert pred_path.exists()
    df = pd.read_parquet(pred_path)
    assert {"p_home_win", "p_away_win", "game_id"}.issubset(df.columns)

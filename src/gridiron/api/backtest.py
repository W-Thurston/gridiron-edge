from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gridiron.api.pipelines import walk_forward
from gridiron.data.schedules import load_schedules_standardized
from gridiron.evaluation.reports import save_backtest_artifacts, summarize_by_week, write_manifest
from gridiron.utils.io import ensure_dir, write_parquet
from gridiron.utils.time import SnapshotPolicy
from gridiron.utils.validation import schedules_schema


@dataclass
class BacktestConfig:
    train_seasons: list[int]
    predict_seasons: list[int]
    snapshot_policy: SnapshotPolicy
    mode: str
    artifacts_dir: Path
    schedules_parquet_path: Path | None = None
    schedules_csv_url: str = ""
    force_remote: bool = False


def run_backtest(cfg: BacktestConfig) -> dict:
    t0 = time.time()

    # Load standardized schedules using configured source
    seasons = sorted(set(cfg.train_seasons + cfg.predict_seasons))
    used_source = None

    if (
        (not cfg.force_remote)
        and cfg.schedules_parquet_path
        and cfg.schedules_parquet_path.exists()
    ):
        print(f"[schedules] Reading local parquet → {cfg.schedules_parquet_path}")
        std = pd.read_parquet(cfg.schedules_parquet_path)
        used_source = str(cfg.schedules_parquet_path)
    else:
        print(f"[schedules] Reading CSV → {cfg.schedules_csv_url}")
        assert (
            cfg.schedules_csv_url is not None
        ), "schedules_csv_url must be set when using remote CSV"
        std = load_schedules_standardized(
            seasons=seasons,
            csv_url=cfg.schedules_csv_url,
            verbose=True,
        )
        used_source = cfg.schedules_csv_url or "<unknown>"

    std = schedules_schema().validate(std)

    # Restrict to regular season by default (Phase 0)
    reg = std[std["game_type"] == "REG"].copy()
    train = reg[reg["season"].isin(cfg.train_seasons)].copy()
    predict = reg[reg["season"].isin(cfg.predict_seasons)].copy()

    preds = walk_forward(train, predict, cfg.snapshot_policy, mode=cfg.mode)

    run_id = f"bt_{int(time.time())}"
    out_dir = ensure_dir(cfg.artifacts_dir / "backtests" / run_id)

    # Save artifacts (predictions, weekly summary, metrics, plots, index.html)
    _ = save_backtest_artifacts(out_dir, preds)

    write_parquet(preds, out_dir / "predictions.parquet")
    by_week = summarize_by_week(preds)
    by_week.to_csv(out_dir / "metrics_by_week.csv", index=False)

    # Minimal summary CSV (kept simple in Phase 0)
    (out_dir / "metrics_summary.csv").write_text(f"metric,value\nn_predictions,{len(preds)}\n")

    write_manifest(
        out_dir,
        {
            "run_id": run_id,
            "config": {
                "train_seasons": cfg.train_seasons,
                "predict_seasons": cfg.predict_seasons,
                "snapshot": cfg.snapshot_policy.name,
                "mode": cfg.mode,
                "schedules_source": used_source,
                "force_remote": cfg.force_remote,
            },
            "timing_sec": round(time.time() - t0, 3),
        },
    )
    return {"run_id": run_id, "artifacts_dir": str(out_dir)}

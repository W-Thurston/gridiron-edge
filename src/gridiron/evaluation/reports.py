from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridiron.utils.io import ensure_dir

# ---------- IO helpers ----------


def write_manifest(path_dir: str | Path, manifest: dict) -> Path:
    ensure_dir(path_dir)
    p = Path(path_dir) / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return p


# ---------- Metrics ----------


def _scored(preds_scored: pd.DataFrame) -> pd.DataFrame:
    """Return rows where outcome is known, with y and pred columns."""
    d = preds_scored.dropna(subset=["home_score", "away_score"]).copy()
    d["y"] = (d["home_score"] > d["away_score"]).astype(int)
    d["pred"] = d["p_home_win"].astype(float)
    return d


def summarize_by_week(preds: pd.DataFrame) -> pd.DataFrame:
    d = _scored(preds)
    # NOTE: include_groups=False silences future warning on >=2.2
    try:
        grp = d.groupby(["season", "week"], as_index=False, sort=True, observed=True)
        return grp.apply(
            lambda g: pd.Series(
                {
                    "n_games": int(len(g)),
                    "acc": float((g["pred"] >= 0.5).astype(int).eq(g["y"]).mean()),
                    "brier": float(((g["pred"] - g["y"]) ** 2).mean()),
                }
            ),
            include_groups=False,
        ).reset_index(drop=True)
    except TypeError:
        # fallback for older pandas
        return (
            d.groupby(["season", "week"], as_index=False, sort=True, observed=True)
            .apply(
                lambda g: pd.Series(
                    {
                        "n_games": int(len(g)),
                        "acc": float((g["pred"] >= 0.5).astype(int).eq(g["y"]).mean()),
                        "brier": float(((g["pred"] - g["y"]) ** 2).mean()),
                    }
                )
            )
            .reset_index(drop=True)
        )


def overall_metrics(preds: pd.DataFrame) -> dict[str, Any]:
    d = _scored(preds)
    if len(d) == 0:
        return {"n_games": 0, "acc": None, "brier": None, "logloss": None}
    # safe log-loss (clip to avoid -inf)
    p = d["pred"].clip(1e-12, 1 - 1e-12)
    y = d["y"].astype(int)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    acc = float((d["pred"] >= 0.5).astype(int).eq(y).mean())
    brier = float(((d["pred"] - y) ** 2).mean())
    return {"n_games": int(len(d)), "acc": acc, "brier": brier, "logloss": float(logloss)}


# ---------- Artifact writers & plots ----------


def _plot_line(df: pd.DataFrame, xcols: list[str], ycol: str, title: str, out_png: Path) -> None:
    # Build an ordinal “year-week” x-axis label for readability
    df = df.copy()
    df["label"] = df["season"].astype(str) + "-W" + df["week"].astype(int).astype(str)
    plt.figure()
    plt.plot(df["label"].values, df[ycol].values, marker="o")
    plt.title(title)
    plt.xlabel("Season-Week")
    plt.ylabel(ycol)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_hist(preds: pd.DataFrame, out_png: Path) -> None:
    d = _scored(preds)
    plt.figure()
    plt.hist(d["pred"].values, bins=20)
    plt.title("Predicted Home Win Probability (Histogram)")
    plt.xlabel("p_home_win")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _write_html_report(
    out_dir: Path,
    overall: dict[str, Any],
    weekly_csv_rel: str,
    preds_parquet_rel: str,
    acc_png_rel: str,
    brier_png_rel: str,
    hist_png_rel: str,
) -> Path:
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gridiron Backtest Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    .kpi {{ display:inline-block; margin-right:24px; padding:12px 16px; border:1px solid #ddd; border-radius:8px; }}
    img {{ max-width: 100%; height: auto; }}
    h2 {{ margin-top: 28px; }}
    code,a {{ color: #0b5fff; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>Gridiron Backtest Report</h1>
  <div>
    <span class="kpi"><b>Games:</b> {overall.get('n_games')}</span>
    <span class="kpi"><b>Acc:</b> {overall.get('acc')}</span>
    <span class="kpi"><b>Brier:</b> {overall.get('brier')}</span>
    <span class="kpi"><b>LogLoss:</b> {overall.get('logloss')}</span>
  </div>

  <h2>Downloads</h2>
  <ul>
    <li><a href="{weekly_csv_rel}">Weekly Summary (CSV)</a></li>
    <li><a href="{preds_parquet_rel}">Predictions (Parquet)</a></li>
  </ul>

  <h2>Weekly Accuracy</h2>
  <img src="{acc_png_rel}" alt="Weekly Accuracy">

  <h2>Weekly Brier</h2>
  <img src="{brier_png_rel}" alt="Weekly Brier Score">

  <h2>Prediction Distribution</h2>
  <img src="{hist_png_rel}" alt="Pred histogram">
</body>
</html>
"""
    p = out_dir / "index.html"
    p.write_text(html)
    return p


def save_backtest_artifacts(out_dir: str | Path, preds: pd.DataFrame) -> dict:
    """
    Saves:
      - preds.parquet
      - weekly_summary.csv
      - metrics_overall.json
      - plots: weekly_acc.png, weekly_brier.png, preds_hist.png
      - index.html (simple report linking everything)
    Returns paths in a dict.
    """
    out_dir = ensure_dir(out_dir)
    out_dir = Path(out_dir)

    # 1) raw predictions
    preds_parquet = out_dir / "preds.parquet"
    preds.to_parquet(preds_parquet, index=False)

    # 2) weekly summary
    weekly = summarize_by_week(preds)
    weekly_csv = out_dir / "weekly_summary.csv"
    weekly.to_csv(weekly_csv, index=False)

    # 3) overall metrics
    overall = overall_metrics(preds)
    metrics_json = out_dir / "metrics_overall.json"
    metrics_json.write_text(json.dumps(overall, indent=2))

    # 4) plots
    acc_png = out_dir / "weekly_acc.png"
    brier_png = out_dir / "weekly_brier.png"
    hist_png = out_dir / "preds_hist.png"

    if len(weekly):
        _plot_line(weekly, ["season", "week"], "acc", "Weekly Accuracy", acc_png)
        _plot_line(weekly, ["season", "week"], "brier", "Weekly Brier Score", brier_png)
    _plot_hist(preds, hist_png)

    # 5) tiny HTML report
    index_html = _write_html_report(
        out_dir,
        overall,
        weekly_csv_rel=weekly_csv.name,
        preds_parquet_rel=preds_parquet.name,
        acc_png_rel=acc_png.name,
        brier_png_rel=brier_png.name,
        hist_png_rel=hist_png.name,
    )

    return {
        "preds_parquet": preds_parquet,
        "weekly_csv": weekly_csv,
        "metrics_json": metrics_json,
        "acc_png": acc_png,
        "brier_png": brier_png,
        "hist_png": hist_png,
        "index_html": index_html,
    }

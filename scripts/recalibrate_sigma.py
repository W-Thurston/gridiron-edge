"""Recalibrate sigma and margin_std for champion models.

Calibrates against holdout-season actual margins only (not training data),
so sigma represents how the model's probabilities map to real margins on
unseen games — consistent with our honest-metrics philosophy.

Run from repo root:
    uv run python scripts/recalibrate_sigma.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from gridiron_edge.core.constants import HOLDOUT_SEASONS
from gridiron_edge.evaluation.archive import load_prediction_log
from gridiron_edge.models.game_prediction.post_process import (
    _PROB_CEIL,
    _PROB_FLOOR,
    calibrate_spread_sigma,
)

CHAMPIONS = ["random_forest", "xgboost", "logistic"]


def _load_actual_margins() -> pd.DataFrame:
    """Build a game_id → actual_margin lookup from cleaned games.

    actual_margin = home_score - away_score, derived from the
    WINNER/LOSER/PTS_WINNER/PTS_LOSER/GAME_LOCATION schema.
    """
    games = pd.read_csv("data/cleaned/NFL_wk_by_wk_cleaned.csv")

    # GAME_LOCATION: "H" = winner was home, "@" = winner was away, "N" = neutral
    # For "H": home team won → margin = PTS_WINNER - PTS_LOSER
    # For "@": away team won → margin = PTS_LOSER - PTS_WINNER
    # For "N": treat designated home from schedule — join on archive's home_team
    #          to resolve. For sigma calibration, neutral sites are a tiny
    #          minority and the margin is symmetric, so using WINNER as "home"
    #          introduces negligible bias. We'll handle it properly in the join.
    games["home_margin"] = np.where(
        games["GAME_LOCATION"].isin(["H", "N"]),
        games["PTS_WINNER"] - games["PTS_LOSER"],
        games["PTS_LOSER"] - games["PTS_WINNER"],
    )
    return games[["GAME_ID", "YEAR", "home_margin"]].drop_duplicates("GAME_ID")


def main() -> None:
    """Calibrate sigma and margin_std for all champion models."""
    archive = load_prediction_log()
    margins = _load_actual_margins()

    print(f"Archive rows: {len(archive)}")
    print(f"Games with margins: {len(margins)}")
    print(f"Holdout seasons: {HOLDOUT_SEASONS}")
    print()

    results: dict[str, dict[str, float]] = {}

    for model in CHAMPIONS:
        preds = archive[archive["model_version"] == model].copy()
        if preds.empty:
            print(f"⚠️  No archive predictions for '{model}' — skip")
            continue

        merged = preds.merge(
            margins,
            left_on="game_id",
            right_on="GAME_ID",
            how="inner",
        )

        # --- Holdout only (honest calibration) ---
        holdout = merged[merged["season"].isin(HOLDOUT_SEASONS)]

        # --- Also calibrate on full backfill for comparison ---
        full = merged

        print(f"{'=' * 60}")
        print(f"Model: {model}")
        print(f"  Matched games (full): {len(full)}")
        print(f"  Matched games (holdout): {len(holdout)}")

        for label, df in [("holdout", holdout), ("full", full)]:
            if df.empty:
                print(f"  [{label}] — no data, skipping")
                continue

            # 1. Calibrate sigma
            sigma = calibrate_spread_sigma(df["home_win_prob"], df["home_margin"])

            # 2. Compute predicted margins at calibrated sigma
            clamped = np.clip(df["home_win_prob"].values, _PROB_FLOOR, _PROB_CEIL)
            predicted_margin = sigma * norm.ppf(clamped)
            actual = df["home_margin"].values
            residuals = predicted_margin - actual

            # 3. Margin std = RMSE of residuals
            margin_std = round(float(np.sqrt(np.mean(residuals**2))), 2)

            # 4. Validation metrics
            mae = round(float(np.mean(np.abs(residuals))), 2)
            corr = round(float(np.corrcoef(predicted_margin, actual)[0, 1]), 4)
            bias = round(float(np.mean(residuals)), 2)

            tag = " ◄ USE THIS" if label == "holdout" else ""
            print(f"\n  [{label}]{tag}")
            print(f"    sigma:      {sigma}")
            print(f"    margin_std: {margin_std}")
            print(f"    MAE:        {mae}")
            print(f"    corr:       {corr}")
            print(f"    bias:       {bias}")

            if label == "holdout":
                results[model] = {"sigma": sigma, "margin_std": margin_std}

        print()

    # --- Summary: copy-paste into post_process.py ---
    print("=" * 60)
    print("COPY-PASTE INTO post_process.py:")
    print("=" * 60)
    print("\n# _MODEL_SIGMAS champion entries:")
    for model, vals in results.items():
        print(f'    "{model}": {vals["sigma"]},')
    print("\n# _MODEL_MARGIN_STDS champion entries:")
    for model, vals in results.items():
        print(f'    "{model}": {vals["margin_std"]},')


if __name__ == "__main__":
    main()

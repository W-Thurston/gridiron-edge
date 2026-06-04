"""Remove old versioned model predictions from the archive.

The PredictorRegistry now only contains unversioned champions
(random_forest, xgboost, logistic) + elo_v2. Old versioned
entries (rf_v1-v3, xgb_v1-v3, logistic_v1-v4, elo_v1) are
dead weight with mostly-NaN enrichment columns.

Run from repo root:
    uv run python scripts/clean_archive.py
"""

from __future__ import annotations

from gridiron_edge.evaluation.archive import load_prediction_log

KEEP_MODELS = {"random_forest", "xgboost", "logistic", "elo_v2"}


def main() -> None:
    """Remove old versioned model predictions from the archive."""
    archive = load_prediction_log()
    print(f"Archive rows before: {len(archive)}")
    print("\nModel version counts (before):")
    print(archive["model_version"].value_counts().to_string())

    cleaned = archive[archive["model_version"].isin(KEEP_MODELS)].copy()
    dropped = len(archive) - len(cleaned)

    print(f"\nDropping {dropped} rows from old versioned models")
    print(f"Archive rows after: {len(cleaned)}")
    print("\nModel version counts (after):")
    print(cleaned["model_version"].value_counts().to_string())

    # Confirm before writing
    resp = input("\nWrite cleaned archive? (y/n): ")
    if resp.strip().lower() == "y":
        path = "data/output/predictions/predictions_log.parquet"
        cleaned.to_parquet(path, index=False)
        print(f"✅ Written to {path}")
    else:
        print("Aborted.")


if __name__ == "__main__":
    main()

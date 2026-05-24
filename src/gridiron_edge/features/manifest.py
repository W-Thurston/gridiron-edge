# src/gridiron_edge/features/manifest.py

"""Feature set manifest — records which features produced the modeling file.

Written alongside ``modeling_file.csv`` every time ``build_model_inputs``
runs. Consumed by any code that loads the feature matrix to train or
predict, ensuring the feature set in use matches what the model expects.

The manifest prevents silent column mismatches when the feature set
changes between a training run and a prediction run. Without it, adding
a new feature to the pipeline produces a CSV with new columns, but a
model trained on the old CSV silently receives a DataFrame with extra
(or missing) columns it doesn't know about.

Schema versioning:
    ``schema_version`` is an integer that increments whenever the set of
    features or their output columns changes. Code that loads the feature
    matrix can assert the version it was trained on still matches.

Manifest file location:
    ``data/modeling/modeling_file_manifest.json``

Example manifest::

    {
        "schema_version": 1,
        "created_at": "2026-05-23T13:17:42",
        "feature_names": ["home_field", "team_elo", "travel"],
        "feature_columns": ["HOME_FIELD", "TEAM_A_ELO", "TEAM_B_ELO", ...],
        "all_columns": ["GAME_ID", "TEAM_A", "TEAM_B", ..., "HOME_FIELD", ...],
        "row_count": 14552,
    }
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from logging import Logger
from pathlib import Path
from typing import Any

import pandas as pd

logger: Logger = logging.getLogger(__name__)

# Bump this whenever the feature set or feature output columns change.
# Models trained on a previous version will detect the mismatch and
# surface a clear error rather than silently producing wrong predictions.
CURRENT_SCHEMA_VERSION: int = 2

_MANIFEST_FILENAME: str = "modeling_file_manifest.json"


def _manifest_path(modeling_dir: Path) -> Path:
    """Return the manifest path for a given modeling directory."""
    return modeling_dir / _MANIFEST_FILENAME


def write_manifest(
    df: pd.DataFrame,
    *,
    feature_names: list[str],
    feature_columns: list[str],
    modeling_dir: Path,
    schema_version: int = CURRENT_SCHEMA_VERSION,
) -> Path:
    """Write a feature set manifest alongside the modeling file.

    Called by ``build_model_inputs`` immediately after writing
    ``modeling_file.csv``.

    Args:
        df: The full modeling DataFrame that was written to disk.
        feature_names: Ordered list of feature keys that were applied
            (e.g. ``["home_field", "team_elo", "travel"]``).
        feature_columns: Flat list of all columns produced by those
            features (e.g. ``["HOME_FIELD", "TEAM_A_ELO", ...]``).
        modeling_dir: Directory where the modeling CSV lives.
        schema_version: Integer schema version. Increment when the
            feature set or column schema changes.

    Returns:
        Absolute path to the written manifest file.
    """
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_names": feature_names,
        "feature_columns": feature_columns,
        "all_columns": df.columns.tolist(),
        "row_count": len(df),
    }

    path: Path = _manifest_path(modeling_dir)
    path.write_text(json.dumps(manifest, indent=2))
    logger.debug("Feature manifest written to %s", path)
    return path


def read_manifest(modeling_dir: Path) -> dict[str, Any]:
    """Read the feature set manifest from disk.

    Args:
        modeling_dir: Directory containing the modeling CSV and manifest.

    Returns:
        Manifest dict with keys: schema_version, created_at,
        feature_names, feature_columns, all_columns, row_count.

    Raises:
        FileNotFoundError: If no manifest exists in the directory.
    """
    path: Path = _manifest_path(modeling_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No feature manifest found at {path}. "
            "Run 'gridiron features model-inputs' to rebuild the modeling file."
        )
    return json.loads(path.read_text())


def validate_columns(
    df: pd.DataFrame,
    *,
    expected_columns: list[str],
    context: str = "",
) -> None:
    """Assert that a DataFrame contains exactly the expected columns.

    Raises a clear, actionable error if columns are missing or unexpected,
    rather than letting a downstream KeyError surface deep in model code.

    Args:
        df: DataFrame to validate (e.g. loaded from ``modeling_file.csv``).
        expected_columns: List of column names the caller requires.
        context: Optional string describing the caller (e.g. model name)
            for inclusion in the error message.

    Raises:
        ValueError: If any expected columns are missing from ``df``.
    """
    actual: set[str] = set(df.columns)
    expected: set[str] = set(expected_columns)

    missing: set[str] = expected - actual
    if missing:
        prefix: str = f"[{context}] " if context else ""
        raise ValueError(
            f"{prefix}Feature matrix is missing expected columns: "
            f"{sorted(missing)}. "
            "The feature set may have changed. "
            "Run 'gridiron features model-inputs' to rebuild, "
            "then retrain any models that depend on these columns."
        )


def validate_schema_version(
    manifest: dict[str, Any],
    *,
    required_version: int,
    context: str = "",
) -> None:
    """Assert that the manifest schema version matches what the caller requires.

    Args:
        manifest: Output of ``read_manifest()``.
        required_version: Schema version the caller was built against.
        context: Optional caller description for error messages.

    Raises:
        ValueError: If the manifest version does not match.
    """
    actual_version: int = manifest.get("schema_version", 0)
    if actual_version != required_version:
        prefix: str = f"[{context}] " if context else ""
        raise ValueError(
            f"{prefix}Feature schema version mismatch: "
            f"expected {required_version}, found {actual_version}. "
            "The feature set has changed since this model was trained. "
            "Retrain the model against the current feature matrix."
        )

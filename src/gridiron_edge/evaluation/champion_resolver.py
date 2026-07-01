# src/gridiron_edge/evaluation/champion_resolver.py
"""Persistent champion manifest read + write API.

The manifest at ``data/output/champions/champions.json`` records the current
champion ``(model_name, model_type)`` pair for every ``model_name``. It is
written during ``full-retrain`` (see the ``promote-champions`` stage) and
read by CLI consumers, the API layer, and downstream analysis.

Per D21, the manifest is a static artifact: consumers read; nothing is
computed at request time. Writing the manifest is a separate concern
handled by ``full-retrain``'s promotion stage.

Schema:
    {
        "schema_version": 1,
        "updated_at": "2026-07-01T14:23:00Z",
        "models": {
            "win_prob": {
                "model_type": "random_forest",
                "promoted_at": "2026-07-01T14:20:00Z",
                "source_run_id": "20260701_142000",
                "metrics": {
                    "brier": 0.213,
                    "ece": 0.041,
                    "auc": 0.721
                }
            },
            "total": {
                "model_type": "xgboost",
                "promoted_at": "2026-07-01T14:20:00Z",
                "source_run_id": "20260701_142000",
                "metrics": {
                    "mae": 10.24,
                    "rmse": 12.87
                }
            },
            "qb_pass_yards": {
                "model_type": "elasticnet",
                "promoted_at": "2026-07-01T14:20:00Z",
                "source_run_id": "20260701_142000",
                "metrics": {
                    "mae": 63.4,
                    "r2": 0.118,
                    "coverage": 0.938
                }
            }
        }
    }

The ``metrics`` field is task-flexible — game classification models use
Brier/ECE/AUC; regression models use MAE/RMSE/R². Consumers should treat
it as informational only, not depend on specific keys.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from gridiron_edge.core.settings import get_settings

CURRENT_SCHEMA_VERSION = 1


class ChampionNotFoundError(Exception):
    """Raised when the champion manifest is missing or has no entry for a model.

    Downstream consumers should catch this to degrade gracefully when the
    manifest hasn't been written yet (e.g., fresh checkout, first full-retrain
    hasn't run).
    """


def _manifest_path(repo: Path | None = None) -> Path:
    """Return the manifest path.

    Creates the parent directory if needed. Follows the same pattern as
    ``evaluation/archive.py::_archive_path`` for consistency.

    Args:
        repo: Repository root override. Defaults to ``get_settings().repo_root``.

    Returns:
        Absolute path to ``data/output/champions/champions.json``.
    """
    root: Path = repo or get_settings().repo_root
    directory: Path = root / "data" / "output" / "champions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "champions.json"


def write_manifest(
    entries: dict[str, dict[str, Any]],
    *,
    source_run_id: str,
    repo: Path | None = None,
) -> Path:
    """Atomically write the champion manifest.

    Args:
        entries: Mapping of ``model_name`` → manifest entry. Each entry
            must contain ``model_type``, ``promoted_at``, and ``metrics``.
            ``source_run_id`` is stamped by this function using the value
            passed in, so callers should not include it per-entry.
        source_run_id: Identifier for the write operation (e.g. the
            full-retrain timestamp). Applied to every entry so all
            entries in one write share provenance.
        repo: Repository root override.

    Returns:
        Absolute path to the written manifest.

    Notes:
        Write is atomic via ``os.replace``: the manifest is written to
        a sibling ``.tmp`` file and renamed on success. Readers never
        observe a partially-written manifest.
    """
    path: Path = _manifest_path(repo)
    tmp_path: Path = path.with_suffix(".json.tmp")

    stamped_entries: dict[str, dict[str, Any]] = {}
    for model_name, entry in entries.items():
        stamped_entries[model_name] = {
            "model_type": entry["model_type"],
            "promoted_at": entry["promoted_at"],
            "source_run_id": source_run_id,
            "metrics": dict(entry.get("metrics", {})),
        }

    manifest: dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "updated_at": datetime.now(UTC).isoformat(),
        "models": stamped_entries,
    }

    tmp_path.write_text(json.dumps(manifest, indent=2))
    tmp_path.replace(path)  # atomic on POSIX + Windows
    return path


def read_manifest(repo: Path | None = None) -> dict[str, Any]:
    """Load the champion manifest from disk.

    Args:
        repo: Repository root override.

    Returns:
        The manifest as a dict.

    Raises:
        ChampionNotFoundError: If the manifest file does not exist.
    """
    path: Path = _manifest_path(repo)
    if not path.exists():
        raise ChampionNotFoundError(
            f"Champion manifest not found at {path}. "
            f"Run `gridiron full-retrain` or "
            f"`gridiron evaluate select-model --write-manifest` to create it."
        )
    return json.loads(path.read_text())


def resolve_current_champion(
    model_name: str,
    *,
    repo: Path | None = None,
) -> tuple[str, str]:
    """Return the current champion ``(model_name, model_type)`` for ``model_name``.

    This is the primary read API used by CLI commands (``output predictions``,
    ``weekly-predict``, edges report), the API layer, and downstream analysis
    scripts. It returns just the identity — for full metadata, use
    :func:`resolve_current_champion_with_metadata`.

    Args:
        model_name: The model purpose (e.g. ``"win_prob"``, ``"total"``,
            ``"qb_pass_yards"``).
        repo: Repository root override.

    Returns:
        Tuple of ``(model_name, model_type)``. The ``model_name`` is echoed
        back for symmetry with the composite-identity convention used
        elsewhere in the codebase.

    Raises:
        ChampionNotFoundError: If the manifest doesn't exist, or if it
            exists but has no entry for ``model_name``.
    """
    manifest: dict[str, Any] = read_manifest(repo=repo)
    models: dict[str, Any] = manifest.get("models", {})
    if model_name not in models:
        available: list[str] = sorted(models.keys())
        raise ChampionNotFoundError(
            f"No champion registered for model_name={model_name!r}. Available: {available}"
        )
    return (model_name, models[model_name]["model_type"])


def resolve_current_champion_with_metadata(
    model_name: str,
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Return the full manifest entry for ``model_name``.

    Use this when consumers need ``promoted_at``, ``source_run_id``, or
    ``metrics`` in addition to the identity — e.g., displaying "champion
    promoted 3 days ago" or "champion selected on Brier=0.213" in the UI.

    Args:
        model_name: The model purpose.
        repo: Repository root override.

    Returns:
        The full manifest entry as a dict, containing at least
        ``model_type``, ``promoted_at``, ``source_run_id``, and ``metrics``.

    Raises:
        ChampionNotFoundError: If the manifest doesn't exist or has no
            entry for ``model_name``.
    """
    manifest: dict[str, Any] = read_manifest(repo=repo)
    models: dict[str, Any] = manifest.get("models", {})
    if model_name not in models:
        available: list[str] = sorted(models.keys())
        raise ChampionNotFoundError(
            f"No champion registered for model_name={model_name!r}. Available: {available}"
        )
    entry: dict[str, Any] = models[model_name]
    return dict(entry)  # defensive copy


def list_current_champions(repo: Path | None = None) -> dict[str, tuple[str, str]]:
    """Return all currently-registered ``(model_name, model_type)`` pairs.

    Useful for introspection and for the ``full-retrain`` baseline report to
    identify which models are current champions.

    Args:
        repo: Repository root override.

    Returns:
        Dict mapping ``model_name`` to ``(model_name, model_type)`` tuples.
        Returns empty dict if manifest doesn't exist (this is not an error
        for this discovery-shaped API).
    """
    try:
        manifest: dict[str, Any] = read_manifest(repo=repo)
    except ChampionNotFoundError:
        return {}

    models: dict[str, Any] = manifest.get("models", {})
    return {model_name: (model_name, entry["model_type"]) for model_name, entry in models.items()}

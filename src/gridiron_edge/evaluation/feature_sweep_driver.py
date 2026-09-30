# src/gridiron_edge/evaluation/feature_sweep_driver.py

"""Checkpointed driver for the consolidated feature sweep.

Runs every family through ``feature_sweep_run.run_family`` and keeps a
per-family checkpoint after every completed step, so an interrupted sweep
resumes without repeating finished steps. Checkpoints are local scratch
state (pickled raw predictions) in a caller-chosen directory, not a
product: nothing is written under ``data/`` and no champion or prediction
artifact is touched.

Public API
----------
ALL_FAMILIES            the five sweep families
run_sweep               modeling frame -> {family label: FamilyRun}
load_runs               checkpoint directory -> {family label: FamilyRun}
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import pickle
from typing import Final

from pandas import DataFrame

from gridiron_edge.evaluation.feature_sweep_fit import FamilySpec
from gridiron_edge.evaluation.feature_sweep_rows import MatchedRows, build_matched_rows
from gridiron_edge.evaluation.feature_sweep_run import (
    FamilyRun,
    make_feature_provider,
    run_family,
)
from gridiron_edge.evaluation.feature_sweep_steps import build_sweep_steps
from gridiron_edge.models.game_prediction.base import GameModelType

logger = logging.getLogger(__name__)

CHECKPOINT_SUBDIR: Final[str] = "checkpoints"

ALL_FAMILIES: Final[tuple[FamilySpec, ...]] = (
    FamilySpec("total", GameModelType.XGBOOST, "expanded"),
    FamilySpec("total", GameModelType.RANDOM_FOREST, "expanded"),
    FamilySpec("win", GameModelType.XGBOOST, "expanded"),
    FamilySpec("win", GameModelType.RANDOM_FOREST, "expanded"),
    FamilySpec("win", GameModelType.LOGISTIC, "combined"),
)

TARGET_COLUMNS: Final[dict[str, str]] = {"win": "HOME_WIN", "total": "ACTUAL_TOTAL"}


def _checkpoint_path(output_dir: Path, label: str) -> Path:
    return output_dir / CHECKPOINT_SUBDIR / f"{label.replace('/', '__')}.pkl"


def _save(path: Path, run: FamilyRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(run, handle)
    tmp.replace(path)


def _load(path: Path) -> FamilyRun | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_runs(output_dir: Path) -> dict[str, FamilyRun]:
    """Load every family checkpoint in ``output_dir``."""
    runs: dict[str, FamilyRun] = {}
    for path in sorted((output_dir / CHECKPOINT_SUBDIR).glob("*.pkl")):
        run = _load(path)
        if run is not None:
            runs[run.spec.label] = run
    return runs


def run_sweep(
    df: DataFrame,
    repo: Path,
    output_dir: Path,
    *,
    families: tuple[FamilySpec, ...] = ALL_FAMILIES,
    on_step: Callable[[str, str, float], None] | None = None,
) -> tuple[dict[str, FamilyRun], dict[str, MatchedRows]]:
    """Run (or resume) the sweep for ``families``.

    Args:
        df: Canonical modeling frame.
        repo: Repository root (read-only, for EPA-window rebuilds).
        output_dir: Directory for checkpoints; created if missing.
        families: Families to run, in order.
        on_step: Optional ``(family, step, seconds)`` progress callback.

    Returns:
        ``(runs, matched_rows)`` keyed by family label and target.
    """
    matched: dict[str, MatchedRows] = {}
    runs: dict[str, FamilyRun] = {}
    for spec in families:
        if spec.target not in matched:
            matched[spec.target] = build_matched_rows(df, TARGET_COLUMNS[spec.target])
        rows = matched[spec.target]
        path = _checkpoint_path(output_dir, spec.label)
        resume = _load(path)
        steps = build_sweep_steps(spec.feature_set_key)
        if resume is not None and len(resume.steps) >= len(steps):
            logger.info("%s already complete; skipping", spec.label)
            runs[spec.label] = resume
            continue
        provider = make_feature_provider(rows, spec, repo)
        runs[spec.label] = run_family(
            spec,
            rows,
            steps,
            provider,
            on_step=on_step,
            resume=resume,
            checkpoint=lambda run, path=path: _save(path, run),
        )
        _save(path, runs[spec.label])
    return runs, matched

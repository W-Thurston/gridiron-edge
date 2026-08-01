# src/gridiron_edge/models/game_prediction/weekly_spread_product.py

"""Derived spread attachment for the schedule-complete weekly win product."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.models.game_prediction.post_process import (
    load_model_calibrations,
    win_prob_to_spread,
)
from gridiron_edge.models.game_prediction.weekly_win_product import WeeklyWinStatus


class WeeklySpreadStatus(StrEnum):
    """Availability state for one scheduled game's derived spread."""

    AVAILABLE = "available"
    WIN_UNAVAILABLE = "win_unavailable"
    CALIBRATION_UNAVAILABLE = "calibration_unavailable"


@dataclass(frozen=True)
class SpreadCalibration:
    """Exact persisted spread calibration for one win model identity."""

    model_name: str
    model_type: str
    calibration_key: str
    sigma: float
    margin_std: float
    updated_at: str

    def __post_init__(self) -> None:
        """Validate strict persisted-calibration invariants."""
        for field_name, value in (
            ("model_name", self.model_name),
            ("model_type", self.model_type),
            ("calibration_key", self.calibration_key),
            ("updated_at", self.updated_at),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
        if self.sigma <= 0:
            raise ValueError("sigma must be greater than 0.")
        if self.margin_std <= 0:
            raise ValueError("margin_std must be greater than 0.")


_SPREAD_COLUMNS: Final[tuple[str, ...]] = (
    "spread_status",
    "model_spread",
    "spread_uncertainty",
    "spread_source_event_id",
    "spread_model_name",
    "spread_model_type",
    "spread_calibration_key",
    "spread_calibration_updated_at",
)

_REQUIRED_WIN_COLUMNS: Final[tuple[str, ...]] = (
    "win_status",
    "home_win_prob",
    "win_model_name",
    "win_model_type",
    "win_event_id",
)


def calibration_key(model_name: str, model_type: str) -> str:
    """Return the persisted calibration key for one model identity."""
    return f"{model_name}_{model_type}"


def _number(payload: dict[str, object], field_name: str) -> float | None:
    """Return one finite numeric payload value, or None."""
    value = payload.get(field_name)
    if not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not pd.notna(numeric) or numeric <= 0:
        return None
    return numeric


def parse_spread_calibration(
    calibrations: dict[str, dict[str, object]],
    *,
    model_name: str,
    model_type: str,
) -> SpreadCalibration | None:
    """Parse one exact persisted calibration without using fallbacks."""
    key = calibration_key(model_name, model_type)
    payload = calibrations.get(key)
    if payload is None:
        return None

    sigma = _number(payload, "sigma")
    margin_std = _number(payload, "margin_std")
    updated_at_value = payload.get("updated_at")
    if sigma is None or margin_std is None or not isinstance(updated_at_value, str):
        return None
    updated_at = updated_at_value.strip()
    if not updated_at:
        return None

    return SpreadCalibration(
        model_name=model_name,
        model_type=model_type,
        calibration_key=key,
        sigma=sigma,
        margin_std=margin_std,
        updated_at=updated_at,
    )


def load_spread_calibration(
    model_name: str,
    model_type: str,
    *,
    repo: Path | None = None,
) -> SpreadCalibration | None:
    """Load one exact persisted calibration without default substitution."""
    return parse_spread_calibration(
        load_model_calibrations(repo),
        model_name=model_name,
        model_type=model_type,
    )


def _require_win_columns(win_product: DataFrame) -> None:
    """Require the canonical win-product contract."""
    missing = sorted(set(_REQUIRED_WIN_COLUMNS) - set(win_product.columns))
    if missing:
        raise ValueError("Win product is missing required columns: " + ", ".join(missing))


def _blocked_values(status: WeeklySpreadStatus) -> dict[str, object]:
    """Return nullable derived-spread fields for one blocked row."""
    return {
        "spread_status": status.value,
        "model_spread": pd.NA,
        "spread_uncertainty": pd.NA,
        "spread_source_event_id": pd.NA,
        "spread_model_name": pd.NA,
        "spread_model_type": pd.NA,
        "spread_calibration_key": pd.NA,
        "spread_calibration_updated_at": pd.NA,
    }


def _available_values(
    row: Series,
    calibration: SpreadCalibration,
) -> dict[str, object]:
    """Derive one spread using the selected win model calibration."""
    if str(row["win_model_name"]) != calibration.model_name:
        raise ValueError("Spread calibration model_name does not match win forecast.")
    if str(row["win_model_type"]) != calibration.model_type:
        raise ValueError("Spread calibration model_type does not match win forecast.")

    home_win_probability = row["home_win_prob"]
    if pd.isna(home_win_probability):
        raise ValueError("Available win row must contain home_win_prob.")
    probability = float(home_win_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("home_win_prob must be between 0 and 1.")

    event_id = str(row["win_event_id"])
    if not event_id.strip() or event_id == "<NA>":
        raise ValueError("Available win row must contain win_event_id.")

    return {
        "spread_status": WeeklySpreadStatus.AVAILABLE.value,
        "model_spread": win_prob_to_spread(
            probability,
            sigma=calibration.sigma,
        ),
        "spread_uncertainty": calibration.margin_std,
        "spread_source_event_id": event_id,
        "spread_model_name": calibration.model_name,
        "spread_model_type": calibration.model_type,
        "spread_calibration_key": calibration.calibration_key,
        "spread_calibration_updated_at": calibration.updated_at,
    }


def attach_derived_spreads(
    win_product: DataFrame,
    calibrations: dict[tuple[str, str], SpreadCalibration],
) -> DataFrame:
    """Attach a calibrated derived spread or explicit blocker to every row.

    Spread convention is NFL home-line convention: negative means the home
    team is favored, positive means the away team is favored, and zero is
    pick'em. The function preserves the win-product row count and ordering.
    """
    _require_win_columns(win_product)
    source = win_product.copy()
    spread_values: list[dict[str, object]] = []

    for _, row in source.iterrows():
        if str(row["win_status"]) != WeeklyWinStatus.AVAILABLE.value:
            spread_values.append(_blocked_values(WeeklySpreadStatus.WIN_UNAVAILABLE))
            continue

        model_name = str(row["win_model_name"])
        model_type = str(row["win_model_type"])
        calibration = calibrations.get((model_name, model_type))
        if calibration is None:
            spread_values.append(_blocked_values(WeeklySpreadStatus.CALIBRATION_UNAVAILABLE))
            continue

        spread_values.append(_available_values(row, calibration))

    spread_frame = DataFrame(spread_values, index=source.index)
    return pd.concat([source, spread_frame], axis=1)


def load_and_attach_derived_spreads(
    win_product: DataFrame,
    *,
    repo: Path | None = None,
) -> DataFrame:
    """Load exact persisted calibrations and attach schedule-complete spreads."""
    _require_win_columns(win_product)
    registry = load_model_calibrations(repo)
    calibrations: dict[tuple[str, str], SpreadCalibration] = {}

    available = win_product.loc[
        win_product["win_status"] == WeeklyWinStatus.AVAILABLE.value,
        ["win_model_name", "win_model_type"],
    ].drop_duplicates()

    for _, identity in available.iterrows():
        model_name = str(identity["win_model_name"])
        model_type = str(identity["win_model_type"])
        calibration = parse_spread_calibration(
            registry,
            model_name=model_name,
            model_type=model_type,
        )
        if calibration is not None:
            calibrations[(model_name, model_type)] = calibration

    return attach_derived_spreads(win_product, calibrations)

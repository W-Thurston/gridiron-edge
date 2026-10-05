# src/gridiron_edge/evaluation/weekly_recap_starters.py

"""The weekly recap's starting-QB section: what each forecast assumed against who started.

For every scheduled game and side the section records three predictors of the starter, as the
forecast saw them (``DECISIONS.md`` D77): the assumed starter, nflverse's schedule value at
composition, and the carried-forward starter (D65). It scores each against the actual starter.

* **Record.** A product row carries its own record. A row without one (a schema-1 product, or
  a row marked ``sidecar_unavailable``) is scored from a stored capture of the sidecar its run
  was bound to, through the same resolver that produced the product's record, and is labelled
  ``sidecar_capture``. Without a capture it reads ``not_recorded``.
* **Actual starter.** nflverse's ``*_qb_id`` for the completed game when that player has at
  least one pass attempt for the team that week in the player game logs, otherwise the team's
  top passer that week. nflverse can keep a stale pre-game value after the game; the attempt
  check replaces it while keeping a starter who left mid-game.

Pure: no I/O. The recap builder supplies the product, the captures and the game logs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, cast

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.features.team.qb_profile import resolve_starting_qb_provenance
from gridiron_edge.models.game_prediction.weekly_starting_qb import (
    STATUS_COLUMN,
    StartingQbStatus,
    has_starting_qb_columns,
)

SIDES: Final[tuple[str, ...]] = ("away", "home")
_PASSER_COLUMNS: Final[tuple[str, ...]] = (
    "team",
    "player_id",
    "player_display_name",
    "attempts",
)
_LOG_COLUMNS: Final[tuple[str, ...]] = (
    *_PASSER_COLUMNS,
    "season",
    "season_type",
    "week",
)
_SIDE_STRING_FIELDS: Final[tuple[str, ...]] = (
    "starter_status",
    "starter_record_source",
    "assumed_qb_id",
    "assumed_qb_name",
    "assumed_qb_source",
    "assumed_qb_as_of",
    "nflverse_qb_id",
    "nflverse_qb_name",
    "carried_forward_qb_id",
    "carried_forward_qb_name",
    "actual_qb_id",
    "actual_qb_name",
    "actual_qb_source",
)
# Predictor name, the column holding its pick, the column holding whether it was right.
_PREDICTORS: Final[tuple[tuple[str, str, str], ...]] = (
    ("assumed", "assumed_qb_id", "assumed_correct"),
    ("nflverse", "nflverse_qb_id", "nflverse_correct"),
    ("carry_forward", "carried_forward_qb_id", "carry_forward_correct"),
)
STARTER_STRING_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{side}_{field}" for side in SIDES for field in _SIDE_STRING_FIELDS
)
STARTER_BOOLEAN_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{side}_{flag}" for side in SIDES for _, _, flag in _PREDICTORS
)
ACTUAL_STARTER_CONVENTION: Final[str] = (
    "nflverse *_qb_id when that player has a pass attempt for the team that week in the player "
    "game logs, otherwise the team's top passer that week"
)


class StarterStatus(StrEnum):
    """Whether one side's starter assumptions can be scored."""

    AVAILABLE = "available"
    # The row carries no starter record and no capture of its run's sidecar exists.
    NOT_RECORDED = "not_recorded"
    # The record exists but resolved no starter for this side.
    STARTER_UNRESOLVED = "starter_unresolved"
    NOT_IN_PRODUCT = "not_in_product"
    # The game logs hold no pass attempt for the team that week.
    ACTUAL_MISSING = "actual_missing"


class StarterRecordSource(StrEnum):
    """Where a side's starter record came from."""

    PRODUCT_RECORD = "product_record"
    SIDECAR_CAPTURE = "sidecar_capture"


class ActualStarterSource(StrEnum):
    """Where the actual starter came from."""

    NFLVERSE_SCHEDULE = "nflverse_schedule"
    GAME_LOG_TOP_PASSER = "game_log_top_passer"


@dataclass(frozen=True, slots=True)
class SidecarCapture:
    """A digest-verified copy of the QB/coach sidecar one forecast run was bound to."""

    digest: str
    sidecar: DataFrame


@dataclass(frozen=True, slots=True)
class PredictorAccuracy:
    """How often one predictor named the actual starter, over the slots it made a pick for."""

    count: int
    correct: int
    accuracy: float | None


@dataclass(frozen=True, slots=True)
class StarterSummary:
    """The week's starter slots by status, and each predictor's accuracy over scored slots."""

    slots: int
    scored: int
    from_capture: int
    not_recorded: int
    starter_unresolved: int
    not_in_product: int
    actual_missing: int
    actual_from_game_logs: int
    assumed: PredictorAccuracy
    nflverse: PredictorAccuracy
    carry_forward: PredictorAccuracy


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def runs_without_starter_record(product: DataFrame) -> tuple[str, ...]:
    """Return the runs of product rows that carry no starter record, sorted.

    These are the runs whose sidecar captures can stand in for the missing record.
    """
    if has_starting_qb_columns(product):
        rows = product.loc[product[STATUS_COLUMN] == StartingQbStatus.SIDECAR_UNAVAILABLE.value]
    else:
        rows = product
    return tuple(sorted(set(rows["win_run_id"].dropna().astype(str))))


def week_passers(player_game_logs: DataFrame, *, season_int: int, week: int) -> DataFrame:
    """Return the week's regular-season passers (at least one attempt) in a canonical order."""
    missing = sorted(set(_LOG_COLUMNS) - set(player_game_logs.columns))
    if missing:
        raise ValueError("Player game logs are missing required columns: " + ", ".join(missing))
    scoped = player_game_logs.loc[
        (player_game_logs["season"].astype(int) == season_int)
        & (player_game_logs["season_type"] == "REG")
        & (player_game_logs["week"].astype(int) == week)
        & (player_game_logs["attempts"].fillna(0) > 0),
        list(_PASSER_COLUMNS),
    ]
    rows = scoped.assign(
        team=scoped["team"].astype(str),
        player_id=scoped["player_id"].astype(str),
        player_display_name=scoped["player_display_name"].astype(str),
        attempts=scoped["attempts"].astype(float),
    )
    if rows.duplicated(["team", "player_id"]).any():
        raise ValueError("Player game logs hold a player twice for one team and week.")
    return rows.sort_values(["team", "player_id"]).reset_index(drop=True)


def player_names(
    player_game_logs: DataFrame,
    *,
    season_int: int,
    week: int,
) -> dict[str, str]:
    """Map player ids to display names from the season's game logs up to ``week``."""
    scoped = player_game_logs.loc[
        (player_game_logs["season"].astype(int) == season_int)
        & (player_game_logs["week"].astype(int) <= week),
        ["player_id", "player_display_name"],
    ].dropna()
    return dict(
        zip(
            scoped["player_id"].astype(str),
            scoped["player_display_name"].astype(str),
            strict=True,
        )
    )


# ---------------------------------------------------------------------------
# Per-game columns
# ---------------------------------------------------------------------------


def _text(value: object) -> str | None:
    if value is None or pd.isna(cast(Any, value)):
        return None
    text = str(value)
    return text or None


def _actual(
    team: str,
    nflverse_id: str | None,
    passers_by_team: Mapping[str, DataFrame],
) -> tuple[str | None, str | None, str | None]:
    """Return the actual starter's id, name and source, or nulls when the logs hold none."""
    passers = passers_by_team.get(team)
    if passers is None or passers.empty:
        return None, None, None
    names = dict(zip(passers["player_id"], passers["player_display_name"], strict=True))
    if nflverse_id is not None and nflverse_id in names:
        return nflverse_id, names[nflverse_id], ActualStarterSource.NFLVERSE_SCHEDULE.value
    top = passers.sort_values(["attempts", "player_id"], ascending=[False, True]).iloc[0]
    return (
        str(top["player_id"]),
        str(top["player_display_name"]),
        ActualStarterSource.GAME_LOG_TOP_PASSER.value,
    )


def _capture_records(
    captures: Mapping[str, SidecarCapture],
) -> dict[tuple[str, str, str], dict[str, str | None]]:
    """Resolve each capture once: (run, game, side) -> the record the product would carry."""
    records: dict[tuple[str, str, str], dict[str, str | None]] = {}
    for run_id, capture in captures.items():
        provenance = resolve_starting_qb_provenance(capture.sidecar)
        for item in provenance.to_dict("records"):
            records[(run_id, str(item["GAME_ID"]), str(item["SIDE"]).lower())] = {
                "qb_id": _text(item["RESOLVED_QB_ID"]),
                "qb_name": None,
                "qb_source": _text(item["QB_SOURCE"]),
                "qb_projection_as_of": _text(item["PROJECTION_AS_OF"]),
                "qb_nflverse_id": _text(item["NFLVERSE_QB_ID"]),
                "qb_carried_forward_id": _text(item["CARRIED_FORWARD_QB_ID"]),
            }
    return records


def _capture_names(captures: Mapping[str, SidecarCapture]) -> dict[str, str]:
    names: dict[str, str] = {}
    for run_id in sorted(captures):
        sidecar = captures[run_id].sidecar
        for side in ("AWAY", "HOME"):
            if f"{side}_QB_NAME" not in sidecar.columns:
                continue
            pairs = sidecar[[f"{side}_QB_ID", f"{side}_QB_NAME"]].dropna()
            names.update(
                zip(pairs.iloc[:, 0].astype(str), pairs.iloc[:, 1].astype(str), strict=True)
            )
    return names


def _side_record(
    product_row: Series | None,
    side: str,
    *,
    recorded: bool,
    capture_records: Mapping[tuple[str, str, str], dict[str, str | None]],
) -> tuple[dict[str, str | None] | None, StarterRecordSource | None]:
    """Return one side's record and where it came from, or ``None`` when there is none."""
    if product_row is None:
        return None, None
    if recorded and str(product_row[STATUS_COLUMN]) != StartingQbStatus.SIDECAR_UNAVAILABLE.value:
        fields = (
            "qb_id",
            "qb_name",
            "qb_source",
            "qb_projection_as_of",
            "qb_nflverse_id",
            "qb_carried_forward_id",
        )
        return (
            {field: _text(product_row[f"{side}_{field}"]) for field in fields},
            StarterRecordSource.PRODUCT_RECORD,
        )
    key = (str(product_row["win_run_id"]), str(product_row["game_id"]), side)
    record = capture_records.get(key)
    return (None, None) if record is None else (record, StarterRecordSource.SIDECAR_CAPTURE)


def starter_columns(
    schedule: DataFrame,
    product: DataFrame,
    *,
    captures: Mapping[str, SidecarCapture],
    passers: DataFrame,
    names: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Return, per scheduled game id, the starter columns of both sides.

    ``captures`` maps a run id to the verified capture of the sidecar that run was bound to;
    ``passers`` is :func:`week_passers`; ``names`` maps player ids to display names (the
    product's own recorded names take precedence for the assumed starter).
    """
    recorded = has_starting_qb_columns(product)
    product_rows = {str(row["game_id"]): row for _, row in product.iterrows()}
    capture_records = _capture_records(captures)
    known = {**_capture_names(captures), **names}
    passers_by_team = {str(team): rows for team, rows in passers.groupby("team", sort=True)}

    columns: dict[str, dict[str, object]] = {}
    for _, game in schedule.iterrows():
        game_id = str(game["game_id"])
        product_row = product_rows.get(game_id)
        row: dict[str, object] = {}
        for side in SIDES:
            actual_id, actual_name, actual_source = _actual(
                str(game[f"{side}_team"]), _text(game[f"{side}_qb_id"]), passers_by_team
            )
            record, source = _side_record(
                product_row, side, recorded=recorded, capture_records=capture_records
            )
            if product_row is None:
                status = StarterStatus.NOT_IN_PRODUCT
            elif record is None:
                status = StarterStatus.NOT_RECORDED
            elif record["qb_id"] is None:
                status = StarterStatus.STARTER_UNRESOLVED
            elif actual_id is None:
                status = StarterStatus.ACTUAL_MISSING
            else:
                status = StarterStatus.AVAILABLE
            values = record or {}
            assumed_id = values.get("qb_id")
            nflverse_id = values.get("qb_nflverse_id")
            carried_id = values.get("qb_carried_forward_id")
            side_values: dict[str, object] = {
                "starter_status": status.value,
                "starter_record_source": None if source is None else source.value,
                "assumed_qb_id": assumed_id,
                "assumed_qb_name": values.get("qb_name") or _name(known, assumed_id),
                "assumed_qb_source": values.get("qb_source"),
                "assumed_qb_as_of": values.get("qb_projection_as_of"),
                "nflverse_qb_id": nflverse_id,
                "nflverse_qb_name": _name(known, nflverse_id),
                "carried_forward_qb_id": carried_id,
                "carried_forward_qb_name": _name(known, carried_id),
                "actual_qb_id": actual_id,
                "actual_qb_name": actual_name,
                "actual_qb_source": actual_source,
            }
            for _, pick_field, flag in _PREDICTORS:
                pick = side_values[pick_field]
                scored = status is StarterStatus.AVAILABLE and pick is not None
                side_values[flag] = (pick == actual_id) if scored else None
            row |= {f"{side}_{field}": value for field, value in side_values.items()}
        columns[game_id] = row
    return columns


def _name(names: Mapping[str, str], player_id: str | None) -> str | None:
    return None if player_id is None else names.get(player_id)


# ---------------------------------------------------------------------------
# Week summary
# ---------------------------------------------------------------------------


def _side_values(games: DataFrame, field: str) -> Series:
    return pd.concat([games[f"{side}_{field}"] for side in SIDES], ignore_index=True)


def starter_summary(games: DataFrame) -> StarterSummary:
    """Count the week's starter slots by status and score each predictor over scored slots."""
    statuses = _side_values(games, "starter_status")
    scored = statuses == StarterStatus.AVAILABLE.value

    def accuracy(flag: str) -> PredictorAccuracy:
        values = _side_values(games, flag)[scored].dropna().astype(bool)
        count = len(values)
        correct = int(values.sum())
        return PredictorAccuracy(count, correct, correct / count if count else None)

    def count(status: StarterStatus) -> int:
        return int((statuses == status.value).sum())

    return StarterSummary(
        slots=len(statuses),
        scored=int(scored.sum()),
        from_capture=int(
            (
                scored
                & (
                    _side_values(games, "starter_record_source")
                    == StarterRecordSource.SIDECAR_CAPTURE.value
                )
            ).sum()
        ),
        not_recorded=count(StarterStatus.NOT_RECORDED),
        starter_unresolved=count(StarterStatus.STARTER_UNRESOLVED),
        not_in_product=count(StarterStatus.NOT_IN_PRODUCT),
        actual_missing=count(StarterStatus.ACTUAL_MISSING),
        actual_from_game_logs=int(
            (
                _side_values(games, "actual_qb_source")
                == ActualStarterSource.GAME_LOG_TOP_PASSER.value
            ).sum()
        ),
        assumed=accuracy("assumed_correct"),
        nflverse=accuracy("nflverse_correct"),
        carry_forward=accuracy("carry_forward_correct"),
    )


__all__ = [
    "ACTUAL_STARTER_CONVENTION",
    "STARTER_BOOLEAN_COLUMNS",
    "STARTER_STRING_COLUMNS",
    "ActualStarterSource",
    "PredictorAccuracy",
    "SidecarCapture",
    "StarterRecordSource",
    "StarterStatus",
    "StarterSummary",
    "player_names",
    "runs_without_starter_record",
    "starter_columns",
    "starter_summary",
    "week_passers",
]

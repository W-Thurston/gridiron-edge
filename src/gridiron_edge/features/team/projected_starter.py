# src/gridiron_edge/features/team/projected_starter.py

"""Pure projected-starting-QB resolver (``DECISIONS.md`` D75).

The resolver turns one team's depth chart quarterbacks and weekly availability
statuses into an ordered candidate list and a projected starter. It returns the
inputs it used, not just a name, so later work (hot-swapping a quarterback, a
scenario engine) can reuse the same list. Nothing here reads or writes anything
but its arguments.

Public API
----------
Candidate                     dataclass  - one chart quarterback with its status
ordered_candidates            chart + statuses -> candidate list
project_starter               candidate list -> projected starter id
prepare_snapshots             timestamped depth chart -> quarterback rows by team
latest_snapshot_chart         one team's rows + cutoff -> latest chart and its time
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd
from pandas import DataFrame, Timestamp

#: Statuses that remove a quarterback from consideration in the primary rule.
UNAVAILABLE_PRIMARY: Final[frozenset[str]] = frozenset({"Out", "Doubtful"})
#: The variant also skips Questionable; reported for information only.
UNAVAILABLE_VARIANT: Final[frozenset[str]] = UNAVAILABLE_PRIMARY | {"Questionable"}


@dataclass(frozen=True)
class Candidate:
    """One chart quarterback with the inputs the resolver used.

    Attributes:
        gsis_id: Player identifier.
        chart_rank: Depth chart rank (1 is the top quarterback).
        status: Reported game status, or ``None`` when not on the report.
        available: Whether the status leaves the player eligible.
    """

    gsis_id: str
    chart_rank: int
    status: str | None
    available: bool


def ordered_candidates(
    chart: Sequence[tuple[int, str]],
    statuses: Mapping[str, str | None],
    unavailable: frozenset[str],
) -> list[Candidate]:
    """Order chart quarterbacks by rank and attach their availability.

    Args:
        chart: ``(chart_rank, gsis_id)`` pairs for one team and week.
        statuses: Reported game status by player id.
        unavailable: Statuses that make a player ineligible.

    Returns:
        Candidates by ascending rank; a repeated player keeps its best rank and
        ties keep their input order.
    """
    seen: set[str] = set()
    candidates: list[Candidate] = []
    for rank, player_id in sorted(chart, key=lambda item: item[0]):
        if player_id in seen:
            continue
        seen.add(player_id)
        status: str | None = statuses.get(player_id)
        candidates.append(Candidate(player_id, rank, status, status not in unavailable))
    return candidates


def project_starter(candidates: Sequence[Candidate]) -> str | None:
    """Return the first available candidate, else the top of the chart, else ``None``."""
    for candidate in candidates:
        if candidate.available:
            return candidate.gsis_id
    return candidates[0].gsis_id if candidates else None


def prepare_snapshots(snapshots: DataFrame) -> dict[str, DataFrame]:
    """Index the timestamped depth chart's quarterback rows by team.

    Args:
        snapshots: Depth chart in the timestamped format (``dt``, ``team``,
            ``pos_abb``, ``pos_rank``, ``gsis_id``).

    Returns:
        Quarterback rows per team with a parsed UTC ``when`` column.
    """
    qbs = snapshots[snapshots["pos_abb"] == "QB"].copy()
    qbs["when"] = pd.to_datetime(qbs["dt"], utc=True)
    return {str(team): frame for team, frame in qbs.groupby("team")}


def latest_snapshot_chart(
    team_snapshots: DataFrame,
    *,
    cutoff: Timestamp | None = None,
) -> tuple[list[tuple[int, str]], Timestamp | None]:
    """Return one team's latest chart at or before ``cutoff`` and its snapshot time.

    Args:
        team_snapshots: One team's rows from ``prepare_snapshots``.
        cutoff: Latest snapshot time to consider; every snapshot when ``None``.

    Returns:
        ``(chart, as_of)`` where ``chart`` is ``(rank, gsis_id)`` pairs and
        ``as_of`` is the snapshot time, or ``([], None)`` when none is visible.
    """
    visible = team_snapshots if cutoff is None else team_snapshots[team_snapshots["when"] <= cutoff]
    if visible.empty:
        return [], None
    as_of: Timestamp = visible["when"].max()
    latest = visible[visible["when"] == as_of]
    chart = [
        (int(rank), player)
        for rank, player in zip(latest["pos_rank"], latest["gsis_id"], strict=True)
        if isinstance(player, str)
    ]
    return chart, as_of

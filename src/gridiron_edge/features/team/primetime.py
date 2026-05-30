# src/gridiron_edge/features/team/primetime.py

"""Primetime game flag feature.

Marks whether a game is broadcast in a nationally televised primetime slot.
Primetime games carry two signals:

1. **Selection bias** — the NFL schedule-makers assign the most appealing
   matchups to primetime. Teams appearing frequently in primetime are
   implicitly endorsed as high-quality by people with access to information
   the model doesn't have (injury reports, locker room sentiment, etc.).

2. **Performance effects** — documented evidence that teams perform
   differently under primetime conditions: heightened intensity, different
   crowd energy, disrupted weekly routine (especially Thursday Night Football
   with its compressed recovery window).

Produces:

    IS_PRIMETIME    int     1 if the game is in a primetime slot, 0 otherwise.

Primetime slots (derived from GAME_DAY_OF_WEEK and GAMETIME):

    Monday Night Football (MNF)   — all Monday games regardless of time
    Sunday Night Football (SNF)   — Sunday games with kickoff >= 20:00
    Thursday Night Football (TNF) — Thursday games with kickoff >= 20:00
    Saturday Night Football       — Saturday games with kickoff >= 20:00
                                    (late-season + playoff primetime)

Sunday afternoon games (13:00, 16:05, 16:25, 16:30) are NOT primetime.
Saturday day games (09:00, 12:00) are NOT primetime.
Games with a missing or unknown gametime (International Series or TBD) are
treated as non-primetime (conservative default; they're usually not SNF/MNF
slots).

Design notes:
    - IS_PRIMETIME is a game-level feature — the same value appears in
      both the TEAM_A and TEAM_B rows for a given game. The model learns
      its effect on win probability directly.
    - Thursday games also benefit from the rest/schedule stress features
      (SHORT_WEEK flag) which capture the compressed recovery. IS_PRIMETIME
      captures the separate broadcast/selection-bias signal.
    - No external data required — derived entirely from GAME_DAY_OF_WEEK
      and GAMETIME columns already in the canonical games CSV.
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING, Final

import pandas as pd

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

logger: Logger = logging.getLogger(__name__)

# Kickoff hour (24-hour) at or above which an evening game is primetime.
# 20:00 covers 8:00 PM kickoffs — the standard SNF/TNF/MNF slot.
_PRIMETIME_HOUR: Final[int] = 20

_PRIMETIME_DAYS_ANY_TIME: Final[frozenset[str]] = frozenset({"Monday"})
_PRIMETIME_DAYS_EVENING: Final[frozenset[str]] = frozenset({"Sunday", "Thursday", "Saturday"})


@FeatureRegistry.register("primetime")
class PrimetimeFeature:
    """Primetime game flag: IS_PRIMETIME.

    Reads GAME_DAY_OF_WEEK and GAMETIME from the canonical games CSV and
    sets IS_PRIMETIME=1 for MNF (all Monday games), SNF (Sunday evenings),
    TNF (Thursday evenings), and Saturday night games.
    """

    spec = FeatureSpec(name="primetime", produces=["IS_PRIMETIME"])

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute the primetime flag and join onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame with at least a GAME_ID column.
            datasets: Provides games() for day and gametime columns.

        Returns:
            Input DataFrame with IS_PRIMETIME appended. Values are 0 or 1.
        """
        games: pd.DataFrame = datasets.games()

        if "GAME_DAY_OF_WEEK" not in games.columns or "GAMETIME" not in games.columns:
            logger.warning(
                "primetime: GAME_DAY_OF_WEEK or GAMETIME not found in games dataset. "
                "Setting IS_PRIMETIME=0 for all rows."
            )
            df = df.copy()
            df["IS_PRIMETIME"] = 0
            return df

        flags: pd.DataFrame = (
            games[["GAME_ID", "GAME_DAY_OF_WEEK", "GAMETIME"]].drop_duplicates("GAME_ID").copy()
        )
        flags["IS_PRIMETIME"] = flags.apply(_is_primetime, axis=1).astype(int)

        return df.merge(
            flags[["GAME_ID", "IS_PRIMETIME"]],
            on="GAME_ID",
            how="left",
        ).assign(IS_PRIMETIME=lambda x: x["IS_PRIMETIME"].fillna(0).astype(int))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _is_primetime(row: pd.Series) -> bool:  # type: ignore[type-arg]
    """Return True if a game row falls in a primetime slot.

    Args:
        row: Row from the games DataFrame with GAME_DAY_OF_WEEK and
            GAMETIME columns.

    Returns:
        True if the game is MNF, SNF, TNF, or Saturday night.
    """
    day: str = str(row.get("GAME_DAY_OF_WEEK", ""))
    gametime: str = str(row.get("GAMETIME", ""))

    if day in _PRIMETIME_DAYS_ANY_TIME:
        return True

    if day in _PRIMETIME_DAYS_EVENING and gametime:
        try:
            hour = int(gametime.split(":")[0])
            return hour >= _PRIMETIME_HOUR
        except (ValueError, IndexError):
            return False

    return False

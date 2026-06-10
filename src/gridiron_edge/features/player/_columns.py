# src/gridiron_edge/features/player/_columns.py
"""Feature column definitions for the prop model feature pipeline.

Builds the authoritative list of feature columns programmatically from
the component modules so it stays in sync automatically.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Game context feature columns (not derived from other modules)
# ---------------------------------------------------------------------------

GAME_CONTEXT_COLS: Final[list[str]] = [
    "is_home",
    "game_spread",
    "over_under",
    "implied_team_total",
    "is_dome",
    "rest_days",
]


def _build_prop_feature_cols() -> list[str]:
    """Build the complete prop feature column list from component modules."""
    from gridiron_edge.features.player.matchup import (
        _MATCHUP_STATS,
        DEFAULT_MATCHUP_WINDOW,
    )
    from gridiron_edge.features.player.rolling import (
        DEFAULT_WINDOWS as ROLLING_WINDOWS,
    )
    from gridiron_edge.features.player.rolling import (
        ROLLING_STAT_COLS,
    )
    from gridiron_edge.features.player.usage import (
        _SHARE_COLS,
    )
    from gridiron_edge.features.player.usage import (
        DEFAULT_WINDOWS as USAGE_WINDOWS,
    )

    cols: list[str] = []

    # Rolling: stat x window x {mean, std}
    for stat in ROLLING_STAT_COLS:
        for w in ROLLING_WINDOWS:
            cols.append(f"{stat}_L{w}_mean")
            cols.append(f"{stat}_L{w}_std")

    # Matchup: stat x {allowed, rank}
    mw = DEFAULT_MATCHUP_WINDOW
    for _positions, _raw_col, name in _MATCHUP_STATS:
        cols.append(f"opp_{name}_allowed_L{mw}")
        cols.append(f"opp_{name}_rank_L{mw}")

    # Usage: share x window
    for share_col in _SHARE_COLS:
        for w in USAGE_WINDOWS:
            cols.append(f"{share_col}_L{w}")

    # Game context
    cols.extend(GAME_CONTEXT_COLS)

    return cols


PROP_FEATURE_COLS: Final[list[str]] = _build_prop_feature_cols()

# tests/unit/api/test_serializers_projections.py

"""Unit tests for projections serializer."""

from __future__ import annotations

import pandas as pd

from gridiron_edge.api.schemas.projections import ProjectionsList
from gridiron_edge.api.serializers.projections import serialize_projections

LONG_TO_SHORT = {
    "Seattle Seahawks": "SEA",
    "Buffalo Bills": "BUF",
    "Baltimore Ravens": "BAL",
}


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TEAM": "SEA",
                "AVG_WINS": 10.87025,
                "P_MAKE_PLAYOFFS": 0.7762,
                "P_REACH_DIV": 0.5497,
                "P_REACH_CONF": 0.3274,
                "P_REACH_SB": 0.1866,
                "P_WIN_SB": 0.1038,
            },
            {
                "TEAM": "BUF",
                "AVG_WINS": 10.3247,
                "P_MAKE_PLAYOFFS": 0.728,
                "P_REACH_DIV": 0.4856,
                "P_REACH_CONF": 0.2747,
                "P_REACH_SB": 0.1582,
                "P_WIN_SB": 0.0875,
            },
        ],
    )


class TestSerializeProjections:
    def test_empty_df_marks_items_unavailable(self) -> None:
        result = serialize_projections(
            pd.DataFrame(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
        )
        assert result.items == []
        assert result.total == 0
        assert result.response_meta is not None
        assert "items" in result.response_meta.field_status

    def test_populated(self) -> None:
        result = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            "2025-11-24T18:30:00Z",
        )
        assert result.total == 2
        assert result.season == "2025-2026"
        assert result.computed_at == "2025-11-24T18:30:00Z"
        # Sort by P_WIN_SB descending, so SEA first
        assert result.items[0].abbr == "SEA"
        assert result.items[0].name == "Seattle Seahawks"
        assert result.items[0].win_sb == 0.1038
        assert result.items[0].make_playoffs == 0.7762

    def test_marks_pending_and_blocked_fields(self) -> None:
        result: ProjectionsList = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
        )
        fs = result.response_meta.field_status
        assert "n_simulations" in fs
        assert "items.clinched" in fs
        assert "items.eliminated" in fs
        # week_over_week_delta is now populated from Elo state, not blocked.
        assert "items.week_over_week_delta" not in fs

    def test_unknown_abbr_falls_back_to_abbr_as_name(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "TEAM": "XXX",
                    "AVG_WINS": 5.0,
                    "P_MAKE_PLAYOFFS": 0.1,
                    "P_REACH_DIV": 0.0,
                    "P_REACH_CONF": 0.0,
                    "P_REACH_SB": 0.0,
                    "P_WIN_SB": 0.0,
                },
            ],
        )
        result = serialize_projections(df, LONG_TO_SHORT, "2025-2026", None)
        # Unknown abbreviation should default name to the abbr
        assert result.items[0].abbr == "XXX"
        assert result.items[0].name == "XXX"

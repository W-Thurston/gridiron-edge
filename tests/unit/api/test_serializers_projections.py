# tests/unit/api/test_serializers_projections.py

"""Unit tests for projections serializer."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from gridiron_edge.api.schemas.projections import ProjectionsList, TeamProjectionRow
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
                "elo_delta": 12.0,
            },
            {
                "TEAM": "BUF",
                "AVG_WINS": 10.3247,
                "P_MAKE_PLAYOFFS": 0.728,
                "P_REACH_DIV": 0.4856,
                "P_REACH_CONF": 0.2747,
                "P_REACH_SB": 0.1582,
                "P_WIN_SB": 0.0875,
                "elo_delta": -4.0,
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
            n_simulations=None,
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
            n_simulations=None,
        )
        assert result.total == 2
        assert result.season == "2025-2026"
        assert result.computed_at == "2025-11-24T18:30:00Z"
        # Sort by P_WIN_SB descending, so SEA first
        assert result.items[0].abbr == "SEA"
        assert result.items[0].name == "Seattle Seahawks"
        assert result.items[0].win_sb == 0.1038
        assert result.items[0].make_playoffs == 0.7762
        assert result.items[0].elo_delta == 12.0

    def test_marks_pending_status_fields(self) -> None:
        result: ProjectionsList = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=None,
        )
        fs = result.response_meta.field_status
        assert "items.clinched" in fs
        assert "items.eliminated" in fs

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
        result = serialize_projections(
            df,
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=None,
        )
        # Unknown abbreviation should default name to the abbr
        assert result.items[0].abbr == "XXX"
        assert result.items[0].name == "XXX"


class TestNSimulations:
    def test_populates_n_simulations(self) -> None:
        result = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )
        assert result.n_simulations == 10000

    def test_null_n_simulations_stays_null(self) -> None:
        result = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=None,
        )
        assert result.n_simulations is None

    def test_no_longer_marks_n_simulations_pending(self) -> None:
        """The field_status marker on n_simulations should be gone
        now that we populate the field."""
        result = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )
        fs = result.response_meta.field_status
        assert "n_simulations" not in fs

    def test_populated_elo_delta_has_no_unavailable_status(self) -> None:
        result: ProjectionsList = serialize_projections(
            _make_df(),
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )

        assert "items.elo_delta" not in result.response_meta.field_status

    def test_all_null_elo_deltas_mark_no_prior_snapshot(self) -> None:
        df: DataFrame = _make_df()
        df["elo_delta"] = None

        result = serialize_projections(
            df,
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )

        status = result.response_meta.field_status["items.elo_delta"]
        assert status.status == "blocked"
        assert status.blocker == "no_prior_snapshot"
        assert status.roadmap == "data"
        assert all(item.elo_delta is None for item in result.items)

    def test_partial_null_elo_deltas_do_not_mark_entire_field_unavailable(self) -> None:
        df: DataFrame = _make_df()
        df.loc[df["TEAM"] == "BUF", "elo_delta"] = None

        result: ProjectionsList = serialize_projections(
            df,
            LONG_TO_SHORT,
            "2025-2026",
            None,
            n_simulations=10000,
        )

        assert "items.elo_delta" not in result.response_meta.field_status
        by_team: dict[str, TeamProjectionRow] = {item.abbr: item for item in result.items}
        assert by_team["SEA"].elo_delta == 12.0
        assert by_team["BUF"].elo_delta is None

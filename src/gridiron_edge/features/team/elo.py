# src/gridiron_edge/features/team/elo.py

"""Canonical Away/Home Elo feature generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import Series

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from pandas import DataFrame

    from gridiron_edge.datasets.accessor import DatasetAccessor


_ELO_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "NFL_TEAM",
    "NFL_YEAR",
    "NFL_WEEK",
)

_ELO_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *_ELO_IDENTITY_COLUMNS,
    "ELO",
)

_HOME_AWAY_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "AWAY_TEAM",
    "HOME_TEAM",
    "YEAR",
    "WEEK_NUM",
)


def _require_columns(
    frame: DataFrame,
    required: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Require an established input schema."""
    missing: list[str] = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _validate_elo_identity(
    elo: DataFrame,
) -> None:
    """Reject duplicate team, season, and week Elo identities."""
    duplicated: Series = elo.duplicated(
        subset=list(_ELO_IDENTITY_COLUMNS),
        keep=False,
    )
    if not duplicated.any():
        return

    identities = (
        elo.loc[
            duplicated,
            list(_ELO_IDENTITY_COLUMNS),
        ]
        .drop_duplicates()
        .sort_values(
            list(_ELO_IDENTITY_COLUMNS),
            kind="stable",
        )
    )

    formatted: list[str] = [
        (f"{row['NFL_TEAM']}/{row['NFL_YEAR']}/{row['NFL_WEEK']}")
        for _, row in identities.iterrows()
    ]

    raise ValueError("Elo state contains duplicate identities: " + ", ".join(formatted))


@FeatureRegistry.register("home_away_elo")
class HomeAwayEloFeature:
    """Join exact weekly Elo ratings for canonical Away and Home teams."""

    spec = FeatureSpec(
        name="home_away_elo",
        produces=[
            "AWAY_ELO",
            "HOME_ELO",
        ],
    )

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach Away and Home Elo ratings without dropping games.

        Every input row remains in the output. Missing team, season, or
        week ratings produce null Elo values.

        Args:
            df: One-row-per-game frame using canonical Away/Home identity.
            datasets: Repository-scoped dataset accessor.

        Returns:
            A new frame containing ``AWAY_ELO`` and ``HOME_ELO``.

        Raises:
            ValueError: If the game frame or Elo state has a malformed
                schema, or if Elo identity is ambiguous.
        """
        _require_columns(
            df,
            _HOME_AWAY_INPUT_COLUMNS,
            label="Home/away game frame",
        )

        elo: DataFrame = datasets.elo_state().copy()

        _require_columns(
            elo,
            _ELO_REQUIRED_COLUMNS,
            label="Elo state",
        )
        _validate_elo_identity(elo)

        source: DataFrame = df.copy()
        source["_HOME_AWAY_ELO_ORDER"] = range(len(source))

        away = elo.loc[
            :,
            list(_ELO_REQUIRED_COLUMNS),
        ].rename(
            columns={
                "NFL_TEAM": "AWAY_TEAM",
                "NFL_YEAR": "YEAR",
                "NFL_WEEK": "WEEK_NUM",
                "ELO": "AWAY_ELO",
            }
        )

        joined: DataFrame = source.merge(
            away,
            how="left",
            on=[
                "AWAY_TEAM",
                "YEAR",
                "WEEK_NUM",
            ],
            sort=False,
            validate="many_to_one",
        )

        home = elo.loc[
            :,
            list(_ELO_REQUIRED_COLUMNS),
        ].rename(
            columns={
                "NFL_TEAM": "HOME_TEAM",
                "NFL_YEAR": "YEAR",
                "NFL_WEEK": "WEEK_NUM",
                "ELO": "HOME_ELO",
            }
        )

        joined = joined.merge(
            home,
            how="left",
            on=[
                "HOME_TEAM",
                "YEAR",
                "WEEK_NUM",
            ],
            sort=False,
            validate="many_to_one",
        )

        return (
            joined.sort_values(
                "_HOME_AWAY_ELO_ORDER",
                kind="stable",
            )
            .drop(
                columns=[
                    "_HOME_AWAY_ELO_ORDER",
                ]
            )
            .reset_index(
                drop=True,
            )
        )

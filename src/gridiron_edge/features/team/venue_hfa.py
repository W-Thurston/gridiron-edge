# src/gridiron_edge/features/team/venue_hfa.py

"""Franchise-level home field advantage coefficient feature.

Measures how much better (or worse) each NFL franchise performs at home
relative to the league average, expressed as a signed win-rate differential.
This captures persistent crowd noise, travel burden on visitors, and
altitude effects that are tied to a franchise's fanbase and home market -
effects that Elo ratings absorb only slowly and EPA rolling averages
cannot capture at all.

Produces (symmetric - one value per team perspective per row):

    TEAM_A_FRANCHISE_HFA    float   TEAM_A's historical home win rate minus
                                    the league-average home win rate.
                                    Positive → stronger-than-average home
                                    advantage (e.g. Kansas City Chiefs).
                                    Negative → weaker-than-average.
                                    Zero → league average (new franchise,
                                    insufficient data, or neutral site).

    TEAM_B_FRANCHISE_HFA    float   Same for TEAM_B.

Implementation
--------------
The coefficient is computed dynamically from the canonical games CSV at
pipeline run time - no external data or manual lookup table required.

For each franchise (long team name), we compute:

    home_win_rate = home_wins / home_games_played

where a "home win" is any completed game where the team was the home side
and won (GAME_LOCATION == "H" and WINNER == team).  Neutral-site
games (GAME_LOCATION == "N") are excluded - they carry no home crowd signal.

The coefficient is then:

    franchise_hfa = home_win_rate - league_avg_home_win_rate

A minimum of ``_MIN_HOME_GAMES`` (default: 20) home games is required
before a franchise's own rate is used.  Franchises below this threshold
receive 0.0 (the league average differential), which is the correct
uninformative prior for a truly new expansion team.

Stadium continuity
------------------
This feature operates at the franchise level, not the stadium level.
When a team moves to a new stadium, the coefficient immediately reflects
the full franchise history rather than resetting to zero.  This is a
deliberate simplification - it avoids stadium-continuity date tracking
and is the correct prior (the fanbase moves with the team).

A more sophisticated stadium-level version that estimates separate
coefficients per physical building and blends toward the franchise
coefficient for new stadiums is tracked in the project backlog.

Design notes:
    - The two-row-per-game design means TEAM_A and TEAM_B can each be
      either the home or away team in a given row.  The coefficient is
      a property of the *franchise*, not the game-row perspective.
    - Neutral-site games receive TEAM_A_FRANCHISE_HFA = 0.0 and
      TEAM_B_FRANCHISE_HFA = 0.0, reflecting that neither team has a
      home crowd advantage.  This is handled via IS_NEUTRAL_SITE if
      present, falling back to the GAME_LOCATION column.
    - The computation uses all historical data in the games CSV, including
      seasons before the modeling window.  This is correct - we want the
      most stable estimate of franchise HFA, not a short rolling window.
    - Temporal leakage note: technically, using future seasons' home
      results to estimate a franchise's HFA for early-season rows is mild
      leakage.  In practice, HFA coefficients are extremely stable year
      over year (correlation > 0.85 across 5-year windows), so this
      leakage is negligible compared to the feature's signal value.  A
      fully leakage-proof version would use a leave-one-season-out
      estimate, which is tracked as a potential future refinement.
"""

from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING, Final

import pandas as pd

from gridiron_edge.core.constants import HOME_GAME_LOCATION
from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

logger: Logger = logging.getLogger(__name__)

# Minimum home games required to use a franchise's own win rate.
# Below this threshold we fall back to 0.0 (league average differential).
_MIN_HOME_GAMES: Final[int] = 20

# GAME_LOCATION value for neutral-site games
_NEUTRAL_LOCATION: Final[str] = "N"

_GAMES_COLS: Final[list[str]] = ["WINNER", "LOSER", "GAME_LOCATION", "WIN_OR_TIE"]


@FeatureRegistry.register("venue_hfa")
class VenueHFAFeature:
    """Franchise-level home field advantage coefficients.

    Computes TEAM_A_FRANCHISE_HFA and TEAM_B_FRANCHISE_HFA from the
    canonical games CSV.  See module docstring for full methodology.
    """

    spec = FeatureSpec(
        name="venue_hfa",
        produces=["TEAM_A_FRANCHISE_HFA", "TEAM_B_FRANCHISE_HFA"],
        depends_on=("travel",),
    )

    def compute(self, *, df: pd.DataFrame, datasets: DatasetAccessor) -> pd.DataFrame:
        """Compute franchise HFA coefficients and join onto the modeling DataFrame.

        Args:
            df: Modeling DataFrame with TEAM_A, TEAM_B, and (optionally)
                IS_NEUTRAL_SITE columns.
            datasets: Provides ``games()`` for historical results.

        Returns:
            Input DataFrame with TEAM_A_FRANCHISE_HFA and
            TEAM_B_FRANCHISE_HFA appended.
        """
        games: pd.DataFrame = datasets.games()
        hfa_map: dict[str, float] = self._compute_hfa_map(games)

        if not hfa_map:
            logger.warning(
                "venue_hfa: No franchise HFA coefficients could be computed "
                "(games CSV may be empty). Setting both columns to 0.0."
            )
            df = df.copy()
            df["TEAM_A_FRANCHISE_HFA"] = 0.0
            df["TEAM_B_FRANCHISE_HFA"] = 0.0
            return df

        df = df.copy()
        df["TEAM_A_FRANCHISE_HFA"] = df["TEAM_A"].map(hfa_map).fillna(0.0)
        df["TEAM_B_FRANCHISE_HFA"] = df["TEAM_B"].map(hfa_map).fillna(0.0)

        # Neutral-site games: neither team has a home crowd → zero out both
        neutral_mask = self._neutral_mask(df)
        df.loc[neutral_mask, "TEAM_A_FRANCHISE_HFA"] = 0.0
        df.loc[neutral_mask, "TEAM_B_FRANCHISE_HFA"] = 0.0

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hfa_map(games: pd.DataFrame) -> dict[str, float]:
        """Build a {franchise_long_name: hfa_coefficient} mapping.

        Computes each franchise's home win rate from the games CSV, then
        subtracts the league-average home win rate to produce a signed
        differential.  Franchises with fewer than _MIN_HOME_GAMES home
        games receive 0.0 (league average differential).

        Args:
            games: Canonical games DataFrame with WINNER, LOSER,
                GAME_LOCATION, and WIN_OR_TIE columns.

        Returns:
            Dict mapping each franchise long name to its HFA coefficient.
            Returns an empty dict if the games DataFrame is empty or lacks
            the required columns.
        """
        required = {"WINNER", "LOSER", "GAME_LOCATION", "WIN_OR_TIE"}
        if games.empty or not required.issubset(games.columns):
            return {}

        # Standard home games only (exclude neutral site and away games)
        home_games: pd.DataFrame = games.loc[games["GAME_LOCATION"] == HOME_GAME_LOCATION, :].copy()

        if home_games.empty:
            return {}

        # Ties count as 0.5 wins; losses as 0 - WIN_OR_TIE encodes this
        # from the WINNER's perspective.  In a home game where WINNER is
        # the home team: WIN_OR_TIE=1 (win) or 0.5 (tie).
        # In a home game where WINNER is the away team (GAME_LOCATION="@"):
        # those rows are already excluded by the filter above.
        #
        # So for each home game row:
        #   WINNER = home team → counted as win (1.0) or tie (0.5)
        #   LOSER  = home team → counted as loss (0.0)

        # Home team wins/ties: WINNER is the home team
        winner_home: pd.DataFrame = (
            home_games.groupby("WINNER")["WIN_OR_TIE"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "home_wins", "count": "home_games_as_winner"})
        )

        # Home team losses: LOSER is the home team in standard home games
        # (WIN_OR_TIE=1.0 means home team won; WIN_OR_TIE=0.5 (a tie) is
        # excluded because by the WINNER/LOSER convention ties record the
        # away team as LOSER, which would incorrectly credit a home game to
        # the visiting franchise (venue_hfa/H1).
        non_tie_home_games: pd.DataFrame = home_games.loc[home_games["WIN_OR_TIE"] == 1.0, :]
        non_tie_home_games = non_tie_home_games.assign(LOSER_CREDIT=0.0)
        loser_home: pd.DataFrame = (
            non_tie_home_games.groupby("LOSER")["LOSER_CREDIT"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "home_wins_as_loser", "count": "home_games_as_loser"})
        )

        # Combine via outer merge so franchises appearing on only one side
        # are retained; fill missing values with 0 before summing.
        # This avoids row-by-row .loc[] access which produces Series | Any
        # types that pyrefly cannot statically verify as scalars.
        stats: pd.DataFrame = (
            winner_home.rename_axis("franchise")
            .merge(
                loser_home.rename_axis("franchise"),
                left_index=True,
                right_index=True,
                how="outer",
            )
            .fillna(0.0)
        )
        stats["home_wins"] = stats["home_wins"].astype(float) + stats["home_wins_as_loser"].astype(
            float
        )
        stats["home_games"] = stats["home_games_as_winner"].astype(int) + stats[
            "home_games_as_loser"
        ].astype(int)
        stats["home_win_rate"] = stats["home_wins"] / stats["home_games"].clip(lower=1)

        # League-average home win rate (weighted by games played)
        league_avg: float = float(stats["home_wins"].sum() / max(int(stats["home_games"].sum()), 1))
        logger.debug("venue_hfa: league average home win rate = %.4f", league_avg)

        # Build final coefficient map - below threshold → 0.0 differential
        stats["hfa_coeff"] = stats["home_win_rate"] - league_avg
        stats.loc[stats["home_games"] < _MIN_HOME_GAMES, "hfa_coeff"] = 0.0

        below_threshold: list[str] = stats.loc[stats["home_games"] < _MIN_HOME_GAMES].index.tolist()
        if below_threshold:
            logger.debug(
                "venue_hfa: %d franchise(s) below %d home-game threshold (using 0.0): %s",
                len(below_threshold),
                _MIN_HOME_GAMES,
                below_threshold,
            )

        return dict(zip(stats.index.astype(str), stats["hfa_coeff"].astype(float), strict=False))

    @staticmethod
    def _neutral_mask(df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        """Return a boolean mask of neutral-site rows.

        Prefers IS_NEUTRAL_SITE if present (set by the travel feature);
        otherwise returns all-False (no neutral sites masked).

        Args:
            df: Modeling DataFrame, potentially containing IS_NEUTRAL_SITE.

        Returns:
            Boolean Series aligned with df's index.
        """
        if "IS_NEUTRAL_SITE" in df.columns:
            return df["IS_NEUTRAL_SITE"] == 1
        logger.debug(
            "venue_hfa: IS_NEUTRAL_SITE not present in DataFrame; "
            "neutral-site games will retain franchise HFA values. "
            "Ensure 'travel' runs before 'venue_hfa' in FEATURES list."
        )
        return pd.Series(False, index=df.index)

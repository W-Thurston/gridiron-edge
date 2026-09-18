"""Elo state table construction.

Builds the canonical ``NFL_Team_Elo.csv`` used by downstream predict,
features, and viz modules. Delegates the simulation to the canonical
:mod:`ratings.elo.simulator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.core.console import console
from gridiron_edge.core.constants import (
    EXPANSION_TEAMS as EXPANSION_START_YEAR,
)
from gridiron_edge.ratings.elo.simulator import (
    EloSimulationResult,
    transition_to_next_season,
)

# Earliest season covered by the canonical historical games contract.
_CANONICAL_HISTORY_START_SEASON: int = 1999


@dataclass(frozen=True)
class EloTableConfig:
    """Configuration parameters for Elo table construction."""

    k: float = 20.0
    initial_elo: float = 1500.0
    expansion_elo: float = 1300.0
    offseason_regress_frac: float = 1 / 3.0
    divisor: float = 480.0


def _season_start(year: object) -> int:
    """Return the starting year from one canonical NFL season label."""
    text: str = str(year).strip()
    parts: list[str] = text.split("-")

    if len(parts) != 2:
        raise ValueError(f"Invalid NFL season label {text!r}. Expected format YYYY-YYYY.")

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid NFL season label {text!r}. Expected numeric years.") from exc

    if end != start + 1:
        raise ValueError(
            f"Invalid NFL season label {text!r}. "
            "Ending year must be one greater than starting year."
        )

    return start


def _next_season_label(year: str) -> str:
    """Derive the next season label from one historical season label."""
    start: int = _season_start(year)
    return f"{start + 1}-{start + 2}"


def validate_complete_elo_history(
    games: DataFrame,
) -> None:
    """Validate canonical completed-game history for Elo reconstruction.

    The represented NFL seasons must begin at the canonical 1999 history floor
    and remain contiguous through the latest represented season. The latest
    season may be partial because reconstruction occurs during the active season.

    Args:
        games: Canonical one-row-per-game completed history.

    Raises:
        ValueError: If the history is empty, malformed, duplicated,
            incomplete, late-starting, or discontinuous.
    """
    required: set[str] = {
        "GAME_ID",
        "YEAR",
        "WEEK_NUM",
        "AWAY_TEAM",
        "HOME_TEAM",
        "AWAY_SCORE",
        "HOME_SCORE",
    }
    missing: list[str] = sorted(required - set(games.columns))
    if missing:
        raise ValueError("Canonical Elo history is missing required columns: " + ", ".join(missing))

    if games.empty:
        raise ValueError("Canonical Elo history must not be empty.")

    game_ids: Series[str] = games["GAME_ID"].astype("string")
    invalid_game_ids: Series[bool] = game_ids.isna() | game_ids.str.strip().eq("")
    if invalid_game_ids.any():
        raise ValueError("Canonical Elo history contains null or empty game identities.")

    duplicated_game_ids: Series[bool] = game_ids.duplicated(keep=False)
    if duplicated_game_ids.any():
        duplicates: list[str] = sorted(
            game_ids.loc[duplicated_game_ids].astype(str).unique().tolist()
        )
        raise ValueError(
            "Canonical Elo history contains duplicate game IDs: " + ", ".join(duplicates)
        )

    season_labels: Series[str] = games["YEAR"].astype("string")
    invalid_seasons: Series[bool] = season_labels.isna() | season_labels.str.strip().eq("")
    if invalid_seasons.any():
        raise ValueError("Canonical Elo history contains null or empty season identities.")

    season_starts: dict[str, int] = {}
    for label in season_labels.astype(str).str.strip().unique().tolist():
        season_starts[label] = _season_start(label)

    represented: list[int] = sorted(set(season_starts.values()))
    expected = list(
        range(
            _CANONICAL_HISTORY_START_SEASON,
            represented[-1] + 1,
        )
    )

    if represented[0] != _CANONICAL_HISTORY_START_SEASON:
        raise ValueError(
            "Canonical Elo history must begin with season "
            f"{_CANONICAL_HISTORY_START_SEASON}-"
            f"{_CANONICAL_HISTORY_START_SEASON + 1}; "
            f"earliest represented season is "
            f"{represented[0]}-{represented[0] + 1}."
        )

    if represented != expected:
        missing_seasons: list[int] = sorted(set(expected) - set(represented))
        formatted: str = ", ".join(f"{season}-{season + 1}" for season in missing_seasons)
        raise ValueError("Canonical Elo history is missing intermediate season(s): " + formatted)

    for column in ("AWAY_TEAM", "HOME_TEAM"):
        teams: Series[str] = games[column].astype("string")
        invalid_teams: Series[bool] = teams.isna() | teams.str.strip().eq("")
        if invalid_teams.any():
            raise ValueError(f"Canonical Elo history contains null or empty {column} identities.")

    same_team: Series[bool] = (
        games["AWAY_TEAM"].astype(str).str.strip() == games["HOME_TEAM"].astype(str).str.strip()
    )
    if same_team.any():
        game_ids_with_same_team: list[str] = sorted(
            games.loc[same_team, "GAME_ID"].astype(str).tolist()
        )
        raise ValueError(
            "Canonical Elo history has identical Away and Home teams for games: "
            + ", ".join(game_ids_with_same_team)
        )

    weeks: Series = pd.to_numeric(
        games["WEEK_NUM"],
        errors="coerce",
    )
    invalid_weeks: Series[bool] = weeks.isna() | (weeks < 1) | (weeks % 1 != 0)
    if invalid_weeks.any():
        raise ValueError("Canonical Elo history contains invalid week identities.")

    away_scores: Series = pd.to_numeric(
        games["AWAY_SCORE"],
        errors="coerce",
    )
    home_scores: Series = pd.to_numeric(
        games["HOME_SCORE"],
        errors="coerce",
    )

    away_present: Series[bool] = away_scores.notna()
    home_present: Series[bool] = home_scores.notna()
    if not away_present.equals(home_present):
        raise ValueError(
            "Canonical Elo history requires Away and Home scores to be present together."
        )

    if not away_present.all():
        raise ValueError("Canonical Elo history must contain completed games only.")

    if (away_scores < 0).any() or (home_scores < 0).any():
        raise ValueError("Canonical Elo history contains negative game scores.")


def _max_week_for_year(
    games: pd.DataFrame,
    year: str,
) -> int:
    """Return the maximum completed week for one season, or zero."""
    subset = games.loc[
        games["YEAR"] == year,
        "WEEK_NUM",
    ]
    return int(subset.max()) if not subset.empty else 0


def _latest_season_ratings_by_team(
    elo: dict[tuple[str, str, int], float],
    *,
    year: str,
    teams: set[str],
) -> dict[str, float]:
    """Return each team's latest available Elo state in one season.

    Postseason teams finish on different weeks, so a single global final
    state week cannot represent every returning team. Missing state for a
    returning team is rejected rather than silently replaced with initial Elo.
    """
    latest: dict[str, tuple[int, float]] = {}
    for (team, state_year, week), rating in elo.items():
        if state_year != year or team not in teams:
            continue
        current = latest.get(team)
        if current is None or week > current[0]:
            latest[team] = (week, rating)

    missing = sorted(teams - set(latest))
    if missing:
        raise ValueError(
            "No prior-season Elo state found for returning team(s): " + ", ".join(missing)
        )

    return {team: latest[team][1] for team in teams}


def _add_next_season_week_one(
    elo: dict[tuple[str, str, int], float],
    *,
    games: DataFrame,
    sorted_years: list[str],
    teams_by_year: dict[str, set[str]],
    cfg: EloTableConfig,
) -> None:
    """Append one deterministic synthetic next-season Week 1 state.

    The transition begins from the final postgame state of the latest
    historical season. Returning teams receive the same offseason
    regression used between historical seasons. Expansion teams whose
    configured start season matches the derived next season receive the
    configured expansion rating.

    The input Elo mapping is updated in place. Existing historical rows
    are not altered.
    """
    if not sorted_years:
        return

    latest_year: str = sorted_years[-1]
    if _max_week_for_year(games, latest_year) < 22:
        return

    next_year: str = _next_season_label(latest_year)

    returning_teams: set[str] = teams_by_year.get(
        latest_year,
        set(),
    )

    final_ratings = _latest_season_ratings_by_team(
        elo,
        year=latest_year,
        teams=returning_teams,
    )

    transitioned: dict[str, float] = transition_to_next_season(
        final_ratings,
        returning_teams=returning_teams,
        expansion_start=EXPANSION_START_YEAR,
        next_year=next_year,
        regress_frac=cfg.offseason_regress_frac,
        initial_elo=cfg.initial_elo,
        expansion_elo=cfg.expansion_elo,
    )

    for team, rating in transitioned.items():
        key = (
            team,
            next_year,
            1,
        )
        if key not in elo:
            elo[key] = rating


def build_elo_state_table_all_years(
    games: pd.DataFrame,
    *,
    cfg: EloTableConfig | None = None,
) -> pd.DataFrame:
    """Build the full Elo state table from historical game results."""
    from gridiron_edge.evaluation.tune import _prepare_games
    from gridiron_edge.ratings.elo.simulator import simulate_elo_history

    cfg = cfg or EloTableConfig()

    validate_complete_elo_history(games)
    games_prepared, sorted_years, teams_by_year = _prepare_games(games)

    result: EloSimulationResult = simulate_elo_history(
        games_prepared,
        sorted_years,
        teams_by_year,
        EXPANSION_START_YEAR,
        k_early=cfg.k,
        k_mid=cfg.k,
        k_week18=cfg.k,
        k_post=cfg.k,
        divisor=cfg.divisor,
        regress_frac=cfg.offseason_regress_frac,
        initial_elo=cfg.initial_elo,
        expansion_elo=cfg.expansion_elo,
    )

    elo_dict: dict[tuple[str, str, int], float] = dict(result.elo)

    _add_next_season_week_one(
        elo_dict,
        games=games_prepared,
        sorted_years=sorted_years,
        teams_by_year=teams_by_year,
        cfg=cfg,
    )

    rows: list[dict[str, float | int | str]] = [
        {"NFL_TEAM": team, "NFL_YEAR": year, "NFL_WEEK": week, "ELO": elo}
        for (team, year, week), elo in elo_dict.items()
    ]

    if not rows:
        return pd.DataFrame(columns=["NFL_TEAM", "NFL_YEAR", "NFL_WEEK", "ELO"])

    df_out: DataFrame = (
        pd.DataFrame(rows).sort_values(["NFL_YEAR", "NFL_WEEK", "NFL_TEAM"]).reset_index(drop=True)
    )

    if console.verbose:
        n_teams: int = df_out["NFL_TEAM"].nunique()
        n_seasons: int = df_out["NFL_YEAR"].nunique()
        print(f"  Elo table: {len(df_out):,} rows  {n_teams} teams  {n_seasons} seasons")

    return df_out

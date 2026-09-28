# src/gridiron_edge/ratings/qb_elo.py

"""Starting-quarterback rating recomputed from complete game history.

A two-component Elo: a game's strength for each side is a team component plus
the starting QB's component. Both components move by the same outcome error
(``result - expected``), so a QB is credited only for what the team component
does not already explain, instead of double-counting a QB term layered on the
persisted team Elo. The QB component starts at 0 for a debut, follows the
player across teams, learns quickly early in a career (K decays with prior
starts), and regresses toward 0 each offseason.

The replay uses each completed game's actual starter. It is self-contained,
does not read the persisted team Elo, and is recomputed rather than persisted
(``DECISIONS.md`` D69). Lookup semantics match Glicko-2: a (season, week) sees
only strictly earlier completed games, and a week beyond the last completed
game gets the last state unchanged.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

from gridiron_edge.core.constants import EXPANSION_TEAMS
from gridiron_edge.ratings.elo.core import DEFAULT_ELO_DIVISOR, elo_win_probability
from gridiron_edge.ratings.history import (
    CutoffKey,
    away_outcome,
    prepare_history,
    season_label,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas import DataFrame


@dataclass(frozen=True)
class QbEloConfig:
    """QB/team Elo parameters, fit on 1999-2022 only (``DECISIONS.md`` D70)."""

    k_team: float = 30.0
    k_qb_debut: float = 40.0
    k_qb_floor: float = 20.0
    k_qb_halflife_starts: float = 8.0
    qb_regress_frac: float = 1 / 6
    team_regress_frac: float = 1 / 3
    divisor: float = DEFAULT_ELO_DIVISOR
    initial_team: float = 1500.0
    expansion_team: float = 1300.0


class QbState(NamedTuple):
    """A quarterback's rating component and prior start count."""

    rating: float
    starts: int


@dataclass(frozen=True)
class QbEloPrediction:
    """One replayed game's pre-game Away win probability and result."""

    game_id: str
    season: int
    away_win_probability: float
    away_result: float


@dataclass
class _Snapshot:
    team: dict[str, float]
    qb: dict[str, QbState]


_STARTER_COLUMNS = ("GAME_ID", "AWAY_QB_ID", "HOME_QB_ID")


class QbEloHistory:
    """Full QB/team Elo replay with as-of lookup by (season, week)."""

    def __init__(
        self,
        games: DataFrame,
        starters: DataFrame,
        config: QbEloConfig | None = None,
    ) -> None:
        self._cfg = config or QbEloConfig()
        history = prepare_history(games)
        self._keys: list[CutoffKey] = []
        self._after: list[_Snapshot] = []
        self.predictions: list[QbEloPrediction] = []
        self._replay(history, self._starter_lookup(history, starters))

    @staticmethod
    def _starter_lookup(history: DataFrame, starters: DataFrame) -> DataFrame:
        missing = sorted(set(_STARTER_COLUMNS) - set(starters.columns))
        if missing:
            raise ValueError("QB starters are missing required columns: " + ", ".join(missing))
        joined = history.merge(
            starters.loc[:, list(_STARTER_COLUMNS)].assign(
                GAME_ID=lambda frame: frame["GAME_ID"].astype(str)
            ),
            on="GAME_ID",
            how="left",
            validate="one_to_one",
        )
        unknown = joined["AWAY_QB_ID"].isna() | joined["HOME_QB_ID"].isna()
        if unknown.any():
            raise ValueError(
                "Completed games have no recorded starting QB: "
                + ", ".join(joined.loc[unknown, "GAME_ID"].head(10).tolist())
            )
        return joined

    def _qb_k(self, starts: int) -> float:
        cfg = self._cfg
        return cfg.k_qb_floor + (cfg.k_qb_debut - cfg.k_qb_floor) / (
            1.0 + starts / cfg.k_qb_halflife_starts
        )

    def _team_default(self, team: str, season: int) -> float:
        if EXPANSION_TEAMS.get(team) == season_label(season):
            return self._cfg.expansion_team
        return self._cfg.initial_team

    def _transition(self, snapshot: _Snapshot, season: int) -> _Snapshot:
        cfg = self._cfg
        mean = sum(snapshot.team.values()) / len(snapshot.team) if snapshot.team else 0.0
        team = {
            name: mean * cfg.team_regress_frac + rating * (1.0 - cfg.team_regress_frac)
            for name, rating in snapshot.team.items()
        }
        label = season_label(season)
        for name, start in EXPANSION_TEAMS.items():
            if start == label:
                team[name] = cfg.expansion_team
        qb = {
            qb_id: QbState(state.rating * (1.0 - cfg.qb_regress_frac), state.starts)
            for qb_id, state in snapshot.qb.items()
        }
        return _Snapshot(team, qb)

    def _replay(self, history: DataFrame, joined: DataFrame) -> None:
        cfg = self._cfg
        current = _Snapshot({}, {})
        previous_season: int | None = None

        for (season, week), week_games in joined.groupby(["SEASON", "WEEK"], sort=True):
            season, week = int(season), int(week)  # noqa: PLW2901
            if previous_season is not None and season != previous_season:
                current = self._transition(current, season)
            previous_season = season

            team = dict(current.team)
            qb = dict(current.qb)
            for game_id, away, home, away_qb, home_qb, away_score, home_score in zip(
                week_games["GAME_ID"].astype(str),
                week_games["AWAY_TEAM"].astype(str),
                week_games["HOME_TEAM"].astype(str),
                week_games["AWAY_QB_ID"].astype(str),
                week_games["HOME_QB_ID"].astype(str),
                week_games["AWAY_SCORE"].to_numpy(dtype=float),
                week_games["HOME_SCORE"].to_numpy(dtype=float),
                strict=True,
            ):
                for name in (away, home):
                    team.setdefault(name, self._team_default(name, season))
                a_state = qb.get(away_qb, QbState(0.0, 0))
                h_state = qb.get(home_qb, QbState(0.0, 0))

                p_away, _ = elo_win_probability(
                    team[away] + a_state.rating,
                    team[home] + h_state.rating,
                    divisor=cfg.divisor,
                )
                result = away_outcome(float(away_score), float(home_score))
                error = result - p_away

                self.predictions.append(QbEloPrediction(game_id, season, p_away, result))
                # A team and a QB each play at most once per week, so in-place
                # updates never feed one game's result into another's prediction.
                team[away] += cfg.k_team * error
                team[home] -= cfg.k_team * error
                qb[away_qb] = QbState(
                    a_state.rating + self._qb_k(a_state.starts) * error,
                    a_state.starts + 1,
                )
                qb[home_qb] = QbState(
                    h_state.rating - self._qb_k(h_state.starts) * error,
                    h_state.starts + 1,
                )

            current = _Snapshot(team, qb)
            self._keys.append((season, week))
            self._after.append(current)

    def _snapshot_as_of(self, season: int, week: int) -> _Snapshot:
        index = bisect_left(self._keys, (season, week)) - 1
        if index < 0:
            return _Snapshot({}, {})
        snapshot = self._after[index]
        last_season = self._keys[index][0]
        if season > last_season + 1:
            raise ValueError(
                f"Cannot look ahead to season {season_label(season)}: "
                f"last completed game is in {season_label(last_season)}."
            )
        return self._transition(snapshot, season) if season > last_season else snapshot

    def qb_state_as_of(
        self,
        season: int,
        week: int,
        qb_ids: Iterable[str],
    ) -> dict[str, QbState]:
        """Return each QB's component and prior starts before ``(season, week)``.

        A QB with no prior start is ``QbState(0.0, 0)``.

        Raises:
            ValueError: If ``season`` is more than one season past the last
                completed game.
        """
        snapshot = self._snapshot_as_of(season, week)
        return {qb_id: snapshot.qb.get(qb_id, QbState(0.0, 0)) for qb_id in qb_ids}

    def prediction_frame(self) -> DataFrame:
        """Return replayed pre-game predictions, for fitting and diagnostics."""
        return pd.DataFrame(self.predictions)

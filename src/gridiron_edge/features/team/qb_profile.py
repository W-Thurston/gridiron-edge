# src/gridiron_edge/features/team/qb_profile.py

"""Canonical Away/Home starting-QB profile features.

Resolves each side's starting QB from the QB/coach identity sidecar, then
derives that QB's own pregame career and rolling performance from
``player_game_logs`` - the per-player per-game aggregation nflverse
already publishes (``passing_epa``, ``passing_cpoe``, ``attempts``), not
raw play-by-play (which does not carry passer identity - see
``DECISIONS.md``).

Starting-QB identity is only known for games already played. For a game
with no recorded starter yet (every unplayed game), identity falls back to
that team's most recent *known* prior starter - one code path used for
both historical and live rows, so training and live serving never
diverge. See ``DECISIONS.md`` for the carry-forward policy this
implements.

All career/rolling aggregates use strictly prior games for the resolved
QB (across whichever teams they played for - career stats travel with the
player), with no lookahead, mirroring ``features/team/epa.py``'s rolling
convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd
from pandas import DataFrame, Series

from gridiron_edge.features.base import FeatureSpec
from gridiron_edge.features.registry import FeatureRegistry
from gridiron_edge.models.game_prediction.game_schema import (
    away_feature_name,
    home_feature_name,
)

if TYPE_CHECKING:
    from gridiron_edge.datasets.accessor import DatasetAccessor

# Default rolling window, matching features/team/epa.py's DEFAULT_ROLLING_WINDOW.
DEFAULT_ROLLING_WINDOW: Final[int] = 4

QB_PROFILE_METRICS: Final[tuple[str, ...]] = (
    "qb_epa_play_career",
    "qb_epa_play_rolling",
    "qb_cpoe_rolling",
    "qb_cpoe_rolling_available",
    "qb_experience",
)

# The subset of QB_PROFILE_METRICS that _build_target_qb_profile computes
# directly (qb_cpoe_rolling_available is derived afterward by the CPOE
# availability policy; qb_experience is merged and typed separately).
_PROFILE_VALUE_METRICS: Final[tuple[str, ...]] = (
    "qb_epa_play_career",
    "qb_epa_play_rolling",
    "qb_cpoe_rolling",
)

_HOME_AWAY_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
)

QB_SOURCE_NFLVERSE_SCHEDULE = "nflverse_schedule"
QB_SOURCE_CARRIED_FORWARD = "carried_forward"

_SIDECAR_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_QB_ID",
    "HOME_QB_ID",
)

_LOGS_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "player_id",
    "position",
    "season",
    "week",
    "attempts",
    "passing_epa",
    "passing_cpoe",
)

_AWAY_METRIC_COLS: Final[list[str]] = [
    away_feature_name(metric.upper()) for metric in QB_PROFILE_METRICS
]
_HOME_METRIC_COLS: Final[list[str]] = [
    home_feature_name(metric.upper()) for metric in QB_PROFILE_METRICS
]
_AWAY_CHANGE_FLAG: Final[str] = away_feature_name("QB_CHANGE_FLAG")
_HOME_CHANGE_FLAG: Final[str] = home_feature_name("QB_CHANGE_FLAG")

# Prior for a known starter with no prior pass-attempt history (a debut).
# Attempt-weighted EPA per attempt of QB debut games with at least 10 attempts,
# seasons 2000-2022 only (1999 debuts are left-censored, 2023+ is holdout):
# -0.1293 over 143 debuts. The all-QB mean (+0.0288) would overstate a debut
# starter by about 0.16, so the debut mean is used. See DECISIONS.md D73.
DEBUT_QB_EPA_PER_ATTEMPT: Final[float] = -0.1293

# Metrics that are zero-filled (not NaN) when entirely unavailable: an
# availability indicator and a count both mean "none" as 0, not "unknown".
_ZERO_FILL_METRICS: Final[tuple[str, ...]] = ("QB_CPOE_ROLLING_AVAILABLE", "QB_EXPERIENCE")


def _empty_metric_values() -> dict[str, float | int]:
    """Default Away/Home metric values when no profile can be resolved at all."""
    values: dict[str, float | int] = {
        column: float("nan") for column in (*_AWAY_METRIC_COLS, *_HOME_METRIC_COLS)
    }
    for metric in _ZERO_FILL_METRICS:
        values[away_feature_name(metric)] = 0
        values[home_feature_name(metric)] = 0
    values[_AWAY_CHANGE_FLAG] = 0
    values[_HOME_CHANGE_FLAG] = 0
    return values


def _require_columns(frame: DataFrame, required: tuple[str, ...], *, label: str) -> None:
    """Require an established input schema."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _season_numbers(year_text: Series) -> Series:
    """Convert canonical season labels ("2024-2025") to starting-season ints."""
    if year_text.isna().any():
        raise ValueError("YEAR must not contain nulls.")

    stripped = year_text.astype(str).str.strip()
    if stripped.eq("").any():
        raise ValueError("YEAR must not contain empty values.")

    season_text = stripped.str.split("-", n=1).str[0]
    try:
        return season_text.astype(int)
    except ValueError as exc:
        raise ValueError("YEAR must begin with a numeric season.") from exc


def _projected_ids(sidecar: DataFrame, column: str) -> Series:
    """Return a side's projected starter ids, null when the sidecar has no projection."""
    if column in sidecar.columns:
        return sidecar[column]
    return Series(pd.NA, index=sidecar.index, dtype="string")


def _projected_text(sidecar: DataFrame, column: str) -> Series:
    """Return a projection text column, null when the sidecar does not carry it."""
    if column in sidecar.columns:
        return sidecar[column]
    return Series(pd.NA, index=sidecar.index, dtype="string")


def resolve_starting_qb_provenance(sidecar: DataFrame) -> DataFrame:
    """Resolve one starter per (game, side) with where each id came from.

    Every sidecar row already carries the actual starter for played games
    and, for an unplayed game, whatever the nflverse schedule publishes (often
    null). Forward-filling each team's QB identity chronologically leaves
    known values untouched and replaces only the nulls with the most recent
    known starter - the carry-forward policy in ``DECISIONS.md`` D65. An
    unplayed game with a projected starter (``*_QB_ID_PROJECTED``,
    ``DECISIONS.md`` D75) uses that id for its own row instead; played games
    never carry a projection.

    Besides the resolved id and the change flag, the frame records the
    provenance the weekly product persists (``DECISIONS.md`` D77): the source,
    the projection snapshot time, the sidecar's own value, and the team's most
    recent earlier known starter, so what was assumed is reproducible from the
    same computation that feeds the features.
    """
    _require_columns(sidecar, _SIDECAR_REQUIRED_COLUMNS, label="QB/coach sidecar")

    sides = []
    for side, team_column in (("AWAY", "AWAY_TEAM"), ("HOME", "HOME_TEAM")):
        frame = sidecar.loc[:, ["GAME_ID", "YEAR", "WEEK_NUM", team_column, f"{side}_QB_ID"]]
        frame = frame.rename(columns={team_column: "TEAM", f"{side}_QB_ID": "QB_ID"})
        frame["PROJECTED_QB_ID"] = _projected_text(sidecar, f"{side}_QB_ID_PROJECTED")
        frame["PROJECTION_SOURCE"] = _projected_text(sidecar, f"{side}_QB_PROJECTION_SOURCE")
        frame["PROJECTION_AS_OF"] = _projected_text(sidecar, f"{side}_QB_PROJECTION_AS_OF")
        frame["SIDE"] = side
        sides.append(frame)

    history = pd.concat(sides, ignore_index=True)
    history["SEASON"] = _season_numbers(history["YEAR"])
    history = history.sort_values(
        ["TEAM", "SEASON", "WEEK_NUM", "GAME_ID"],
        kind="stable",
        ignore_index=True,
    )

    # The forward-fill base is the raw id, so games after a projected one stay anchored to
    # the last known starter; a projected id then replaces the resolved id for its own game.
    carried = history.groupby("TEAM", sort=False)["QB_ID"].ffill()
    projected = history["PROJECTED_QB_ID"].notna()
    history["RESOLVED_QB_ID"] = history["PROJECTED_QB_ID"].where(projected, carried)
    history["PREV_RESOLVED_QB_ID"] = history.groupby("TEAM", sort=False)["RESOLVED_QB_ID"].shift(1)
    history["QB_CHANGE_FLAG"] = (
        history["PREV_RESOLVED_QB_ID"].notna()
        & history["RESOLVED_QB_ID"].notna()
        & history["PREV_RESOLVED_QB_ID"].ne(history["RESOLVED_QB_ID"])
    ).astype(int)

    history["NFLVERSE_QB_ID"] = history["QB_ID"]
    history["CARRIED_FORWARD_QB_ID"] = carried.groupby(history["TEAM"], sort=False).shift(1)
    history["QB_SOURCE"] = pd.Series(pd.NA, index=history.index, dtype="string")
    history.loc[history["QB_ID"].notna(), "QB_SOURCE"] = QB_SOURCE_NFLVERSE_SCHEDULE
    history.loc[history["QB_ID"].isna() & carried.notna(), "QB_SOURCE"] = QB_SOURCE_CARRIED_FORWARD
    history.loc[projected, "QB_SOURCE"] = history.loc[projected, "PROJECTION_SOURCE"]
    history.loc[~projected, "PROJECTION_AS_OF"] = pd.NA

    return history.loc[
        :,
        [
            "GAME_ID",
            "TEAM",
            "SIDE",
            "SEASON",
            "WEEK_NUM",
            "RESOLVED_QB_ID",
            "QB_CHANGE_FLAG",
            "QB_SOURCE",
            "PROJECTION_AS_OF",
            "NFLVERSE_QB_ID",
            "CARRIED_FORWARD_QB_ID",
        ],
    ]


def resolve_starting_qb_history(sidecar: DataFrame) -> DataFrame:
    """Build one carried-forward starter per (team, season, week).

    The feature-facing view of :func:`resolve_starting_qb_provenance`: the
    resolved id and the change flag, nothing else.
    """
    return resolve_starting_qb_provenance(sidecar).loc[
        :,
        ["GAME_ID", "TEAM", "SIDE", "SEASON", "WEEK_NUM", "RESOLVED_QB_ID", "QB_CHANGE_FLAG"],
    ]


def _qb_game_log_history(player_game_logs: DataFrame) -> DataFrame:
    """One chronologically ordered row per QB pass-attempt game."""
    _require_columns(player_game_logs, _LOGS_REQUIRED_COLUMNS, label="Player game logs")

    logs = player_game_logs.loc[
        (player_game_logs["position"] == "QB") & (player_game_logs["attempts"] > 0),
        ["player_id", "season", "week", "attempts", "passing_epa", "passing_cpoe"],
    ].copy()
    logs["season"] = logs["season"].astype(int)
    logs["week"] = logs["week"].astype(int)

    return logs.sort_values(
        ["player_id", "season", "week"],
        kind="stable",
        ignore_index=True,
    )


def _qb_profile_entering(
    history: DataFrame,
    *,
    player_id: object,
    season: int,
    week: int,
    window: int,
) -> tuple[float, float, float, int]:
    """Return career EPA/play, rolling EPA/play, rolling CPOE, and experience.

    ``experience`` is the count of strictly prior QB pass-attempt games for
    this player. Every other value is ``NaN`` when the player has no prior
    games (no history yet, including a null/unresolved identity).
    """
    # pyrefly: ignore [no-matching-overload]
    if pd.isna(player_id):
        return (float("nan"), float("nan"), float("nan"), 0)

    prior = history.loc[
        (history["player_id"] == player_id)
        & (
            (history["season"] < season)
            | ((history["season"] == season) & (history["week"] < week))
        ),
        :,
    ]

    experience = len(prior)
    if prior.empty:
        return (float("nan"), float("nan"), float("nan"), experience)

    career_attempts = float(prior["attempts"].sum())
    career_epa_per_play = (
        float(prior["passing_epa"].sum()) / career_attempts if career_attempts > 0 else float("nan")
    )

    recent = prior.tail(window)
    recent_attempts = float(recent["attempts"].sum())
    rolling_epa_per_play = (
        float(recent["passing_epa"].sum()) / recent_attempts
        if recent_attempts > 0
        else float("nan")
    )

    cpoe_values = recent["passing_cpoe"].dropna()
    rolling_cpoe = float(cpoe_values.mean()) if not cpoe_values.empty else float("nan")

    return (career_epa_per_play, rolling_epa_per_play, rolling_cpoe, experience)


def _build_target_qb_profile(
    logs: DataFrame,
    targets: DataFrame,
    *,
    window: int,
) -> DataFrame:
    """Compute QB profile metrics for explicit (player_id, season, week) targets."""
    normalized_targets = targets.drop_duplicates(
        subset=["player_id", "season", "week"],
        keep="first",
        ignore_index=True,
    )

    rows: list[dict[str, object]] = []
    for target in normalized_targets.itertuples(index=False):
        career_epa, rolling_epa, rolling_cpoe, experience = _qb_profile_entering(
            logs,
            player_id=target.player_id,
            # pyrefly: ignore [bad-argument-type]
            season=int(target.season),
            # pyrefly: ignore [bad-argument-type]
            week=int(target.week),
            window=window,
        )
        rows.append(
            {
                "player_id": target.player_id,
                "season": target.season,
                "week": target.week,
                "qb_epa_play_career": career_epa,
                "qb_epa_play_rolling": rolling_epa,
                "qb_cpoe_rolling": rolling_cpoe,
                "qb_experience": experience,
            }
        )

    return DataFrame(
        rows,
        columns=[
            "player_id",
            "season",
            "week",
            "qb_epa_play_career",
            "qb_epa_play_rolling",
            "qb_cpoe_rolling",
            "qb_experience",
        ],
    )


def _apply_debut_epa_prior(frame: DataFrame, columns: list[tuple[str, list[str]]]) -> None:
    """Fill missing career/rolling QB EPA for a known debut starter, in place.

    Only a starter whose identity is resolved but who has no prior history
    receives the debut prior; an unresolved starter stays missing so a real
    identity gap still fails closed (``DECISIONS.md`` D65, D73). The paired
    ``QB_EXPERIENCE`` value (0) already marks these rows.
    """
    for id_col, value_cols in columns:
        known = frame[id_col].notna()
        for value_col in value_cols:
            frame.loc[known & frame[value_col].isna(), value_col] = DEBUT_QB_EPA_PER_ATTEMPT


def _apply_cpoe_availability_policy(frame: DataFrame, columns: list[tuple[str, str]]) -> None:
    """Fill missing rolling QB CPOE with a paired availability indicator, in place.

    Mirrors ``features/team/epa.py``'s ``_apply_cpoe_availability_policy``
    (``DECISIONS.md`` D60) applied to a per-QB rolling value instead of a
    team aggregate: CPOE is null before nflverse's completion-probability
    model existed, and separately whenever a QB has no prior game history
    at all.
    """
    for cpoe_col, available_col in columns:
        frame[available_col] = frame[cpoe_col].notna().astype(int)
        frame[cpoe_col] = frame[cpoe_col].fillna(0.0)


@FeatureRegistry.register("home_away_qb_profile")
class HomeAwayQbProfileFeature:
    """Join Away/Home starting-QB career and rolling profile metrics."""

    spec = FeatureSpec(
        name="home_away_qb_profile",
        produces=[
            *_AWAY_METRIC_COLS,
            *_HOME_METRIC_COLS,
            _AWAY_CHANGE_FLAG,
            _HOME_CHANGE_FLAG,
        ],
    )

    def __init__(self, *, window: int = DEFAULT_ROLLING_WINDOW) -> None:
        """Configure the rolling window for QB EPA/CPOE."""
        if window < 1:
            raise ValueError("window must be at least 1.")
        self.window = window

    def compute(
        self,
        *,
        df: pd.DataFrame,
        datasets: DatasetAccessor,
    ) -> pd.DataFrame:
        """Attach Away and Home starting-QB profile metrics.

        Every input row remains in the output. A team with no resolvable
        starting-QB identity (no sidecar row, or no prior known starter at
        all) produces null profile values and a zero change flag for that
        side.
        """
        _require_columns(df, _HOME_AWAY_INPUT_COLUMNS, label="Home/away game frame")

        source = df.copy().drop(
            columns=[*_AWAY_METRIC_COLS, *_HOME_METRIC_COLS, _AWAY_CHANGE_FLAG, _HOME_CHANGE_FLAG],
            errors="ignore",
        )
        source["_QB_PROFILE_ORDER"] = range(len(source))
        source["SEASON"] = _season_numbers(source["YEAR"])

        sidecar = datasets.qb_coach_sidecar()

        if sidecar.empty:
            result = source.assign(**_empty_metric_values())
            return result.drop(columns=["_QB_PROFILE_ORDER", "SEASON"]).reset_index(drop=True)

        starter_history = resolve_starting_qb_history(sidecar)

        away_starters = (
            starter_history.loc[starter_history["SIDE"] == "AWAY", :]
            .rename(
                columns={
                    "TEAM": "AWAY_TEAM",
                    "RESOLVED_QB_ID": "AWAY_QB_ID_RESOLVED",
                    "QB_CHANGE_FLAG": _AWAY_CHANGE_FLAG,
                }
            )
            .loc[
                :,
                [
                    "GAME_ID",
                    "AWAY_TEAM",
                    "SEASON",
                    "WEEK_NUM",
                    "AWAY_QB_ID_RESOLVED",
                    _AWAY_CHANGE_FLAG,
                ],
            ]
        )
        home_starters = (
            starter_history.loc[starter_history["SIDE"] == "HOME", :]
            .rename(
                columns={
                    "TEAM": "HOME_TEAM",
                    "RESOLVED_QB_ID": "HOME_QB_ID_RESOLVED",
                    "QB_CHANGE_FLAG": _HOME_CHANGE_FLAG,
                }
            )
            .loc[
                :,
                [
                    "GAME_ID",
                    "HOME_TEAM",
                    "SEASON",
                    "WEEK_NUM",
                    "HOME_QB_ID_RESOLVED",
                    _HOME_CHANGE_FLAG,
                ],
            ]
        )

        resolved = source.merge(
            away_starters,
            how="left",
            on=["GAME_ID", "AWAY_TEAM", "SEASON", "WEEK_NUM"],
            sort=False,
            validate="one_to_one",
        ).merge(
            home_starters,
            how="left",
            on=["GAME_ID", "HOME_TEAM", "SEASON", "WEEK_NUM"],
            sort=False,
            validate="one_to_one",
        )
        resolved[_AWAY_CHANGE_FLAG] = resolved[_AWAY_CHANGE_FLAG].fillna(0).astype(int)
        resolved[_HOME_CHANGE_FLAG] = resolved[_HOME_CHANGE_FLAG].fillna(0).astype(int)

        player_game_logs = datasets.player_game_logs()
        if player_game_logs.empty:
            fill_values = {
                column: float("nan") for column in (*_AWAY_METRIC_COLS, *_HOME_METRIC_COLS)
            }
            for metric in _ZERO_FILL_METRICS:
                fill_values[away_feature_name(metric)] = 0
                fill_values[home_feature_name(metric)] = 0
            result = resolved.assign(**fill_values)
            return result.drop(
                columns=[
                    "_QB_PROFILE_ORDER",
                    "SEASON",
                    "AWAY_QB_ID_RESOLVED",
                    "HOME_QB_ID_RESOLVED",
                ]
            ).reset_index(drop=True)

        logs = _qb_game_log_history(player_game_logs)

        away_targets = resolved.loc[:, ["SEASON", "WEEK_NUM", "AWAY_QB_ID_RESOLVED"]].rename(
            columns={"WEEK_NUM": "week", "SEASON": "season", "AWAY_QB_ID_RESOLVED": "player_id"}
        )
        home_targets = resolved.loc[:, ["SEASON", "WEEK_NUM", "HOME_QB_ID_RESOLVED"]].rename(
            columns={"WEEK_NUM": "week", "SEASON": "season", "HOME_QB_ID_RESOLVED": "player_id"}
        )
        targets = pd.concat([away_targets, home_targets], ignore_index=True)
        # Rows with no resolvable starter (null player_id) never match a real
        # QB history row; _build_target_qb_profile still emits an all-NaN row
        # for them via _qb_profile_entering's None/NaN guard.
        profile = _build_target_qb_profile(logs, targets, window=self.window)

        away_rename = {
            "player_id": "AWAY_QB_ID_RESOLVED",
            "season": "SEASON",
            "week": "WEEK_NUM",
        } | {metric: away_feature_name(metric.upper()) for metric in _PROFILE_VALUE_METRICS}
        result = resolved.merge(
            profile.rename(columns=away_rename),
            how="left",
            on=["AWAY_QB_ID_RESOLVED", "SEASON", "WEEK_NUM"],
            sort=False,
            validate="many_to_one",
        )
        result[away_feature_name("QB_EXPERIENCE")] = result["qb_experience"].fillna(0).astype(int)
        result = result.drop(columns=["qb_experience"])

        home_rename = {
            "player_id": "HOME_QB_ID_RESOLVED",
            "season": "SEASON",
            "week": "WEEK_NUM",
        } | {metric: home_feature_name(metric.upper()) for metric in _PROFILE_VALUE_METRICS}
        result = result.merge(
            profile.rename(columns=home_rename),
            how="left",
            on=["HOME_QB_ID_RESOLVED", "SEASON", "WEEK_NUM"],
            sort=False,
            validate="many_to_one",
        )
        result[home_feature_name("QB_EXPERIENCE")] = result["qb_experience"].fillna(0).astype(int)
        result = result.drop(columns=["qb_experience"])

        _apply_debut_epa_prior(
            result,
            [
                (
                    "AWAY_QB_ID_RESOLVED",
                    [
                        away_feature_name("QB_EPA_PLAY_CAREER"),
                        away_feature_name("QB_EPA_PLAY_ROLLING"),
                    ],
                ),
                (
                    "HOME_QB_ID_RESOLVED",
                    [
                        home_feature_name("QB_EPA_PLAY_CAREER"),
                        home_feature_name("QB_EPA_PLAY_ROLLING"),
                    ],
                ),
            ],
        )
        _apply_cpoe_availability_policy(
            result,
            [
                (
                    away_feature_name("QB_CPOE_ROLLING"),
                    away_feature_name("QB_CPOE_ROLLING_AVAILABLE"),
                ),
                (
                    home_feature_name("QB_CPOE_ROLLING"),
                    home_feature_name("QB_CPOE_ROLLING_AVAILABLE"),
                ),
            ],
        )

        return (
            result.sort_values("_QB_PROFILE_ORDER", kind="stable")
            .drop(
                columns=[
                    "_QB_PROFILE_ORDER",
                    "SEASON",
                    "AWAY_QB_ID_RESOLVED",
                    "HOME_QB_ID_RESOLVED",
                ]
            )
            .reset_index(drop=True)
        )

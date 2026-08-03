"""Reviewed synchronization for season-scoped stadium metadata."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pandas import DataFrame

_STADIUM_COLUMNS: Final[tuple[str, ...]] = (
    "HOME_TEAM",
    "YEAR",
    "STADIUM",
    "LATITUDE",
    "LONGITUDE",
    "ROOF",
    "SURFACE",
    "ALTITUDE",
)
_UPDATE_COLUMNS: Final[tuple[str, ...]] = (
    "ACTION",
    "REVIEW_STATUS",
    "HOME_TEAM",
    "YEAR",
    "SOURCE_YEAR",
    "SOURCE_STADIUM",
    "STADIUM",
    "LATITUDE",
    "LONGITUDE",
    "ROOF",
    "SURFACE",
    "ALTITUDE",
    "NOTE",
)
_SPECIAL_HOME_TEAMS: Final[frozenset[str]] = frozenset({"Alternate", "International"})
_REVIEW_STATUSES: Final[frozenset[str]] = frozenset(
    {"proposed", "approved", "rejected", "unresolved"}
)


def _require_columns(frame: DataFrame, required: tuple[str, ...], *, label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def _season_start(value: object) -> int:
    text = str(value).strip()
    try:
        return int(text.split("-", maxsplit=1)[0])
    except ValueError as exc:
        raise ValueError(f"Invalid season label: {text!r}.") from exc


def _normal_teams(stadiums: DataFrame) -> frozenset[str]:
    return frozenset(
        str(value).strip()
        for value in stadiums["HOME_TEAM"].dropna()
        if str(value).strip() and str(value).strip() not in _SPECIAL_HOME_TEAMS
    )


def load_stadium_aliases(path: Path) -> DataFrame:
    """Load and validate an optional explicit stadium-alias artifact."""
    columns = ["SOURCE_STADIUM", "CANONICAL_STADIUM"]
    if not path.exists():
        return DataFrame(columns=columns)

    aliases = pd.read_csv(path, dtype="string")
    _require_columns(
        aliases,
        ("SOURCE_STADIUM", "CANONICAL_STADIUM"),
        label="Stadium aliases",
    )
    aliases = aliases.loc[:, columns].copy()
    for column in columns:
        aliases[column] = aliases[column].fillna("").str.strip()
    if aliases[columns].eq("").any().any():
        raise ValueError("Stadium aliases must not contain empty identities.")
    if aliases["SOURCE_STADIUM"].duplicated().any():
        raise ValueError("Stadium aliases contain duplicate source identities.")
    if aliases["SOURCE_STADIUM"].eq(aliases["CANONICAL_STADIUM"]).any():
        raise ValueError("Stadium aliases must not map a stadium to itself.")
    return aliases.sort_values("SOURCE_STADIUM", kind="stable").reset_index(drop=True)


def validate_stadium_reference(stadiums: DataFrame) -> None:
    """Validate canonical stadium identities and coordinate contracts."""
    _require_columns(stadiums, _STADIUM_COLUMNS, label="Stadium reference")
    if stadiums.empty:
        raise ValueError("Stadium reference must not be empty.")

    home_teams = stadiums["HOME_TEAM"].fillna("").astype(str).str.strip()
    years = stadiums["YEAR"].fillna("").astype(str).str.strip()
    names = stadiums["STADIUM"].fillna("").astype(str).str.strip()
    if home_teams.eq("").any() or years.eq("").any() or names.eq("").any():
        raise ValueError("Stadium reference identities must not be empty.")

    normal = ~home_teams.isin(_SPECIAL_HOME_TEAMS)
    origin_coordinates = stadiums.loc[
        normal,
        [
            "HOME_TEAM",
            "YEAR",
            "LATITUDE",
            "LONGITUDE",
            "ALTITUDE",
        ],
    ].drop_duplicates()

    origin_counts = origin_coordinates.groupby(
        ["HOME_TEAM", "YEAR"],
        dropna=False,
    ).size()

    if origin_counts.gt(1).any():
        raise ValueError(
            "Stadium reference contains conflicting franchise-season origin coordinates."
        )

    coordinate_rows = stadiums.loc[
        :,
        ["STADIUM", "LATITUDE", "LONGITUDE", "ALTITUDE"],
    ].drop_duplicates()
    conflicts = coordinate_rows.groupby("STADIUM", dropna=False).size()
    if conflicts.gt(1).any():
        raise ValueError("Stadium reference contains conflicting coordinate identities.")


def audit_stadium_coverage(
    stadiums: DataFrame,
    schedule: DataFrame,
    *,
    season: str,
) -> DataFrame:
    """Return missing franchise origins and unresolved scheduled game sites."""
    validate_stadium_reference(stadiums)
    _require_columns(
        schedule,
        ("season", "away_team", "home_team", "stadium"),
        label="Upcoming schedule",
    )
    scoped = schedule.loc[schedule["season"].astype(str) == season].copy()
    teams = sorted(
        set(scoped["away_team"].dropna().astype(str))
        | set(scoped["home_team"].dropna().astype(str))
    )
    existing = set(
        stadiums.loc[stadiums["YEAR"].astype(str) == season, "HOME_TEAM"].dropna().astype(str)
    )
    known_sites = set(stadiums["STADIUM"].dropna().astype(str).str.strip())

    rows: list[dict[str, object]] = []
    for team in teams:
        if team not in existing:
            rows.append(
                {
                    "ISSUE": "missing_franchise_origin",
                    "HOME_TEAM": team,
                    "YEAR": season,
                    "STADIUM": pd.NA,
                    "GAME_COUNT": int(
                        (
                            scoped["away_team"].astype(str).eq(team)
                            | scoped["home_team"].astype(str).eq(team)
                        ).sum()
                    ),
                }
            )
    for site in sorted(scoped["stadium"].dropna().astype(str).str.strip().unique()):
        if site not in known_sites:
            rows.append(
                {
                    "ISSUE": "unresolved_game_site",
                    "HOME_TEAM": pd.NA,
                    "YEAR": season,
                    "STADIUM": site,
                    "GAME_COUNT": int(scoped["stadium"].astype(str).str.strip().eq(site).sum()),
                }
            )
    return DataFrame(
        rows,
        columns=["ISSUE", "HOME_TEAM", "YEAR", "STADIUM", "GAME_COUNT"],
    )


def prepare_stadium_updates(
    stadiums: DataFrame,
    schedule: DataFrame,
    *,
    season: str,
    aliases: DataFrame | None = None,
) -> DataFrame:
    """Prepare reviewed carry-forward and explicit stadium-alias proposals."""
    validate_stadium_reference(stadiums)
    _require_columns(
        schedule,
        ("season", "away_team", "home_team", "stadium"),
        label="Upcoming schedule",
    )
    scoped = schedule.loc[schedule["season"].astype(str) == season].copy()
    normal_teams = _normal_teams(stadiums)
    required_teams = sorted(
        set(scoped["away_team"].dropna().astype(str))
        | set(scoped["home_team"].dropna().astype(str))
    )
    unknown = sorted(set(required_teams) - set(normal_teams))
    if unknown:
        raise ValueError("Upcoming schedule contains unknown NFL teams: " + ", ".join(unknown))

    alias_map: dict[str, str] = {}
    if aliases is not None and not aliases.empty:
        _require_columns(
            aliases,
            ("SOURCE_STADIUM", "CANONICAL_STADIUM"),
            label="Stadium aliases",
        )
        if aliases["SOURCE_STADIUM"].duplicated().any():
            raise ValueError("Stadium aliases contain duplicate source identities.")
        alias_map = dict(
            zip(
                aliases["SOURCE_STADIUM"].astype(str),
                aliases["CANONICAL_STADIUM"].astype(str),
                strict=True,
            )
        )

    existing_target = set(
        stadiums.loc[stadiums["YEAR"].astype(str) == season, "HOME_TEAM"].dropna().astype(str)
    )
    work = stadiums.copy()
    work["_START_YEAR"] = work["YEAR"].map(_season_start)
    target_start = _season_start(season)
    prior = work.loc[
        (~work["HOME_TEAM"].astype(str).isin(_SPECIAL_HOME_TEAMS))
        & (work["_START_YEAR"] < target_start),
        :,
    ]
    latest = (
        prior.sort_values(["HOME_TEAM", "_START_YEAR"], kind="stable")
        .groupby("HOME_TEAM", sort=True)
        .tail(1)
        .set_index("HOME_TEAM")
    )
    known_sites = set(stadiums["STADIUM"].astype(str))
    rows: list[dict[str, object]] = []

    for team in required_teams:
        if team in existing_target:
            continue
        if team not in latest.index:
            rows.append(
                {
                    "ACTION": "unresolved",
                    "REVIEW_STATUS": "unresolved",
                    "HOME_TEAM": team,
                    "YEAR": season,
                    "NOTE": "No prior franchise origin exists.",
                }
            )
            continue

        source = latest.loc[team]
        home_sites = sorted(
            scoped.loc[scoped["home_team"].astype(str) == team, "stadium"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        source_stadium = str(source["STADIUM"])
        current_site = source_stadium if source_stadium in home_sites else None
        action = "carry_forward"
        note = "Latest franchise stadium remains a scheduled home site."

        if current_site is None:
            alias_matches = [site for site in home_sites if alias_map.get(site) == source_stadium]
            if len(alias_matches) == 1:
                current_site = alias_matches[0]
                action = "alias_existing"
                note = f"Approved alias of {source_stadium}."
            else:
                action = "unresolved"
                note = "Normal franchise home site requires alias review."

        rows.append(
            {
                "ACTION": action,
                "REVIEW_STATUS": "proposed" if current_site is not None else "unresolved",
                "HOME_TEAM": team,
                "YEAR": season,
                "SOURCE_YEAR": source["YEAR"],
                "SOURCE_STADIUM": source_stadium,
                "STADIUM": current_site if current_site is not None else pd.NA,
                "LATITUDE": source["LATITUDE"] if current_site is not None else pd.NA,
                "LONGITUDE": source["LONGITUDE"] if current_site is not None else pd.NA,
                "ROOF": source["ROOF"] if current_site is not None else pd.NA,
                "SURFACE": source["SURFACE"] if current_site is not None else pd.NA,
                "ALTITUDE": source["ALTITUDE"] if current_site is not None else pd.NA,
                "NOTE": note,
            }
        )

    scheduled_sites = sorted(scoped["stadium"].dropna().astype(str).unique())
    for site in scheduled_sites:
        if site in known_sites:
            continue
        if any(str(row.get("STADIUM", "")) == site for row in rows):
            continue
        rows.append(
            {
                "ACTION": "unresolved",
                "REVIEW_STATUS": "unresolved",
                "HOME_TEAM": "International" if site in alias_map else pd.NA,
                "YEAR": season,
                "SOURCE_STADIUM": alias_map.get(site, pd.NA),
                "STADIUM": site,
                "NOTE": "New scheduled game site requires explicit classification and metadata.",
            }
        )

    return DataFrame(rows).reindex(columns=list(_UPDATE_COLUMNS))


def _remove_already_applied_updates(
    stadiums: DataFrame,
    approved: DataFrame,
) -> DataFrame:
    """Remove exact existing rows and reject conflicting identities."""
    remaining: list[dict[Hashable, Any]] = []

    for row in approved.to_dict(orient="records"):
        home_team = str(row["HOME_TEAM"])
        year = str(row["YEAR"])
        stadium = str(row["STADIUM"])

        if home_team in _SPECIAL_HOME_TEAMS:
            existing = stadiums.loc[
                (stadiums["HOME_TEAM"].astype(str) == home_team)
                & (stadiums["YEAR"].astype(str) == year)
                & (stadiums["STADIUM"].astype(str) == stadium),
                :,
            ]
            identity = f"{home_team}/{year}/{stadium}"
        else:
            existing = stadiums.loc[
                (stadiums["HOME_TEAM"].astype(str) == home_team)
                & (stadiums["YEAR"].astype(str) == year),
                :,
            ]
            identity = f"{home_team}/{year}"

        if existing.empty:
            remaining.append(row)
            continue

        candidate = DataFrame(
            [row],
            columns=list(_STADIUM_COLUMNS),
        ).astype(
            stadiums.dtypes.to_dict(),
        )

        exact = (
            existing.loc[
                :,
                list(_STADIUM_COLUMNS),
            ]
            .eq(candidate.iloc[0])
            .all(axis=1)
        )

        if exact.any():
            continue

        raise ValueError(f"Approved stadium update conflicts with an existing identity: {identity}")

    return DataFrame(
        remaining,
        columns=approved.columns,
    )


def _validate_approved_updates(
    stadiums: DataFrame,
    updates: DataFrame,
    *,
    nfl_teams: frozenset[str],
) -> DataFrame:
    _require_columns(updates, _UPDATE_COLUMNS, label="Stadium updates")
    invalid_statuses = sorted(set(updates["REVIEW_STATUS"].astype(str)) - _REVIEW_STATUSES)
    if invalid_statuses:
        raise ValueError("Unknown review statuses: " + ", ".join(invalid_statuses))
    approved = updates.loc[updates["REVIEW_STATUS"].astype(str) == "approved"].copy()
    if approved.empty:
        return approved

    allowed = set(nfl_teams) | set(_SPECIAL_HOME_TEAMS)
    unknown = sorted(set(approved["HOME_TEAM"].astype(str)) - allowed)
    if unknown:
        raise ValueError(
            "Approved stadium HOME_TEAM values must be NFL teams, Alternate, or International: "
            + ", ".join(unknown)
        )
    required_values = [
        "HOME_TEAM",
        "YEAR",
        "STADIUM",
        "LATITUDE",
        "LONGITUDE",
        "ROOF",
        "SURFACE",
        "ALTITUDE",
    ]
    if approved[required_values].isna().any().any():
        raise ValueError("Approved stadium updates require complete metadata.")
    latitudes = pd.to_numeric(approved["LATITUDE"], errors="coerce")
    longitudes = pd.to_numeric(approved["LONGITUDE"], errors="coerce")
    if latitudes.isna().any() or not latitudes.between(-90, 90).all():
        raise ValueError("Approved stadium latitude must be between -90 and 90.")
    if longitudes.isna().any() or not longitudes.between(-180, 180).all():
        raise ValueError("Approved stadium longitude must be between -180 and 180.")
    normal = ~approved["HOME_TEAM"].astype(str).isin(_SPECIAL_HOME_TEAMS)
    if approved.loc[normal].duplicated(["HOME_TEAM", "YEAR"]).any():
        raise ValueError("Approved updates contain duplicate franchise-season origins.")

    return _remove_already_applied_updates(
        stadiums,
        approved,
    )


def apply_approved_stadium_updates(
    stadiums: DataFrame,
    updates: DataFrame,
    *,
    path: Path,
) -> DataFrame:
    """Atomically append approved rows without modifying historical data."""
    validate_stadium_reference(stadiums)
    approved = _validate_approved_updates(
        stadiums,
        updates,
        nfl_teams=_normal_teams(stadiums),
    )
    additions = approved.loc[
        :,
        list(_STADIUM_COLUMNS),
    ].copy()

    if additions.empty:
        combined = stadiums.copy()
    else:
        combined = pd.concat(
            [
                stadiums.copy(),
                additions,
            ],
            ignore_index=True,
        ).drop_duplicates(ignore_index=True)

        combined = combined.astype(
            stadiums.dtypes.to_dict(),
        )

    validate_stadium_reference(combined)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    return combined

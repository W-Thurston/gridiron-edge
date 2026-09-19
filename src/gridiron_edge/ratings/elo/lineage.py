# src/gridiron_edge/ratings/elo/lineage.py
"""Persist and verify exact Elo reconstruction lineage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import pandas as pd
from pandas import DataFrame

from gridiron_edge.datasets.registry import dataset_path

ELO_LINEAGE_SCHEMA_VERSION: int = 1

_LINEAGE_FILENAME: str = "NFL_Team_Elo.metadata.json"
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_GAMES_REQUIRED_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "YEAR",
    "WEEK_NUM",
    "AWAY_TEAM",
    "HOME_TEAM",
    "AWAY_SCORE",
    "HOME_SCORE",
)

_ELO_REQUIRED_COLUMNS: tuple[str, ...] = (
    "NFL_TEAM",
    "NFL_YEAR",
    "NFL_WEEK",
    "ELO",
)


@dataclass(frozen=True)
class EloArtifactReference:
    """Exact identity and scope of one persisted CSV artifact."""

    relative_path: str
    content_digest: str
    row_count: int
    columns: tuple[str, ...]
    first_season: str
    latest_season: str
    latest_week: int

    def __post_init__(self) -> None:
        """Validate persisted artifact-reference fields."""
        path = Path(self.relative_path)
        if not self.relative_path.strip() or path.is_absolute() or ".." in path.parts:
            raise ValueError("Elo lineage artifact path must be a safe relative path.")

        if not _DIGEST_PATTERN.fullmatch(self.content_digest):
            raise ValueError("Elo lineage content_digest must be a lowercase SHA-256 digest.")

        if isinstance(self.row_count, bool) or self.row_count < 1:
            raise ValueError("Elo lineage row_count must be a positive integer.")

        if not self.columns or any(not column.strip() for column in self.columns):
            raise ValueError("Elo lineage columns must contain nonempty identities.")

        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Elo lineage columns must not contain duplicates.")

        _season_start(self.first_season)
        _season_start(self.latest_season)

        if _season_start(self.latest_season) < _season_start(self.first_season):
            raise ValueError("Elo lineage latest_season must not precede first_season.")

        if isinstance(self.latest_week, bool) or self.latest_week < 1:
            raise ValueError("Elo lineage latest_week must be a positive integer.")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "relative_path": self.relative_path,
            "content_digest": self.content_digest,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "first_season": self.first_season,
            "latest_season": self.latest_season,
            "latest_week": self.latest_week,
        }


@dataclass(frozen=True)
class EloLineage:
    """Exact source and output evidence for one Elo reconstruction."""

    schema_version: int
    generated_at: datetime
    source_games: EloArtifactReference
    elo_state: EloArtifactReference

    def __post_init__(self) -> None:
        """Validate the complete lineage contract."""
        if self.schema_version != ELO_LINEAGE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Elo lineage schema_version: {self.schema_version}.")

        if self.generated_at.tzinfo is None:
            raise ValueError("Elo lineage generated_at must be timezone-aware.")

        if self.source_games.relative_path != ("data/cleaned/NFL_wk_by_wk_cleaned.csv"):
            raise ValueError(
                "Elo lineage source_games path does not match the canonical games artifact."
            )

        if self.elo_state.relative_path != ("data/cleaned/NFL_Team_Elo.csv"):
            raise ValueError(
                "Elo lineage elo_state path does not match the canonical Elo artifact."
            )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "generated_at": (self.generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z")),
            "source_games": self.source_games.to_dict(),
            "elo_state": self.elo_state.to_dict(),
        }


def elo_lineage_path(
    repo: Path,
) -> Path:
    """Return the canonical Elo lineage sidecar path."""
    return dataset_path(repo, "elo_state").parent / _LINEAGE_FILENAME


def file_content_digest(
    path: Path,
) -> str:
    """Return the SHA-256 digest of exact persisted file bytes."""
    digest = sha256()

    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def build_elo_lineage(
    *,
    repo: Path,
    generated_at: datetime,
) -> EloLineage:
    """Build lineage from the current persisted games and Elo artifacts."""
    games_path = dataset_path(repo, "games")
    elo_path = dataset_path(repo, "elo_state")

    if not games_path.is_file():
        raise FileNotFoundError(f"Canonical games artifact is missing: {games_path}")
    if not elo_path.is_file():
        raise FileNotFoundError(f"Canonical Elo artifact is missing: {elo_path}")

    games = pd.read_csv(games_path)
    elo = pd.read_csv(elo_path)

    return EloLineage(
        schema_version=ELO_LINEAGE_SCHEMA_VERSION,
        generated_at=generated_at,
        source_games=_artifact_reference(
            repo=repo,
            path=games_path,
            frame=games,
            season_column="YEAR",
            week_column="WEEK_NUM",
            required_columns=_GAMES_REQUIRED_COLUMNS,
            label="Canonical games",
        ),
        elo_state=_artifact_reference(
            repo=repo,
            path=elo_path,
            frame=elo,
            season_column="NFL_YEAR",
            week_column="NFL_WEEK",
            required_columns=_ELO_REQUIRED_COLUMNS,
            label="Canonical Elo",
        ),
    )


def write_elo_lineage(
    lineage: EloLineage,
    *,
    repo: Path,
) -> Path:
    """Write one validated Elo lineage sidecar atomically."""
    path = elo_lineage_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        lineage.to_dict(),
        indent=2,
        sort_keys=True,
    )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        temporary.write_text(
            payload + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return path


def load_elo_lineage(
    *,
    repo: Path,
) -> EloLineage:
    """Strictly load the persisted Elo lineage sidecar."""
    path = elo_lineage_path(repo)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Elo lineage contains malformed JSON: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Elo lineage root must be a JSON object.")

    _require_exact_keys(
        raw,
        {
            "schema_version",
            "generated_at",
            "source_games",
            "elo_state",
        },
        label="Elo lineage",
    )

    schema_version = _integer(
        raw["schema_version"],
        "schema_version",
    )
    if schema_version != ELO_LINEAGE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Elo lineage schema_version: {schema_version}.")

    return EloLineage(
        schema_version=schema_version,
        generated_at=_timestamp(
            raw["generated_at"],
            "generated_at",
        ),
        source_games=_reference(
            raw["source_games"],
            "source_games",
        ),
        elo_state=_reference(
            raw["elo_state"],
            "elo_state",
        ),
    )


def verify_current_elo_lineage(
    *,
    repo: Path,
) -> bool:
    """Return whether current games and Elo artifacts match lineage.

    Missing lineage or referenced artifacts are ordinary semantic
    unavailability. Malformed or unsupported lineage remains an error.
    """
    if not elo_lineage_path(repo).is_file():
        return False

    lineage = load_elo_lineage(repo=repo)

    games_path = _resolve_relative(
        repo,
        lineage.source_games.relative_path,
    )
    elo_path = _resolve_relative(
        repo,
        lineage.elo_state.relative_path,
    )

    if not games_path.is_file() or not elo_path.is_file():
        return False

    games = pd.read_csv(games_path)
    elo = pd.read_csv(elo_path)

    current_games = _artifact_reference(
        repo=repo,
        path=games_path,
        frame=games,
        season_column="YEAR",
        week_column="WEEK_NUM",
        required_columns=_GAMES_REQUIRED_COLUMNS,
        label="Canonical games",
    )
    current_elo = _artifact_reference(
        repo=repo,
        path=elo_path,
        frame=elo,
        season_column="NFL_YEAR",
        week_column="NFL_WEEK",
        required_columns=_ELO_REQUIRED_COLUMNS,
        label="Canonical Elo",
    )

    return current_games == lineage.source_games and current_elo == lineage.elo_state


def _artifact_reference(
    *,
    repo: Path,
    path: Path,
    frame: DataFrame,
    season_column: str,
    week_column: str,
    required_columns: tuple[str, ...],
    label: str,
) -> EloArtifactReference:
    """Build one exact persisted-artifact reference."""
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} artifact is missing required columns: " + ", ".join(missing))

    if frame.empty:
        raise ValueError(f"{label} artifact must not be empty.")

    seasons = frame[season_column].astype("string")
    if seasons.isna().any() or seasons.str.strip().eq("").any():
        raise ValueError(f"{label} artifact contains invalid season identities.")

    normalized_seasons = sorted(
        {value.strip() for value in seasons.tolist()},
        key=_season_start,
    )

    weeks = pd.to_numeric(
        frame[week_column],
        errors="coerce",
    )
    invalid_weeks = weeks.isna() | (weeks < 1) | (weeks % 1 != 0)
    if invalid_weeks.any():
        raise ValueError(f"{label} artifact contains invalid week identities.")

    latest_season = normalized_seasons[-1]
    latest_week = int(
        weeks.loc[frame[season_column].astype(str).str.strip().eq(latest_season)].max()
    )

    return EloArtifactReference(
        relative_path=_relative_path(
            repo,
            path,
        ),
        content_digest=file_content_digest(path),
        row_count=len(frame),
        columns=tuple(column for column in frame.columns),
        first_season=normalized_seasons[0],
        latest_season=latest_season,
        latest_week=latest_week,
    )


def _reference(
    value: object,
    label: str,
) -> EloArtifactReference:
    """Strictly deserialize one artifact reference."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")

    _require_exact_keys(
        value,
        {
            "relative_path",
            "content_digest",
            "row_count",
            "columns",
            "first_season",
            "latest_season",
            "latest_week",
        },
        label=label,
    )

    columns_value = value["columns"]
    if not isinstance(columns_value, list) or any(
        not isinstance(column, str) for column in columns_value
    ):
        raise ValueError(f"{label}.columns must be a list of strings.")

    return EloArtifactReference(
        relative_path=_text(
            value["relative_path"],
            f"{label}.relative_path",
        ),
        content_digest=_text(
            value["content_digest"],
            f"{label}.content_digest",
        ),
        row_count=_integer(
            value["row_count"],
            f"{label}.row_count",
        ),
        columns=tuple(columns_value),
        first_season=_text(
            value["first_season"],
            f"{label}.first_season",
        ),
        latest_season=_text(
            value["latest_season"],
            f"{label}.latest_season",
        ),
        latest_week=_integer(
            value["latest_week"],
            f"{label}.latest_week",
        ),
    )


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    """Require one mapping to contain its exact schema keys."""
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if missing or unexpected:
        raise ValueError(
            f"{label} fields do not match schema; missing={missing}, unexpected={unexpected}."
        )


def _relative_path(
    repo: Path,
    path: Path,
) -> str:
    """Return one repository-contained POSIX relative path."""
    resolved_repo = repo.resolve()
    resolved_path = path.resolve()

    try:
        relative = resolved_path.relative_to(resolved_repo)
    except ValueError as exc:
        raise ValueError("Elo lineage artifact escapes the repository root.") from exc

    return relative.as_posix()


def _resolve_relative(
    repo: Path,
    relative_path: str,
) -> Path:
    """Resolve one safe lineage path inside the repository."""
    candidate = Path(relative_path)
    if not relative_path.strip() or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Elo lineage artifact path must be a safe relative path.")

    root = repo.resolve()
    resolved = (root / candidate).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Elo lineage artifact escapes the repository root.") from exc

    return resolved


def _season_start(
    season: str,
) -> int:
    """Return the starting year from a canonical NFL season label."""
    text = season.strip()
    parts = text.split("-")

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


def _timestamp(
    value: object,
    label: str,
) -> datetime:
    """Require one timezone-aware ISO timestamp."""
    text = _text(value, label)

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp.") from exc

    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware.")

    return parsed


def _text(
    value: object,
    label: str,
) -> str:
    """Require one nonempty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")

    return value.strip()


def _integer(
    value: object,
    label: str,
) -> int:
    """Require one non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")

    return value

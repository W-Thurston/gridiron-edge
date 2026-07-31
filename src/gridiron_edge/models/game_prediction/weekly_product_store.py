# src/gridiron_edge/models/game_prediction/weekly_product_store.py

"""Immutable storage for validated weekly game products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pandas import DataFrame

from gridiron_edge.core.settings import get_settings
from gridiron_edge.datasets.registry import dataset_path
from gridiron_edge.evaluation.forecast_contracts import WeeklyProductIdentity
from gridiron_edge.models.game_prediction.product_validation import (
    validate_weekly_game_product,
)

WEEKLY_PRODUCT_SCHEMA_VERSION: Final[int] = 1
_INDEX_FILENAME: Final[str] = "index.json"
_PRODUCTS_DIRECTORY: Final[str] = "products"
_CURRENT_FILENAME: Final[str] = "current.json"
_CURRENT_SCHEMA_VERSION: Final[int] = 1
_STORAGE_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "product_schema_version",
    "product_id",
    "product_run_id",
    "product_generated_at",
)


@dataclass(frozen=True)
class WeeklyProductRecord:
    """Indexed metadata for one immutable weekly product artifact."""

    product_id: str
    run_id: str
    season: str
    week: int
    generated_at: datetime
    row_count: int
    columns: tuple[str, ...]
    artifact: str

    def __post_init__(self) -> None:
        """Validate indexed product metadata."""
        for field_name, value in (
            ("product_id", self.product_id),
            ("run_id", self.run_id),
            ("season", self.season),
            ("artifact", self.artifact),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
        if self.week < 1:
            raise ValueError("week must be at least 1.")
        if self.row_count < 1:
            raise ValueError("row_count must be at least 1.")
        if not self.columns:
            raise ValueError("columns must not be empty.")
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware UTC.")
        if self.generated_at.utcoffset() != timedelta(0):
            raise ValueError("generated_at must use UTC.")


@dataclass(frozen=True)
class WeeklyProductSelection:
    """Explicit current-product selection for one weekly scope."""

    season: str
    week: int
    product_id: str
    selected_at: datetime

    def __post_init__(self) -> None:
        """Validate current-selection metadata."""
        if not self.season.strip():
            raise ValueError("season must not be empty.")
        if self.week < 1:
            raise ValueError("week must be at least 1.")
        _validate_product_id(self.product_id)
        if self.selected_at.tzinfo is None:
            raise ValueError("selected_at must be timezone-aware UTC.")
        if self.selected_at.utcoffset() != timedelta(0):
            raise ValueError("selected_at must use UTC.")


def weekly_product_root(repo: Path | None = None) -> Path:
    """Return the registered weekly-product storage root."""
    root = repo or get_settings().repo_root
    return dataset_path(root, "weekly_products")


def _validate_product_id(product_id: str) -> str:
    """Validate a product ID for safe use as one artifact filename."""
    normalized = product_id.strip()
    if not normalized:
        raise ValueError("product_id must not be empty.")
    if Path(normalized).name != normalized or normalized in {".", ".."}:
        raise ValueError("product_id must be a single filename-safe value.")
    return normalized


def weekly_product_artifact_path(
    product_id: str,
    *,
    repo: Path | None = None,
) -> Path:
    """Return the immutable Parquet path for one product ID."""
    normalized = _validate_product_id(product_id)
    return weekly_product_root(repo) / _PRODUCTS_DIRECTORY / f"{normalized}.parquet"


def _index_path(repo: Path | None = None) -> Path:
    return weekly_product_root(repo) / _INDEX_FILENAME


def _empty_index() -> dict[str, object]:
    return {
        "schema_version": WEEKLY_PRODUCT_SCHEMA_VERSION,
        "products": {},
    }


def _read_index(repo: Path | None = None) -> dict[str, Any]:
    """Read and validate the weekly-product index."""
    path = _index_path(repo)
    if not path.exists():
        return _empty_index()

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Weekly product index must contain a JSON object.")
    schema_version = raw.get("schema_version")
    if schema_version != WEEKLY_PRODUCT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported weekly product index schema version: "
            f"{schema_version!r}; expected {WEEKLY_PRODUCT_SCHEMA_VERSION}."
        )
    products = raw.get("products")
    if not isinstance(products, dict):
        raise ValueError("Weekly product index 'products' must be an object.")
    return raw


def _write_index(index: dict[str, Any], *, repo: Path | None = None) -> None:
    """Atomically write the weekly-product index."""
    path = _index_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True))
    temporary.replace(path)


def _current_path(repo: Path | None = None) -> Path:
    return weekly_product_root(repo) / _CURRENT_FILENAME


def _selection_key(season: str, week: int) -> str:
    """Return the canonical current-selection key for one weekly scope."""
    if not season.strip():
        raise ValueError("season must not be empty.")
    if week < 1:
        raise ValueError("week must be at least 1.")
    return f"{season}_week_{week:02d}"


def _empty_current() -> dict[str, object]:
    return {
        "schema_version": _CURRENT_SCHEMA_VERSION,
        "selections": {},
    }


def _read_current(repo: Path | None = None) -> dict[str, Any]:
    """Read and validate the explicit current-product manifest."""
    path = _current_path(repo)
    if not path.exists():
        return _empty_current()

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Weekly product current manifest must contain a JSON object.")
    schema_version = raw.get("schema_version")
    if schema_version != _CURRENT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported weekly product current schema version: "
            f"{schema_version!r}; expected {_CURRENT_SCHEMA_VERSION}."
        )
    selections = raw.get("selections")
    if not isinstance(selections, dict):
        raise ValueError("Weekly product current 'selections' must be an object.")
    return raw


def _write_current(current: dict[str, Any], *, repo: Path | None = None) -> None:
    """Atomically write the explicit current-product manifest."""
    path = _current_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(current, indent=2, sort_keys=True))
    temporary.replace(path)


def _parse_generated_at(value: object, *, context: str) -> datetime:
    """Parse one timezone-aware UTC timestamp."""
    if not isinstance(value, str):
        raise ValueError(f"{context} generated_at must be an ISO datetime string.")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{context} generated_at must be a valid datetime.")
    if timestamp.tzinfo is None:
        raise ValueError(f"{context} generated_at must be timezone-aware UTC.")
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"{context} generated_at must use UTC.")
    return timestamp.to_pydatetime()


def _record_from_payload(product_id: str, payload: object) -> WeeklyProductRecord:
    """Parse one index payload into an immutable record."""
    if not isinstance(payload, dict):
        raise ValueError(f"Weekly product index entry {product_id!r} must be an object.")
    required = {
        "product_id",
        "run_id",
        "season",
        "week",
        "generated_at",
        "row_count",
        "columns",
        "artifact",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"Weekly product index entry {product_id!r} is missing: " + ", ".join(missing)
        )
    columns = payload["columns"]
    if not isinstance(columns, list) or not all(isinstance(value, str) for value in columns):
        raise ValueError(f"Weekly product index entry {product_id!r} has invalid columns.")
    return WeeklyProductRecord(
        product_id=str(payload["product_id"]),
        run_id=str(payload["run_id"]),
        season=str(payload["season"]),
        week=int(payload["week"]),
        generated_at=_parse_generated_at(
            payload["generated_at"],
            context=f"Weekly product index entry {product_id!r}",
        ),
        row_count=int(payload["row_count"]),
        columns=tuple(columns),
        artifact=str(payload["artifact"]),
    )


def _record_payload(record: WeeklyProductRecord) -> dict[str, object]:
    return {
        "product_id": record.product_id,
        "run_id": record.run_id,
        "season": record.season,
        "week": record.week,
        "generated_at": record.generated_at.isoformat(),
        "row_count": record.row_count,
        "columns": list(record.columns),
        "artifact": record.artifact,
    }


def _artifact_from_record(
    record: WeeklyProductRecord,
    *,
    repo: Path | None,
) -> Path:
    """Resolve an indexed artifact while preventing path escape."""
    root = weekly_product_root(repo).resolve()
    artifact = (root / record.artifact).resolve()
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Weekly product artifact escapes storage root: {record.artifact!r}."
        ) from exc
    return artifact


def _stamp_product(
    product: DataFrame,
    identity: WeeklyProductIdentity,
) -> DataFrame:
    """Validate and prepend storage-owned product identity columns."""
    conflicts = sorted(set(_STORAGE_IDENTITY_COLUMNS).intersection(product.columns))
    if conflicts:
        raise ValueError(
            "Weekly product already contains storage identity columns: " + ", ".join(conflicts)
        )
    if product.empty:
        raise ValueError("Weekly product must contain at least one row.")

    validated = validate_weekly_game_product(product)
    if not (validated["season"].astype(str) == identity.season).all():
        raise ValueError("Weekly product rows must match identity season.")
    if not (validated["week"].astype(int) == identity.week).all():
        raise ValueError("Weekly product rows must match identity week.")

    generated_at = pd.Timestamp(identity.generated_at)
    stamped = validated.copy()
    stamped.insert(0, "product_generated_at", generated_at)
    stamped.insert(0, "product_run_id", identity.run_id)
    stamped.insert(0, "product_id", identity.product_id)
    stamped.insert(0, "product_schema_version", WEEKLY_PRODUCT_SCHEMA_VERSION)
    return stamped


def _load_indexed_artifact(
    record: WeeklyProductRecord,
    *,
    repo: Path | None,
) -> DataFrame:
    """Load and validate one indexed weekly product artifact."""
    artifact = _artifact_from_record(record, repo=repo)
    if not artifact.exists():
        raise FileNotFoundError(
            f"Weekly product artifact is missing for product_id={record.product_id!r}: {artifact}"
        )

    stored = pd.read_parquet(artifact)
    if tuple(stored.columns) != record.columns:
        raise ValueError(f"Weekly product artifact column mismatch for {record.product_id!r}.")
    if len(stored) != record.row_count:
        raise ValueError(
            f"Weekly product artifact row-count mismatch for {record.product_id!r}: "
            f"expected {record.row_count}, found {len(stored)}."
        )

    versions = set(stored["product_schema_version"].tolist())
    if versions != {WEEKLY_PRODUCT_SCHEMA_VERSION}:
        raise ValueError(f"Weekly product artifact schema mismatch for {record.product_id!r}.")
    if set(stored["product_id"].astype(str)) != {record.product_id}:
        raise ValueError(f"Weekly product artifact product_id mismatch for {record.product_id!r}.")
    if set(stored["product_run_id"].astype(str)) != {record.run_id}:
        raise ValueError(f"Weekly product artifact run_id mismatch for {record.product_id!r}.")

    generated = pd.to_datetime(stored["product_generated_at"], utc=True, errors="coerce")
    # pyrefly: ignore [missing-attribute]
    if generated.isna().any():
        raise ValueError(
            f"Weekly product artifact generated_at is invalid for {record.product_id!r}."
        )
    expected_generated = pd.Timestamp(record.generated_at)
    # pyrefly: ignore [missing-attribute]
    if not (generated == expected_generated).all():
        raise ValueError(
            f"Weekly product artifact generated_at mismatch for {record.product_id!r}."
        )
    stored["product_generated_at"] = generated

    if not (stored["season"].astype(str) == record.season).all():
        raise ValueError(f"Weekly product artifact season mismatch for {record.product_id!r}.")
    if not (stored["week"].astype(int) == record.week).all():
        raise ValueError(f"Weekly product artifact week mismatch for {record.product_id!r}.")

    domain = stored.drop(columns=list(_STORAGE_IDENTITY_COLUMNS))
    validate_weekly_game_product(domain)
    return stored


def write_weekly_product(
    product: DataFrame,
    *,
    identity: WeeklyProductIdentity,
    repo: Path | None = None,
) -> Path:
    """Write one immutable, indexed weekly product artifact."""
    product_id = _validate_product_id(identity.product_id)
    stamped = _stamp_product(product, identity)
    root = weekly_product_root(repo)
    artifact = weekly_product_artifact_path(product_id, repo=repo)
    index = _read_index(repo)
    products = index["products"]
    if not isinstance(products, dict):
        raise ValueError("Weekly product index 'products' must be an object.")

    existing_payload = products.get(product_id)
    if artifact.exists() or existing_payload is not None:
        if not artifact.exists() or existing_payload is None:
            raise ValueError(f"Weekly product store is inconsistent for product_id={product_id!r}.")
        record = _record_from_payload(product_id, existing_payload)
        existing = _load_indexed_artifact(record, repo=repo)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_name(f"{artifact.name}.tmp")
        stamped.to_parquet(temporary, index=False)
        incoming = pd.read_parquet(temporary)
        temporary.unlink(missing_ok=True)
        if not existing.equals(incoming):
            raise ValueError(
                f"Weekly product ID cannot be reused with different content: {product_id}"
            )
        return artifact

    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_name(f"{artifact.name}.tmp")
    stamped.to_parquet(temporary, index=False)
    normalized = pd.read_parquet(temporary)

    relative_artifact = artifact.relative_to(root).as_posix()
    record = WeeklyProductRecord(
        product_id=product_id,
        run_id=identity.run_id,
        season=identity.season,
        week=identity.week,
        generated_at=identity.generated_at,
        row_count=len(normalized),
        columns=tuple(normalized.columns),
        artifact=relative_artifact,
    )

    temporary.replace(artifact)
    products[product_id] = _record_payload(record)
    _write_index(index, repo=repo)
    return artifact


def load_weekly_product(
    product_id: str,
    *,
    repo: Path | None = None,
) -> DataFrame:
    """Load one exact immutable weekly product without model computation."""
    normalized_id = _validate_product_id(product_id)
    index = _read_index(repo)
    products = index["products"]
    if not isinstance(products, dict):
        raise ValueError("Weekly product index 'products' must be an object.")
    payload = products.get(normalized_id)
    if payload is None:
        artifact = weekly_product_artifact_path(normalized_id, repo=repo)
        if artifact.exists():
            raise ValueError(
                f"Weekly product artifact exists without index entry: {normalized_id!r}."
            )
        raise FileNotFoundError(f"Weekly product is not indexed: product_id={normalized_id!r}.")
    record = _record_from_payload(normalized_id, payload)
    if record.product_id != normalized_id:
        raise ValueError(f"Weekly product index key and product_id disagree: {normalized_id!r}.")
    return _load_indexed_artifact(record, repo=repo)


def list_weekly_products(
    *,
    season: str | None = None,
    week: int | None = None,
    repo: Path | None = None,
) -> tuple[WeeklyProductRecord, ...]:
    """List indexed products with optional exact scope filters."""
    index = _read_index(repo)
    products = index["products"]
    if not isinstance(products, dict):
        raise ValueError("Weekly product index 'products' must be an object.")
    records = [
        _record_from_payload(str(product_id), payload) for product_id, payload in products.items()
    ]
    if season is not None:
        records = [record for record in records if record.season == season]
    if week is not None:
        records = [record for record in records if record.week == week]
    return tuple(sorted(records, key=lambda record: record.product_id))


def _selection_from_payload(
    key: str,
    payload: object,
) -> WeeklyProductSelection:
    """Parse one explicit current-product selection."""
    if not isinstance(payload, dict):
        raise ValueError(f"Weekly product selection {key!r} must be an object.")
    required = {"season", "week", "product_id", "selected_at"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Weekly product selection {key!r} is missing: " + ", ".join(missing))
    selected_at = _parse_generated_at(
        payload["selected_at"],
        context=f"Weekly product selection {key!r}",
    )
    selection = WeeklyProductSelection(
        season=str(payload["season"]),
        week=int(payload["week"]),
        product_id=str(payload["product_id"]),
        selected_at=selected_at,
    )
    if key != _selection_key(selection.season, selection.week):
        raise ValueError(f"Weekly product selection key mismatch: {key!r}.")
    return selection


def select_current_weekly_product(
    product_id: str,
    *,
    season: str,
    week: int,
    selected_at: datetime,
    repo: Path | None = None,
) -> WeeklyProductSelection:
    """Explicitly select one indexed immutable product for a weekly scope."""
    normalized_id = _validate_product_id(product_id)
    key = _selection_key(season, week)
    if selected_at.tzinfo is None:
        raise ValueError("selected_at must be timezone-aware UTC.")
    if selected_at.utcoffset() != timedelta(0):
        raise ValueError("selected_at must use UTC.")

    records = list_weekly_products(repo=repo)
    record = next(
        (candidate for candidate in records if candidate.product_id == normalized_id),
        None,
    )
    if record is None:
        raise FileNotFoundError(f"Weekly product is not indexed: product_id={normalized_id!r}.")
    if record.season != season or record.week != week:
        raise ValueError(
            "Weekly product scope does not match current selection: "
            f"product={record.season} week {record.week}, "
            f"selection={season} week {week}."
        )

    load_weekly_product(normalized_id, repo=repo)
    current = _read_current(repo)
    selections = current["selections"]
    if not isinstance(selections, dict):
        raise ValueError("Weekly product current 'selections' must be an object.")
    selections[key] = {
        "season": season,
        "week": week,
        "product_id": normalized_id,
        "selected_at": selected_at.isoformat(),
    }
    _write_current(current, repo=repo)
    return WeeklyProductSelection(
        season=season,
        week=week,
        product_id=normalized_id,
        selected_at=selected_at,
    )


def get_current_weekly_product_selection(
    *,
    season: str,
    week: int,
    repo: Path | None = None,
) -> WeeklyProductSelection:
    """Return the explicit current-product selection for one weekly scope."""
    key = _selection_key(season, week)
    current = _read_current(repo)
    selections = current["selections"]
    if not isinstance(selections, dict):
        raise ValueError("Weekly product current 'selections' must be an object.")
    payload = selections.get(key)
    if payload is None:
        raise FileNotFoundError(
            f"No current weekly product selected for season={season!r}, week={week}."
        )
    return _selection_from_payload(key, payload)


def load_current_weekly_product(
    *,
    season: str,
    week: int,
    repo: Path | None = None,
) -> DataFrame:
    """Load only the explicitly selected current product for one weekly scope."""
    selection = get_current_weekly_product_selection(
        season=season,
        week=week,
        repo=repo,
    )
    product = load_weekly_product(selection.product_id, repo=repo)
    if not (product["season"].astype(str) == season).all():
        raise ValueError("Selected current weekly product season mismatch.")
    if not (product["week"].astype(int) == week).all():
        raise ValueError("Selected current weekly product week mismatch.")
    return product

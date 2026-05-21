# src/gridiron_edge/core/settings.py

from dataclasses import dataclass
import os
from pathlib import Path

from gridiron_edge.core.paths import repo_root


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration for gridiron_edge.

    All paths are absolute and derived from the repository root at
    import time. The ``owm_api_key`` is read from the ``OWM_API_KEY``
    environment variable and will be ``None`` if unset.

    Attributes:
        repo_root: Absolute path to the repository root.
        owm_api_key: OpenWeatherMap API key, or ``None`` if not configured.
        data_raw: Directory for raw scraped data files.
        data_cleaned: Directory for cleaned canonical datasets.
        data_modeling: Directory for derived modeling artifacts.
        data_output: Directory for output reports and Excel files.
        ranks_excel: Path to the Ranks and Betting Excel workbook.
    """

    repo_root: Path
    owm_api_key: str | None
    data_raw: Path
    data_cleaned: Path
    data_modeling: Path
    data_output: Path
    ranks_excel: Path


def get_settings() -> Settings:
    """Build a ``Settings`` instance from the current environment.

    Reads ``OWM_API_KEY`` from the environment and resolves all data
    directory paths relative to the repository root.

    Returns:
        A frozen ``Settings`` dataclass with all paths resolved.
    """
    root: Path = repo_root()
    return Settings(
        repo_root=root,
        owm_api_key=os.environ.get("OWM_API_KEY"),
        data_raw=root / "data" / "raw",
        data_cleaned=root / "data" / "cleaned",
        data_modeling=root / "data" / "modeling",
        data_output=root / "data" / "output",
        ranks_excel=root / "data" / "output" / "Ranks_and_Betting.xlsx",
    )


def ensure_data_dirs(settings: Settings | None = None) -> Settings:
    """Create standard data directories if they do not already exist.

    Args:
        settings: Optional pre-built ``Settings`` instance. If ``None``,
            calls ``get_settings()`` to build one.

    Returns:
        The resolved ``Settings`` instance (passed-in or freshly built).
    """
    s: Settings = settings or get_settings()
    for path in (s.data_raw, s.data_cleaned, s.data_modeling, s.data_output):
        path.mkdir(parents=True, exist_ok=True)
    return s

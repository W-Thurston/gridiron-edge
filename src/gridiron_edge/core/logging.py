# src/gridiron_edge/core/logging.py

import logging
import sys


def setup_logging(*, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging for CLI runs."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("gridiron")

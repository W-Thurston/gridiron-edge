# src/gridiron_edge/core/logging.py

"""Logging configuration for Gridiron Edge CLI runs.

Compact mode (default): WARNING level - only unexpected issues appear.
Verbose mode (--verbose): DEBUG level - all logger.info() and logger.debug()
calls are visible, giving a full trace of what each function is doing.
"""

from __future__ import annotations

import logging
import sys
import warnings


def setup_logging(*, verbose: bool = False) -> logging.Logger:
    """Configure root logging for a CLI run.

    Args:
        verbose: If True, sets log level to DEBUG so all internal logger
            calls are visible. If False, only WARNING and above are shown,
            keeping the console clean for compact mode.

    Returns:
        The ``gridiron`` root logger.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s  %(message)s" if verbose else "%(message)s",
        stream=sys.stdout,
        force=True,
    )
    # Silence noisy third-party loggers even in verbose mode
    for noisy in ("scrapy", "twisted", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass

    return logging.getLogger("gridiron")

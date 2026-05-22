# src/gridiron_edge/core/console.py

"""Gridiron Edge console output utilities.

Provides a consistent, timing-aware output system for CLI commands with
two verbosity modes:

- **Compact** (default): single line per step with elapsed time and checkmark.
- **Verbose** (``--verbose``): includes step details, row counts, and file paths.

Usage::

    from gridiron_edge.core.console import console, step

    console.header("run-data-pipeline", subtitle="Season 2025 · weekly refresh")

    with step("Fetch nflverse games") as s:
        path = fetch_nflverse_games_refresh()
        s.set_detail(f"written to {path.name}")
        s.set_rows(272)

    console.summary()

The ``console`` singleton reads the ``GRIDIRON_VERBOSE`` environment variable
on initialisation and is updated by the CLI when ``--verbose`` is passed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── ANSI colour codes (disabled when not a TTY) ────────────────────────────


def _supports_colour() -> bool:
    """Return True if the terminal supports ANSI colour codes."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOUR = _supports_colour()

_RESET = "\033[0m" if _COLOUR else ""
_BOLD = "\033[1m" if _COLOUR else ""
_DIM = "\033[2m" if _COLOUR else ""
_GREEN = "\033[32m" if _COLOUR else ""
_YELLOW = "\033[33m" if _COLOUR else ""
_RED = "\033[31m" if _COLOUR else ""
_CYAN = "\033[36m" if _COLOUR else ""
_WHITE = "\033[97m" if _COLOUR else ""

_TICK = "✓"
_CROSS = "✗"
_ARROW = "→"
_BAR = "━"

_WIDTH = 60


# ── StepResult ─────────────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Mutable result container set by the code inside a ``step()`` block.

    The ``step()`` context manager passes one of these to the caller via
    ``as s:``. The caller fills in detail and row count after its work runs.

    Attributes:
        name: Step label (set by the context manager, read-only for callers).
        elapsed: Wall-clock seconds (set on context exit, read-only for callers).
        ok: True if the step completed without exception.
        detail: Short description shown in verbose mode (e.g. file path, counts).
        rows: Optional integer row count shown in verbose mode.
        skipped: True if the step was skipped (shown as dimmed in output).
    """

    name: str
    elapsed: float = 0.0
    ok: bool = True
    detail: str = ""
    rows: int | None = None
    skipped: bool = False

    def set_detail(self, text: str) -> None:
        """Set the verbose detail line for this step.

        Args:
            text: Short description (e.g. ``"4,856 rows written"``).
        """
        self.detail = text

    def set_rows(self, n: int) -> None:
        """Set the row count for this step.

        Args:
            n: Number of rows read, written, or processed.
        """
        self.rows = n


# ── Console singleton ───────────────────────────────────────────────────────


class Console:
    """Singleton console output controller.

    Maintains verbosity state and a log of completed steps for the
    end-of-pipeline summary.

    Attributes:
        verbose: When True, prints detail lines and file paths per step.
    """

    def __init__(self) -> None:
        self.verbose: bool = os.environ.get("GRIDIRON_VERBOSE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._steps: list[StepResult] = []
        self._pipeline_start: float = 0.0

    def set_verbose(self, verbose: bool) -> None:
        """Update verbosity (called by the CLI after parsing --verbose flag).

        Args:
            verbose: Whether to enable verbose output.
        """
        self.verbose = verbose

    # ── Structural output ───────────────────────────────────────────────────

    def header(self, title: str, *, subtitle: str = "") -> None:
        """Print a pipeline header banner.

        Args:
            title: Main title (e.g. ``"run-data-pipeline"``).
            subtitle: Optional context line (e.g. ``"Season 2025 · weekly refresh"``).
        """
        bar = _BAR * _WIDTH
        print(f"\n{_BOLD}{_CYAN}{bar}{_RESET}")
        print(f"{_BOLD}{_WHITE}  GRIDIRON EDGE  ·  {title}{_RESET}")
        if subtitle:
            print(f"{_DIM}  {subtitle}{_RESET}")
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}")
        self._pipeline_start = time.perf_counter()
        self._steps = []

    def summary(self) -> None:
        """Print end-of-pipeline summary with total time and step outcomes."""
        total = time.perf_counter() - self._pipeline_start
        n_ok = sum(1 for s in self._steps if s.ok and not s.skipped)
        n_skip = sum(1 for s in self._steps if s.skipped)
        n_fail = sum(1 for s in self._steps if not s.ok)

        bar = _BAR * _WIDTH
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}")

        parts: list[str] = []
        if n_fail:
            parts.append(f"{_RED}{_CROSS} {n_fail} failed{_RESET}")
        parts.append(f"{_GREEN}{_TICK} {n_ok} completed{_RESET}")
        if n_skip:
            parts.append(f"{_DIM}{n_skip} skipped{_RESET}")
        parts.append(f"{_DIM}{total:.1f}s total{_RESET}")

        print("  " + "  ·  ".join(parts))
        print(f"{_BOLD}{_CYAN}{bar}{_RESET}\n")

    # ── Step output ─────────────────────────────────────────────────────────

    def _print_step_compact(self, result: StepResult) -> None:
        """Print a single-line step result in compact mode."""
        label = result.name.ljust(40)
        elapsed = f"{result.elapsed:5.1f}s"

        if result.skipped:
            print(f"  {_DIM}{_ARROW} {label}  {elapsed}  —{_RESET}")
        elif result.ok:
            print(f"  {_GREEN}{_TICK}{_RESET} {label}  {_DIM}{elapsed}{_RESET}")
        else:
            print(f"  {_RED}{_CROSS}{_RESET} {label}  {_DIM}{elapsed}{_RESET}")

    def _print_step_verbose(self, result: StepResult) -> None:
        """Print a multi-line step result in verbose mode."""
        label = result.name.ljust(40)
        elapsed = f"{result.elapsed:5.1f}s"

        if result.skipped:
            print(f"  {_DIM}{_ARROW} {label}  {elapsed}  skipped{_RESET}")
            return

        icon: str = f"{_GREEN}{_TICK}{_RESET}" if result.ok else f"{_RED}{_CROSS}{_RESET}"

        print(f"  {icon} {_BOLD}{result.name}{_RESET}  {_DIM}{elapsed}{_RESET}")

        if result.rows is not None:
            print(f"       {_DIM}rows:    {result.rows:,}{_RESET}")
        if result.detail:
            print(f"       {_DIM}detail:  {result.detail}{_RESET}")


# ── Context manager ─────────────────────────────────────────────────────────


console = Console()


@contextmanager
def step(name: str, *, skip: bool = False) -> Iterator[StepResult]:
    """Context manager for a timed, named pipeline step.

    Prints a running indicator (in verbose mode), then the result line with
    elapsed time after the block exits. Exceptions are caught, the step is
    marked as failed, and the exception is re-raised.

    Args:
        name: Human-readable step label.
        skip: If True, the block body is not executed and the step is marked
            as skipped. Use this for conditionally disabled pipeline stages.

    Yields:
        A mutable ``StepResult`` the caller can populate with detail and counts.

    Example::

        with step("Fetch nflverse games") as s:
            path = fetch_nflverse_games_refresh()
            s.set_detail(path.name)
            s.set_rows(272)
    """
    result = StepResult(name=name, skipped=skip)

    if skip:
        console._steps.append(result)
        if console.verbose:
            console._print_step_verbose(result)
        else:
            console._print_step_compact(result)
        yield result
        return

    if console.verbose:
        print(f"  {_DIM}{_ARROW} {name}...{_RESET}", flush=True)

    t0 = time.perf_counter()
    try:
        yield result
        result.ok = True
    except Exception:
        result.ok = False
        result.elapsed = time.perf_counter() - t0
        console._steps.append(result)
        if console.verbose:
            console._print_step_verbose(result)
        else:
            console._print_step_compact(result)
        raise
    else:
        result.elapsed = time.perf_counter() - t0
        console._steps.append(result)
        if console.verbose:
            console._print_step_verbose(result)
        else:
            console._print_step_compact(result)

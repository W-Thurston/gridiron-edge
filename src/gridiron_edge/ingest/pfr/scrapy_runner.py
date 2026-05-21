# src/gridiron_edge/ingest/pfr/scrapy_runner.py

"""Run Scrapy spiders without Twisted's ReactorNotRestartable error.

- First crawl in a process: in-process (Scrapy logs stream immediately).
- Later crawls in the same process: isolated subprocess (fork on Linux/WSL,
  spawn on Windows).
"""

from __future__ import annotations

from collections.abc import Sequence
import multiprocessing as mp
import os
import sys
from typing import Any

from scrapy.crawler import CrawlerRunner
from scrapy.settings import Settings
from twisted.internet import defer, reactor

SpiderSpec = tuple[type, dict[str, Any]]

# Set after the first reactor.run() in this interpreter; reactor cannot restart.
_reactor_used_in_this_process: bool = False


def _build_settings() -> Settings:
    """Build Scrapy settings from the project settings module.

    Reads ``SCRAPY_SETTINGS_MODULE`` from the environment, defaulting to
    ``PFR_scraper.settings`` if not set.

    Returns:
        A configured Scrapy ``Settings`` instance.
    """
    os.environ.setdefault("SCRAPY_SETTINGS_MODULE", "PFR_scraper.settings")
    settings = Settings()
    settings.setmodule(os.environ["SCRAPY_SETTINGS_MODULE"], priority="project")
    return settings


def _spider_label(specs: Sequence[SpiderSpec]) -> str:
    """Build a human-readable label listing spider names from a spec sequence.

    Args:
        specs: Sequence of ``(spider_class, kwargs)`` pairs.

    Returns:
        Comma-separated string of spider names.
    """
    names = [getattr(cls, "name", cls.__name__) for cls, _ in specs]
    return ", ".join(names)


def _run_spiders_in_child(specs: list[SpiderSpec]) -> None:
    """Run all spiders in the current process using Twisted's reactor.

    Should only be called once per interpreter lifetime due to Twisted's
    ``ReactorNotRestartable`` constraint.

    Args:
        specs: List of ``(spider_class, kwargs)`` pairs to crawl sequentially.
    """
    runner = CrawlerRunner(_build_settings())

    @defer.inlineCallbacks  # type: ignore[misc]
    def _crawl_all() -> Any:  # noqa: ANN401
        for spider_cls, kwargs in specs:
            yield runner.crawl(spider_cls, **kwargs)
        reactor.stop()  # type: ignore[attr-defined]

    _crawl_all()
    reactor.run()  # type: ignore[attr-defined]


def _run_spiders_subprocess(specs: list[SpiderSpec]) -> None:
    """Run spiders in an isolated subprocess to avoid ReactorNotRestartable.

    Spawns a child process that runs ``_run_spiders_in_child`` and waits
    for it to complete.

    Args:
        specs: List of ``(spider_class, kwargs)`` pairs to crawl sequentially.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero exit code.
    """
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    proc = ctx.Process(  # type: ignore[attr-defined]
        target=_run_spiders_in_child,
        args=(specs,),
        daemon=False,
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        msg = f"Scrapy crawl subprocess exited with code {proc.exitcode}"
        raise RuntimeError(msg)


def run_spiders(specs: Sequence[SpiderSpec]) -> None:
    """Run one or more spiders, handling the reactor-safe lifecycle automatically.

    On the first call within a process, runs spiders in-process for immediate
    log streaming. On subsequent calls, spawns a subprocess to avoid Twisted's
    ``ReactorNotRestartable`` constraint.

    Args:
        specs: Sequence of ``(spider_class, kwargs)`` pairs. Each spider is
            crawled sequentially within the same run.
    """
    global _reactor_used_in_this_process  # noqa: PLW0603

    crawl_specs = list(specs)
    if not crawl_specs:
        return

    label = _spider_label(crawl_specs)
    if not _reactor_used_in_this_process:
        print(f"> Scrapy crawl starting in-process: {label}", flush=True)
        _run_spiders_in_child(crawl_specs)
        _reactor_used_in_this_process = True
        print(f"> Scrapy crawl finished: {label}", flush=True)
        return

    print(f"> Scrapy crawl starting (subprocess): {label}", flush=True)
    _run_spiders_subprocess(crawl_specs)
    print(f"> Scrapy crawl finished (subprocess): {label}", flush=True)

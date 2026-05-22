# src/gridiron_edge/ingest/pfr/scrapy_runner.py

"""Run Scrapy spiders without Twisted's ReactorNotRestartable error.

Strategy:
- First crawl in a process: ``CrawlerProcess`` run directly on the calling
  thread with ``install_signal_handlers=False`` to avoid the ValueError that
  occurs when Twisted tries to register SIGINT/SIGTERM handlers from a
  non-main thread or subprocess.
- Later crawls in the same process: isolated subprocess (``fork`` on
  Linux/WSL, ``spawn`` on Windows) to avoid ``ReactorNotRestartable``.

Scrapy 2.13+ defaults to the asyncio reactor. ``CrawlerProcess`` owns the
reactor lifecycle and stops it cleanly after all spiders finish, which avoids
the silent hang that occurs when ``reactor.run()`` is called directly and
``reactor.stop()`` is never reached due to an unhandled Deferred error.
"""

from __future__ import annotations

from collections.abc import Sequence
import multiprocessing as mp
import os
import sys
from typing import Any

from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from scrapy.utils.reactor import install_reactor

SpiderSpec = tuple[type, dict[str, Any]]

# Set after the first in-process crawl; reactor cannot restart.
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


def _run_spiders_in_process(specs: list[SpiderSpec]) -> None:
    """Run all spiders in the current process using CrawlerProcess.

    ``CrawlerProcess`` manages the reactor lifecycle end-to-end: it installs
    the asyncio reactor, starts it, and stops it cleanly once all crawlers
    finish. This avoids the silent hang that occurs when ``reactor.run()`` is
    called directly and ``reactor.stop()`` is never reached.

    ``install_signal_handlers=False`` is required because ``signal.signal()``
    only works from the main thread of the main interpreter. Twisted tries to
    install SIGINT/SIGTERM handlers on reactor startup — passing False skips
    this so the call is safe from any calling context.

    Must only be called once per interpreter lifetime. Subsequent calls must
    use ``_run_spiders_subprocess`` to avoid ``ReactorNotRestartable``.

    Args:
        specs: List of ``(spider_class, kwargs)`` pairs to crawl sequentially.
    """
    install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")

    process = CrawlerProcess(_build_settings())
    for spider_cls, kwargs in specs:
        process.crawl(spider_cls, **kwargs)
    process.start(install_signal_handlers=False)


def _run_spiders_subprocess(specs: list[SpiderSpec]) -> None:
    """Run spiders in an isolated subprocess to avoid ReactorNotRestartable.

    Spawns a child process that runs ``_run_spiders_in_process`` and waits
    for it to complete.

    Args:
        specs: List of ``(spider_class, kwargs)`` pairs to crawl sequentially.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero exit code.
    """
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    proc = ctx.Process(  # type: ignore[attr-defined]
        target=_run_spiders_in_process,
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

    On the first call within a process, runs spiders in-process via
    ``CrawlerProcess`` with signal handler installation disabled. On
    subsequent calls, spawns a subprocess to avoid Twisted's
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
        _run_spiders_in_process(crawl_specs)
        _reactor_used_in_this_process = True
        print(f"> Scrapy crawl finished: {label}", flush=True)
        return

    print(f"> Scrapy crawl starting (subprocess): {label}", flush=True)
    _run_spiders_subprocess(crawl_specs)
    print(f"> Scrapy crawl finished (subprocess): {label}", flush=True)

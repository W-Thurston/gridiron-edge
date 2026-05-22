# Gridiron Edge — Project Plan

See [HANDOFF.md](HANDOFF.md) for how to run the project today.

**Status key:** `[x]` done · `[ ]` not started · `[~]` in progress

---

## Architecture

| Layer | Module |
|-------|--------|
| Game + schedule ingest | `gridiron_edge.ingest.nflverse` (nflverse/nfl_data_py) |
| Weather ingest | `gridiron_edge.ingest.pfr.collector_impl` (OpenWeatherMap) |
| Odds ingest + ledger | `gridiron_edge.ingest.odds` (DraftKings → Parquet) |
| Transform | `gridiron_edge.transform.clean` |
| Modeling features | `gridiron_edge.features.pipeline` |
| Elo ratings | `gridiron_edge.ratings.elo` |
| Season simulation | `gridiron_edge.sim` |
| Visualisation | `gridiron_edge.viz` |
| CLI | `uv run gridiron` |

---

## Phase 1–9 — Core refactor ✅

Original migration from `data_pipelines/` + `model_pipelines/` + `utils/` into `src/gridiron_edge/`. All complete.

---

## Phase 10 — Season simulation ✅

- [x] `sim/season.py` — Monte Carlo regular season simulation
- [x] `sim/playoffs.py` — NFL tiebreaker logic, conference seeding, playoff bracket
- [x] `viz/charts.py` — playoff probability table (plottable)
- [x] CLI: `gridiron sim run`

---

## Phase 11 — Stabilise ✅

- [x] Integration tests (Elo fit, features pipeline, outputs)
- [x] Fix TEAM_B home coord merge in `metrics/travel/travel.py`
- [x] Collector paths via `datasets/registry.py` + `core/settings.py`

---

## Phase 12 — Tooling + code quality ✅

- [x] Migrate Poetry → uv (hatchling build backend)
- [x] Ruff: full lint + format suite (Google docstrings, all rule categories)
- [x] Pyrefly: static type checking (Python 3.12)
- [x] Google-style docstrings on all public classes and functions
- [x] Type annotations across all non-PFR_scraper modules
- [x] Fix all real bugs surfaced by linting/type-checking pass

---

## Phase 13 — nflverse migration ✅

- [x] Replace PFR/Scrapy with nfl_data_py (bypasses Cloudflare)
- [x] `ingest/nflverse/games.py`: fetch historical schedules as Parquet
- [x] `ingest/nflverse/schedule.py`: fetch upcoming (unplayed) games
- [x] `transform/clean/games_nflverse.py`: nflverse → canonical games schema
- [x] `transform/clean/schedule_nflverse.py`: nflverse → canonical schedule schema
- [x] Raw storage in Parquet; canonical output remains CSV
- [x] CLI: `gridiron ingest nflverse-games [--season N] [--all-years]`
- [x] CLI: `gridiron ingest nflverse-upcoming [--season N]`
- [x] `run-data-pipeline --upcoming-season` flag for season-boundary use case
- [x] All season args optional — defaults to current season via date inference

---

## Phase 14 — Console output system ✅

- [x] `core/console.py`: timed step context manager, header/summary banners
- [x] Compact mode (default): one line per step with elapsed time and ✓/✗/—
- [x] Verbose mode (`-v`): step detail, row counts, file paths, debug logs
- [x] `core/logging.py`: WARNING in compact, DEBUG in verbose
- [x] tqdm progress bar hidden in compact mode
- [x] Lazy CLI imports: `--help` renders in <1s

---

## Phase 15 — Excel retirement + viz refactor ✅

- [x] `ingest/odds/store.py`: append-only long-format Parquet odds ledger
- [x] `ingest/odds/draftkings.py`: writes to ledger + snapshot (not Excel)
- [x] `viz/predictions.py`: weekly matchup PNG + static HTML (migrated from notebook)
  - Logo paths by full long name
  - Time separators built explicitly
  - 24hr → 12hr time conversion
  - DK underdog highlight optional
- [x] `ratings/elo/predict.py`: writes versioned CSV (not Excel)
- [x] `viz/excel.py`: replaced with `write_elo_rankings_csv()` → CSV
- [x] `core/settings.py`: removed `ranks_excel` field
- [x] `tests/integration/test_excel_output.py`: updated to test CSV output
- [x] CLI: `gridiron output predictions --year --week [--format]`
- [x] CLI: `gridiron output ranks` → CSV
- [x] CLI: `gridiron ingest dk-odds --season --week`
- [ ] HTML viz graphical fixes (deferred)

---

## Phase 16 — Phase E cleanup ✅

- [x] Delete `src/PFR_scraper/` (entire Scrapy package)
- [x] Delete `scrapy_runner.py`, `historical.py`, `upcoming.py`, `scrapy.cfg`
- [x] `collector_impl.py`: remove spider imports + spider-calling methods
- [x] `collector.py`: weather + DK odds facade only
- [x] `transform/clean/games.py` + `schedule.py`: shims → nflverse cleaners
- [x] `pyproject.toml`: remove scrapy; add requests, urllib3, pyarrow, matplotlib, pytz

---

## Validation checklist

```bash
uv sync
uv run pytest
uv run gridiron --help
uv run gridiron run-data-pipeline --all-years --upcoming-season 2026 --build-elo --fit-elo-all-years
uv run gridiron output predictions --year 2026-2027 --week 1
uv run gridiron sim run
uv run gridiron output ranks --year 2025-2026 --week 22
```

```bash
# Code quality gates
uv run ruff check src/ --fix
uvx pyrefly check
uv run pytest
```

---

## Phase 17 — Next priorities

- [ ] Validate `sim run` output end-to-end with current data
- [ ] HTML viz graphical fixes
- [ ] `models/game_prediction/` — train and predict ML game prediction models
- [ ] `analytics/matchup_reports.py` — team matchup report generation
- [ ] `analytics/team_insights.py` — season-level team analytics
- [ ] Multi-sportsbook odds (FanDuel, BetMGM) — add new ingest modules
- [ ] Betting tracker — bet log CLI + P&L tracking
- [ ] Run full manual validation checklist on local data
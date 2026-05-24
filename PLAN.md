# Gridiron Edge — Project Plan

See [HANDOFF.md](HANDOFF.md) for how to run the project today.

**Status key:** `[x]` done · `[ ]` not started · `[~]` in progress

---

## Vision

Gridiron Edge is a long-term **football intelligence and betting research platform** — not just a prediction model.

The system evolves in layers:

| Layer | Description | Status |
|-------|-------------|--------|
| Data Platform | Ingest and store football reality | ✅ Operational |
| Football State Representation | Represent what teams actually are | 🔜 Phase 19 |
| Prediction Engine | What is likely to happen | 🔜 Phase 20 |
| Market Intelligence | What does Vegas think | 🔜 Phase 21 |
| Decision Engine | What should actually be bet | 🔜 Future |
| Explainability & Insights | Why does the model think this | 🔜 Future |

**Core philosophy:** Betting edge is discovered through measurement, infrastructure, and systematic disagreement with the market — not through chasing more complex models. Build evaluation infrastructure before building better models.

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
| Evaluation | `gridiron_edge.evaluation` |
| Models | `gridiron_edge.models` (Predictor + Trainable protocols, ArtifactStore) |
| Analytics | `gridiron_edge.analytics` *(planned)* |
| CLI | `uv run gridiron` (split into `cli/` sub-modules) |

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
- [x] `ratings/elo/predict.py`: writes versioned CSV (not Excel)
- [x] `viz/excel.py`: replaced with `write_elo_rankings_csv()` → CSV
- [x] CLI: `gridiron output predictions --year --week [--format]`
- [x] CLI: `gridiron output ranks` → CSV
- [x] CLI: `gridiron ingest dk-odds --season --week`

---

## Phase 16 — Scrapy retirement + cleanup ✅

- [x] Delete `src/PFR_scraper/` (entire Scrapy package)
- [x] Delete `scrapy_runner.py`, `historical.py`, `upcoming.py`, `scrapy.cfg`
- [x] `collector_impl.py`: remove spider imports + spider-calling methods
- [x] `collector.py`: weather + DK odds facade only
- [x] `transform/clean/games.py` + `schedule.py`: shims → nflverse cleaners
- [x] `pyproject.toml`: remove scrapy; add requests, urllib3, pyarrow, matplotlib, pytz

---

## Phase 17 — Dead code removal + CLI fix ✅

- [x] Add missing `gridiron output predictions` CLI command (was documented but never wired)
- [x] Remove empty stub files: `pipelines/`, `ratings/elo/rating.py`, `ratings/elo/features.py`, root-level `simulate_season.py`, and other orphaned placeholders
- [x] Fix vulture/ruff findings: noqa annotations on side-effect imports, RET504, unused `ctx` parameter
- [x] All quality gates pass: ruff, pyrefly, pytest (14/14)

---

## Phase 18 — Evaluation infrastructure + architectural foundations ✅

**Evaluation framework:**

- [x] Prediction archive — append-only Parquet log at `data/output/predictions/predictions_log.parquet`
  - Deduped by `(game_id, model_version)`; auto-appended on every `output predictions` run
- [x] `evaluation/metrics.py` — Brier score, log loss, calibration table, accuracy
- [x] `evaluation/backfill.py` — single generic `backfill_model(model_version)` covering all registered models
- [x] Historical backfill covering full 1999-present history including Super Bowls
- [x] Fix elo_v1 Super Bowl exclusion bug (pre-2021 wk22 was silently dropped by legacy Elo CSV path)
- [x] Elo parameter grid search — `evaluation/tune.py`
  - elo_v2: flat-K search (100 combinations); best: K=40, div=350, regress=0.40; Brier 0.2269
  - elo_v3: zone-based K search (63,504 combinations, ~8h); best: k_early=40, k_mid=40, k_week18=50, k_post=60, div=360, regress=0.40; Brier 0.2269
  - elo_v2 and elo_v3 statistically tied; zone-based K adds no meaningful improvement over flat-K
  - elo_v1 remains best-calibrated at high confidence (60-70%+ range); recommended for production predictions
  - tqdm progress bar with live best-score postfix; holdout = last 3 seasons; results auto-saved to Parquet
- [x] CLI: `gridiron evaluate backfill / summary / calibration / tune [--v3] [--apply] [--save]`

**Architectural foundations:**

- [x] `models/base.py` — `Predictor` + `Trainable` protocols (runtime_checkable); `PredictorSpec` with `trainable` flag
- [x] `models/artifact.py` — `ArtifactStore` + `ModelMetadata`; immutable versioned artifact store under `data/models/`
- [x] `models/registry.py` — `PredictorRegistry` with `is_trainable()`, `trainable_names()`
- [x] `models/elo/predictor.py` — `EloV1`, `EloV2`, `EloV3` predictors registered at import time
- [x] `features/manifest.py` — feature set manifest written alongside `modeling_file.csv`
- [x] `datasets/loaders.py` — `load_modeling_file()` with schema version + column validation
- [x] CLI split into `cli/` sub-modules: ingest, transform, ratings, features, output, sim, evaluate, models
- [x] CLI: `gridiron models train / list / info`
- [x] Test coverage: 44 tests (added 30 for archive, metrics, manifest)

**Performance:**

- [x] Elo table rebuild: 47.7s → 1.3s — dict-based engine replaces pandas boolean-index loop
- [x] Feature pipeline: 73.0s → 5.0s — vectorised haversine distance + timezone cache
- [x] Full `run-data-pipeline --all-years`: 127s → 8s

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
uv run gridiron evaluate backfill --overwrite
uv run gridiron evaluate summary --group-by model_version
uv run gridiron evaluate calibration
uv run gridiron models list
```

```bash
# Code quality gates
uv run ruff check src/ --fix
uvx pyrefly check
uv run pytest
```

---

## Architectural debt — tracked items

Known seams that will cause friction if left unaddressed. Each is tagged with the phase where it becomes blocking.

| Item | Blocking at | Notes |
|------|------------|-------|
| Odds `game_id` resolver | Phase 21 | DK odds ledger uses internal event IDs, not canonical `YYYY_WW_AWAY_HOME` format; join will fail silently |
| Test coverage for `backfill.py` + `tune.py` | Ongoing | `archive.py`, `metrics.py`, `manifest.py` now covered; backfill and tune still need tests |
| `datasets/registry.py` self-registration | Long-term | Flat dict grows linearly with datasets; low urgency until >20 datasets |

---

## Phase 19 — Richer football state representation 🔜

**Goal:** Move beyond a single Elo number per team toward matchup-aware representations.
All Phase 18 prerequisites are complete — feature manifest, model artifact store, evaluation framework.

- [ ] Ingest nflverse play-by-play data (`nfl_data_py.import_pbp_data`)
- [ ] Compute team-level EPA/play (offense + defense, pass + rush splits)
- [ ] Add EPA features to `features/team/` pipeline
- [ ] Bump `CURRENT_SCHEMA_VERSION` in `features/manifest.py` when feature columns change
- [ ] Investigate success rate, explosive play rate, turnover-luck adjustments
- [ ] Evaluate whether EPA features improve Brier score vs Elo-only baseline via `evaluate summary --group-by model_version`

---

## Phase 20 — ML game prediction models 🔜

**Goal:** Build and evaluate a data-driven model alongside Elo.
All Phase 18 prerequisites are complete — artifact store, Trainable protocol, evaluation framework.

- [ ] `models/game_prediction/predictor.py` — implement `Predictor` + `Trainable` protocols
  - `train()`: load feature matrix, fit model, save artifact via `ArtifactStore`, return `ModelMetadata`
  - `predict_historical()`: load artifact, run inference on games, return archive-schema rows
  - `predict_upcoming()`: load artifact, run inference on upcoming schedule
- [ ] Start with logistic regression (`logistic_v1`) — interpretable baseline before XGBoost/neural
- [ ] Register with `@PredictorRegistry.register` — backfill, evaluate, and compare work automatically
- [ ] Compare Brier score vs elo_v1 baseline via `evaluate summary --group-by model_version`
- [ ] CLI: `gridiron models train logistic_v1` — end-to-end train + save + report holdout Brier

---

## Phase 21 — Market intelligence layer 🔜

**Goal:** Systematically compare model predictions against the market.
**Prerequisite:** Odds `game_id` resolver must be built before any odds-prediction join works.

- [ ] Odds `game_id` resolver — map DK internal event IDs to canonical `YYYY_WW_AWAY_HOME` format using team names + dates
- [ ] Historical odds database — store spread, moneyline, totals, opening + closing lines
- [ ] Multi-sportsbook ingest — FanDuel, BetMGM alongside existing DraftKings
- [ ] Implied probability extraction with de-vig (power/Shin method)
- [ ] Edge calculation: `model_prob - implied_prob` per game
- [ ] Market disagreement analysis: where does model consistently differ from consensus?
- [ ] CLI: `gridiron market edge --year --week`

---

## Phase 22 — Betting tracker + analytics 🔜

**Goal:** Close the loop from prediction to outcome tracking.

- [ ] Bet log CLI — record bets placed with stake, odds, market, sportsbook
- [ ] P&L tracking — ROI by model, season, market type, confidence tier
- [ ] `analytics/matchup_reports.py` — per-game matchup breakdown report
- [ ] `analytics/team_insights.py` — season-level team analytics and trends
- [ ] CLI: `gridiron bets log` + `gridiron bets summary`

---

## Backlog / Future

Items with long-term value but no near-term dependency:

- **Drive/possession simulation** — expand Monte Carlo toward play-level outcome distributions
- **Injury + roster intelligence** — injury-adjusted team strength, player availability impacts
- **Weather effects** — integrate OWM data into prediction features (ingest exists, feature does not)
- **Ensemble systems** — stack Elo + EPA model + ML model with uncertainty estimates
- **Public dashboards / automated reports** — shareable weekly output
- **Live game intelligence** — real-time win probability updates (major architecture change)
- **`datasets/registry.py` self-registration** — replace flat dict with auto-discovery when dataset count grows past ~20
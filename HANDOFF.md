# Gridiron Edge — Handoff

How everything works right now. Assumes you know what the project does — see [README.md](README.md) for the one-paragraph version and [PLAN.md](PLAN.md) for what's coming next.

---

## Architecture

| Layer | Module |
|-------|--------|
| Game + schedule ingest | `gridiron_edge.ingest.nflverse` (nflverse/nfl_data_py) |
| PBP ingest | `gridiron_edge.ingest.nflverse.pbp` |
| Weather ingest | `gridiron_edge.ingest.weather.openweather` (OpenWeatherMap) |
| Odds ingest + ledger | `gridiron_edge.ingest.odds` (DraftKings → Parquet) |
| Transform | `gridiron_edge.transform.clean` |
| Shared constants | `gridiron_edge.core.constants` — single source for `HOME_GAME_LOCATION`, `AWAY_WIN_LOCATION`, `HOLDOUT_SEASONS`, `EXPANSION_TEAMS` |
| Feature column definitions | `gridiron_edge.models.game_prediction._columns` — pure-data leaf, no pandas |
| Feature engineering functions | `gridiron_edge.models.game_prediction._features` — `FEATURE_SETS`, `_prepare_data`, `_is_trained` |
| EPA window hyperparameter infra | `gridiron_edge.models.game_prediction._epa_window` — `WindowData`, `_rebuild_features_with_window` |
| Feature pipeline + dep validation | `gridiron_edge.features.pipeline` (schema v3) + `features.registry.validate_ordering()` |
| Elo ratings | `gridiron_edge.ratings.elo` — `core.elo_win_probability(divisor=)`, `EloTableConfig(divisor=)` |
| Simulation types + config | `gridiron_edge.sim._types` — `SimulationConfig(divisor=)`, `SimPaths`, `TeamIndex`, `SimulationResults` |
| Simulation engine (numba) | `gridiron_edge.sim._engine` — Elo kernels, record accumulation, Monte Carlo |
| Simulation orchestration | `gridiron_edge.sim.season` — data loading, output builders, `run_full_simulation` |
| Visualisation | `gridiron_edge.viz` |
| Evaluation | `gridiron_edge.evaluation` — `archive`, `metrics`, `select`, `backfill`, `tune` |
| Models | `gridiron_edge.models` — Predictor + Trainable protocols, ArtifactStore |
| CLI | `gridiron_edge.cli` — `main.py` stage-list pipeline, sub-apps per domain |

---

## Repository layout

| Path | Role |
|------|------|
| `src/gridiron_edge/core/` | Settings, logging, console, shared constants |
| `src/gridiron_edge/ingest/` | All data ingestion (nflverse, weather, odds) |
| `src/gridiron_edge/transform/` | Raw → canonical schema mappers |
| `src/gridiron_edge/features/` | Feature registry, pipeline, dependency validation |
| `src/gridiron_edge/ratings/elo/` | Elo table, fit, predict, evaluate |
| `src/gridiron_edge/models/` | Predictor protocol, artifact store, model registry, game prediction variants |
| `src/gridiron_edge/evaluation/` | Metrics, backfill, archive, tuning, model selection |
| `src/gridiron_edge/sim/` | Monte Carlo season + playoff simulation |
| `src/gridiron_edge/viz/` | Predictions image/HTML, playoff table, rankings CSV |
| `src/gridiron_edge/cli/` | Typer app + sub-commands |
| `data/` | Generated at runtime — not committed |

**Data layout:**

```
data/
  raw/          nflverse Parquet files (games, upcoming schedule, PBP)
  cleaned/      Canonical CSVs (games, schedule, Elo state, stadiums)
  modeling/     Feature matrix (base + full, Parquet)
  output/
    predictions/{year}/          week_NN_predictions.png + .html + .csv
    predictions/predictions_log.parquet   prediction archive (all models)
    rankings/                    elo_rankings_{year}_wkNN.csv
    sim/                         playoff probability tables
  odds/
    dk_odds_log.parquet          Full historical ledger (all pulls)
    dk_odds_current.parquet      Latest pull snapshot (for viz)
```

---

## Setup

```bash
uv sync
uv run gridiron --help
```

Python `>=3.12,<4`. All dependencies managed via `uv` / `pyproject.toml`.

---

## Configuration

| Setting | How |
|---------|-----|
| OpenWeather API key | `--owm-api-key` flag or env var `OWM_API_KEY` |
| Elo divisor | `--divisor` on `gridiron sim run` (default 480; use tuned value after `evaluate tune elo`) |
| Data paths | `core/settings.py` + `datasets/registry.py` |
| Shared constants | `core/constants.py` |

---

## Key design decisions

### `GAME_LOCATION` schema

Three values only — `"H"` (home win), `"@"` (away win), `"N"` (neutral site). The old PFR-era `"NULL_VALUE"` sentinel was retired. Missing data fields (GAMETIME, STADIUM, ROOF, SURFACE) use `""`.

### Elo divisor is parameterised end-to-end

`core.py` → `EloTableConfig` → `SimulationConfig` → `--divisor` CLI flag. The tuner (`evaluate tune elo`) finds the optimal divisor; set it consistently across table building and simulation. Default is 480 (classic NFL Elo). elo_v2 optimum is 350.

### Feature dependency validation

`FeatureSpec.depends_on` declares ordering constraints. `validate_ordering()` is called at pipeline import time — a mis-ordering raises `ValueError` immediately rather than silently producing wrong columns at training time.

### Prediction archive `is_backfilled`

`predictions_log.parquet` has a boolean `is_backfilled` column. Historical backfill predictions set it to `True`; live pre-game predictions set it to `False`. Filter on this rather than `predicted_at` for live-vs-backfill analysis.

### Weather ingest is idempotent

`fetch_weather` reads existing `weather_enriched.csv`, computes the set difference of `GAME_ID`s, and only calls the OWM API for games not already enriched. Safe to re-run.

### sim/season.py decomposition

`sim/` is split into three files with a clean dependency hierarchy:
- `_types.py` — pure data, no I/O (constants, dataclasses)
- `_engine.py` — numba kernels (imports from `_types` only)
- `season.py` — orchestration (imports from both)

Numba cannot call regular Python functions at JIT time, so the Elo formula is duplicated in `_engine.py`. A comment cross-references `ratings/elo/core.py` — if the formula changes, update both.

---

## Primary workflows

### Full bootstrap (first run or season reset)

```bash
uv run gridiron run-data-pipeline \
  --all-years \
  --upcoming-season 2026 \
  --fit-elo-all-years \
  --season-year 2025-2026
```

 ~135s. Fetches all history (1999–present), 2026 upcoming schedule, rebuilds Elo, fetches weather (idempotent), builds feature matrix.

### Weekly refresh (during season)

```bash
uv run gridiron run-data-pipeline
```

Runs all stages. Re-fetches current season games + upcoming schedule, incremental Elo fit, rebuilds features.

### Stage control

`--only` and `--skip` are mutually exclusive. Valid stage names:

`fetch-games` · `clean-games` · `fetch-upcoming` · `clean-upcoming` · `fetch-weather` · `fetch-odds` · `build-epa` · `build-elo` · `build-features`

```bash
# Features only
uv run gridiron run-data-pipeline --only build-features

# Skip odds and weather
uv run gridiron run-data-pipeline --skip fetch-odds --skip fetch-weather
```

### Step-by-step

```bash
uv run gridiron ingest nflverse-games [--all-years]
uv run gridiron ingest nflverse-upcoming --season 2026
uv run gridiron transform clean-games
uv run gridiron transform clean-upcoming
uv run gridiron ratings elo fit [--all-years]
uv run gridiron features model-inputs [--all-years]
```

### Outputs

```bash
uv run gridiron output predictions --year 2026-2027 --week 1
uv run gridiron output ranks --year 2026-2027 --week 1
uv run gridiron sim run [--n-sims 10000] [--divisor 350]
```

### Model training and evaluation

```bash
uv run gridiron models train logistic_v1 [--overwrite]
uv run gridiron models train random_forest_v1 [--overwrite]
uv run gridiron evaluate backfill --model-version elo_v2
uv run gridiron evaluate select-model
uv run gridiron evaluate report
```

### Archive migration (one-time, pre-thermonuclear-review archives only)

```bash
python -c "from gridiron_edge.evaluation.archive import migrate_archive; migrate_archive()"
```

Adds `is_backfilled` column to existing prediction archives. Idempotent.

---

## File contract

| File | Contents |
|------|----------|
| `data/cleaned/NFL_wk_by_wk_cleaned.csv` | Canonical historical games — `GAME_LOCATION` = `"H"/"@"/"N"` |
| `data/cleaned/NFL_upcoming_schedule_cleaned.csv` | Canonical upcoming schedule |
| `data/cleaned/NFL_Team_Elo.csv` | Elo ratings state table |
| `data/modeling/base_modeling_file.parquet` | Base modeling rows (pre-features) |
| `data/modeling/modeling_file.parquet` | Full feature matrix |
| `data/cleaned/NFL_stadium_reference.csv` | Stadium geo reference — add new venues here for weather coverage |
| `data/output/predictions/predictions_log.parquet` | Prediction archive — `is_backfilled` flags historical vs live |
| `data/odds/dk_odds_log.parquet` | Full DK odds history (long format) |
| `data/odds/dk_odds_current.parquet` | Latest DK odds snapshot for viz |

---

## Where to read code

| What | Where |
|------|-------|
| CLI entry + `run-data-pipeline` | `cli/main.py` |
| Shared constants | `core/constants.py` |
| Feature pipeline + ordering | `features/pipeline.py` |
| Feature dependency validation | `features/registry.py` — `validate_ordering()` |
| Feature column definitions | `models/game_prediction/_columns.py` |
| Feature engineering functions | `models/game_prediction/_features.py` |
| EPA window hyperparameter infra | `models/game_prediction/_epa_window.py` |
| Elo core formula (parameterised) | `ratings/elo/core.py` |
| Simulation types + config | `sim/_types.py` |
| Simulation engine (numba) | `sim/_engine.py` |
| Simulation orchestration | `sim/season.py` |
| Prediction archive schema | `evaluation/archive.py` |
| Model selection + reporting | `evaluation/select.py` |
| Weather ingest (idempotent) | `ingest/weather/openweather.py` |
| DK odds ingest | `ingest/odds/draftkings.py` |
| DK game_id resolution | ingest/odds/_game_id.py |

All paths relative to `src/gridiron_edge/`.

---

## Code quality gates


```bash
# Normal dev loop (runs on every commit via pre-commit hook)
uv run ruff check . --fix && uvx pyrefly check && uv run pytest -m "unit and not slow"

```

Pre-commit hooks enforce ruff + pyrefly + unit tests on every commit.
Pre-push hooks add integration + e2e tests.
Use uv run gridiron -v <command> for verbose output.

---

### Testing architecture

Three-tier test pyramid with auto-applied pytest markers:

| Tier        | Directory            | Runs When                      | Speed         |
| ----------- | -------------------- | ------------------------------ | ------------- |
| Unit        | `tests/unit/`        | Every commit (pre-commit hook) | \~1s each     |
| Integration | `tests/integration/` | Every push (pre-push hook)     | \~5-15s each  |
| E2E         | `tests/e2e/`         | Every push (pre-push hook)     | \~30-60s each |

Additional markers: `@pytest.mark.slow` (excluded by default), `@pytest.mark.network` (real API calls).

**Markers are auto-applied by directory** — no `@pytest.mark.unit` decorators needed. Root `conftest.py` tags tests via `pytest_collection_modifyitems`.

**Shared fixtures:**

* `tests/fixtures/dataframes.py` — 9 DataFrame factories (`make_games()`, `make_modeling_rows()`, `make_stadiums()`, etc.)
* `tests/fixtures/repos.py` — composable `MiniRepoBuilder` for integration/e2e tests
* `tests/fixtures/dk_payload_fixture.py` — DraftKings API response fixture

**Running tests:**

```bash
uv run pytest -m "unit and not slow"           # fast dev loop (~35s)
uv run pytest -m "integration or e2e"          # cross-module + pipeline tests
uv run pytest --cov --cov-report=term-missing  # with coverage report
```

**Deferred test areas** (to be added with their respective workstreams):

* Numba sim kernels (`test_engine.py`, `test_playoffs.py`) — defer to sim workstream
* DK API mocking (`test_draftkings.py` full) — defer to odds workstream

---

### Coverage baseline (as of W0 completion)

412 tests | 40.04% line coverage | threshold: `fail_under = 40`

| Tier | Coverage | Modules |
|------|----------|---------|
| **Core business logic** | 80-100% | features/*, datasets/*, core/*, evaluation/metrics, ratings/elo/core |
| **Integration-heavy** | 40-80% | pipeline, archive, backfill, diagnostics, odds/store, artifact |
| **Deferred** | 0-30% | sim/*, viz/*, CLI, model training, draftkings, elo predictor, ETL |

**Strategy:** Ratchet `fail_under` up as workstreams add tests for their modules.
Each workstream is expected to bring its modules to 80%+ coverage.

---

## Known sharp edges

### Missing stadium coordinates (2026-2027)

12 new/renamed stadia for the 2026-2027 season are not yet in `NFL_stadium_reference.csv`. Weather ingest skips affected games. Add rows with columns `STADIUM`, `HOME_TEAM`, `YEAR`, `LATITUDE`, `LONGITUDE`, `ALTITUDE`:

Bernabeu · Caesars Superdome · Estadio Banorte · EverBank Stadium · FC Bayern Munich Stadium · Highmark Stadium · Huntington Bank Field · Maracana Stadium · Melbourne Cricket Ground · Northwest Stadium · Stade de France · Tottenham Hotspur Stadium

### Off-season `current_nfl_season()`

Returns `year - 1` when `month < 6`. In May 2026 it returns 2025, treating you as if in the 2025-2026 season. Pass `--season 2026` or `--upcoming-season 2026` explicitly when fetching 2026-2027 data.

### Sim requires a populated upcoming schedule

`gridiron sim run` raises `FileNotFoundError` with an actionable message when the upcoming schedule CSV is empty. Run `gridiron ingest nflverse-upcoming --season 2026` then `gridiron transform clean-upcoming` first.

### Elo state table must precede predictions

`output predictions` merges Elo onto the upcoming schedule by week. If the Elo table doesn't have entries for the requested week yet, predictions will be empty. Run `ratings elo fit` first.

### nflverse data cadence

nflverse updates nightly after each game day. The cleanest weekly snapshot is Thursday (after the NFL incorporates Mon–Wed stat corrections). Historical coverage goes back to 1999.

---

## Operational checklist (weekly, during season)

1. `uv run gridiron run-data-pipeline` — refresh data + features
2. `uv run gridiron ingest dk-odds` — pull current week odds
3. `uv run gridiron output predictions --year YYYY-YYYY+1 --week N`
4. `uv run gridiron sim run`
5. `uv run gridiron output ranks --year YYYY-YYYY+1 --week N`

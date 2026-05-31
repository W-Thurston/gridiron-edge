# Gridiron Edge — Changelog

What has been built and when. Newest first.

---

## W3: Market Intelligence Foundation — 2026-05-31

### New package: `market/`
- Pure-math leaf package at `src/gridiron_edge/market/` — no data dependencies,
  no pandas, no I/O

### `market/odds_math.py`
- `american_to_decimal()`: American → decimal odds conversion
- `american_to_implied_prob()`: American → raw implied probability (includes vig)
- `decimal_to_american()`: decimal → American; even-money normalises to +100
- `hold_pct()`: bookmaker overround for two-way markets
- `no_vig()`: fair probabilities via power method (default) or additive rescaling
- `_power_devig()`: bisection solver for `raw_a^k + raw_b^k = 1` — no scipy

### `market/kelly.py`
- `kelly_fraction()`: full-Kelly optimal fraction; returns 0 when edge ≤ 0
- `kelly_stake()`: dollar amount using fractional Kelly (default quarter-Kelly)
- Input validation: probability must be in (0, 1), bankroll ≥ 0, fraction in [0, 1]

### Tests added (64)
- `test_odds_math.py` (42) — conversions, roundtrips, extreme odds (±10000),
  hold percentage, no-vig additive vs power, sums-to-one, fair-probs-not-above-raw
- `test_kelly.py` (22) — positive/negative/zero edge, fractional staking,
  zero bankroll, guard rails on probability/bankroll/fraction

### Deferred
- `market/consensus.py` — deferred until multi-book data available (W7)

## W1: Quick Wins & Unblocking — 2026-05-31

### DK Unicode Minus Fix
- `ingest/odds/draftkings.py` → `_norm_display_odds_american()`: handle Unicode
  minus (U+2212) before `isdigit()` check and `int()` conversion. DraftKings API
  returns `"−150"` with U+2212 instead of ASCII hyphen; this caused `ValueError`
  on all negative odds parsing.

### DK `game_id` Resolver
- New module: `ingest/odds/_game_id.py`
- `team_long_to_short()`: reverse lookup from `NFLVERSE_SHORT_TO_LONG`, with
  historical relocation codes (`OAK`, `SD`, `STL`) deprioritized so current
  codes (`LV`, `LAC`, `LA`) always win
- `build_game_id()`: constructs canonical `YYYY_WW_AWAY_HOME` format
- `resolve_dk_game_ids()`: vectorized column addition supporting both
  intermediate (`home_team`/`away_team`) and wide (`team`/`opponent`/`location`)
  DataFrame formats

### End-to-End Odds Join Validation
- Integration test confirms predictions ↔ odds join on `game_id` at 100% match
  rate on synthetic data, with left-join surfacing unmatched games as nulls

### Tests added (25)
- `test_draftkings_parse.py` (9) — Unicode minus, positive, int/float passthrough,
  fallback keys, non-numeric string, missing keys
- `test_game_id.py` (13) — team lookup, all 32 teams resolve, build_game_id format,
  week padding, unknown teams → None, both DataFrame formats, column preservation
- `test_odds_join.py` (3) — canonical format validation, inner join match rate,
  left join null surfacing


## W0 Complete: Test Framework Build-Out — 2026-05-31

### Summary
Professional three-tier testing infrastructure (unit → integration → e2e)
with automated quality gates, shared fixtures, and 412 tests at 40% coverage.

### Phases completed
- **Phase 0** — Foundation: directory restructure, auto-markers, shared fixtures,
  pre-commit/pre-push hooks, coverage config
- **Phase 1** — Core & Datasets: 60 tests covering constants, paths, settings,
  registry, loaders, writers, accessor
- **Phase 2** — Missing Features: 63 tests covering all 11 feature modules,
  feature registry, FeatureSpec protocol
- **Phase 3** — Models & Evaluation: 35 tests covering Predictor/Trainable
  protocols, model registry, artifact store, backfill, select, tune, diagnostics
- **Phase 4** — Ingest, Transform, Sim: 65 tests covering odds store, nflverse
  helpers, sim types/constants, geo/haversine, DK fixture validation
- **Phase 5** — Integration & E2E: 28 tests covering dataset roundtrips,
  artifact roundtrips, CLI workflows, full prediction pipeline via MiniRepoBuilder
- **Deferred resolution** — Added test_tune.py (16 tests), test_diagnostics.py
  (8 tests), removed slow training tests that exercised sklearn/xgboost internals

### Coverage baseline
- 412 tests, 0 failed, 0 deselected
- 40.04% line coverage (threshold: 40%, ratchet up over time)
- Core business logic (features, datasets, evaluation) at 80-100%
- Sim, viz, CLI, and model training code deferred to respective workstreams

### Deferred test areas (to be added with respective workstreams)
- Numba sim kernels: `test_engine.py`, `test_playoffs.py` (sim workstream)
- DK API mocking: full `test_draftkings.py` (odds workstream)
- Elo predictor: `test_elo_predictor.py` (elo workstream)
- Transform ETL: `test_epa_transform.py` (data pipeline workstream)
- Cosmetic: migrate inline imports → top-level; migrate local helpers → shared fixtures


### Test Framework Build-Out — 2026-05-31

Established professional three-tier testing infrastructure.

**Test directory restructure**
- Restructured `tests/` into `unit/`, `integration/`, `e2e/` subdirectories
- Tests auto-tagged by directory via `pytest_collection_modifyitems` in root conftest — no manual `@pytest.mark` decorators needed
- Existing tests moved to `tests/unit/` with zero import changes required

**Shared fixtures**
- `tests/fixtures/dataframes.py` — 9 centralized DataFrame factories: `make_games`, `make_modeling_rows`, `make_stadiums`, `make_elo_state`, `make_epa_by_game`, `make_weather_enriched`, `make_eval_df`, `make_predictions`, `make_accessor`
- `tests/fixtures/repos.py` — composable `MiniRepoBuilder` class (builder pattern: `.with_games().with_stadiums().with_elo_state().build()`)
- Replaces duplicated `_make_games()`, `_make_eval_df()`, `mini_repo` patterns across 8+ test files

**Pre-commit / pre-push hooks:**
- Added `.pre-commit-config.yaml` with two stages:
  - `pre-commit`: ruff lint + format, pyrefly type check, unit tests
  - `pre-push`: integration + e2e tests
- Installed via `pre-commit install` + `pre-commit install --hook-type pre-push`
- Safety valve: `|| test $? -eq 5` allows commits during incremental marker migration

**Pytest configuration:**
- Added markers to `pyproject.toml`: `unit`, `integration`, `e2e`, `slow`, `network`
- `--strict-markers` enforced — no typos in marker names
- Coverage config added: `fail_under = 60`, `show_missing = true`

**Fixed drifted tests**
- `test_home_field_feature`: `GAME_LOCATION` `"NULL_VALUE"` → `"H"` (aligned with constants consolidation)
- `test_weather`: `_make_modeling_row` returns DataFrame not dict; `test_null_value_string_gives_nan` assertion updated
- `test_tree_models`: imports updated for `_epa_window` module extraction (`_rebuild_features_with_window`, `_EPA_WINDOW_OPTIONS`)
- `test_features_pipeline`: `pd.read_csv` → `pd.read_parquet` for `modeling_base`/`modeling_full`
- Model training tests (`TestRandomForestV1Training`, `TestXGBoostV1Training`) marked `@pytest.mark.slow` (~15min each)

**Tooling**
- `mirror_repo_to_sharepoint.py` — mirrors repo to SharePoint-synced folder for Copilot indexing. Copies `.py` files as `.py.txt` with SOURCE headers; preserves `.md`/`.json`/`.yaml` as-is. Supports `--clean`, `--dry-run`, `--extra-ext`.


## Thermonuclear Code Quality Review — 2026-05-30

Eight review batches across the full codebase, followed by six implementation passes and full pipeline validation. All changes committed in four atomic commits.

### Pass 1+2 — Constants consolidation + Elo engine

**Constants — single source of truth in `core/constants.py`:**
- `HOME_GAME_LOCATION = "H"`, `AWAY_WIN_LOCATION = "@"`, `HOLDOUT_SEASONS`, `EXPANSION_TEAMS` — all previously defined independently in 2–4 files each
- Retired the PFR-era `"NULL_VALUE"` home-game sentinel → `"H"` for `GAME_LOCATION`; `""` for all missing data fields (GAMETIME, STADIUM, ROOF, SURFACE, GAME_DATE, GAME_DAY_OF_WEEK) across the transform layer
- All consumers updated: `venue_hfa`, `home_field`, `record`, `primetime`, `backfill`, `tune`, `elo/predictor`, `metrics`, `schedule_nflverse`, `games_nflverse`, `_nflverse_common`
- Deleted dead placeholder packages: `datasets/contracts/`, `analytics/`, `config/`

**Elo engine — parameterised divisor:**
- `ratings/elo/core.py`: `elo_win_probability(divisor=DEFAULT_ELO_DIVISOR)` and `update_elo(divisor=)` — divisor no longer hardcoded to 480
- `EloTableConfig` gains `divisor: float = 480.0`; `_build_elo_dict` passes it through
- `tune.py`: `_win_prob` deleted — `_simulate_and_score` delegates to `core.elo_win_probability`
- `SimulationConfig` gains `divisor: float = 480.0`; numba `_elo_win_prob`/`_elo_update` in `sim/_engine.py` accept divisor as a parameter
- `gridiron sim run` gains `--divisor` flag

### Batch 1-8 code review fixes

Individual file-level fixes from all 8 review batches:
- `DatasetSpec`: dropped redundant `key` field (14 instantiations updated)
- `FeatureRegistry`: duplicate-name guard + descriptive `KeyError` in `register()`/`get()`
- `features/team/epa.py`: vectorised inner EPA rolling loop; extracted `_join_team_epa` helper; `EPA_COLS` made public
- `ratings/elo/table.py`: deleted backwards-compat alias `update_elo_state_table_incremental`
- `evaluation/diagnostics.py`: filled `_MODEL_COLORS` gaps for logistic_v4, random_forest_v1/v2, xgboost_v2
- `evaluation/metrics.py`: removed duplicate `_archive_path` and `load_prediction_log` — now imports from `archive.py`
- `viz/excel.py` → `viz/rankings.py`: renamed; `cli/output.py` updated
- `metrics/travel/geo.py`: `Tude` type alias renamed to `CoordinateValue`
- `backfill.py`, `tune.py`, `metrics.py`: local `_AWAY_WIN_LOCATION` definitions removed, imported from `core.constants`

### Pass 3 — File decomposition

**`sim/season.py`** (1235 lines) split into three files:
- `sim/_types.py` — constants, all config dataclasses (`SimulationConfig`, `SimPaths`, `TeamIndex`, `ScheduleArrays`, `SimulationResults`), `_log_phase`, `format_record`. Pure-data leaf — no I/O, no numba.
- `sim/_engine.py` — numba kernels: `_elo_win_prob`, `_elo_update`, `apply_actuals_to_matrices`, `simulate_remaining_regular_season`, `precompute_game_counts`
- `sim/season.py` — data loading, output builders, `run_full_simulation` (~734 lines)
- `sim/__init__.py` — public API re-exports; sync assertions validate `playoffs.py` constants match `_types.py` at import time
- `viz/charts.py` — import updated from `sim.season` → `sim._types`

**`models/game_prediction/_shared.py`** (333 lines) split:
- `_columns.py` — schema version, all column lists, `FeatureSet` dataclass; pure-data leaf
- `_features.py` — feature engineering functions, `FEATURE_SETS` dict, `_prepare_data`, `_is_trained`
- `_shared.py` — thin re-export shim (33 lines)
- `logistic.py` and `tree.py` updated to import from new modules directly

**`models/game_prediction/tree.py`** (984 lines):
- `_epa_window.py` extracted — `_EPA_RAW_COLS`, `_EPA_COL_MAP`, `_EPA_WINDOW_OPTIONS`, `WindowData` NamedTuple, `_rebuild_features_with_window`, `_get_cached_window_data`
- `tree.py` reduced to 820 lines

**Final line counts:** no file exceeds 820 lines. `playoffs.py` ↔ `_types.py` constant sync is machine-checked at import time.

### Pass 4 — Feature dependency enforcement

- `features/base.py`: `FeatureSpec` gains `depends_on: Sequence[str] = ()` field
- `features/registry.py`: `validate_ordering(feature_names)` — raises `ValueError` at import time if ordering violates any `depends_on` constraint
- `features/pipeline.py`: calls `validate_ordering(FEATURES)` at module level
- Dependencies declared: `travel` → `home_field`; `venue_hfa` → `travel`; `schedule_strength` → `team_elo`

### Pass 5 — CLI stage-list pattern

- `cli/main.py`: 10 boolean flags replaced with `--skip STAGE` / `--only STAGE` repeatable options
- `ALL_STAGES` defines the canonical stage vocabulary: `fetch-games`, `clean-games`, `fetch-upcoming`, `clean-upcoming`, `fetch-weather`, `fetch-odds`, `build-epa`, `build-elo`, `build-features`
- Dead `build-epa` stage fixed — was declared but never executed
- `PLR0912`/`PLR0915` suppressions moved to `_run_pipeline_stages` where they belong; `run_data_pipeline` is now clean
- `evaluation/select.py` introduced — `collect_model_metrics`, `rank_models`, `compute_report_data` extracted from `cli/evaluate.py`

### Pass 6 — Archive schema migration

- `evaluation/archive.py`: `is_backfilled: bool` column added to schema; `build_archive_rows` and `append_to_prediction_log` gain `is_backfilled` parameter; `write_archive_rows` and `load_prediction_log` backward-compatible; `migrate_archive()` added
- `models/elo/predictor.py`: `_BACKFILL_TS` constant deleted; predictions use actual timestamp + `is_backfilled=True`
- `logistic.py`, `tree.py`: inline `datetime(1970, 1, 1)` sentinels replaced with actual timestamp + `is_backfilled=True`

### Post-commit fixes

- `ingest/weather/openweather.py` — `fetch_weather` now reads existing `weather_enriched.csv` and fetches only games not already enriched. Idempotent — safe to re-run without duplicating rows.
- `sim/season.py` — `run_full_simulation` raises `FileNotFoundError` with actionable message when the upcoming schedule CSV is empty, instead of a cryptic `IndexError`.

---

## Phase 20d — Tree-based models (RF + XGBoost)

- `models/game_prediction/tree.py` — Random Forest and XGBoost variants registered alongside logistic models
- `models/game_prediction/logistic.py` — v3 and v4 logistic variants added
- `PredictorRegistry` — `register` + `get` + `trainable_names()` pattern generalised
- `evaluation/tune.py` — hyperparameter grid search for Elo K/divisor and EPA window
- `evaluation/diagnostics.py` — calibration plots, model comparison charts

---

## Phase 20c — Model reporting

- `evaluation/select.py` — `select_model` + `generate_report` pipeline
- `cli/evaluate.py` — `evaluate report`, `evaluate select-model`, `evaluate calibration` commands
- Full model characterisation: Brier score, log loss, calibration, accuracy per season

---

## Phase 20b — Model evaluation infrastructure

- `evaluation/metrics.py` — Brier score, log loss, calibration table, accuracy
- `evaluation/backfill.py` — `backfill_model(model_version)` covering all registered models
- `evaluation/archive.py` — append-only prediction log at `predictions_log.parquet`
- `cli/evaluate.py` — `evaluate backfill`, `evaluate summary` commands

---

## Phase 20a — Prediction engine

- `models/game_prediction/logistic.py` — logistic v1 + v2 registered predictors
- `models/base.py` — `Predictor` + `Trainable` protocols
- `models/registry.py` — `PredictorRegistry`
- `models/artifact.py` — `ArtifactStore` (joblib-based)
- `cli/models.py` — `models train`, `models list` commands

---

## Phase 19 — Football state representation (EPA, rest, travel, records)

- `features/team/epa.py` — rolling EPA features from PBP data
- `features/team/rest.py` — days rest, short week, post-bye flags
- `features/team/travel.py` — km traveled, timezone shift
- `features/team/record.py` — win/loss/tie record, win streak
- `features/team/schedule_strength.py` — SOS, SOV
- `ingest/nflverse/pbp.py` — play-by-play ingestion
- `transform/clean/epa.py` — PBP → game-level EPA aggregation
- Schema v3 modeling file with all Phase 19 features

---

## Phase 18 — Evaluation infrastructure

- Prediction archive — append-only Parquet log
- `evaluation/metrics.py` — Brier score, log loss, calibration, accuracy
- `evaluation/backfill.py` — generic backfill covering all registered models
- `evaluation/tune.py` — Elo parameter grid search
- `datasets/manifest.py` — schema versioning for modeling files

---

## Phase 15-17 — Excel retirement, Scrapy retirement, dead code removal

- `ingest/odds/` — DraftKings odds ingest + append-only Parquet ledger
- `ingest/odds/store.py` — long-format odds storage with dedup
- `viz/predictions.py` — weekly matchup PNG + static HTML (migrated from notebook)
- `viz/rankings.py` — Elo rankings CSV (was Excel)
- Scrapy / PFR scraper fully deleted
- Dead stub files removed; all ruff/pyrefly gates passing

---

## Phase 13-14 — nflverse migration + console system

- Replaced PFR/Scrapy with `nfl_data_py` — bypasses Cloudflare
- `ingest/nflverse/` — game + schedule + upcoming ingestion
- `transform/clean/games_nflverse.py` + `schedule_nflverse.py` — canonical schema mappers
- `core/console.py` — timed step context manager, header/summary banners, verbose mode
- `core/logging.py` — WARNING in compact mode, DEBUG in verbose

---

## Phases 1-12 — Core refactor + tooling

Original migration from `data_pipelines/` + `model_pipelines/` + `utils/` into `src/gridiron_edge/`. uv migration, Ruff + Pyrefly quality gates, Google-style docstrings, full type annotation pass. See git history for full detail.

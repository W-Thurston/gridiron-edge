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
| Evaluation | `gridiron_edge.evaluation` — `archive`, `metrics`, `select`, `backfill`, `tune`, `champion` |
| Models | `gridiron_edge.models` — Predictor + Trainable protocols, ArtifactStore |
| CLI | `gridiron_edge.cli` — `main.py` stage-list pipeline, sub-apps per domain |
| Market math | `gridiron_edge.market` — `odds_math`, `kelly`, `edge`, `recommendations`, `clv` |
| Post-processing enrichment | `gridiron_edge.models.game_prediction.post_process` — spread, bands, tier, scores |
| Prediction pipeline | `gridiron_edge.models.game_prediction.pipeline` — composable predict → enrich orchestrator |
| Total points model | `gridiron_edge.models.game_prediction.total` — RF regressor, shares feature set with win models |
| Edge calculation | `gridiron_edge.market.edge` — pure scalar EV, cover-prob, edge detection (ML/spread/total). Frozen dataclasses. |
| Edge recommendations | `gridiron_edge.market.recommendations` — joins predictions ↔ odds, builds 18-column edge report, ranks by EV |
| Closing Line Value | `gridiron_edge.market.clv` — probability-based and point-based CLV, opening/closing odds extraction, CLV report |
| Edge CLI | `gridiron_edge.cli.edges` — gridiron edges report (weekly), gridiron edges clv (historical CLV) |
| Bet ledger | `gridiron_edge.betting.ledger` — append-only Parquet bet log with PnL and CLV on settlement |
| Bankroll management | `gridiron_edge.betting.bankroll` — transaction log (deposit/withdraw/bet/settle), current_balance, balance_history |
| Performance analytics | `gridiron_edge.betting.performance` — pure DataFrame analytics: record, ROI, CLV, EV, streaks, summary |
| Betting CLI | `gridiron_edge.cli.betting` — 8 commands: log, settle, list, summary, balance, export, deposit, with |

---

## Repository layout

| Path | Role |
|------|------|
| `src/gridiron_edge/core/` | Settings, logging, console, shared constants |
| `src/gridiron_edge/ingest/` | All data ingestion (nflverse, weather, odds) |
| `src/gridiron_edge/transform/` | Raw → canonical schema mappers |
| `src/gridiron_edge/features/` | Feature registry, pipeline, dependency validation |
| `src/gridiron_edge/market/` | Odds math, Kelly staking, edge detection, edge reports, CLV analysis |
| `src/gridiron_edge/betting/` | Bet ledger, bankroll transaction log, performance analytics |
| `src/gridiron_edge/ratings/elo/` | Elo table, fit, predict, evaluate |
| `src/gridiron_edge/models/` | Predictor protocol, artifact store, model registry, game prediction variants |
| `src/gridiron_edge/evaluation/` | Metrics, backfill, archive, tuning, model selection |
| `src/gridiron_edge/sim/` | Monte Carlo season + playoff simulation |
| `src/gridiron_edge/viz/` | Predictions image/HTML, playoff table, rankings CSV |
| `src/gridiron_edge/cli/` | Typer app + sub-commands |
| `data/` | Generated at runtime — not committed |
| `data/models/{version}/` | Trained model artifacts (joblib + metadata JSON) |

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

### EPA aggregation is the single PBP funnel

All PBP-derived game-level features flow through **one transform**:

- `transform/clean/epa.py` → `_agg_side()`
- Output: `epa_by_game.parquet`

The feature layer (`features/team/epa.py`) reads this dataset and applies rolling windows.

**Adding a new PBP feature requires only:**
1. Add computation to `_agg_side()`
2. Add column name to `EPA_COLS`

Everything else auto-propagates through:
`_columns.py → feature sets → model inputs`

This pattern was fully validated during Phase 20e (22 EPA metrics, 107 total features).

### `GAME_LOCATION` schema

Three values only — `"H"` (home win), `"@"` (away win), `"N"` (neutral site). The old PFR-era `"NULL_VALUE"` sentinel was retired. Missing data fields (GAMETIME, STADIUM, ROOF, SURFACE) use `""`.

### Elo divisor is parameterised end-to-end

`core.py` → `EloTableConfig` → `SimulationConfig` → `--divisor` CLI flag. The tuner (`evaluate tune elo`) finds the optimal divisor; set it consistently across table building and simulation. Default is 480 (classic NFL Elo). elo_v2 optimum is 350.

### Feature dependency validation

`FeatureSpec.depends_on` declares ordering constraints. `validate_ordering()` is called at pipeline import time — a mis-ordering raises `ValueError` immediately rather than silently producing wrong columns at training time.

### Prediction archive `is_backfilled`

`predictions_log.parquet` has a boolean `is_backfilled` column. Historical backfill predictions set it to `True`; live pre-game predictions set it to `False`. Filter on this rather than `predicted_at` for live-vs-backfill analysis.

### PBP ingest column expansion

`sack` was added to `_KEEP_COLUMNS` during Phase 20e (Batch 3).

When `_KEEP_COLUMNS` changes:
- Existing PBP parquet files must be deleted
- Re-ingest required via: `gridiron ingest pbp --all-years`

This ensures new columns are physically present in stored parquet files.

### Weather ingest is idempotent

`fetch_weather` reads existing `weather_enriched.csv`, computes the set difference of `GAME_ID`s, and only calls the OWM API for games not already enriched. Safe to re-run.

### Market package is a pure-math leaf

The market package is layered: `odds_math.py` and `kelly.py` are pure-math
leaves (no pandas, no I/O). `edge.py` adds scipy.stats.norm for probit
cover probabilities but remains scalar-only. `recommendations.py` and
`clv.py` use pandas for data joins but take DataFrames as arguments — no
file I/O. `cli/edges.py` is the thin CLI wiring that loads data and calls
the library. CLV reuses `pivot_odds_to_wide` from recommendations via
`_pivot_and_suffix()` to stay DRY.

### sim/season.py decomposition

`sim/` is split into three files with a clean dependency hierarchy:
- `_types.py` — pure data, no I/O (constants, dataclasses)
- `_engine.py` — numba kernels (imports from `_types` only)
- `season.py` — orchestration (imports from both)

Numba cannot call regular Python functions at JIT time, so the Elo formula is duplicated in `_engine.py`. A comment cross-references `ratings/elo/core.py` — if the formula changes, update both.

### Post-processing enrichment is a separate step, not inside models

All derived outputs (spread, bands, tier, projected scores) are computed in
`post_process.py` after the model produces `home_win_prob`.  This keeps models
clean and composable — any model that outputs a win probability gets the full
enrichment for free via `enrich_predictions()`.

### Prediction pipeline is composable

`pipeline.py` orchestrates: load features → win model → total model →
build game rows → enrich.  Adding a new model type means adding one inference
call, not rewriting the pipeline.  `_predict_historical_tree()` and
`_predict_historical_logistic()` delegate to `predict_games()`.

### Total model is a supporting model, not a standalone predictor

The total model (`total.py`) trains a `RandomForestRegressor` on the same
107-feature set but targets `actual_total` instead of `RESULT`.  It is NOT
registered in `PredictorRegistry` — it feeds into `enrich_predictions()` via
the pipeline rather than operating as an independent predictor.

### VEGAS_LINE sign convention

nflverse `VEGAS_LINE` uses **positive = home favored** (PFR convention).
Our `model_spread` uses **negative = home favored** (probit convention).
They are exact negations.  Always negate `VEGAS_LINE` before comparing
to `model_spread`.

### Archive schema is soft-versioned

The prediction archive (`predictions_log.parquet`) uses NaN fill for columns
missing from older archives.  No schema version column — `load_prediction_log()`
adds missing columns on load.  String columns (e.g. `confidence_tier`) get
empty string instead of NaN.

### Bankroll is decoupled from the bet ledger

`ledger.py` and `bankroll.py` are independent modules with no imports
between them. The CLI (`cli/betting.py`) orchestrates both: `bet log`
calls `log_bet()` then `record_bet_placed()`, and `bet settle` calls
`settle_bet()` then `record_bet_settled()`. This keeps each module
testable in isolation and avoids circular dependencies.

### Gross return model for bankroll

On settlement, the bankroll receives the *gross return* (stake + pnl):
won = stake × decimal_odds, lost = 0, push = stake. This means
`bet_placed` always deducts the full stake, and `bet_settled` credits
back whatever is returned. The running balance is the cumulative sum
of all signed transactions.

#### Champion/challenger model promotion

Models are registered as unversioned champions (random_forest, xgboost,
logistic) rather than versioned variants (rf_v1, rf_v2, etc.).
`gridiron models train <name>` auto-compares the newly trained model
against the existing champion using three gates:

1. **Brier** must improve by ≥ 0.002 (primary metric)
2. **ECE** must not degrade by > 0.01 (calibration guardrail)
3. **AUC** must not degrade by > 0.01 (discrimination guardrail)

If all gates pass, the challenger is promoted. If any gate fails, the
old champion is restored from backup. Use `--force` to override,
`--no-promote` for dry-run comparison. Logic lives in
`evaluation/champion.py`.

#### Temporal CV for model training

All model families use TimeSeriesSplit(n_splits=5) for cross-validation
during hyperparameter search. Early folds with fewer than
MIN_CV_TRAIN_ROWS (4000) rows are skipped to avoid undersized training
sets biasing HP selection. The training data is sorted chronologically
in `_prepare_data` before splitting. This replaced StratifiedKFold
(shuffle=True) which had temporal leakage.

---

### Workflows (End-to-End Data Flows)

These trace how data moves through the system for each major operation.
Use them to understand what happens when a command runs, where to add
new functionality, and where to look when something breaks.


#### Data Pipeline (`gridiron run-data-pipeline`)

Runs 9 stages in order. Each stage can be skipped (`--skip`) or
isolated (`--only`).

```
fetch-games          nflverse API -> data/raw/ (Parquet)
    |
clean-games          raw -> data/cleaned/NFL_wk_by_wk_cleaned.csv
    |                (canonical schema: WINNER/LOSER/GAME_LOCATION/VEGAS_LINE)
fetch-upcoming       nflverse API -> raw upcoming schedule
    |
clean-upcoming       raw -> data/cleaned/NFL_upcoming_schedule_cleaned.csv
    |
fetch-weather        OpenWeatherMap API -> data/cleaned/weather_enriched.csv
    |                (idempotent -- only fetches missing GAME_IDs)
fetch-odds           DraftKings API -> data/odds/dk_odds_log.parquet
    |
build-epa            PBP data -> transform/clean/epa.py._agg_side()
    |                -> data/cleaned/epa_by_game.parquet
    |                (ALL PBP-derived features funnel through _agg_side)
build-elo            games + config -> ratings/elo/table.py
    |                -> data/cleaned/NFL_Team_Elo.csv
    |
build-features       games + all cleaned data
                     -> features/pipeline.py.build_model_inputs()
                     -> data/modeling/modeling_file.parquet
```

**Feature pipeline detail** (`build-features` stage):

```
base_modeling_file.parquet (games + teams + metadata)
    |
    v
run_features() applies 11 registered features IN ORDER:
    home_field -> team_elo -> travel -> epa -> rest -> weather
    -> divisional -> venue_hfa -> record -> schedule_strength -> primetime
    |
    |  Order enforced by FeatureSpec.depends_on + validate_ordering()
    |  at import time. Mis-ordering raises ValueError immediately.
    |
    v
modeling_file.parquet (94 columns, ~14K rows)
    Two rows per game (one per team, TEAM_A / TEAM_B perspective)
    HOME_FIELD column indicates which perspective (0 = away, 1 = home)
    RESULT column: 1 = TEAM_A won, 0 = TEAM_A lost, 0.5 = tie
    YEAR, WEEK_NUM, GAME_ID for identification
```

**Adding a new PBP-derived feature:**
1. Add computation to `transform/clean/epa.py._agg_side()`
2. Add column name to `EPA_COLS` in `features/team/epa.py`
3. Everything else auto-propagates: `_columns.py` -> feature sets -> model inputs


#### Model Training (`gridiron models train <version>`)

```
PredictorRegistry.get(version)     look up registered model class
    |
    v
model.train(games, repo=repo)      dispatches to model-specific trainer
    |
    |  Tree models: _train_random_forest() / _train_xgboost()
    |  Logistic:    _train_logistic()
    |  Elo:         no training artifact -- simulation-based
    |
    |-- load_modeling_file()        loads data/modeling/modeling_file.parquet
    |
    |-- _prepare_data(df, feature_fn)
    |       Filters ties (RESULT == 0.5)
    |       Applies feature_fn -> feature matrix (e.g., 107 columns)
    |       Drops rows with any NaN features
    |       Splits on HOLDOUT_SEASONS (YEAR column)
    |       Returns: (x_train, y_train, x_hold, y_hold,
    |                 train_seasons, holdout_seasons)
    |
    |-- Randomized HP search with tqdm progress bar
    |       Tree: TimeSeriesSplit CV on training set (folds < 4000 rows skipped)
    |       Evaluates Brier score (win models) or MAE (total model)
    |
    |-- Retrain best params on full training set
    |
    +-- ArtifactStore.save(version, model, metadata)
            -> data/models/{version}/model.joblib
            -> data/models/{version}/metadata.json
            Champion/challenger: train auto-compares to existing champion.
            Promotes if Brier improves ≥ 0.002, ECE doesn't degrade > 0.01,
            AUC doesn't degrade > 0.01.  --force overrides gates.
```

**Total model training** follows the same pattern but:
- Target is `actual_total` (PTS_WINNER + PTS_LOSER) instead of `RESULT`
- Uses `RandomForestRegressor` instead of classifier
- Uses `TimeSeriesSplit` instead of `StratifiedKFold` for CV
- Evaluates MAE instead of Brier score
- Called directly: `from total import train_total_model; train_total_model()`


#### Historical Prediction (`predict_games` in `pipeline.py`)

This is the core prediction flow used by backfill and evaluation.

```
predict_games(model_version, feature_fn, repo)
    |
    |-- Step 1: Load features
    |       load_modeling_file(repo, required_schema_version=4)
    |       feature_fn(df) -> feature matrix
    |       Filter NaN rows -> df_valid, x_feat
    |
    |-- Step 2: Win probability inference
    |       ArtifactStore.load(model_version) -> sklearn pipeline
    |       pipeline.predict_proba(x_feat)[:, 1] -> probabilities
    |
    |-- Step 3: Total points inference (optional)
    |       predict_total(df_valid) -> predicted combined scores
    |       Falls back gracefully if total model not trained
    |
    |-- Step 4: Build game-level rows
    |       build_game_predictions(df_valid, probs, totals=totals)
    |       Filters to away-team rows (HOME_FIELD == 0)
    |       Deduplicates on GAME_ID -> one row per game
    |       Maps columns: TEAM_A -> away_team, TEAM_B -> home_team
    |       Attaches: predicted_at, is_backfilled, model_version,
    |                 season, week, away/home_win_prob, model_total
    |
    +-- Step 5: Enrich
            enrich_predictions(result, model_version, recalibrate)
            |
            |-- Isotonic recalibration (if calibrator exists)
            |       Adjusts home_win_prob and away_win_prob in place
            |
            |-- Spread derivation
            |       model_spread = -sigma * Phi_inv(home_win_prob)
            |       sigma looked up per model version from _MODEL_SIGMAS
            |
            |-- Uncertainty bands
            |       spread +/- z * margin_std -> probit -> (win_prob_lo, win_prob_hi)
            |       margin_std looked up per model from _MODEL_MARGIN_STDS
            |       z = 1.645 (90% credible interval)
            |
            |-- Confidence tier
            |       band_width = win_prob_hi - win_prob_lo
            |       < 0.65 -> "High" | < 0.82 -> "Moderate" | else -> "Low"
            |
            +-- Projected scores (if model_total present)
                    home = (total - spread) / 2
                    away = (total + spread) / 2

        Returns: DataFrame with 21 columns (13 base + 8 enrichment)
```

**Who calls `predict_games`:**
- `_predict_historical_tree()` delegates to `predict_games()`
- `_predict_historical_logistic()` delegates to `predict_games()`
- Elo models use their own simulation path but call `enrich_predictions()` on the output
- `_predict_upcoming_tree/logistic()` use a different data path (schedule + feature registry) but also call `enrich_predictions()` before returning


#### Backfill & Evaluation

```
gridiron evaluate backfill --model-version random_forest_v3
    |
    v
backfill_model(model_version)
    |
    |-- PredictorRegistry.get(model_version) -> predictor instance
    |
    |-- predictor.predict_historical(games)
    |       -> enriched predictions DataFrame (21 columns)
    |       (calls predict_games internally for tree/logistic)
    |
    +-- write_archive_rows(predictions)
            -> data/output/predictions/predictions_log.parquet
            Deduplicates on (game_id, model_version)
            Most recent prediction wins

gridiron evaluate select-model
    |
    |-- collect_model_metrics() -> loads archive, joins to actuals
    |
    +-- rank_models() -> ranked table by Brier score
```


#### Archive Schema

The prediction archive (`predictions_log.parquet`) has 21 columns:

| Group | Columns |
|-------|---------|
| **Identity** | `predicted_at`, `is_backfilled`, `model_version`, `season`, `week`, `game_id`, `game_date`, `away_team`, `home_team` |
| **Ratings** | `away_elo`, `home_elo` |
| **Predictions** | `away_win_prob`, `home_win_prob` |
| **Enrichment** | `model_spread`, `model_total`, `projected_home_score`, `projected_away_score`, `margin_std`, `win_prob_lo`, `win_prob_hi`, `confidence_tier` |

Backward compatible: old archives missing enrichment columns get NaN
(or empty string for `confidence_tier`) on load.

**Sign conventions:**
- `model_spread`: **negative = home favored** (probit convention)
- `VEGAS_LINE` (nflverse): **positive = home favored** (PFR convention)
- Always negate `VEGAS_LINE` before comparing to `model_spread`

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
uv run gridiron models train random_forest
uv run gridiron models train xgboost --force
uv run gridiron models train logistic --no-promote
uv run gridiron evaluate backfill --model-version random_forest
uv run gridiron evaluate backfill --model-version xgboost
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
| Market math | `market/odds_math.py`, `market/kelly.py`, `market/edge.py`, `market/recommendations.py`, `market/clv.py` |
| Bet tracking | `betting/ledger.py`, `betting/bankroll.py`, `betting/performance.py`, `cli/betting.py` |
| EPA window hyperparameter infra | `models/game_prediction/_epa_window.py` |
| Elo core formula (parameterised) | `ratings/elo/core.py` |
| Simulation types + config | `sim/_types.py` |
| Simulation engine (numba) | `sim/_engine.py` |
| Simulation orchestration | `sim/season.py` |
| Prediction archive schema | `evaluation/archive.py` |
| Model selection + reporting | `evaluation/select.py` |
| Weather ingest (idempotent) | `ingest/weather/openweather.py` |
| DK odds ingest | `ingest/odds/draftkings.py` |
| DK game_id resolution | `ingest/odds/_game_id.py` |
| Champion/challenger promotion | `evaluation/champion.py` |
| Post-processing (spread, bands, tier) | `models/game_prediction/post_process.py` |
| Prediction pipeline orchestrator | `models/game_prediction/pipeline.py` |
| Total points model | `models/game_prediction/total.py` |
| Prediction archive schema | `evaluation/archive.py` — `_ARCHIVE_COLUMNS` |

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
3. `- uv run gridiron edges report --week N --season YYYY-YYYY+1 — generate edge report`
4. `uv run gridiron sim run`
5. `uv run gridiron output ranks --year YYYY-YYYY+1 --week N`
6. `uv run gridiron evaluate backfill --model-version random_forest`
7. `gridiron bet log --game-id {ID} --market {TYPE} --side {SIDE} --odds {ODDS} --stake {AMT} --book {BOOK}`
8. `gridiron bet settle {BET_ID} {won|lost|push}`
9. `gridiron bet summary`
10. `gridiron bet balance`

# Project Handoff: `NFL_Predict` / Gridiron Edge

How to run the project, how it is laid out, and what commonly breaks.

---

## What this project does

- **Goal**: NFL game prediction support — win probabilities, Elo rankings, weekly matchup visualisations, season simulation, odds tracking, and model evaluation.
- **Data source**: [nflverse](https://github.com/nflverse/nflverse-data) via `nfl_data_py` (replaces PFR/Scrapy).
- **Entry point**: `uv run gridiron` → [`src/gridiron_edge/cli/main.py`](src/gridiron_edge/cli/main.py) (via shim at `cli.py`)

---

## Repository layout

| Path | Role |
|------|------|
| [`src/gridiron_edge/`](src/gridiron_edge/) | Main package |
| [`src/gridiron_edge/cli/`](src/gridiron_edge/cli/) | CLI sub-modules (ingest, transform, ratings, features, output, sim, evaluate) |
| [`src/gridiron_edge/ingest/nflverse/`](src/gridiron_edge/ingest/nflverse/) | nflverse game + schedule ingestion |
| [`src/gridiron_edge/ingest/pfr/collector_impl.py`](src/gridiron_edge/ingest/pfr/collector_impl.py) | Weather (OWM) + DraftKings odds |
| [`src/gridiron_edge/ingest/odds/`](src/gridiron_edge/ingest/odds/) | Odds ledger (Parquet, long format) |
| [`src/gridiron_edge/transform/clean/`](src/gridiron_edge/transform/clean/) | nflverse → canonical schema mappers |
| [`src/gridiron_edge/features/`](src/gridiron_edge/features/) | Feature registry + pipeline |
| [`src/gridiron_edge/ratings/elo/`](src/gridiron_edge/ratings/elo/) | Elo table, fit, predict, evaluate |
| [`src/gridiron_edge/models/`](src/gridiron_edge/models/) | Predictor protocol, registry, Elo predictors |
| [`src/gridiron_edge/evaluation/`](src/gridiron_edge/evaluation/) | Prediction archive, metrics, backfill, tune |
| [`src/gridiron_edge/sim/`](src/gridiron_edge/sim/) | Monte Carlo season + playoff simulation |
| [`src/gridiron_edge/viz/`](src/gridiron_edge/viz/) | Predictions image/HTML, playoff table, rankings CSV |
| [`src/gridiron_edge/core/`](src/gridiron_edge/core/) | Settings, logging, console output |
| [`data/`](data/) | Generated at runtime (not committed) |

**Data layout:**

```
data/
  raw/          nflverse Parquet files (games, upcoming schedule)
  cleaned/      Canonical CSVs (games, schedule, Elo state, stadiums)
  modeling/     Feature matrix (base + full)
  output/
    predictions/
      {year}/              week_NN_predictions.png + .html
      predictions_log.parquet   All archived model predictions
    rankings/              elo_rankings_{year}_wkNN.csv
    sim/                   playoff probability tables
    tune/                  elo_v2_tune_results.parquet, elo_v3_tune_results.parquet
  odds/
    dk_odds_log.parquet    Full historical ledger (all pulls)
    dk_odds_current.parquet   Latest pull snapshot (for viz)
```

---

## Setup

```bash
uv sync
uv run gridiron --help
```

**Python**: `>=3.12,<4`.

---

## Configuration

| Secret / setting | How |
|------------------|-----|
| OpenWeather | Env var `OWM_API_KEY` for `gridiron ingest weather` |
| Data paths | [`datasets/registry.py`](src/gridiron_edge/datasets/registry.py) + [`core/settings.py`](src/gridiron_edge/core/settings.py) |

---

## Primary workflows

### Full bootstrap (first run or season reset)

```bash
uv run gridiron run-data-pipeline \
  --all-years \
  --upcoming-season 2026 \
  --build-elo \
  --fit-elo-all-years
```

Fetches all history (1999-present), 2026 upcoming schedule, rebuilds Elo, builds feature matrix. ~8s.

### Weekly refresh (during season)

```bash
# No flags needed - auto-detects current season
uv run gridiron run-data-pipeline
```

Re-fetches current season games + upcoming schedule, refreshes Elo incrementally, rebuilds features.

### Specific season

```bash
uv run gridiron run-data-pipeline --season 2025
```

### Step-by-step

```bash
uv run gridiron ingest nflverse-games              # current season
uv run gridiron ingest nflverse-games --all-years  # full history
uv run gridiron ingest nflverse-upcoming           # current season upcoming
uv run gridiron transform clean-games
uv run gridiron transform clean-upcoming
uv run gridiron ratings elo fit [--all-years]
uv run gridiron features model-inputs
```

### Output commands

```bash
# Weekly matchup predictions (PNG + HTML, auto-archived)
uv run gridiron output predictions --year 2026-2027 --week 1

# Elo rankings CSV
uv run gridiron output ranks --year 2026-2027 --week 1

# Season simulation
uv run gridiron sim run [--n-sims 10000]
```

### Evaluation commands

```bash
# Populate prediction archive for all historical games
uv run gridiron evaluate backfill                        # elo_v1 (default)
uv run gridiron evaluate backfill --model-version elo_v2
uv run gridiron evaluate backfill --overwrite            # re-run from scratch

# Accuracy summary
uv run gridiron evaluate summary                         # by season, elo_v1
uv run gridiron evaluate summary --group-by model_version  # compare all models
uv run gridiron evaluate summary --group-by week

# Calibration table
uv run gridiron evaluate calibration
uv run gridiron evaluate calibration --model-version elo_v2

# Elo parameter tuning (saves results to data/output/tune/)
uv run gridiron evaluate tune           # elo_v2: flat-K, 100 combinations (~45s)
uv run gridiron evaluate tune --v3      # elo_v3: zone-K, ~63k combinations (~8h)
uv run gridiron evaluate tune --apply   # run search then archive best params as new version
```

### DK odds

```bash
# Pull current week odds -> appends to ledger + writes snapshot
uv run gridiron ingest dk-odds --season 2026-2027 --week 1
```

### Weather

```bash
export OWM_API_KEY="YOUR_KEY"
uv run gridiron ingest weather --season-year "2026-2027"
```

---

## File contract (key artifacts)

| File | Purpose |
|------|---------|
| `data/raw/NFL_wk_by_wk_nflverse.parquet` | Raw nflverse games (all seasons) |
| `data/cleaned/NFL_wk_by_wk_cleaned.csv` | Canonical historical games |
| `data/raw/NFL_upcoming_schedule_nflverse.parquet` | Raw upcoming schedule |
| `data/cleaned/NFL_upcoming_schedule_cleaned.csv` | Canonical upcoming schedule |
| `data/cleaned/NFL_Team_Elo.csv` | Elo ratings state table |
| `data/modeling/base_modeling_file.csv` | Base modeling rows |
| `data/modeling/modeling_file.csv` | Full feature matrix |
| `data/cleaned/NFL_stadium_reference.csv` | Stadium / geo reference |
| `data/odds/dk_odds_log.parquet` | Full DK odds history (long format) |
| `data/odds/dk_odds_current.parquet` | Latest DK odds snapshot for viz |
| `data/output/predictions/predictions_log.parquet` | All archived model predictions |
| `data/output/tune/elo_v3_tune_results.parquet` | elo_v3 grid search results (when complete) |

---

## Where to read code

- CLI entrypoint: [`src/gridiron_edge/cli/main.py`](src/gridiron_edge/cli/main.py)
- CLI sub-modules: [`src/gridiron_edge/cli/`](src/gridiron_edge/cli/) (one file per command group)
- nflverse ingest: [`src/gridiron_edge/ingest/nflverse/`](src/gridiron_edge/ingest/nflverse/)
- Weather + DK odds collector: [`src/gridiron_edge/ingest/pfr/collector_impl.py`](src/gridiron_edge/ingest/pfr/collector_impl.py)
- Odds ledger: [`src/gridiron_edge/ingest/odds/store.py`](src/gridiron_edge/ingest/odds/store.py)
- Features: [`src/gridiron_edge/features/pipeline.py`](src/gridiron_edge/features/pipeline.py)
- Elo: [`src/gridiron_edge/ratings/elo/`](src/gridiron_edge/ratings/elo/)
- Predictor protocol + registry: [`src/gridiron_edge/models/`](src/gridiron_edge/models/)
- Prediction archive: [`src/gridiron_edge/evaluation/archive.py`](src/gridiron_edge/evaluation/archive.py)
- Evaluation metrics: [`src/gridiron_edge/evaluation/metrics.py`](src/gridiron_edge/evaluation/metrics.py)
- Elo tuner: [`src/gridiron_edge/evaluation/tune.py`](src/gridiron_edge/evaluation/tune.py)
- Simulation: [`src/gridiron_edge/sim/`](src/gridiron_edge/sim/)
- Predictions viz: [`src/gridiron_edge/viz/predictions.py`](src/gridiron_edge/viz/predictions.py)
- Console output: [`src/gridiron_edge/core/console.py`](src/gridiron_edge/core/console.py)

---

## Adding a new prediction model

The `Predictor` protocol makes adding new models mechanical. Steps:

1. Create `src/gridiron_edge/models/{model_type}/predictor.py`
2. Implement `predict_historical(games, *, repo)` and `predict_upcoming(schedule, *, repo)`
3. Decorate the class with `@PredictorRegistry.register`
4. Add the import to `evaluation/backfill.py`'s import block

The evaluation framework (backfill, summary, calibration, compare) then works automatically for the new model with no further changes.

---

## Code quality

```bash
uv run ruff check src/ --fix   # lint
uv run ruff format src/        # format
uvx pyrefly check              # type check
uv run pytest                  # tests (14/14)
```

All four must pass before committing. Config in [`pyproject.toml`](pyproject.toml) (Ruff) and [`pyrefly.toml`](pyrefly.toml) (Pyrefly).

Use `uv run gridiron -v <command>` for verbose output with step detail, row counts, and debug logs. `GRIDIRON_VERBOSE=1` also works.

---

## Known sharp edges

### nflverse data cadence

nflverse updates nightly after each game day. The cleanest weekly snapshot is Thursday (after the NFL incorporates Mon-Wed stat corrections). Historical coverage goes back to 1999.

### DK odds `--week` default

`fetch_dk_odds()` defaults `week=1` if not provided. Always pass `--week` explicitly when pulling mid-season to ensure the ledger is tagged correctly.

### Elo and upcoming week

`output predictions` merges Elo onto the upcoming schedule. If the Elo state table doesn't yet have entries for the requested week, predictions will be empty. Run `ratings elo fit` first.

### `collector_impl.py` scope

`collector_impl.py` now handles only weather (OpenWeatherMap) and DraftKings odds. All historical game + schedule ingestion goes through `ingest/nflverse/`. Do not add Scrapy or PFR dependencies back — that path was retired.

### Week 18 predictions

Week 18 Brier score is historically higher (~0.255 vs ~0.228 average) because teams rest starters when playoff seeding is locked. This is expected and correct — do not exclude week 18 from evaluation. The evaluation output notes this automatically.

### elo_v3 tune job

The elo_v3 grid search (~63k combinations) takes ~8 hours. Results auto-save to `data/output/tune/elo_v3_tune_results.parquet` so the terminal can be closed safely. To apply the best params after the run:

```bash
uv run gridiron evaluate tune --v3 --apply
```

### Odds game_id mismatch (known debt)

The DK odds ledger uses internal DraftKings event IDs, not the canonical `YYYY_WW_AWAY_HOME` game_id format. Joining odds against predictions by `game_id` will fail silently until the resolver is built (Phase 21 prerequisite).

---

## Operational checklist (weekly)

1. `uv run gridiron run-data-pipeline` — refresh games + upcoming + Elo + features
2. `uv run gridiron ingest dk-odds --season YYYY-YYYY+1 --week N` — pull odds
3. `uv run gridiron output predictions --year YYYY-YYYY+1 --week N` — generate viz + archive
4. `uv run gridiron sim run` — update playoff probabilities
5. `uv run gridiron output ranks --year YYYY-YYYY+1 --week N` — rankings CSV

---

## Further work

See [`PLAN.md`](PLAN.md) for full phase history and next priorities.
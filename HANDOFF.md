# Project Handoff: `NFL_Predict` / Gridiron Edge

How to run the project, how it is laid out, and what commonly breaks.

See [PLAN.md](PLAN.md) for current priorities and roadmap.
See [CHANGELOG.md](CHANGELOG.md) for completed phase history and decisions.

---

## What this project does

- **Goal:** NFL game prediction support — win probabilities, Elo rankings,
  weekly matchup visualisations, season simulation, and odds tracking.
- **Data source:** [nflverse](https://github.com/nflverse/nflverse-data) via
  `nfl_data_py` (replaced PFR/Scrapy at Phase 13).
- **Production model:** `random_forest_v2` (Brier 0.21078, AUC 0.71820,
  auto-selected). 51 features: Elo, EPA rolling window, rest, weather,
  travel, divisional flag, franchise HFA.
- **Entry point:** `uv run gridiron` → `src/gridiron_edge/cli.py`

---

## Repository layout

| Path | Role |
|------|------|
| `src/gridiron_edge/` | Main package |
| `src/gridiron_edge/ingest/nflverse/` | nflverse game + schedule ingestion |
| `src/gridiron_edge/ingest/pfr/collector_impl.py` | Weather (OWM) + DraftKings odds |
| `src/gridiron_edge/ingest/odds/` | Odds ledger (Parquet, long format) |
| `src/gridiron_edge/transform/clean/` | nflverse → canonical schema mappers |
| `src/gridiron_edge/features/` | Feature registry + pipeline (schema v3) |
| `src/gridiron_edge/ratings/elo/` | Elo table, fit, predict, evaluate |
| `src/gridiron_edge/models/game_prediction/` | ML models (logistic, RF, XGBoost) |
| `src/gridiron_edge/evaluation/` | Metrics, backfill, diagnostics, model selection |
| `src/gridiron_edge/sim/` | Monte Carlo season + playoff simulation |
| `src/gridiron_edge/viz/` | Predictions image/HTML, playoff table, rankings CSV |
| `src/gridiron_edge/core/` | Settings, logging, console output |
| `data/` | Generated at runtime (not committed) |

**Data layout:**

```
data/
  raw/            nflverse Parquet files (games, upcoming schedule, PBP)
  cleaned/        Canonical CSVs (games, schedule, Elo state, stadiums, weather)
  modeling/       Feature matrix (base_modeling_file.parquet + modeling_file.parquet)
  models/         Trained model artifacts (one directory per version)
  output/
    predictions/{year}/   week_NN_predictions.png + .html + .csv
    predictions/predictions_log.parquet   Full prediction archive
    rankings/             elo_rankings_{year}_wkNN.csv
    sim/                  playoff probability tables
  odds/
    dk_odds_log.parquet       Full historical ledger (all pulls)
    dk_odds_current.parquet   Latest pull snapshot (for viz)
```

---

## Setup

```bash
uv sync
uv run gridiron --help
```

**Python:** `>=3.12,<4`.

---

## Configuration

| Secret / setting | How |
|------------------|-----|
| OpenWeather API | Env var `OWM_API_KEY` for `gridiron ingest weather` |
| Data paths | `datasets/registry.py` + `core/settings.py` |

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

Fetches all history (1999–present), 2026 upcoming schedule, rebuilds Elo,
builds feature matrix. ~115s.

### Weekly refresh (during season)

```bash
uv run gridiron run-data-pipeline
```

Auto-detects current season. Re-fetches games + upcoming schedule, refreshes
Elo incrementally, rebuilds features.

### Step-by-step

```bash
uv run gridiron ingest nflverse-games [--all-years]
uv run gridiron ingest nflverse-upcoming
uv run gridiron transform clean-games
uv run gridiron transform clean-upcoming
uv run gridiron ratings elo fit [--all-years]
uv run gridiron features model-inputs [--all-years]
```

### Output

```bash
uv run gridiron output predictions --year 2026-2027 --week 1   # PNG + HTML
uv run gridiron output ranks --year 2026-2027 --week 1         # Elo rankings CSV
uv run gridiron sim run [--n-sims 10000]                       # Season simulation
```

### Odds

```bash
uv run gridiron ingest dk-odds --season 2026-2027 --week 1
```

### Weather

```bash
export OWM_API_KEY="YOUR_KEY"
uv run gridiron ingest weather --season-year "2026-2027"
```

---

## Model workflows

### Training a model

```bash
uv run gridiron models train random_forest_v2
uv run gridiron models train xgboost_v2
uv run gridiron models list    # confirm artifact exists
```

### Evaluation

```bash
uv run gridiron evaluate backfill --model-version random_forest_v2
uv run gridiron evaluate select-model       # composite ranking, all models
uv run gridiron evaluate report             # full characterisation of best model
uv run gridiron evaluate diagnostics --model-version random_forest_v2 --compare
```

### Adding a new model variant

New variants require one call in `tree.py` or `logistic.py`:

```python
# tree.py
RandomForestV3Predictor = _make_tree_variant(
    "random_forest_v3",
    "RF — description of what's new",
    feature_set=FEATURE_SETS["expanded"],
    model_type="rf",
)
```

Then add the name to `predictor.py` re-exports and run the quality gates.
No new class body, no new training function, no new `_make_*_features`.

### Adding a new feature

1. Create `src/gridiron_edge/features/team/your_feature.py` implementing
   `FeatureSpec` + `compute()` decorated with `@FeatureRegistry.register("name")`
2. Add `import gridiron_edge.features.team.your_feature  # noqa: F401` to
   `features/pipeline.py`
3. Add `"name"` to the `FEATURES` list in `pipeline.py` (order matters — see
   comments in that file)
4. If new columns are added to the games CSV, bump `CURRENT_SCHEMA_VERSION`
   in `features/manifest.py`
5. Rebuild: `uv run gridiron features model-inputs --all-years`

---

## Model architecture

**Feature pipeline (schema v3, 51 features for v2 models):**

| Group | Features |
|-------|---------|
| Home field | `HOME_FIELD` (binary) |
| Elo | `ELO_DIFF`, `TEAM_A_ELO`, `TEAM_B_ELO` |
| EPA (rolling, 8 metrics × 2 teams) | `TEAM_A/B_OFF_EPA_PER_PLAY`, `OFF_PASS_EPA`, `OFF_RUSH_EPA`, `OFF_SUCCESS_RATE`, `DEF_*` |
| Rest | `TEAM_A/B_DAYS_REST`, `SHORT_WEEK`, `POST_BYE` |
| Weather | `IS_DOME`, `WIND_SPEED_MPH`, `TEMP_F`, `PRECIP_FLAG` |
| Travel | `TEAM_A/B_KM_TRAVELED`, `TZ_SHIFT`, `ALTITUDE`, `IS_NEUTRAL_SITE` |
| Venue | `IS_DIV_GAME`, `TEAM_A/B_FRANCHISE_HFA` |

**Named feature sets** (`_shared.FEATURE_SETS`):

| Key | Features | Used by |
|-----|---------|---------|
| `"diff"` | 10 differential | logistic_v1 |
| `"raw"` | 22 raw | logistic_v2 |
| `"combined"` | 32 combined | logistic_v3/v4, rf_v1, xgb_v1 |
| `"expanded"` | 51 expanded | rf_v2, xgb_v2 |

**Holdout strategy:** Last 3 seasons (`2023-2024`, `2024-2025`, `2025-2026`)
withheld from all training. Never touch these for any tuning decision.

---

## File contract (key artifacts)

| File | Purpose |
|------|---------|
| `data/raw/NFL_wk_by_wk_nflverse.parquet` | Raw nflverse games |
| `data/cleaned/NFL_wk_by_wk_cleaned.csv` | Canonical historical games |
| `data/cleaned/NFL_wk_by_wk_w_weather.csv` | Games + weather (reconciled IDs) |
| `data/raw/NFL_upcoming_schedule_nflverse.parquet` | Raw upcoming schedule |
| `data/cleaned/NFL_upcoming_schedule_cleaned.csv` | Canonical upcoming schedule |
| `data/cleaned/NFL_Team_Elo.csv` | Elo ratings state table |
| `data/cleaned/NFL_stadium_reference.csv` | Stadium / geo reference |
| `data/modeling/base_modeling_file.parquet` | Base modeling rows |
| `data/modeling/modeling_file.parquet` | Full feature matrix (schema v3) |
| `data/output/predictions/predictions_log.parquet` | Full prediction archive |
| `data/odds/dk_odds_log.parquet` | Full DK odds history |
| `data/odds/dk_odds_current.parquet` | Latest DK odds snapshot |

---

## Where to read code

| What | Where |
|------|-------|
| CLI entry | `src/gridiron_edge/cli.py` |
| nflverse ingest | `src/gridiron_edge/ingest/nflverse/` |
| Weather + DK collector | `src/gridiron_edge/ingest/pfr/collector_impl.py` |
| Odds ledger | `src/gridiron_edge/ingest/odds/store.py` |
| Feature constants + sets | `src/gridiron_edge/models/game_prediction/_shared.py` |
| Feature pipeline + FEATURES list | `src/gridiron_edge/features/pipeline.py` |
| Model factory (tree) | `src/gridiron_edge/models/game_prediction/tree.py` |
| Model factory (logistic) | `src/gridiron_edge/models/game_prediction/logistic.py` |
| Model registry entry point | `src/gridiron_edge/models/game_prediction/predictor.py` |
| Elo | `src/gridiron_edge/ratings/elo/` |
| Simulation | `src/gridiron_edge/sim/` |
| Evaluation metrics | `src/gridiron_edge/evaluation/metrics.py` |
| Predictions viz | `src/gridiron_edge/viz/predictions.py` |
| Console output | `src/gridiron_edge/core/console.py` |

---

## Code quality gates

```bash
uv run ruff check src/ --fix   # lint + auto-fix
uv run ruff format src/        # format
uvx pyrefly check              # static type check
uv run pytest                  # tests
```

All four must pass before committing. Config in `pyproject.toml` (Ruff) and
`pyrefly.toml` (Pyrefly). Use `uv run gridiron -v <command>` for verbose
output. `GRIDIRON_VERBOSE=1` also works.

---

## Known sharp edges

**nflverse data cadence** — updates nightly. Cleanest weekly snapshot is
Thursday (after Mon–Wed stat corrections). Historical coverage: 1999–present.

**Weather file game IDs** — `NFL_wk_by_wk_w_weather.csv` uses NFLverse
historical IDs (e.g. `1999_01_BAL_STL`). Was reconciled from retrofitted IDs
via `scripts/reconcile_weather_ids.py` (one-time, now deleted). 24 corrupt
Super Bowl artifact IDs were purged. The weather backfill covers 1999–present;
773 games remain without weather data (pre-OWM history).

**Stadium reference** — `gridiron ingest nflverse-upcoming` warns when an
upcoming schedule contains a stadium absent from `NFL_stadium_reference.csv`.
Add the missing stadium with coordinates before running weather ingest.
Stadium name ≠ new building — name changes don't require a new row.

**DK odds `--week` default** — always pass `--week` explicitly. Defaults to
week=1 if omitted, which mis-tags mid-season pulls in the ledger.

**Elo and upcoming week** — run `ratings elo fit` before `output predictions`.
If Elo state doesn't have entries for the requested week, predictions will be
empty.

**`collector_impl.py` scope** — handles only weather (OWM) and DraftKings
odds. All game + schedule ingestion goes through `ingest/nflverse/`. Do not
add Scrapy or PFR dependencies.

**Feature pipeline order** — `FEATURES` list in `pipeline.py` is ordered.
`travel` must run before `venue_hfa` (HFA reads `IS_NEUTRAL_SITE`). See
comments in `pipeline.py`.

**Side-effect imports in `pipeline.py`** — every feature module import must
have `# noqa: F401`. Without it ruff strips the import and the feature never
registers, producing a silent `KeyError` at runtime.

---

## Operational checklist (weekly)

1. `uv run gridiron run-data-pipeline` — refresh games + upcoming + features
2. `uv run gridiron ingest dk-odds --season YYYY-YYYY+1 --week N` — pull odds
3. `uv run gridiron output predictions --year YYYY-YYYY+1 --week N` — viz
4. `uv run gridiron sim run` — playoff probabilities
5. `uv run gridiron output ranks --year YYYY-YYYY+1 --week N` — rankings CSV
6. `uv run gridiron evaluate report` — confirm model still auto-selects correctly
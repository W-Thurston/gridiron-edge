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
| Football State Representation | Represent what teams actually are | ✅ Phase 19 |
| Prediction Engine | What is likely to happen | 🔜 Phase 20 (in progress) |
| Model Evaluation | Which model should I use and why | ✅ Phase 20b |
| Model Reporting | Full characterisation of chosen model | ✅ Phase 20c |
| Tree-based Models | Non-linear signal capture (RF + XGBoost) | 🔜 Phase 20d |
| Feature Engineering | Interactions, transforms, domain features | 🔜 Phase 20e (conditional) |
| Market Intelligence | What does Vegas think | 🔜 Phase 21 |
| Decision Engine | What should actually be bet | 🔜 Future |
| Explainability & Insights | Why does the model think this | 🔜 Future |

**Core philosophy:** Betting edge is discovered through measurement, infrastructure, and systematic disagreement with the market — not through chasing more complex models. Understand existing models deeply before building more complex ones.

---

## Architecture

| Layer | Module |
|-------|--------|
| Game + schedule ingest | `gridiron_edge.ingest.nflverse` (nflverse/nfl_data_py) |
| PBP ingest | `gridiron_edge.ingest.nflverse.pbp` (nflverse/nfl_data_py) |
| Weather ingest | `gridiron_edge.ingest.pfr.collector_impl` (OpenWeatherMap) |
| Odds ingest + ledger | `gridiron_edge.ingest.odds` (DraftKings → Parquet) |
| Transform | `gridiron_edge.transform.clean` |
| Modeling features | `gridiron_edge.features.pipeline` (schema v2) |
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
- [x] `evaluation/metrics.py` — Brier score, log loss, calibration table, accuracy
- [x] `evaluation/backfill.py` — single generic `backfill_model(model_version)` covering all registered models
- [x] Historical backfill covering full 1999-present history including Super Bowls
- [x] Elo parameter grid search — `evaluation/tune.py`
  - elo_v2: flat-K (K=40, div=350, regress=0.40); Brier 0.2269
  - elo_v3: zone-based K (k_early=40, k_mid=40, k_week18=50, k_post=60, div=360); Brier 0.2269
  - elo_v1 remains best-calibrated at high confidence; recommended for production predictions
- [x] CLI: `gridiron evaluate backfill / summary / calibration / tune`

**Architectural foundations:**

- [x] `models/base.py` — `Predictor` + `Trainable` protocols; `PredictorSpec` with `trainable` flag
- [x] `models/artifact.py` — `ArtifactStore` + `ModelMetadata`; immutable versioned artifact store
- [x] `models/registry.py` — `PredictorRegistry` with `is_trainable()`, `trainable_names()`
- [x] `models/elo/predictor.py` — `EloV1`, `EloV2`, `EloV3` predictors registered at import time
- [x] `features/manifest.py` — feature set manifest written alongside `modeling_file.csv`
- [x] `datasets/loaders.py` — `load_modeling_file()` with schema version + column validation
- [x] CLI split into `cli/` sub-modules: ingest, transform, ratings, features, output, sim, evaluate, models
- [x] Test coverage: 44 tests

**Performance:**

- [x] Elo table rebuild: 47.7s → 1.3s
- [x] Feature pipeline: 73.0s → 5.0s
- [x] Full `run-data-pipeline --all-years`: 127s → 8s

---

## Phase 19 — Richer football state representation ✅

- [x] `ingest/nflverse/pbp.py` — permanent PBP cache in `data/raw/pbp/` (~20MB/season, ~540MB total)
- [x] `transform/clean/epa.py` — aggregate PBP to game-level team EPA stats (`epa_by_game.parquet`)
- [x] `features/team/epa.py` — rolling window EPA feature registered as `"epa"`
  - 8 metrics per team: off/def EPA per play, pass/rush splits, success rate
  - Default rolling window = 4 games (to be tuned as part of Phase 20 hyperparameter search)
- [x] `features/manifest.py` — `CURRENT_SCHEMA_VERSION` bumped to 2
- [x] CLI: `gridiron ingest pbp [--all-years]`, `gridiron transform aggregate-epa`
- [x] CLI: `gridiron run-data-pipeline --build-epa`

---

## Phase 20 — ML game prediction models 🔜 (in progress)

**Goal:** Build and evaluate ML models alongside Elo. Understand what each model is learning and where it struggles before adding complexity.

**Completed:**

- [x] `models/game_prediction/predictor.py` — logistic regression variants (logistic_v1 through v4)
  - `logistic_v1`: differential features only (10 features) — Brier **0.22059**, AUC 0.68279
  - `logistic_v2`: raw features, both teams (22 features) — Brier **0.22059**, AUC 0.68279
  - `logistic_v3`: differential + raw combined (32 features) — Brier **0.22057**, AUC 0.68289 ← best
  - `logistic_v4`: differential + raw, ElasticNet regularised, 28 features (21 non-zero) — Brier **0.22058**, AUC 0.68288
  - All logistic variants beat elo_v1–v3 on every metric; EPA features add genuine signal
  - Marginal spread between v1–v4 suggests logistic ceiling reached; signals XGBoost as next tier
- [x] Training: `LogisticRegressionCV`, 5-fold CV, `StandardScaler` pipeline; holdout = last 3 seasons
- [x] `pyproject.toml`: add `scikit-learn>=1.4.0`, `joblib>=1.3.0`
- [x] CLI: `gridiron models train / list / info`
- [x] `evaluate backfill` for all registered models
- [x] `evaluate select-model` — composite ranking across all models; recommendation with reason

**Current model rankings (all models, holdout 2023–2026):**

| model_version | n_games | brier | ece | auc | composite_rank |
|---|---|---|---|---|---|
| logistic_v3 | 7011 | 0.22057 | 0.01538 | 0.68289 | 3 ← recommended |
| logistic_v4 | 7011 | 0.22058 | 0.01672 | 0.68288 | 6 |
| logistic_v2 | 7011 | 0.22059 | 0.01743 | 0.68279 | 9 |
| logistic_v1 | 7011 | 0.22059 | 0.01767 | 0.68279 | 10 |
| elo_v2 | 7276 | 0.22685 | 0.07222 | 0.67679 | 16 |
| elo_v3 | 7276 | 0.22693 | 0.07238 | 0.67654 | 19 |
| elo_v1 | 7276 | 0.23094 | 0.07125 | 0.67080 | 19 |

**Next — tree-based models (Phase 20d):**

- [ ] See Phase 20d below

---

## Phase 20b — Model evaluation framework ✅

**Goal:** Build a systematic, quantitative framework for choosing between models and auditing their quality.

**Completed:**

**Quantitative metrics** (`evaluation/metrics.py`):
- [x] Brier score, log loss, accuracy, ROC-AUC, ECE (Expected Calibration Error)
- [x] Calibration table — predicted probability bucket vs actual win rate
- [x] `build_evaluation_df` — joins prediction archive to game outcomes
- [x] `summarise` — grouped Brier/accuracy summary by season, week, or model

**Evaluation CLI** (`cli/evaluate.py`):
- [x] `gridiron evaluate summary [--group-by season|week|model_version]`
- [x] `gridiron evaluate calibration [--buckets N]`
- [x] `gridiron evaluate backfill [--model-version X] [--overwrite]`
- [x] `gridiron evaluate tune [--v3] [--apply]` — Elo grid search
- [x] `gridiron evaluate diagnostics [--model-version X] [--compare]` — calibration/ROC/Brier plots
- [x] `gridiron evaluate select-model [--criteria brier,ece,auc] [--top N]` — composite ranking with recommendation

**What `select-model` does:**
- Loads all registered models with archived predictions
- Ranks each model on configurable criteria (default: Brier, ECE, AUC)
- Computes composite rank (sum of per-criterion ranks); lowest wins
- Prints ranked table and recommendation with reason
- Notes if primary criterion is tied and advises visual confirmation via `diagnostics --compare`

---

## Phase 20c — Model report command ✅

**Goal:** Close the gap between "which model wins" and "is this model safe to deploy." A single `evaluate report` command that auto-selects the best model and immediately characterises its performance in depth — no manual stitching of multiple commands required.

**The problem this solves:**

`select-model` tells you *which* model won on aggregate metrics. That's necessary but not sufficient for deployment confidence. Three additional questions must be answered:

1. **Does it fall apart at high confidence?** A model that's well-calibrated on average but wrong 40% of the time when predicting 75%+ win probability is dangerous for betting. Aggregate Brier hides this.
2. **Is performance stable over seasons?** A model trained through 2022-2023 that's quietly degrading in 2024-2025 has concept drift. Aggregate metrics pool all seasons and conceal the trend.
3. **What are its systematic blind spots?** The worst individual calls reveal patterns (road underdogs in week 1, divisional rivalry games, etc.) that aggregate metrics cannot surface.

**New CLI command:**

```
gridiron evaluate report [--model-version X] [--top-misses N] [--season YYYY-YYYY]
```

- Without `--model-version`: auto-selects the recommended model via `select-model` logic
- With `--model-version`: analyses that specific model
- `--top-misses N`: number of worst individual calls to surface (default 10)
- `--season`: filter analysis to a specific season

**Output sections (all printed to terminal; no new plot files):**

```
[1. Model Selection]
   Ranking table (same as select-model)
   → Recommendation: logistic_v3  (best Brier, ECE, AUC)

[2. Confidence-Stratified Brier]
   confidence_tier  n_games  brier    actual_win_rate  calibration_gap
   50–60%            2841    0.238    0.534            +0.003
   60–70%            2104    0.213    0.651            +0.005
   70–80%             912    0.193    0.741            -0.002
   80–100%            154    0.201    0.798            -0.021
   ⚠  High-confidence tier (80%+): model predicts 85% avg, teams win 80% — slight overconfidence

[3. Season-over-Season Brier]
   season      n_games  brier    delta_vs_mean
   2023-2024      570   0.219    -0.006  ✓
   2024-2025      570   0.223    -0.002  ✓
   2025-2026      570   0.228    +0.003  ~  (slight decay — monitor)

[4. Top N Misses]
   season     week  away  home  predicted  outcome  error
   2025-2026   3    NYJ   MIA   0.76       L        0.76
   ...
   Patterns: 4/10 worst misses are road dogs week 1–3 (early-season EPA instability?)
```

**New functions in `evaluation/metrics.py`:**

- `brier_by_confidence_tier(df, tiers)` — Brier + calibration gap per predicted-prob bucket
- `brier_by_season(df)` — per-season Brier with delta vs mean; trend indicator
- `biggest_misses(df, n)` — top N games by `|predicted_prob - outcome|`, with game context

**Design constraints:**

- Terminal-only output — no new PNG files; keeps the command fast and scriptable
- Auto-selection is identical to `select-model` logic (shared `_collect_model_metrics` + ranking); no divergence
- Confidence tiers default to `(0.5, 0.6)`, `(0.6, 0.7)`, `(0.7, 0.8)`, `(0.8, 1.0)` but are configurable
- "Patterns" summary in the misses section is a simple heuristic (e.g. count early-week games, count road underdogs) — not NLP; avoids over-engineering
- All three new metric functions are unit-testable on synthetic DataFrames; add to `tests/evaluation/test_metrics.py`

**What this does NOT include (deferred):**

- CLV tracking — needs odds `game_id` resolver (Phase 21 dependency)
- Market disagreement analysis — same dependency
- SHAP values — add complexity with low marginal value until XGBoost is built
- Ensemble agreement — low value until there are more meaningfully different models
- Plot output — `diagnostics --compare` already covers visual needs

**Acceptance criteria:**

```bash
uv run gridiron evaluate report                          # auto-selects best model, full report
uv run gridiron evaluate report --model-version elo_v1  # specific model
uv run gridiron evaluate report --top-misses 20         # more misses
uv run gridiron evaluate report --season 2025-2026      # season filter
uv run ruff check src/ --fix && uvx pyrefly check && uv run pytest
```

---

## Phase 20d — Tree-based models: Random Forest + XGBoost 🔜

**Goal:** Determine whether the signal in the current feature set is genuinely non-linear, and by how much tree-based models improve on the logistic ceiling (Brier 0.22057). Both models use the same 32-feature combined set as logistic_v3 to ensure a clean apples-to-apples comparison.

**Why both models?** Random Forest and XGBoost both capture non-linear interactions but via different mechanisms — ensemble averaging vs. boosting. On a medium-small dataset (~12k training rows) the two approaches are genuinely competitive. Building both reveals whether the signal is better captured by averaging (RF) or sequential error correction (XGBoost), and gives a richer picture going into Phase 21.

**Why this feature set?** Tree models are transformation-invariant and discover interaction effects automatically. Manual interaction terms or polynomial transforms would be redundant here — those belong in Phase 20e *only if* the tree results warrant them.

**New infrastructure required:**

- `_rebuild_features_with_window(df, window)` — re-computes rolling EPA features with a configurable window size at training time, rather than using the pre-computed 4-game default from the pipeline. This resolves the "Rolling window tuning" backlog item.
- `_train_random_forest(...)` — shared training function for RF variants
- `_train_xgboost(...)` — shared training function for XGBoost variants
- `pyproject.toml`: add `xgboost>=2.0.0` (scikit-learn already present)

---

### `random_forest_v1` — Random Forest with isotonic calibration

**Architecture:** `StandardScaler` → `RandomForestClassifier` → `CalibratedClassifierCV(method="isotonic")`

Scaling is kept for pipeline consistency even though RF is scale-invariant. Isotonic calibration is applied as a post-processing step because RF probabilities from `predict_proba` are often slightly overconfident — without calibration, ECE for RF typically sits at 0.03–0.05 compared to logistic_v3's 0.015. Isotonic calibration corrects this without distorting the ranking quality (AUC is unaffected).

**Hyperparameter search:** `RandomizedSearchCV`, 5-fold CV, 50 iterations

| Parameter | Search range | Rationale |
|---|---|---|
| `n_estimators` | 100, 200, 300, 500 | More trees → lower variance; diminishing returns past ~300 |
| `max_depth` | 3, 4, 5, 6, `None` | Shallow trees prevent overfitting on ~12k rows |
| `min_samples_leaf` | 5, 10, 20, 30 | Higher = more regularisation; critical at small dataset size |
| `max_features` | `"sqrt"`, `"log2"`, 0.5 | Controls inter-tree correlation |
| `epa_window` | 1, 2, 3, 4, 6, 8 | Rolling EPA window as a hyperparameter |

**Expected performance:** Brier 0.218–0.222. Should beat logistic; likely trails XGBoost. Well-calibrated ECE if isotonic calibration is applied correctly.

**Stored metadata:** best hyperparameters, OOB score, feature importances (top 10 by mean decrease in impurity), calibration method applied.

---

### `xgboost_v1` — Gradient boosted trees

**Architecture:** `StandardScaler` → `XGBClassifier(objective="binary:logistic", eval_metric="logloss")`

XGBoost's `binary:logistic` objective produces well-calibrated probabilities natively in most cases. ECE will be confirmed post-training; isotonic calibration applied only if ECE exceeds 0.025.

**Hyperparameter search:** `RandomizedSearchCV`, 5-fold CV, 75 iterations (larger space warrants more iterations)

| Parameter | Search range | Rationale |
|---|---|---|
| `n_estimators` | 100, 150, 200, 300, 500 | More trees with low `learning_rate` |
| `max_depth` | 2, 3, 4, 5, 6 | Shallow trees are key for tabular data at this size |
| `learning_rate` | 0.01, 0.03, 0.05, 0.1, 0.2 | Lower = better generalisation but slower convergence |
| `subsample` | 0.6, 0.7, 0.8, 1.0 | Row subsampling per tree (stochastic gradient boosting) |
| `colsample_bytree` | 0.6, 0.7, 0.8, 1.0 | Feature subsampling per tree |
| `min_child_weight` | 1, 5, 10, 20 | Minimum sum of instance weight per leaf; key regulariser |
| `gamma` | 0, 0.1, 0.3, 0.5 | Minimum loss reduction required to make a split |
| `epa_window` | 1, 2, 3, 4, 6, 8 | Rolling EPA window as a hyperparameter |

**Expected performance:** Brier 0.215–0.220. Expected best-in-class if genuine non-linear structure exists; may only marginally beat logistic if the signal is largely linear.

**Stored metadata:** best hyperparameters, feature importances (gain-based, top 10), calibration applied (yes/no), overfit gap (train vs holdout Brier).

---

### Decision gate after Phase 20d

The results determine what happens next:

| Outcome | What it means | Next step |
|---|---|---|
| XGBoost Brier < 0.219 (>0.002 improvement over logistic) | Meaningful non-linear structure confirmed | Proceed to Phase 20e feature engineering |
| XGBoost Brier 0.219–0.221 (marginal improvement) | Signal largely linear; tree depth not justified | Skip Phase 20e; proceed directly to Phase 21 |
| XGBoost ≈ logistic | No non-linear signal; dataset may be too small | Phase 20e not warranted; consider `logistic_v5` with explicit cross-terms only |
| RF beats XGBoost | Boosting is overfit; dataset favours averaging | Adjust Phase 20e scope; use RF as production model |

---

### Acceptance criteria

```bash
uv run gridiron models train random_forest_v1
uv run gridiron models train xgboost_v1
uv run gridiron evaluate backfill --model-version random_forest_v1
uv run gridiron evaluate backfill --model-version xgboost_v1
uv run gridiron evaluate select-model          # updated ranking table with all 9 models
uv run gridiron evaluate report                # full report on auto-selected winner
uv run ruff check src/ --fix && uvx pyrefly check && uv run pytest
```

---

## Phase 20e — Feature engineering 🔜 (conditional on Phase 20d results)

**Goal:** If Phase 20d confirms meaningful non-linear structure (XGBoost Brier < 0.219), determine whether additional engineered features or new information sources can push performance further.

**This phase is explicitly conditional.** If tree models do not meaningfully outperform logistic_v3, the signal is largely linear and further feature engineering on the current feature set has limited upside. Phase 21 (market intelligence) would then take priority.

**Three categories of work, ordered by expected value:**

---

### Category A — Domain-driven new features (highest value, independent of model results)

These are new *information sources* that no amount of transformation can substitute for. They belong here regardless of whether Phase 20d shows non-linearity, because the current feature set simply doesn't contain this information.

- **Rest and schedule stress** — days since last game, back-to-back weeks, bye week recency. Captures fatigue effects that EPA rolling averages smooth over. Low implementation cost (schedule data already ingested).
- **Weather effects** — temperature, wind speed, precipitation at game location. OWM ingest already exists (`collector_impl.py`); feature pipeline integration is the missing piece. Matters most for outdoor stadiums in late-season games.
- **Home field strength by venue** — stadium attendance capacity or crowd noise proxy. Simple lookup table; captures that some venues (Arrowhead, CenturyLink) consistently affect outcomes.

These are added as new feature columns in `features/` and re-run through all existing models for clean comparison.

---

### Category B — Explicit interactions and polynomial transforms (medium value, tree-result-dependent)

Only warranted if XGBoost confirms non-linear structure *and* a logistic model variant is desired for interpretability or deployment reasons. The specific interactions that matter would be read from XGBoost's feature importances — no point guessing.

- **`logistic_v5`** — logistic regression with top-K interaction terms derived from XGBoost feature importances. E.g. if XGBoost shows `ELO_DIFF × OFF_EPA_DIFF` as the dominant split, add that cross-term explicitly to a logistic model.
- **Polynomial features** — `EPA²` or `log(EPA)` terms only for features where XGBoost's partial dependence plots show clear non-monotonic relationships.

**Do not build this speculatively.** The feature importances from `xgboost_v1` are the input to this decision.

---

### Category C — QB and advanced EPA intelligence (longer-term, Phase 21+ dependency)

High signal but higher implementation cost and potential data gaps. Listed here rather than Phase 21 because they're model features, not market intelligence.

- **QB Elo** — per-QB rolling performance rating updated after each game. Requires QB-level PBP aggregation and starter tracking (injury replacements). Listed in backlog as "QB intelligence."
- **Advanced EPA extensions** — CPOE (completion percentage over expected), RYOE (rushing yards over expected). Requires additional PBP aggregation in `transform/clean/epa.py`.

These are scoped and planned only after Phase 20d and Phase 21 are complete.

---

### Acceptance criteria (if Phase 20e is triggered)

```bash
# New domain features added and pipeline rebuilt
uv run gridiron run-data-pipeline --build-epa       # recomputes with new features
uv run gridiron models train xgboost_v1             # retrain on expanded feature set
uv run gridiron evaluate select-model               # confirm improvement
uv run gridiron evaluate report
uv run ruff check src/ --fix && uvx pyrefly check && uv run pytest
```

---



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
uv run gridiron evaluate select-model
uv run gridiron evaluate report
uv run gridiron models list
uv run gridiron models train random_forest_v1    # Phase 20d
uv run gridiron models train xgboost_v1          # Phase 20d
```

```bash
# Code quality gates
uv run ruff check src/ --fix
uvx pyrefly check
uv run pytest
```

---

## Architectural debt — tracked items

| Item | Blocking at | Notes |
|------|------------|-------|
| Odds `game_id` resolver | Phase 21 | DK odds ledger uses internal event IDs, not canonical `YYYY_WW_AWAY_HOME` format; join will fail silently |
| Test coverage for `backfill.py` + `tune.py` | Ongoing | `archive.py`, `metrics.py`, `manifest.py` now covered; backfill and tune still need tests |
| `datasets/registry.py` self-registration | Long-term | Flat dict grows linearly with datasets; low urgency until >20 datasets |

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

### Simulation improvements
- **Monte Carlo per-stage playoffs** — simulate each playoff round independently (wild card, divisional, conference, Super Bowl) rather than all at once; allows injecting known results as the postseason progresses
- **Skip full-season sim after week 18** — once the regular season is complete, the sim should use actual standings rather than re-simulating 18 weeks of results
- **Dynamic schedule updates** — handle NFL game relocations mid-season without manual file edits

### Model features
- **QB intelligence** — QB Elo rating, QBR, passer rating as team-level features; high signal for game prediction. Scoped in Phase 20e Category C — requires QB-level PBP aggregation and starter tracking.
- **Advanced EPA metrics** — once base EPA is in (Phase 19), extend to:
  - Defense-adjusted Value over Average (DVOA)
  - Completion Percentage over Expected (CPOE)
  - Rushing Yards Over Expected (RYOE)
  - Weighted EPA (see github.com/greerreNFL)
  Scoped in Phase 20e Category C alongside QB intelligence.
- **Stadium attendance capacity** — venue size as a feature for crowd noise / home field strength. Scoped in Phase 20e Category A as a simple lookup table addition.
- **Stadium-level HFA coefficient** — the current `venue_hfa` feature operates at the franchise level (historical home win rate per franchise). A more sophisticated version would compute separate coefficients per physical stadium building, blending toward the franchise coefficient for new stadiums with insufficient data. Requires reliable stadium open/close date data (name changes ≠ new stadium) to correctly partition a franchise's home game history by building. Implement after stadium continuity dates can be sourced.

### Infrastructure
- **Drive/possession simulation** — expand Monte Carlo toward play-level outcome distributions
- **Injury + roster intelligence** — injury-adjusted team strength, player availability impacts
- **Weather effects** — integrate OWM data into prediction features (ingest exists, feature does not). Scoped in Phase 20e Category A.
- **Ensemble systems** — stack Elo + EPA model + ML model with uncertainty estimates
- **Public dashboards / automated reports** — shareable weekly output
- **Live game intelligence** — real-time win probability updates (major architecture change)
- **`datasets/registry.py` self-registration** — replace flat dict with auto-discovery when dataset count grows past ~20
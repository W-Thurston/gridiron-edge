# Gridiron Edge — Changelog

Decision record for completed work. Each entry captures what was built,
why the approach was chosen, and key results. Full detail lives in git history;
this file is the summary a future reader needs when asking "why does this work
this way?"

Tag each phase completion: `git tag phase-NN` so boundaries are queryable
via `git log phase-N..phase-NN --oneline`.

---

## Phase 20f — Model variant infrastructure (2026-05-29)

**What:** Replaced 8 hand-written predictor class bodies (~35 lines each) with
`_make_tree_variant()` and `_make_logistic_variant()` factory functions in
`tree.py` and `logistic.py`. Added `FeatureSet` frozen dataclass and
`FEATURE_SETS` registry to `_shared.py`. Optimised training loops: window
cache eliminates redundant parquet reads across hyperparameter iterations,
`StratifiedKFold` instantiated once per training run, feature importance
averaging vectorised with `np.array().mean(axis=0)`. Fixed `"feature_set"`
metadata which was hardcoded to `"combined_32"` regardless of variant.
`predictor.py` converted to a thin shim with re-exports for backward
compatibility.

**Why:** Adding a new model variant previously required a new class body,
a re-export in `predictor.py`, a new `_make_*_features` function, and new
registration tests across 4 files. With the factory it is one call. Chose
Option C (programmatic factory) over Option A (Hydra + MLflow) — MLflow
earns its keep when real bets are tracked and experiment lineage has P&L
value; that is a Phase 22+ concern. Option A tracked in backlog.

**Results:** All 8 variants registered, quality gates passing, no external
interface changes. Window cache eliminates up to ~45 redundant parquet reads
per 50-iteration RF search run.

---

## Phase 20e — Feature engineering (2026-05-29)

**What:** Added five new feature modules to `features/team/`: `rest.py`
(`DAYS_REST`, `SHORT_WEEK`, `POST_BYE`), `weather.py` (`IS_DOME`,
`WIND_SPEED_MPH`, `TEMP_F`, `PRECIP_FLAG`), `travel.py` (`KM_TRAVELED`,
`TZ_SHIFT`, `ALTITUDE`, `IS_NEUTRAL_SITE`), `divisional.py` (`IS_DIV_GAME`),
`venue_hfa.py` (`TEAM_A/B_FRANCHISE_HFA`). Schema bumped to v3. Added
`_EXPANDED_FEATURES` (51 total) to `_shared.py`. Registered `random_forest_v2`
and `xgboost_v2` on the expanded feature set. Fixed `pipeline.py` side-effect
imports that were being stripped by ruff. Stadium reference warning added to
`schedule_nflverse.py` — fires when upcoming schedule contains a stadium
absent from the reference CSV.

Also fixed weather archive: one-time `reconcile_weather_ids.py` script
corrected 1,699 retrofitted `GAME_ID` values in `weather_enriched.csv` to
match NFLverse historical convention. 24 corrupt Super Bowl artifact IDs
(same team code on both sides) identified and purged.

**Why:** RF v1 Brier (0.21503) cleared the Phase 20d gate. Features chosen
for data availability first — all derivable from existing sources with no
new ingest required. Franchise-level HFA (not stadium-level) chosen to avoid
stadium continuity date tracking; stadium-level version tracked in backlog.

**Results:**

| model | Brier | ECE | AUC |
|---|---|---|---|
| random_forest_v2 | **0.21078** | 0.02108 | **0.71820** |
| random_forest_v1 | 0.21503 | 0.02836 | 0.70527 |
| xgboost_v2 | 0.21865 | 0.02059 | 0.69200 |
| xgboost_v1 | 0.21857 | 0.02093 | 0.69278 |
| logistic_v3 | 0.22102 | **0.01606** | 0.68241 |

RF benefited substantially; XGBoost did not — boosting was already
saturating on existing signal. `random_forest_v2` auto-selected as
production model.

---

## Phase 20d — Tree-based models (2026-05-28)

**What:** Refactored monolithic `predictor.py` into `_shared.py` (feature
constants, `_prepare_data`, `_is_trained`), `logistic.py` (all logistic
variants + training helpers), and `tree.py` (RF + XGBoost training +
`_rebuild_features_with_window`). Registered `random_forest_v1` and
`xgboost_v1`. Added `xgboost>=2.0.0` to dependencies.

`_rebuild_features_with_window` resolves the rolling-window-as-hyperparameter
backlog item — EPA window is searched over [1,2,3,4,6,8] during training rather
than fixed at 4.

**Why:** Tree models capture non-linear interactions that logistic regression
cannot. RF and XGBoost built together because they differ in mechanism
(ensemble averaging vs. boosting) and the comparison reveals where signal
lives. Isotonic calibration applied unconditionally to RF (RF `predict_proba`
is systematically overconfident); applied conditionally to XGBoost only if
holdout ECE > 0.025.

**Results:**

| model | Brier | ECE | AUC |
|---|---|---|---|
| random_forest_v1 | **0.21503** | 0.02836 | **0.70527** |
| xgboost_v1 | 0.21857 | **0.02093** | 0.69278 |
| logistic_v3 | 0.22102 | 0.01606 | 0.68241 |

Gate cleared (RF Brier 0.21503 < 0.219 threshold). RF beats XGBoost —
dataset favours averaging over boosting. Phase 20e activated.

---

## Phase 20c — Model report command (2026-05-27)

**What:** Added `gridiron evaluate report` — single command that auto-selects
the best model and prints a four-section characterisation: model ranking,
confidence-stratified Brier, season-over-season Brier with drift detection,
and top-N worst individual calls. 19 unit tests added.

**Why:** `select-model` identifies which model wins on aggregate. That is
necessary but insufficient for deployment confidence — a model can look good
on aggregate while being dangerously wrong at high confidence or quietly
drifting across seasons. `report` surfaces both without requiring manual
command stitching.

---

## Phase 20b — Model evaluation framework (2026-05-26)

**What:** Built the full evaluation infrastructure: Brier score, log loss,
accuracy, ROC-AUC, ECE, calibration table, Brier decomposition (Murphy 1973).
`gridiron evaluate summary / calibration / backfill / tune / diagnostics /
select-model`. Composite ranking across all registered models with
recommendation. CLI split into `cli/` sub-modules.

**Why:** Cannot make an informed model choice without a systematic framework.
Composite ranking (sum of per-criterion ranks) chosen over single-metric
selection because Brier, ECE, and AUC each surface different failure modes —
a model that wins on Brier alone may be dangerously miscalibrated.

**Also:** Elo parameter grid search (`evaluation/tune.py`). `elo_v2`
(flat-K=40) and `elo_v3` (zone-based K) added — both Brier 0.2269 vs
`elo_v1` 0.2314. `elo_v1` remains best-calibrated at high confidence.
Performance wins: Elo rebuild 47.7s→1.3s, feature pipeline 73s→5s, full
pipeline 127s→8s. Test coverage: 44 tests.

---

## Phase 20 — Logistic regression models (2026-05-25)

**What:** First ML game prediction models. Four logistic variants:
`logistic_v1` (10 differential features), `logistic_v2` (22 raw features),
`logistic_v3` (32 combined), `logistic_v4` (32 combined, ElasticNet).
`models/base.py` (Predictor + Trainable protocols), `models/artifact.py`
(ArtifactStore + ModelMetadata), `models/registry.py` (PredictorRegistry).

**Why:** Logistic regression first for interpretability and as a baseline
ceiling. Differential features (TEAM_A − TEAM_B) vs raw tested separately
because differential is more interpretable but raw retains inter-team
asymmetry information. ElasticNet variant added because L1 regularisation
prunes weak features — the 21 non-zero features in v4 vs 32 in v3 tells
us which features are genuinely contributing.

**Results:** All logistic variants Brier ~0.221, beating all Elo variants.
Marginal spread between v1–v4 confirms logistic ceiling reached.

---

## Phase 19 — Richer football state representation (2026-05-20)

**What:** PBP ingest (`ingest/nflverse/pbp.py`, permanent cache ~540MB),
EPA aggregation to game level (`transform/clean/epa.py`, `epa_by_game.parquet`),
EPA rolling window feature registered as `"epa"` (8 metrics per team, default
window=4). Schema bumped to v2. CLI: `gridiron ingest pbp`, `gridiron transform
aggregate-epa`.

**Why:** Elo encodes historical strength but is blind to current team form.
EPA (Expected Points Added) captures recent offensive and defensive efficiency.
Rolling window chosen over season average because teams change mid-season;
window size left as a hyperparameter for Phase 20d training search.

---

## Phases 1–18 — Foundation (2025–2026)

**Phases 1–9:** Core refactor — migrated from `data_pipelines/` +
`model_pipelines/` + `utils/` monolith into `src/gridiron_edge/` package
structure.

**Phase 10:** Monte Carlo season simulation + playoff bracket (NFL tiebreaker
logic, conference seeding). CLI: `gridiron sim run`.

**Phase 11:** Stabilisation — integration tests, travel feature coord merge
fix, collector paths via registry.

**Phase 12:** Tooling — Poetry → uv migration, Ruff (full lint + format),
Pyrefly (static typing), Google-style docstrings across all public surfaces.

**Phase 13:** nflverse migration — replaced PFR/Scrapy (Cloudflare-blocked)
with `nfl_data_py`. Canonical games + schedule schema established. Raw
storage in Parquet; canonical output remains CSV.

**Phase 14:** Console output system — timed step context manager, compact/
verbose modes, lazy CLI imports (`--help` < 1s).

**Phase 15:** Excel retirement — append-only Parquet odds ledger, weekly
matchup PNG/HTML (migrated from notebook), versioned CSV rankings output.

**Phase 16:** Scrapy retirement — deleted `PFR_scraper/` entirely,
`collector_impl.py` now handles only weather (OWM) and DraftKings odds.

**Phase 17:** Dead code removal — wired missing `output predictions` CLI
command, removed empty stubs, all quality gates passing (14/14 tests).

**Phase 18:** Evaluation infrastructure + architectural foundations —
prediction archive (append-only Parquet), `evaluation/metrics.py`,
`evaluation/backfill.py`, `models/base.py` protocols, `ArtifactStore`,
`PredictorRegistry`, `features/manifest.py`, `datasets/loaders.py`.
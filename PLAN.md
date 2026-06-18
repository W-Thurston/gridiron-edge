# Gridiron Edge — Development Plan

> **Purpose:** single source of truth for *what to build next* and *why*.
> Updated at the start and close of every workstream.

| Document | Role |
|----------|------|
| **PLAN.md** (this file) | What is planned, what is active, what is deferred |
| **CHANGELOG.md** | What was built and when (completed workstream details) |
| **HANDOFF.md** | How the system works today (architecture, workflows, operations) |
| **ROADMAP.md** | Long-term strategic direction, workstream inventory, architecture decisions |

#### Status key

| Tag | Meaning |
|-----|---------|
| Done | Done — details in CHANGELOG.md |
| In progress | In progress |
| Planned | Planned / blocked |
| Deferred | Deferred |

---

## High-Level Priority Order

Agreed 2026-06-10. See ROADMAP.md for full workstream inventory.

| # | Workstream | Status |
|---|-----------|--------|
| 1 | Champion/Challenger for Props (RF + XGBoost) | Done |
| 2 | Game Model Refactor (align to props pattern) | Done |
| 3 | Integration & E2E Tests | **Active** |
| 4 | Deep Code Review + Test Suite Review | Planned |
| 5 | Scenario / "What If" Engine (W4.5) | Planned |
| 6 | API & Frontend (W8 + W9) | Planned |
| 7 | All External Odds (DK props, historical, line shopping) | Planned |
| 8 | Evaluate remaining work | Planned |

Two code reviews planned: one after workstream 3 (before building more),
one near completion of all workstreams. A separate dedicated test suite
review is included as part of workstream 4.

---

## Workstream 1: Champion/Challenger for Props (RF + XGBoost)

**Goal:** Add RandomForest and XGBoost as prop model types alongside
ElasticNet, build a champion selection system, and improve R² across all
5 stat families.

**Why it matters:** ElasticNet R² ranges from 0.071 (QB pass) to 0.203
(WR rec). Tree models capture nonlinear interactions ElasticNet cannot.
This is the single highest-impact improvement available.

**ROADMAP ref:** W4 extension.

### Current Baseline (ElasticNet only)

| Model | Train | Holdout | MAE | RMSE | R² | Nonzero |
|-------|-------|---------|-----|------|----|---------|
| qb_pass_yards | 5,706 | 1,367 | 58.0 | 72.6 | 0.071 | 37/128 |
| qb_rush_yards | 1,434 | 468 | 16.4 | 20.2 | 0.090 | 52/128 |
| rb_rush_yards | 10,023 | 2,001 | 25.0 | 32.3 | 0.168 | 16/124 |
| wr_rec_yards | 23,831 | 4,535 | 25.1 | 32.9 | 0.203 | 55/120 |
| te_rec_yards | 10,087 | 2,052 | 18.3 | 24.2 | 0.188 | 58/120 |

### Locked Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Multi-model architecture | Factory in base class, `model_type` param on `train()` | Prop subclasses are spec-only — no training logic divergence across model types |
| 2 | Model types | ElasticNet, RandomForest, XGBoost | Ridge is subsumed by ElasticNet (l1_ratio=0) |
| 3 | Scaling | StandardScaler for ElasticNet, None for trees | Tree models don't need feature scaling |
| 4 | HP search strategy | Grid search evaluated on holdout set | Consistent with current approach; HOLDOUT_SEASONS split provides temporal validation |
| 5 | ElasticNet grid | 5 alpha × 5 l1_ratio = 25 combos | Current grid, keep unchanged |
| 6 | RF grid | `n_estimators` ∈ [100, 300, 500], `max_depth` ∈ [8, 12, 16, None], `min_samples_leaf` ∈ [5, 10, 20] = 36 combos | Standard, fast |
| 7 | XGB grid | `n_estimators` ∈ [100, 300, 500], `max_depth` ∈ [4, 6, 8], `learning_rate` ∈ [0.01, 0.05, 0.1], `subsample` ∈ [0.8, 1.0] = 54 combos | Standard, fast |
| 8 | Promotion primary metric | Lowest holdout MAE | Direct measure of prediction accuracy for regression |
| 9 | Guardrail 1 | R² > 0 | Must beat mean baseline |
| 10 | Guardrail 2 | Coverage ∈ [85%, 97%] for 90% nominal interval | Uncertainty estimates not wildly off |
| 11 | Fallback champion | ElasticNet if no model passes guardrails | Known stable baseline |
| 12 | Model persistence | Retrain on demand (no artifacts) | Same as current; artifact storage is a future optimization |
| 13 | Clip ranges | Per-spec via `clip_lo`, `clip_hi` on `PropModelSpec` | Centralized, not scattered in predict methods |
| 14 | Champion comparison | External function in `evaluation/champion.py`, generalized for both classification and regression | Same separation of concerns as game models |
| 15 | Comparison architecture | Single generalized `compare_models()` supporting both Brier (classification) and MAE (regression) gates | Unified champion/challenger pattern across all models |
| 16 | Selection loop location | Outside the trainer (CLI or utility function) | The trainer trains, the evaluator judges |

### Phase A: Prop Model Factory — Steps

#### A1: Add PropModelType enum and factory to base.py

New enum:

    class PropModelType(str, Enum):
        ELASTICNET = "elasticnet"
        RANDOM_FOREST = "random_forest"
        XGBOOST = "xgboost"

New factory method in PropTrainer:

    def _create_model(self, model_type: PropModelType, params: dict) -> tuple[Any, StandardScaler | None]

Returns `(model_instance, scaler_or_none)`. ElasticNet gets a
StandardScaler; RF and XGB get None.

New HP grid method:

    def _get_param_grid(self, model_type: PropModelType) -> list[dict]

Returns the parameter grid for the given model type.

Modified `train()` signature:

    def train(self, *, model_type: str = "elasticnet", repo: Path | None = None) -> PropModelMetadata

Internally calls `_create_model()`, iterates over `_get_param_grid()`,
evaluates each on holdout, selects best, retrains on full training set.

New `_fit()` in base (**no longer abstract**):
- Iterates over param grid with tqdm progress bar
- Trains model + optional scaler with each param combo
- Evaluates MAE on holdout
- Selects best params
- Retrains on full training set with best params

New `_predict()` in base (**no longer abstract**):
- Applies scaler if present
- Calls `model.predict()`
- Clips to `spec.clip_lo` / `spec.clip_hi`

| Status | Step |
|--------|------|
| Not started | A1 |

#### A2: Add clip_lo / clip_hi to PropModelSpec

Update `PropModelSpec` dataclass:

    @dataclass(frozen=True)
    class PropModelSpec:
        name: str
        target_col: str
        position_filter: list[str]
        description: str
        clip_lo: float = 0.0
        clip_hi: float = 600.0

Clip ranges per stat family:

| Model | clip_hi |
|-------|---------|
| qb_pass_yards | 600 |
| qb_rush_yards | 200 |
| rb_rush_yards | 250 |
| wr_rec_yards | 300 |
| te_rec_yards | 250 |

Update all 5 subclass specs to set their clip_hi values.

| Status | Step |
|--------|------|
| Not started | A2 |

#### A3: Strip _fit() / _predict() / _scaler / _model from all 5 subclasses

Each subclass becomes ~15-20 lines: just the `spec` property.
All training + prediction logic lives in the base class.

Subclasses affected:
- `qb_pass_yards.py`
- `qb_rush_yards.py`
- `rb_rush_yards.py`
- `wr_rec_yards.py`
- `te_rec_yards.py`

| Status | Step |
|--------|------|
| Not started | A3 (depends on A1) |

#### A4: Add model_type to PropModelMetadata

Ensure `PropModelMetadata` (or its equivalent) stores `model_type` so
the champion comparison function knows which model type was used.
This should align with what `evaluation/champion.py` expects.

| Status | Step |
|--------|------|
| Not started | A4 |

### Phase B: Generalized Champion/Challenger Gates — Steps

#### B1: Generalize evaluation/champion.py for regression

Currently supports classification only (Brier, ECE, AUC).

Add regression support:

    @dataclass(frozen=True)
    class RegressionPromotionGates:
        max_mae_tolerance: float = 0.0   # challenger MAE must be <= champion MAE
        min_r2: float = 0.0              # must beat mean baseline
        min_coverage: float = 0.85       # 90% interval coverage floor
        max_coverage: float = 0.97       # 90% interval coverage ceiling

Design approach: generalize `compare_models()` to accept a `mode`
parameter ("classification" or "regression") or auto-detect based on
which metrics are present in the metadata. The function selects the
appropriate gate system.

Single `ComparisonResult` return type — same structure for both.

Existing `PromotionGates` (Brier/ECE/AUC) stays for classification.
New `RegressionPromotionGates` added for regression.

    def compare_models(
        champion: ModelMetadata | PropModelMetadata,
        challenger: ModelMetadata | PropModelMetadata,
        *,
        mode: str = "classification",
        classification_gates: PromotionGates = PromotionGates(),
        regression_gates: RegressionPromotionGates = RegressionPromotionGates(),
    ) -> ComparisonResult

`format_comparison()` also generalized to show the right metrics
and gate labels depending on mode.

| Status | Step |
|--------|------|
| Not started | B1 |

#### B2: Champion selection utility function

New function (in `evaluation/champion.py` or `evaluation/prop_champion.py`):

    def select_prop_champion(
        results: list[PropModelMetadata],
        gates: RegressionPromotionGates = RegressionPromotionGates(),
    ) -> tuple[PropModelMetadata, str]

Logic:
1. Filter to models passing all guardrails (R² > 0, coverage in range)
2. Among eligible models, select lowest MAE
3. If no model passes guardrails, return ElasticNet as fallback
4. Return `(champion_metadata, comparison_table_string)`

| Status | Step |
|--------|------|
| Not started | B2 (depends on B1) |

### Phase C: CLI Integration — Steps

#### C1: Add --model-type option to evaluate command

    gridiron props evaluate --model qb_pass_yards --model-type xgboost

If `--model-type` not specified, uses champion model type (default
elasticnet until champion selection is run).

| Status | Step |
|--------|------|
| Not started | C1 |

#### C2: Add champion command

    gridiron props champion --model qb_pass_yards
    gridiron props champion --model all

Trains all 3 model types for the specified stat family (or all 5
families), shows comparison table, reports champion:

    🏈 Champion/Challenger: qb_pass_yards

      Model Type     MAE    RMSE      R²  Coverage  Status
      ─────────────────────────────────────────────────────
      elasticnet    58.0    72.6   0.071    93.8%   Eligible
      random_forest 52.1    66.3   0.142    91.2%   ★ CHAMPION
      xgboost       53.4    67.8   0.128    90.5%   Eligible

| Status | Step |
|--------|------|
| Not started | C2 (depends on A1–A4, B1–B2) |

#### C3: Train all 15 models and validate

Run `gridiron props champion --model all`. Paste results into this
document and CHANGELOG.md. Verify:
- All 15 models train without error
- At least one non-ElasticNet model wins per stat family
- Coverage stays in [85%, 97%] for all models
- R² improves for QB pass (currently 0.071)

| Status | Step |
|--------|------|
| Not started | C3 (depends on C2) |

### Phase A/B/C: Test Plan

**File:** `tests/unit/models/test_prop_champion.py`

| Test | What It Validates |
|------|-------------------|
| `test_model_type_enum_values` | 3 values: elasticnet, random_forest, xgboost |
| `test_create_model_elasticnet` | Returns (ElasticNet, StandardScaler) |
| `test_create_model_rf` | Returns (RandomForestRegressor, None) |
| `test_create_model_xgb` | Returns (XGBRegressor, None) |
| `test_param_grid_elasticnet` | 25 combos |
| `test_param_grid_rf` | 36 combos |
| `test_param_grid_xgb` | 54 combos |
| `test_clip_ranges_from_spec` | Predictions clipped to spec.clip_lo/clip_hi |

**File:** `tests/unit/evaluation/test_champion.py` (extend existing)

| Test | What It Validates |
|------|-------------------|
| `test_regression_gates_default_values` | max_mae_tolerance=0.0, min_r2=0.0, coverage=[0.85, 0.97] |
| `test_compare_models_regression_mode` | Correct gates applied in regression mode |
| `test_regression_champion_selects_lowest_mae` | Given 3 results, picks lowest MAE |
| `test_regression_guardrail_r2` | R² <= 0 model excluded |
| `test_regression_guardrail_coverage_low` | Coverage < 85% excluded |
| `test_regression_guardrail_coverage_high` | Coverage > 97% excluded |
| `test_regression_fallback_elasticnet` | If no model passes, ElasticNet wins |
| `test_format_comparison_regression` | Output contains MAE, R², Coverage labels |
| `test_classification_mode_unchanged` | Existing Brier/ECE/AUC gates still work |

**Updates to existing test files:**
- `test_qb_pass_yards.py` etc.: Remove tests for `_fit()` / `_predict()` (now in base). Verify spec has clip_lo/clip_hi.
- `test_prop_base.py` (new or extend): Model type factory tests.

### Phase A/B/C: Files Inventory

| Action | File |
|--------|------|
| **Modify** | `models/prop_prediction/base.py` — PropModelType enum, factory, HP grids, train(model_type=) |
| **Modify** | `models/prop_prediction/qb_pass_yards.py` — spec-only, remove _fit/_predict/_scaler/_model |
| **Modify** | `models/prop_prediction/qb_rush_yards.py` — same |
| **Modify** | `models/prop_prediction/rb_rush_yards.py` — same |
| **Modify** | `models/prop_prediction/wr_rec_yards.py` — same |
| **Modify** | `models/prop_prediction/te_rec_yards.py` — same |
| **Modify** | `evaluation/champion.py` — RegressionPromotionGates, generalized compare_models(), select_prop_champion() |
| **Modify** | `cli/props.py` — --model-type option, champion command |
| **Create** | `tests/unit/models/test_prop_champion.py` |
| **Modify** | `tests/unit/evaluation/test_champion.py` — regression gate tests |
| **Modify** | `tests/unit/models/test_qb_pass_yards.py` — remove _fit tests, add clip spec |
| **Modify** | `tests/unit/models/test_qb_rush_yards.py` — same |
| **Modify** | `tests/unit/models/test_rb_rush_yards.py` — same |
| **Modify** | `tests/unit/models/test_wr_rec_yards.py` — same |
| **Modify** | `tests/unit/models/test_te_rec_yards.py` — same |

---

### Workstream 2: Game Model Refactor

**Goal:** Refactor `models/game_prediction/` to mirror the `models/prop_prediction/` architecture established in Workstream 1. Eliminate the dynamic-class variant factories, unify the metadata schema, fold the `total` regression model into the same framework as `win_prob`, and align field names so the codebase has one coherent model structure.

**Status:** **Done** (closed 2026-06-19). Details in CHANGELOG.md.

#### Final Results

D5 baseline verification against pre-WS2 Brier targets:

| Model | WS1 Baseline | D5 Actual | Delta | Verdict |
|---|---|---|---|---|
| `win_prob_logistic` | 0.225 | 0.22153 | -0.003 | Improved ✅ |
| `win_prob_random_forest` | 0.220 | 0.21191 | -0.008 | Improved ✅ |
| `win_prob_xgboost` | 0.218 | 0.21598 | -0.002 | Improved ✅ |
| `win_prob_elo` | n/a | 0.23143 | — | Baseline established |
| `total_random_forest` | n/a | MAE 10.24 / RMSE 13.12 / R² 0.056 | — | Baseline established |
| `total_xgboost` | n/a | MAE 10.35 / RMSE 13.19 / R² 0.046 | — | Baseline established |

All three classification Brier baselines exceeded their pre-WS2 targets. Total models established their first formal baselines.

#### Architecture delivered

- `BaseModelMetadata` shared metadata base with `GameModelMetadata` and `PropModelMetadata` subclasses.
- `GamesTrainer` ABC with `WinProbTrainer` and `TotalTrainer` spec-only subclasses.
- `GamesPredictor` base with five thin composite-key subclasses (one per `(model_name, model_type)` pair).
- `WinProbEloPredictor` consolidates the three legacy Elo variants (`elo_v1` / `v2` / `v3`) into one composite-key registration.
- Composite registry keys: `win_prob_logistic`, `win_prob_random_forest`, `win_prob_xgboost`, `win_prob_elo`, `total_random_forest`, `total_xgboost`.
- Nested artifact path scheme: `data/models/{model_name}/{model_type}/`.
- Prediction archive schema migrated from `model_version` (single column) to `model_name` + `model_type` (two columns).
- All classification metrics promoted from `parameters` dict to first-class fields on `GameModelMetadata`.

#### Phased delivery (closed)

| Phase | Description | Status |
|---|---|---|
| D1a | Shared metadata + subclasses | Done |
| D1b | `ArtifactStore` generalization + nested paths | Done |
| D2a | `GamesTrainer` + spec subclasses (alongside legacy) | Done |
| D2b.1 | `GamesPredictor` + composite-key registrations | Done |
| D2b.2 | Archive schema migration + all callers | Done |
| D2b.3 | Legacy code deleted | Done |
| Elo-WS2 | Elo migrated to `win_prob_elo` composite key | Done |
| D3 | Metrics promoted to first-class fields | Done |
| D4 | All 5 game models retrained successfully | Done |
| D5 | Brier baselines verified; CHANGELOG updated | Done |

#### Bugs surfaced during D4+D5 verification (fixed)

- Season string conversion bug in `_features.py` / `total.py` — modeling file already stores seasons as `"YYYY-YYYY"` strings; the incorrect int→str conversion was crashing every training run.
- `GamesTrainer._run_hp_search` was calling `.set_params(**sampled)` on `CalibratedClassifierCV`, which doesn't expose RF hyperparameters. Fixed via `_apply_params()` helper that uses the `estimator__` prefix when wrapping.
- `_parse_composite_key` (in three files) was using `str.partition("_")` to split keys, which broke for any model_name containing underscores (i.e. all `win_prob_*`). Fixed via prefix matching against `get_known_model_names()` from `predictor.py` as single source of truth.
- `evaluation/select.py::collect_model_metrics` was failing on regression models whose `away_win_prob` column is NaN-only. Added explicit skip.
- Scaler was being saved alongside the logistic model but never applied at predict time. Logistic predictions were severely overconfident (std 0.485 vs ~0.15 expected, archive Brier 0.355 vs 0.222 training Brier). Fixed by loading and applying scaler in all four classification + regression predict methods plus `_maybe_predict_totals`.

#### Known follow-ups (post-WS2 scope)

| Item | Notes | When |
|---|---|---|
| Logistic feature names lost across joblib round-trip | sklearn `UserWarning`, cosmetic only; predict path uses `.values` arrays. | Code review (W4) |
| `CalibratedClassifierCV` inner CV uses `StratifiedKFold(shuffle=False)` | Not strictly time-ordered. Matches WS1 baseline approach. Investigate switching to `TimeSeriesSplit`; expect baseline shift. | Dedicated calibration workstream |
| Logistic `n_games` (7,008) vs tree models (5,705) | Different feature_set (`combined_70` vs `expanded_107`) → different NaN drop patterns. Investigate which expanded columns drive unnecessary row drops. | Code review (W4) |
| Elo sigma + margin_std values | `_MODEL_SIGMAS[("win_prob", "elo")] = 13.60` and `_MODEL_MARGIN_STDS[("win_prob", "elo")] = 13.89` carry pre-WS2 calibration values. Refresh after next archive regeneration. | Sigma calibration follow-up |
| No end-to-end fit-load-predict integration test | The scaler-not-applied bug only surfaced in production-data verification. Worth adding to prevent regression. | Workstream 3 (E5) |

---

## Workstream 3: Integration & E2E Tests

**Goal:** Validate the full prop pipeline end-to-end and add cross-module
tests that catch breakages unit tests miss.

**Why:** A column rename in `rolling.py` could silently break the
builder → trainer chain. Integration tests close this gap.

**Depends on:** Workstreams 1 and 2 (tests should cover the unified
architecture including champion commands).

### Locked Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Test tier definitions | Unit (<1s, synthetic), Integration (5-30s, multi-module), E2E (30s-5min, full workflow) | Matches existing three-tier pyramid |
| 2 | Test marking | Auto-markers by directory (already configured) | No changes needed |
| 3 | Integration test data | Synthetic DataFrames via fixture factories | Fast, deterministic, no external deps |
| 4 | E2E test data | Synthetic data written to disk (tmp_path) | Tests full I/O path |
| 5 | CLI testing approach | Mock data layer for integration; disk data for E2E | Separation of concerns |
| 6 | Fixture data scale | 2 seasons × 10 weeks × 4 teams × 4 players/team = ~320 rows | Fast but exercises all feature types |

### Fixture Data Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Seasons | 2 (e.g., 2023, 2024) | Enough for holdout split |
| Weeks per season | 10 | Enough for L6 rolling features |
| Teams | 4 (KC, LV, BUF, MIA) | Enough for matchup variation |
| Players per team | 4 (1 QB, 1 RB, 1 WR, 1 TE) | One per position |
| Total rows | ~320 | Fast but realistic |

Key requirements for fixture data:
- Enough rows for rolling features to compute without all-NaN (≥7 weeks)
- Team variation for matchup features (≥4 teams)
- Team totals > 0 for usage features
- Game IDs that match between player logs and games CSV
- All 4 positions represented

### Implementation Steps

#### E1: Player DataFrame factories

Add to `tests/fixtures/dataframes.py`:

- `make_player_game_logs(n_players, n_weeks, positions)` — returns DataFrame
  matching cleaned `player_game_logs.parquet` schema
- `make_games_csv(n_weeks)` — returns DataFrame matching
  `NFL_wk_by_wk_cleaned.csv` schema
- `make_prop_predictions(n, model_name)` — returns DataFrame with
  predicted_mean + enrichment columns

| Status | Step |
|--------|------|
| Not started | E1 |

#### E2: Prop repo fixture

Add to `tests/fixtures/repos.py`:

- `make_prop_repo(tmp_path)` — writes both files to disk:
  - `data/cleaned/player_game_logs.parquet`
  - `data/cleaned/NFL_wk_by_wk_cleaned.csv`

This fixture is reused by all integration and E2E tests.

| Status | Step |
|--------|------|
| Not started | E2 (depends on E1) |

#### E3: Integration tests — feature pipeline

**File:** `tests/integration/test_prop_feature_pipeline.py`

| Test | What It Validates |
|------|-------------------|
| `test_builder_produces_all_feature_types` | Rolling + matchup + usage + context columns all present |
| `test_builder_position_filter_correct` | QB filter → only QB rows; RB → only RB rows |
| `test_builder_no_lookahead` | Week N features don't use week N data (spot-check shift) |
| `test_builder_no_row_duplication` | No duplicate (player_id, game_id) in output |
| `test_builder_nan_rates_reasonable` | NaN rates per feature < 50% for filtered position |
| `test_feature_columns_match_prop_feature_cols` | Columns match PROP_FEATURE_COLS |
| `test_builder_with_missing_games` | Player logs with no matching game → NaN context, not crash |

| Status | Step |
|--------|------|
| Not started | E3 (depends on E2) |

#### E4: Integration tests — CLI

**File:** `tests/integration/test_props_cli.py`

| Test | What It Validates |
|------|-------------------|
| `test_evaluate_runs_without_error` | `gridiron props evaluate --model qb_pass_yards` exits 0 |
| `test_evaluate_output_contains_mae` | Output contains "MAE" |
| `test_backfill_creates_archive` | After backfill, archive file exists |
| `test_projections_output_has_table` | Output contains player names and predicted values |
| `test_champion_runs_all_types` | Champion command trains 3 model types |
| `test_unknown_model_exits_error` | `--model fake_model` exits with error message |

CLI tests use `typer.testing.CliRunner` with mocked data layer.

| Status | Step |
|--------|------|
| Not started | E4 (depends on E2, W1 C2) |

#### E5: E2E tests

**File:** `tests/e2e/test_prop_pipeline.py`

| Test | What It Validates |
|------|-------------------|
| `test_full_pipeline_data_to_archive` | Synthetic data → clean → features → train → predict → enrich → archive |
| `test_full_pipeline_multiple_models` | Train QB pass + RB rush, verify both produce valid results |
| `test_evaluate_report_structure` | Full eval report has accuracy, bias, coverage sections |

E2E tests write synthetic data to disk via `make_prop_repo()`.

| Status | Step |
|--------|------|
| Not started | E5 (depends on E2) |

#### E6: Full suite validation

Run all tiers, verify green:

    uv run pytest -v

| Status | Step |
|--------|------|
| Not started | E6 (depends on E3–E5) |

### Integration & E2E: Files Inventory

| Action | File |
|--------|------|
| **Modify** | `tests/fixtures/dataframes.py` — add player/game/prop factories |
| **Modify** | `tests/fixtures/repos.py` — add make_prop_repo() |
| **Create** | `tests/integration/test_prop_feature_pipeline.py` |
| **Create** | `tests/integration/test_props_cli.py` |
| **Create** | `tests/e2e/test_prop_pipeline.py` |

---

## Dependency Graph

    Workstream 1: Champion/Challenger for Props
        │
        ├── A1: PropModelType enum + factory in base.py
        │
        ├── A2: clip_lo/clip_hi in PropModelSpec
        │
        ├── A3: Strip _fit/_predict from 5 subclasses
        │       (depends on A1)
        │
        ├── A4: model_type in PropModelMetadata
        │
        ├── B1: Generalize evaluation/champion.py
        │       (independent of A*)
        │
        ├── B2: select_prop_champion() utility
        │       (depends on B1)
        │
        ├── C1: CLI --model-type option
        │       (depends on A1)
        │
        ├── C2: CLI champion command
        │       (depends on A1–A4, B1–B2)
        │
        └── C3: Train all 15 models, validate
                (depends on C2)

    Workstream 2: Game Model Refactor
        │
        ├── D1: Assess shared metadata
        │       (depends on W1 A4)
        │
        ├── D2: Replace _make_tree_variant()
        │       (depends on D1)
        │
        ├── D3: Wire into generalized champion.py
        │       (depends on W1 B1, D2)
        │
        ├── D4: Run full game model test suite
        │       (depends on D2, D3)
        │
        └── D5: Retrain champions, verify metrics
                (depends on D4)

    Workstream 3: Integration & E2E Tests
        │
        ├── E1: Player DataFrame factories
        │
        ├── E2: Prop repo fixture
        │       (depends on E1)
        │
        ├── E3: Feature pipeline integration tests
        │       (depends on E2)
        │
        ├── E4: CLI integration tests
        │       (depends on E2, W1 C2)
        │
        ├── E5: E2E tests
        │       (depends on E2)
        │
        └── E6: Full suite validation
                (depends on E3–E5)

    Order: W1 complete → W2 complete → W3

---

## Future Workstreams (High-Level)

### Workstream 4: Deep Code Review + Test Suite Review

Two-part review session:
1. **Code review:** Pattern consistency across game + prop models, CLI
   output formatting parity, naming conventions, dead code, import
   hygiene, docstring completeness.
2. **Test suite review:** Edge case coverage audit, fixture quality,
   test isolation, missing negative tests, coverage ratchet assessment.

Detailed plan created when workstreams 1–3 complete.

### Workstream 5: Scenario / "What If" Engine (W4.5)

See ROADMAP.md W4.5 for full description. Now unblocked by W4 completion.
Five phases: player impact quantification → team adjustment → usage
redistribution → conditional re-forecasting → CLI interface.

### Workstream 6: API & Frontend (W8 + W9)

FastAPI serving layer + React/Next.js frontend consuming it. See
ROADMAP.md W8/W9.

### Workstream 7: All External Odds

DraftKings prop odds ingest (E2 from W4), historical odds data, multi-book
line shopping (W7). Requires odds source decision (ROADMAP.md §5.2).

### Workstream 8: Evaluate Remaining Work

Assessment of what's left: model ensemble (W12), real-time/live (W10),
feature engineering backlog, NaN research, architectural debt.

---

## Architectural Debt / Housekeeping

| Item | Notes | When |
|------|-------|------|
| `_DEFAULT_TOTAL_STD` hardcoded in `cli/edges.py` | Currently 13.17 (total model holdout RMSE). Wire into model metadata. | Code review (W4) |
| base.py line count | Currently ~550 lines. Will grow with factory methods. Monitor and split if >700. | Code review (W4) |
| Backport factory pattern to game models | Covered by Workstream 2 (D2). | W2 |
| `PROP_FEATURE_COLS` vs `_EXPANDED_FEATURES` naming | Ensure consistent naming convention across game/prop feature lists. | Code review (W4) |

---

## NaN Research Backlog (Deferred)

Current strategy: drop rows with NaN, with `# TODO(nan)` markers at each
drop site. Future investigation items:

- Bayesian shrinkage priors for early-season rolling stats
- Seasonal carry-forward (use last season's final L6 as prior for week 1)
- Multiple imputation for missing game context features
- Missing-indicator pattern (add `feature_is_missing` binary columns)
- Rookie cold-start with draft capital / combine data

Best done after model architecture is stable (post workstream 2).

---

## Changelog

| Date | Change |
|------|--------|
## Changelog

| Date | Change |
|------|--------|
| 2026-06-19 | **Workstream 2 closed.** Game model refactor complete. All 5 WS2 game models retrained on new infrastructure. D5 Brier baselines verified — all three classification champions improved vs pre-WS2 (logistic 0.222 / RF 0.212 / XGB 0.216). Five major architectural improvements shipped: composite-key registry, nested artifact paths, unified GamesTrainer/GamesPredictor, archive schema migration, first-class metric fields. Workstream 3 (Integration & E2E Tests) is now active. |
| 2026-06-10 | **Full rewrite.** New priority order: champion/challenger → game model refactor → integration tests → code review → scenario engine → API/frontend → external odds → evaluate. Three workstreams planned in detail (champion/challenger, game model refactor, integration tests). All design decisions locked. Generalized champion.py (option B) chosen for unified classification + regression gates. |
